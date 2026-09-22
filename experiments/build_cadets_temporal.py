"""Build leakage-free CADETS TemporalData directly from raw CDM JSON archives.

The default path is the publication path: every supported event is retained,
15-minute windows are fused causally, and days are split chronologically.  A
pilot cap is available only for pipeline validation; pilot outputs are marked
in metadata and must not be reported as full-dataset results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytz
import torch
from torch_geometric.data import TemporalData


E3_ARCHIVES = [
    Path(r"D:\download\E3\cadets-20260629T141704Z-3-002\cadets\ta1-cadets-e3-official.json.tar.gz"),
    Path(r"D:\download\E3\cadets-20260629T141704Z-3-001\cadets\ta1-cadets-e3-official-1.json.tar.gz"),
    Path(r"D:\download\E3\cadets-20260629T141704Z-3-001\cadets\ta1-cadets-e3-official-2.json.tar.gz"),
]
E3_GT = Path(
    r"D:\download\E3\ground-truth-usenix-sec-2025\ground-truth-usenix-sec-2025"
    r"\darpa\E3-CADETS"
)
RELATIONS = [
    "EVENT_CONNECT", "EVENT_EXECUTE", "EVENT_OPEN", "EVENT_READ", "EVENT_RECVFROM",
    "EVENT_RECVMSG", "EVENT_SENDMSG", "EVENT_SENDTO", "EVENT_WRITE", "EVENT_CLONE",
]
REVERSED_RELATIONS = {
    "EVENT_EXECUTE", "EVENT_LSEEK", "EVENT_MMAP", "EVENT_OPEN", "EVENT_ACCEPT",
    "EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_READ_SOCKET_PARAMS",
    "EVENT_CHECK_FILE_ATTRIBUTES",
}
STRICT_SPLITS = {
    2: "train", 3: "train", 4: "train", 5: "val",
    6: "test", 7: "test", 8: "test", 9: "test", 10: "test",
    11: "test", 12: "test", 13: "test",
}
ATTACK_WINDOWS = [
    ("2018-04-06 11:20:00", "2018-04-06 12:09:00"),
    ("2018-04-12 13:59:00", "2018-04-12 14:39:00"),
    ("2018-04-13 09:03:00", "2018-04-13 09:16:00"),
]

UUID_PATTERN = re.compile(rb'"uuid":"([0-9A-Fa-f-]{36})"')
SUBJECT_PATTERN = re.compile(rb'"subject":\{"com\.bbn\.tc\.schema\.avro\.cdm18\.UUID":"([0-9A-Fa-f-]{36})"')
OBJECT_PATTERN = re.compile(rb'"predicateObject":\{"com\.bbn\.tc\.schema\.avro\.cdm18\.UUID":"([0-9A-Fa-f-]{36})"')
EVENT_TYPE_PATTERN = re.compile(rb'"type":"(EVENT_[A-Z0-9_]+)"')
TIME_PATTERN = re.compile(rb'"timestampNanos":([0-9]+)')
EVENT_UUID_PATTERN = re.compile(
    rb'cdm18\.Event":\{"uuid":"([0-9A-Fa-f-]{36})"'
)
EXEC_PATTERN = re.compile(rb'"exec":"([^"\\]*)"')
PATH_PATTERN = re.compile(rb'"predicateObjectPath":\{"string":"([^"\\]*)"')
ADDRESS_PATTERN = re.compile(rb'"(?:localAddress|remoteAddress)":"([^"\\]*)"')
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]{2,}")
EASTERN = pytz.timezone("US/Eastern")


def load_malicious_uuids(gt_dir: Path) -> set[str]:
    malicious = set()
    for path in sorted(gt_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for row in csv.reader(file):
                if row:
                    malicious.add(row[0].strip().upper())
    return malicious


def avro_type(raw: bytes) -> int | None:
    if b"cdm18.Subject" in raw:
        return 0
    if b"cdm18.NetFlowObject" in raw or b"FILE_OBJECT_UNIX_SOCKET" in raw:
        return 2
    if b"cdm18.FileObject" in raw:
        return 1
    return None


def one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def semantic_vector(tokens: list[str], size: int) -> list[float]:
    vector = [0.0] * size
    for token in sorted(set(tokens)):
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else vector


def text_tokens(prefix: str, raw_value: bytes) -> list[str]:
    text = raw_value.decode("utf-8", errors="ignore").lower()
    return [f"{prefix}:{token}" for token in TOKEN_PATTERN.findall(text)[:16]]


class TemporalWriter:
    def __init__(self, out_dir: Path, window_minutes: int, embedding_dim: int):
        self.out_dir = out_dir
        self.window_ns = int(window_minutes * 60 * 1_000_000_000)
        self.embedding_dim = embedding_dim
        self.current_key = None
        self.events = []
        self.last_relation = {}
        self.file_counts = Counter()
        self.edge_counts = Counter()

    def add(self, split: str, day: int, event: tuple) -> None:
        timestamp = event[2]
        window_start = timestamp - (timestamp % self.window_ns)
        key = (split, day, window_start)
        if self.current_key is not None and key != self.current_key:
            self.flush()
        self.current_key = key
        src, dst, _, relation, _, _, _, _, _ = event
        pair = (src, dst)
        if self.last_relation.get(pair) == relation:
            return
        self.last_relation[pair] = relation
        self.events.append(event)

    def flush(self) -> None:
        if not self.events or self.current_key is None:
            self.events = []
            self.last_relation = {}
            return
        split, day, window_start = self.current_key
        self.events.sort(key=lambda item: (item[2], item[4]))
        src = torch.tensor([item[0] for item in self.events], dtype=torch.long)
        dst = torch.tensor([item[1] for item in self.events], dtype=torch.long)
        timestamp = torch.tensor([item[2] for item in self.events], dtype=torch.long)
        relation = torch.tensor([item[3] for item in self.events], dtype=torch.long)
        src_types = torch.tensor([item[5] for item in self.events], dtype=torch.long)
        dst_types = torch.tensor([item[6] for item in self.events], dtype=torch.long)
        fields = [torch.nn.functional.one_hot(src_types, 3)]
        if self.embedding_dim:
            fields.append(torch.tensor([item[7] for item in self.events], dtype=torch.float32))
        fields.append(torch.nn.functional.one_hot(relation, len(RELATIONS)))
        fields.append(torch.nn.functional.one_hot(dst_types, 3))
        if self.embedding_dim:
            fields.append(torch.tensor([item[8] for item in self.events], dtype=torch.float32))
        msg = torch.cat(fields, dim=1).float()
        index = self.file_counts[split]
        path = self.out_dir / split / f"graph_{index:05d}.TemporalData"
        torch.save(TemporalData(src=src, dst=dst, t=timestamp, msg=msg, edge_type=relation), path)
        self.file_counts[split] += 1
        self.edge_counts[split] += len(self.events)
        self.events = []
        self.last_relation = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="E3-CADETS", choices=["E3-CADETS"])
    parser.add_argument("--out-dir", default="artifacts/raw_temporal/E3-CADETS/edge_embeds")
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--feature-mode", choices=["only_type", "hashed_semantic"], default="hashed_semantic")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--include-days", nargs="*", type=int, default=[])
    parser.add_argument(
        "--pilot-events-per-day", type=int, default=0,
        help="Cap non-attack supported events per day; nonzero outputs are pilot-only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.rglob("*.TemporalData")):
        raise FileExistsError(f"Refusing to overwrite existing TemporalData in {out_dir}")
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    include_days = set(args.include_days) if args.include_days else set(STRICT_SPLITS)
    malicious = load_malicious_uuids(E3_GT)
    node_types = {}
    node_tokens = {}
    node_ids = {}
    id_to_uuid = {}
    counters = Counter()
    kept_by_day = Counter()
    embedding_dim = args.embedding_dim if args.feature_mode == "hashed_semantic" else 0
    writer = TemporalWriter(out_dir, args.window_minutes, embedding_dim)

    def node_id(uuid: str) -> int:
        if uuid not in node_ids:
            index = len(node_ids)
            node_ids[uuid] = index
            id_to_uuid[index] = uuid
        return node_ids[uuid]

    stop_after_day = max(include_days) if args.include_days else None
    reached_end = False
    for archive in E3_ARCHIVES:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        print(f"Scanning {archive}", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    continue
                for raw in stream:
                    counters["raw_records"] += 1
                    if counters["raw_records"] % 1_000_000 == 0:
                        print(
                            f"  records={counters['raw_records']:,} "
                            f"supported={counters['supported_events']:,}",
                            flush=True,
                        )
                    if b"cdm18.Event" not in raw:
                        kind = avro_type(raw)
                        match = UUID_PATTERN.search(raw)
                        if kind is not None and match:
                            uuid = match.group(1).decode("ascii").upper()
                            node_types[uuid] = kind
                            semantic_tokens = [f"node_type:{kind}"]
                            for address in ADDRESS_PATTERN.findall(raw):
                                semantic_tokens.extend(text_tokens("address", address))
                            exec_value = EXEC_PATTERN.search(raw)
                            if exec_value:
                                semantic_tokens.extend(text_tokens("exec", exec_value.group(1)))
                            node_tokens[uuid] = semantic_tokens
                            node_id(uuid)
                        continue

                    relation_match = EVENT_TYPE_PATTERN.search(raw)
                    subject_match = SUBJECT_PATTERN.search(raw)
                    object_match = OBJECT_PATTERN.search(raw)
                    time_match = TIME_PATTERN.search(raw)
                    event_match = EVENT_UUID_PATTERN.search(raw)
                    if not all([relation_match, subject_match, object_match, time_match, event_match]):
                        counters["malformed_events"] += 1
                        continue
                    relation_name = relation_match.group(1).decode("ascii")
                    if relation_name not in RELATIONS:
                        counters["unsupported_relation"] += 1
                        continue
                    subject_uuid = subject_match.group(1).decode("ascii").upper()
                    object_uuid = object_match.group(1).decode("ascii").upper()
                    timestamp = int(time_match.group(1))
                    local_dt = datetime.fromtimestamp(timestamp / 1e9, EASTERN)
                    day = local_dt.day
                    if stop_after_day is not None and local_dt.year == 2018 and local_dt.month == 4 and day > stop_after_day:
                        reached_end = True
                        break
                    if day not in include_days or day not in STRICT_SPLITS:
                        continue
                    subject_type = node_types.get(subject_uuid, 0)
                    object_type = node_types.get(object_uuid)
                    if object_type is None:
                        counters["unknown_object_type"] += 1
                        continue
                    in_attack_window = any(
                        start <= local_dt.strftime("%Y-%m-%d %H:%M:%S") <= end
                        for start, end in ATTACK_WINDOWS
                    )
                    pilot_cap = args.pilot_events_per_day
                    if pilot_cap and kept_by_day[day] >= pilot_cap and not in_attack_window:
                        counters["pilot_dropped"] += 1
                        continue
                    kept_by_day[day] += int(not in_attack_window)
                    src, dst = node_id(subject_uuid), node_id(object_uuid)
                    src_type, dst_type = subject_type, object_type
                    if relation_name in REVERSED_RELATIONS:
                        src, dst = dst, src
                        src_type, dst_type = dst_type, src_type
                    relation = RELATIONS.index(relation_name)
                    event_uuid = event_match.group(1).decode("ascii").upper()
                    src_semantics = list(node_tokens.get(subject_uuid, ["node_type:0"]))
                    dst_semantics = list(node_tokens.get(object_uuid, [f"node_type:{object_type}"]))
                    exec_value = EXEC_PATTERN.search(raw)
                    if exec_value:
                        src_semantics.extend(text_tokens("exec", exec_value.group(1)))
                    path_value = PATH_PATTERN.search(raw)
                    if path_value:
                        dst_semantics.extend(text_tokens("path", path_value.group(1)))
                    if relation_name in REVERSED_RELATIONS:
                        src_semantics, dst_semantics = dst_semantics, src_semantics
                    src_embedding = semantic_vector(src_semantics, embedding_dim) if embedding_dim else []
                    dst_embedding = semantic_vector(dst_semantics, embedding_dim) if embedding_dim else []
                    writer.add(
                        STRICT_SPLITS[day], day,
                        (src, dst, timestamp, relation, event_uuid, src_type, dst_type,
                         src_embedding, dst_embedding),
                    )
                    counters["supported_events"] += 1
                if reached_end:
                    break
            if reached_end:
                break
        if reached_end:
            break
    writer.flush()

    with (out_dir / "nodeid2msg.json").open("w", encoding="utf-8") as file:
        json.dump({str(key): value for key, value in id_to_uuid.items()}, file)
    gt_path = out_dir / "ground_truth_nodes.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=["node_id", "label", "uuid"])
        csv_writer.writeheader()
        for uuid in sorted(malicious & set(node_ids)):
            csv_writer.writerow({"node_id": node_ids[uuid], "label": 1, "uuid": uuid})

    metadata = {
        "dataset": args.dataset,
        "protocol": "strict_chronological",
        "status": "pilot" if args.pilot_events_per_day else "publication_full",
        "label_leakage_control": "ground truth is never read while constructing model features",
        "splits_by_eastern_day": {str(day): split for day, split in STRICT_SPLITS.items()},
        "included_days": sorted(include_days),
        "relations": RELATIONS,
        "node_type_order": ["subject", "file", "netflow"],
        "feature_mode": args.feature_mode,
        "embedding_dim": embedding_dim,
        "window_minutes": args.window_minutes,
        "pilot_events_per_day": args.pilot_events_per_day,
        "raw_archives": [str(path) for path in E3_ARCHIVES],
        "raw_counters": dict(counters),
        "temporal_files": dict(writer.file_counts),
        "fused_edges": dict(writer.edge_counts),
        "nodes": len(node_ids),
        "malicious_nodes_in_stream": len(malicious & set(node_ids)),
        "ground_truth": str(gt_path),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
