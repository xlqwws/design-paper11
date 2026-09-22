"""Build E5-CADETS TemporalData from official gzip-wrapped CDM20 Avro OCF."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytz

from build_cadets_temporal import (
    RELATIONS, REVERSED_RELATIONS, TemporalWriter, semantic_vector,
)
from cdm_avro import iter_cdm_events, iter_cdm_records


EASTERN = pytz.timezone("US/Eastern")
E5_RAW = Path(r"D:\download\E5")
E5_GT = Path(
    r"D:\download\E3\ground-truth-usenix-sec-2025\ground-truth-usenix-sec-2025"
    r"\darpa\E5-CADETS"
)
STRICT_SPLITS = {
    8: "train", 9: "train", 10: "train", 11: "train", 12: "val",
    13: "test", 14: "test", 15: "test", 16: "test", 17: "test",
}
ATTACK_WINDOWS = [
    ("2019-05-16 09:31:00", "2019-05-16 10:12:00"),
    ("2019-05-17 10:15:00", "2019-05-17 15:33:00"),
]


def uuid_text(value) -> str:
    if isinstance(value, bytes) and len(value) == 16:
        return str(uuid.UUID(bytes=value)).upper()
    if isinstance(value, str):
        try:
            return str(uuid.UUID(value)).upper()
        except ValueError:
            return ""
    return ""


def load_ground_truth(path: Path) -> set[str]:
    output = set()
    for csv_path in path.glob("*.csv"):
        with csv_path.open("r", encoding="utf-8", errors="replace") as file:
            for row in csv.reader(file):
                if row:
                    output.add(row[0].upper())
    return output


def entity_info(kind: str, record: dict):
    if kind == "Subject":
        node_type = 0
    elif kind == "NetFlowObject" or (
        kind == "FileObject" and "UNIX_SOCKET" in str(record.get("type", ""))
    ):
        node_type = 2
    elif kind == "FileObject":
        node_type = 1
    else:
        return None, []
    semantics = [f"node_type:{node_type}"]
    for key in ["cmdLine", "localAddress", "remoteAddress", "localPort", "remotePort"]:
        value = record.get(key)
        if value not in {None, ""}:
            semantics.append(f"{key.lower()}:{str(value).lower()[:160]}")
    return node_type, semantics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(E5_RAW))
    parser.add_argument("--out-dir", default="artifacts/raw_temporal_pilot/E5-CADETS/edge_embeds")
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--feature-mode", choices=["only_type", "hashed_semantic"], default="hashed_semantic")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--pilot-events-per-day", type=int, default=10000)
    parser.add_argument("--pilot-attack-events-per-window", type=int, default=20000)
    parser.add_argument("--max-records-per-shard", type=int, default=0)
    parser.add_argument("--allow-incomplete-raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.rglob("*.TemporalData")):
        raise FileExistsError(f"Refusing to overwrite existing TemporalData in {out_dir}")
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    malicious = load_ground_truth(E5_GT)
    node_types = {}
    node_tokens = {}
    node_ids = {}
    id_to_uuid = {}
    counters = Counter()
    kept_by_day = Counter()
    kept_by_attack_window = Counter()
    embedding_dim = args.embedding_dim if args.feature_mode == "hashed_semantic" else 0
    writer = TemporalWriter(out_dir, args.window_minutes, embedding_dim)

    def node_id(value: str) -> int:
        if value not in node_ids:
            index = len(node_ids)
            node_ids[value] = index
            id_to_uuid[index] = value
        return node_ids[value]

    shards = sorted(
        Path(args.raw_dir).glob("*.gz"), key=lambda path: int(path.name.split(".")[-2])
    )
    manifest = Path(args.raw_dir) / "bins.md5sum"
    expected_names = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                expected_names.append(Path(parts[-1]).name)
    present_names = {path.name for path in shards}
    missing_names = sorted(set(expected_names) - present_names)
    if missing_names and not args.allow_incomplete_raw:
        raise FileNotFoundError(
            f"Incomplete E5 raw corpus: found {len(shards)} files but manifest lists "
            f"{len(set(expected_names))}; {len(missing_names)} are missing. "
            "Use --allow-incomplete-raw only for diagnostic scans."
        )
    for shard in shards:
        print(f"Scanning {shard}", flush=True)
        records = (
            ({"datum": event} for event in iter_cdm_events(shard, args.max_records_per_shard))
            if args.feature_mode == "only_type"
            else iter_cdm_records(shard, args.max_records_per_shard)
        )
        for top_record in records:
            counters["records"] += 1
            record = top_record.get("datum")
            if not isinstance(record, dict):
                continue
            kind = str(record.get("__record_name", ""))
            if kind != "Event":
                if args.feature_mode == "only_type":
                    continue
                entity_uuid = uuid_text(record.get("uuid"))
                node_type, semantics = entity_info(kind, record)
                if entity_uuid and node_type is not None:
                    node_types[entity_uuid] = node_type
                    node_tokens[entity_uuid] = semantics
                continue

            counters["events"] += 1
            relation_name = str(record.get("type", ""))
            if relation_name not in RELATIONS:
                counters["unsupported_relation"] += 1
                continue
            subject_uuid = uuid_text(record.get("subject"))
            object_uuid = uuid_text(record.get("predicateObject"))
            timestamp = record.get("timestampNanos")
            if not subject_uuid or not object_uuid or not isinstance(timestamp, int):
                counters["missing_endpoint_or_time"] += 1
                continue
            local_dt = datetime.fromtimestamp(timestamp / 1e9, EASTERN)
            day = local_dt.day
            if local_dt.year != 2019 or local_dt.month != 5 or day not in STRICT_SPLITS:
                continue
            timestamp_text = local_dt.strftime("%Y-%m-%d %H:%M:%S")
            attack_window_index = next((
                index for index, (start, end) in enumerate(ATTACK_WINDOWS)
                if start <= timestamp_text <= end
            ), None)
            in_attack_window = attack_window_index is not None
            if (
                in_attack_window and args.pilot_attack_events_per_window
                and kept_by_attack_window[attack_window_index] >= args.pilot_attack_events_per_window
            ):
                counters["pilot_attack_window_dropped"] += 1
                continue
            if (
                args.pilot_events_per_day
                and kept_by_day[day] >= args.pilot_events_per_day
                and not in_attack_window
            ):
                counters["pilot_dropped"] += 1
                continue
            kept_by_day[day] += int(not in_attack_window)
            if in_attack_window:
                kept_by_attack_window[attack_window_index] += 1

            subject_type = 0
            if args.feature_mode == "only_type":
                object_type = 2 if any(
                    token in relation_name for token in ("CONNECT", "SEND", "RECV")
                ) else 1
            else:
                object_type = node_types.get(object_uuid)
            if object_type is None:
                object_type = 1
                counters["inferred_object_type"] += 1
            src, dst = node_id(subject_uuid), node_id(object_uuid)
            src_type, dst_type = subject_type, object_type
            src_semantics = list(node_tokens.get(subject_uuid, ["node_type:0"]))
            dst_semantics = list(node_tokens.get(object_uuid, [f"node_type:{object_type}"]))
            properties = record.get("properties")
            if isinstance(properties, dict) and properties.get("exec"):
                src_semantics.append(f"exec:{str(properties['exec']).lower()[:160]}")
            path = record.get("predicateObjectPath")
            if path:
                dst_semantics.append(f"path:{str(path).lower()[:240]}")
            if relation_name in REVERSED_RELATIONS:
                src, dst = dst, src
                src_type, dst_type = dst_type, src_type
                src_semantics, dst_semantics = dst_semantics, src_semantics
            relation = RELATIONS.index(relation_name)
            event_uuid = uuid_text(record.get("uuid")) or f"{shard.name}:{counters['events']}"
            src_embedding = semantic_vector(src_semantics, embedding_dim) if embedding_dim else []
            dst_embedding = semantic_vector(dst_semantics, embedding_dim) if embedding_dim else []
            writer.add(
                STRICT_SPLITS[day], day,
                (src, dst, timestamp, relation, event_uuid, src_type, dst_type,
                 src_embedding, dst_embedding),
            )
            counters["supported_events"] += 1
            if counters["records"] % 1_000_000 == 0:
                print(f"  records={counters['records']:,} kept={counters['supported_events']:,}", flush=True)
    writer.flush()

    with (out_dir / "nodeid2msg.json").open("w", encoding="utf-8") as file:
        json.dump({str(key): value for key, value in id_to_uuid.items()}, file)
    gt_path = out_dir / "ground_truth_nodes.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=["node_id", "label", "uuid"])
        csv_writer.writeheader()
        for entity_uuid in sorted(malicious & set(node_ids)):
            csv_writer.writerow({"node_id": node_ids[entity_uuid], "label": 1, "uuid": entity_uuid})
    metadata = {
        "dataset": "E5-CADETS", "protocol": "strict_chronological",
        "status": (
            "incomplete_raw_diagnostic" if missing_names
            else ("pilot" if args.pilot_events_per_day else "publication_full")
        ),
        "feature_mode": args.feature_mode, "embedding_dim": embedding_dim,
        "pilot_events_per_day": args.pilot_events_per_day,
        "pilot_attack_events_per_window": args.pilot_attack_events_per_window,
        "splits_by_eastern_day": {str(day): split for day, split in STRICT_SPLITS.items()},
        "attack_windows": ATTACK_WINDOWS, "raw_shards": [str(path) for path in shards],
        "manifest_expected_files": len(set(expected_names)),
        "manifest_present_files": len(present_names & set(expected_names)),
        "manifest_missing_files": missing_names,
        "raw_counters": dict(counters), "temporal_files": dict(writer.file_counts),
        "fused_edges": dict(writer.edge_counts), "nodes": len(node_ids),
        "malicious_nodes_in_stream": len(malicious & set(node_ids)),
        "label_leakage_control": "node labels are never used in features; documented attack-window oversampling is pilot-only",
        "ground_truth": str(gt_path),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
