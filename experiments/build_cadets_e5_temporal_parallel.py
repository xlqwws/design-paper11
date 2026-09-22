"""Parallel, causality-preserving E5-CADETS pilot TemporalData builder."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pytz

from build_cadets_temporal import RELATIONS, REVERSED_RELATIONS, TemporalWriter
from build_cadets_e5_temporal import ATTACK_WINDOWS, E5_GT, E5_RAW, STRICT_SPLITS
from cdm_avro import iter_cdm_events


EASTERN = pytz.timezone("US/Eastern")


def _local_ns(text: str) -> int:
    value = EASTERN.localize(datetime.strptime(text, "%Y-%m-%d %H:%M:%S"))
    return int(value.timestamp() * 1_000_000_000)


DAY_RANGES = {
    day: (
        _local_ns(f"2019-05-{day:02d} 00:00:00"),
        _local_ns(f"2019-05-{day + 1:02d} 00:00:00"),
    )
    for day in STRICT_SPLITS
}
ATTACK_RANGES = [(_local_ns(start), _local_ns(end)) for start, end in ATTACK_WINDOWS]


def _uuid_text(value: object) -> str:
    if isinstance(value, bytes) and len(value) == 16:
        return str(uuid.UUID(bytes=value)).upper()
    if isinstance(value, str):
        try:
            return str(uuid.UUID(value)).upper()
        except ValueError:
            return ""
    return ""


def _push_earliest(heap: list, limit: int, candidate: tuple) -> None:
    timestamp, ordinal = candidate[0], candidate[1]
    item = (-timestamp, -ordinal, candidate)
    if limit <= 0 or len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def _scan_shard(payload: tuple) -> dict:
    shard_index, shard_text, per_day, per_attack, max_records = payload
    shard = Path(shard_text)
    buckets = defaultdict(list)
    counters = Counter()
    for ordinal, event in enumerate(iter_cdm_events(shard, max_records)):
        counters["events"] += 1
        relation = str(event.get("type", ""))
        if relation not in RELATIONS:
            counters["unsupported_relation"] += 1
            continue
        timestamp = event.get("timestampNanos")
        if not isinstance(timestamp, int):
            counters["missing_time"] += 1
            continue
        day = next((d for d, (start, end) in DAY_RANGES.items() if start <= timestamp < end), None)
        if day is None:
            counters["outside_protocol_days"] += 1
            continue
        attack_index = next((
            index for index, (start, end) in enumerate(ATTACK_RANGES)
            if start <= timestamp <= end
        ), None)
        bucket = ("attack", attack_index) if attack_index is not None else ("day", day)
        limit = per_attack if attack_index is not None else per_day
        candidate = (
            timestamp, ordinal, shard_index, day, attack_index, relation,
            event.get("subject"), event.get("predicateObject"), event.get("uuid"),
        )
        _push_earliest(buckets[bucket], limit, candidate)
        counters["supported_in_protocol"] += 1
    selected = {
        f"{key[0]}:{key[1]}": [item[2] for item in heap]
        for key, heap in buckets.items()
    }
    return {
        "shard_index": shard_index,
        "shard": shard_text,
        "counters": dict(counters),
        "selected": selected,
    }


def _parse_bucket(text: str) -> tuple[str, int]:
    kind, value = text.split(":", 1)
    return kind, int(value)


def _load_ground_truth(path: Path) -> set[str]:
    output = set()
    for csv_path in sorted(path.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="replace") as file:
            for row in csv.reader(file):
                if row:
                    output.add(row[0].strip().upper())
    return output


def _shard_number(path: Path) -> int:
    token = path.name.split(".")[-2]
    return int(token) if token.isdigit() else 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(E5_RAW))
    parser.add_argument("--out-dir", default="artifacts/raw_temporal_pilot/E5-CADETS/edge_embeds")
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--pilot-events-per-day", type=int, default=5000)
    parser.add_argument("--pilot-attack-events-per-window", type=int, default=10000)
    parser.add_argument("--scan-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-records-per-shard", type=int, default=0)
    parser.add_argument("--allow-incomplete-raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pilot_events_per_day <= 0 or args.pilot_attack_events_per_window <= 0:
        raise ValueError("Parallel builder requires finite positive pilot caps")
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.rglob("*.TemporalData")):
        raise FileExistsError(f"Refusing to overwrite existing TemporalData in {out_dir}")
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    shards = sorted(Path(args.raw_dir).glob("*.gz"), key=_shard_number)
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
    payloads = [
        (index, str(shard), args.pilot_events_per_day,
         args.pilot_attack_events_per_window, args.max_records_per_shard)
        for index, shard in enumerate(shards)
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.scan_workers) as executor:
        futures = {executor.submit(_scan_shard, payload): payload[1] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed shard {result['shard_index'] + 1}/{len(shards)}: "
                f"events={result['counters'].get('events', 0):,}",
                flush=True,
            )

    counters = Counter()
    candidates = defaultdict(list)
    for result in results:
        counters.update(result["counters"])
        for bucket_text, events in result["selected"].items():
            candidates[_parse_bucket(bucket_text)].extend(events)

    selected = []
    kept_by_bucket = {}
    for bucket, events in sorted(candidates.items()):
        limit = (
            args.pilot_attack_events_per_window
            if bucket[0] == "attack" else args.pilot_events_per_day
        )
        events.sort(key=lambda item: (item[0], item[2], item[1]))
        chosen = events[:limit]
        selected.extend(chosen)
        kept_by_bucket[f"{bucket[0]}:{bucket[1]}"] = len(chosen)
    selected.sort(key=lambda item: (item[0], item[2], item[1]))

    malicious = _load_ground_truth(E5_GT)
    node_ids = {}
    id_to_uuid = {}

    def node_id(value: str) -> int:
        if value not in node_ids:
            index = len(node_ids)
            node_ids[value] = index
            id_to_uuid[index] = value
        return node_ids[value]

    writer = TemporalWriter(out_dir, args.window_minutes, 0)
    invalid = 0
    for timestamp, ordinal, shard_index, day, attack_index, relation_name, subject, obj, event_id in selected:
        subject_uuid = _uuid_text(subject)
        object_uuid = _uuid_text(obj)
        if not subject_uuid or not object_uuid:
            invalid += 1
            continue
        src, dst = node_id(subject_uuid), node_id(object_uuid)
        src_type = 0
        dst_type = 2 if any(
            token in relation_name for token in ("CONNECT", "SEND", "RECV")
        ) else 1
        if relation_name in REVERSED_RELATIONS:
            src, dst = dst, src
            src_type, dst_type = dst_type, src_type
        event_uuid = _uuid_text(event_id) or f"{shard_index}:{ordinal}"
        writer.add(
            STRICT_SPLITS[day], day,
            (src, dst, timestamp, RELATIONS.index(relation_name), event_uuid,
             src_type, dst_type, [], []),
        )
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
        "dataset": "E5-CADETS",
        "protocol": "strict_chronological_parallel_scan",
        "status": "incomplete_raw_diagnostic" if missing_names else "pilot",
        "feature_mode": "only_type",
        "embedding_dim": 0,
        "pilot_events_per_day": args.pilot_events_per_day,
        "pilot_attack_events_per_window": args.pilot_attack_events_per_window,
        "scan_workers": args.scan_workers,
        "splits_by_eastern_day": {str(day): split for day, split in STRICT_SPLITS.items()},
        "attack_windows": ATTACK_WINDOWS,
        "raw_shards": [str(path) for path in shards],
        "manifest_expected_files": len(set(expected_names)),
        "manifest_present_files": len(present_names & set(expected_names)),
        "manifest_missing_files": missing_names,
        "raw_counters": dict(counters),
        "kept_by_bucket_before_fusion": kept_by_bucket,
        "invalid_selected_events": invalid,
        "temporal_files": dict(writer.file_counts),
        "fused_edges": dict(writer.edge_counts),
        "nodes": len(node_ids),
        "malicious_nodes_in_stream": len(malicious & set(node_ids)),
        "label_leakage_control": (
            "labels are never used in features; attack-window caps use only documented "
            "benchmark timestamps and are pilot-only"
        ),
        "ground_truth": str(gt_path),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
