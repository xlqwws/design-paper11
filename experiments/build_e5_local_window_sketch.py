"""Parallel full-event window sketch for the locally available E5 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from build_cadets_e5_temporal import E5_RAW
from build_cadets_e5_temporal_parallel import DAY_RANGES, _shard_number
from build_cadets_temporal import RELATIONS, REVERSED_RELATIONS
from cdm_avro import iter_cdm_events


WINDOW_NS = 15 * 60 * 1_000_000_000
BITMAP_SIZE = 32768


def endpoint_bucket(value: object) -> int | None:
    if isinstance(value, bytes) and len(value) == 16:
        number = int.from_bytes(value[:8], "little", signed=False)
    elif isinstance(value, str) and value:
        number = int.from_bytes(
            hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=8).digest(),
            "little", signed=False,
        )
    else:
        return None
    return int((number * 11400714819323198485) % BITMAP_SIZE)


def scan_shard(payload: tuple[int, str]) -> dict:
    index, path_text = payload
    windows: dict[int, dict] = {}
    counters = Counter()
    for event in iter_cdm_events(Path(path_text), 0):
        counters["events"] += 1
        relation_name = str(event.get("type", ""))
        if relation_name not in RELATIONS:
            counters["unsupported_relation"] += 1
            continue
        timestamp = event.get("timestampNanos")
        if not isinstance(timestamp, int) or not (DAY_RANGES[8][0] <= timestamp < DAY_RANGES[8][1]):
            counters["outside_local_day8"] += 1
            continue
        subject = endpoint_bucket(event.get("subject"))
        obj = endpoint_bucket(event.get("predicateObject"))
        if subject is None or obj is None:
            counters["missing_endpoint"] += 1
            continue
        relation = RELATIONS.index(relation_name)
        src_type = 0
        dst_type = 2 if any(token in relation_name for token in ("CONNECT", "SEND", "RECV")) else 1
        if relation_name in REVERSED_RELATIONS:
            subject, obj = obj, subject
            src_type, dst_type = dst_type, src_type
        key = int(timestamp // WINDOW_NS)
        row = windows.setdefault(key, {
            "edges": 0,
            "relations": np.zeros(len(RELATIONS), dtype=np.int64),
            "src_types": np.zeros(3, dtype=np.int64),
            "dst_types": np.zeros(3, dtype=np.int64),
            "nodes": np.zeros(BITMAP_SIZE, dtype=bool),
        })
        row["edges"] += 1
        row["relations"][relation] += 1
        row["src_types"][src_type] += 1
        row["dst_types"][dst_type] += 1
        row["nodes"][subject] = True
        row["nodes"][obj] = True
        counters["supported_day8_events"] += 1
    return {"index": index, "path": path_text, "counters": dict(counters), "windows": windows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(E5_RAW))
    parser.add_argument(
        "--out-dir", default="artifacts/feature_cache/E5-CADETS-Causal/local_day8_window_sketch"
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "window_sketch.npz").exists():
        raise FileExistsError(f"Refusing to overwrite {out / 'window_sketch.npz'}")

    shards = sorted(raw_dir.glob("*.gz"), key=_shard_number)
    manifest = raw_dir / "bins.md5sum"
    expected = {
        Path(line.split()[-1]).name
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    }
    present = {path.name for path in shards}
    missing = sorted(expected - present)
    if not shards:
        raise FileNotFoundError(f"No local E5 shards in {raw_dir}")

    started = time.perf_counter()
    results = []
    payloads = [(index, str(path)) for index, path in enumerate(shards)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(scan_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"completed={result['index'] + 1}/{len(shards)} "
                f"events={result['counters'].get('events', 0):,}",
                flush=True,
            )

    merged: dict[int, dict] = {}
    counters = Counter()
    for result in results:
        counters.update(result["counters"])
        for key, item in result["windows"].items():
            row = merged.setdefault(key, {
                "edges": 0,
                "relations": np.zeros(len(RELATIONS), dtype=np.int64),
                "src_types": np.zeros(3, dtype=np.int64),
                "dst_types": np.zeros(3, dtype=np.int64),
                "nodes": np.zeros(BITMAP_SIZE, dtype=bool),
            })
            row["edges"] += item["edges"]
            row["relations"] += item["relations"]
            row["src_types"] += item["src_types"]
            row["dst_types"] += item["dst_types"]
            row["nodes"] |= item["nodes"]

    keys = np.asarray(sorted(merged), dtype=np.int64)
    np.savez_compressed(
        out / "window_sketch.npz",
        window_keys=keys,
        edge_counts=np.asarray([merged[key]["edges"] for key in keys], dtype=np.int64),
        relation_counts=np.stack([merged[key]["relations"] for key in keys]),
        src_type_counts=np.stack([merged[key]["src_types"] for key in keys]),
        dst_type_counts=np.stack([merged[key]["dst_types"] for key in keys]),
        node_bitmaps=np.stack([merged[key]["nodes"] for key in keys]),
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "dataset": "E5-CADETS-Causal",
        "status": "incomplete_raw_diagnostic",
        "scope": "all_supported_events_in_locally_available_day8_shards",
        "manifest_expected_files": len(expected),
        "manifest_present_files": len(present & expected),
        "manifest_missing_files": missing,
        "workers": args.workers,
        "window_minutes": 15,
        "windows": len(keys),
        "raw_counters": dict(counters),
        "scan_seconds": elapsed,
        "scan_throughput_events_per_second": counters["events"] / max(elapsed, 1e-12),
        "labels_used": False,
        "attack_period_available": False,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
