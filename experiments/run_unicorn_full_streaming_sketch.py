"""Full-edge bounded-memory causal sketch experiment for Unicorn-Wget."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection.conformal import conformal_predictions  # noqa: E402
from run_streamspot_full_streaming_sketch import entropy_rows, finite_threshold, normalized  # noqa: E402
from build_unicorn_wget_temporal import ARCHIVES, natural_key


NODE_BUCKETS = 512
RELATION_BUCKETS = 512
TRIPLE_BUCKETS = 4096


def bucket(text, buckets, cache):
    value = cache.get(text)
    if value is None:
        digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
        value = int.from_bytes(digest, "big") % buckets
        cache[text] = value
    return value


def graph_split(label, benign_index):
    if label:
        return "test"
    return "train" if benign_index < 75 else ("val" if benign_index < 100 else "test")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=r"D:\download\Unicorn Wget")
    parser.add_argument("--out-dir", default="artifacts/tifs_results/Unicorn-Wget/full_streaming_sketch_seed0")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    relation_rows = []
    src_rows = []
    dst_rows = []
    triple_rows = []
    edge_counts = []
    records = []
    relation_cache = {}
    node_cache = {}
    triple_cache = {}
    benign_index = 0
    raw_edges = 0
    started = time.perf_counter()
    for archive_name, label in ARCHIVES:
        archive_path = Path(args.raw_dir) / archive_name
        with tarfile.open(archive_path, "r:gz") as archive:
            members = sorted([
                member for member in archive.getmembers()
                if member.isfile() and "/stream/" in member.name
                and member.name.endswith(".txt") and "/._" not in member.name
                and not Path(member.name).name.startswith("._")
            ], key=lambda member: natural_key(member.name))
            for member in members:
                relation_count = np.zeros(RELATION_BUCKETS, dtype=np.int64)
                src_count = np.zeros(NODE_BUCKETS, dtype=np.int64)
                dst_count = np.zeros(NODE_BUCKETS, dtype=np.int64)
                triple_count = np.zeros(TRIPLE_BUCKETS, dtype=np.int64)
                edges = 0
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                for raw in stream:
                    fields = raw.decode("utf-8", errors="ignore").strip().split()
                    if len(fields) < 3:
                        continue
                    payload = fields[2].split(":")
                    if len(payload) < 3:
                        continue
                    src_type, dst_type, relation_type = payload[:3]
                    src_bucket = bucket(src_type, NODE_BUCKETS, node_cache)
                    dst_bucket = bucket(dst_type, NODE_BUCKETS, node_cache)
                    relation_bucket = bucket(relation_type, RELATION_BUCKETS, relation_cache)
                    triple_text = f"{src_type}|{relation_type}|{dst_type}"
                    triple_bucket = bucket(triple_text, TRIPLE_BUCKETS, triple_cache)
                    src_count[src_bucket] += 1
                    dst_count[dst_bucket] += 1
                    relation_count[relation_bucket] += 1
                    triple_count[triple_bucket] += 1
                    edges += 1
                graph_id = len(records)
                split = graph_split(label, benign_index)
                if not label:
                    benign_index += 1
                relation_rows.append(relation_count)
                src_rows.append(src_count)
                dst_rows.append(dst_count)
                triple_rows.append(triple_count)
                edge_counts.append(edges)
                records.append({
                    "graph_id": graph_id, "label": label, "split": split,
                    "archive": archive_name, "member": member.name, "raw_edges": edges,
                })
                raw_edges += edges
                print(
                    f"graphs={len(records)}/175 raw_edges={raw_edges:,} "
                    f"elapsed_sec={time.perf_counter() - started:.1f}", flush=True,
                )

    relation_count = np.asarray(relation_rows)
    src_count = np.asarray(src_rows)
    dst_count = np.asarray(dst_rows)
    triple_count = np.asarray(triple_rows)
    edge_count = np.asarray(edge_counts)
    size_features = np.log1p(edge_count)[:, None].astype(np.float32)
    structure_features = np.concatenate([
        normalized(relation_count), normalized(src_count), normalized(dst_count),
        normalized(triple_count),
        (np.count_nonzero(relation_count, axis=1) / RELATION_BUCKETS)[:, None],
        (entropy_rows(relation_count) / np.log(RELATION_BUCKETS))[:, None],
        (entropy_rows(src_count + dst_count) / np.log(NODE_BUCKETS))[:, None],
    ], axis=1).astype(np.float32)
    variants = {
        "dual_head_full": np.concatenate([size_features, structure_features], axis=1),
        "normalized_structure_only": structure_features,
        "size_only": size_features,
    }
    train_ids = np.asarray([row["graph_id"] for row in records if row["split"] == "train"])
    val_ids = np.asarray([row["graph_id"] for row in records if row["split"] == "val"])
    test_ids = np.asarray([row["graph_id"] for row in records if row["split"] == "test"])
    labels = np.asarray([records[index]["label"] for index in test_ids], dtype=int)
    results = []
    per_graph = pd.DataFrame([records[index] for index in test_ids])
    for variant, features in variants.items():
        scaler = StandardScaler().fit(features[train_ids])
        train = scaler.transform(features[train_ids])
        validation = scaler.transform(features[val_ids])
        test = scaler.transform(features[test_ids])
        model = NearestNeighbors(
            n_neighbors=min(args.neighbors, len(train)), metric="euclidean", algorithm="brute"
        ).fit(train)
        validation_scores = model.kneighbors(validation, return_distance=True)[0].mean(axis=1)
        test_scores = model.kneighbors(test, return_distance=True)[0].mean(axis=1)
        threshold = finite_threshold(validation_scores, args.calibration_alpha)
        predicted = conformal_predictions(validation_scores, test_scores, args.calibration_alpha)
        results.append({
            "protocol_version": "tifs-full-streaming-sketch-v1",
            "dataset": "Unicorn-Wget", "method": "online_dynamic_full_streaming_sketch_knn",
            "variant": variant, "seed": args.seed, "raw_edges": int(raw_edges),
            "train_graphs": len(train_ids), "calibration_graphs": len(val_ids),
            "test_graphs": len(test_ids), "attack_graphs": int(labels.sum()),
            "feature_dimension": int(features.shape[1]), "neighbors": args.neighbors,
            "calibration_alpha": args.calibration_alpha, "threshold": threshold,
            "reported_graphs": int(predicted.sum()),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "auroc": float(roc_auc_score(labels, test_scores)),
            "average_precision": float(average_precision_score(labels, test_scores)),
            "scan_seconds": time.perf_counter() - started,
            "state_bytes": int(
                relation_count.nbytes + src_count.nbytes + dst_count.nbytes
                + triple_count.nbytes + edge_count.nbytes
            ),
            "decision_time": "graph_window_close_after_last_observed_event",
            "label_leakage_control": "fit on benign train only; threshold on benign validation only",
        })
        per_graph[f"{variant}_score"] = test_scores
        per_graph[f"{variant}_predicted"] = predicted

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "unlabelled_streaming_counts.npz", edge_count=edge_count,
        relation_count=relation_count, src_count=src_count, dst_count=dst_count,
        triple_count=triple_count,
    )
    per_graph.to_csv(out_dir / "per_graph_scores.csv", index=False)
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"results": results}, file, indent=2)
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
