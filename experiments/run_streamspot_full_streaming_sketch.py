"""Full-edge bounded-memory causal sketch experiment for StreamSpot."""

from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection.conformal import conformal_predictions  # noqa: E402


NODE_TYPES = list("abcdefgh")
EDGE_TYPES = list("ABCDEFGHijklmnopqrstuvwxyz")
NODE_INDEX = {value.encode(): index for index, value in enumerate(NODE_TYPES)}
EDGE_INDEX = {value.encode(): index for index, value in enumerate(EDGE_TYPES)}
BENIGN_STARTS = [0, 100, 200, 400, 500]


def split_for_graph(graph_id):
    if 300 <= graph_id <= 399:
        return "test"
    offset = graph_id % 100
    return "train" if offset < 60 else ("val" if offset < 80 else "test")


def finite_threshold(scores, alpha):
    values = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - alpha)))
    return float(values[rank - 1])


def flush_batch(batch, edge_count, relation_count, src_type_count, dst_type_count, triple_count):
    if not batch:
        return
    values = np.asarray(batch, dtype=np.int64)
    graph, src_type, dst_type, relation = values.T
    edge_count += np.bincount(graph, minlength=600)
    relation_flat = graph * len(EDGE_TYPES) + relation
    relation_count += np.bincount(
        relation_flat, minlength=600 * len(EDGE_TYPES)
    ).reshape(relation_count.shape)
    src_flat = graph * len(NODE_TYPES) + src_type
    dst_flat = graph * len(NODE_TYPES) + dst_type
    src_type_count += np.bincount(
        src_flat, minlength=600 * len(NODE_TYPES)
    ).reshape(src_type_count.shape)
    dst_type_count += np.bincount(
        dst_flat, minlength=600 * len(NODE_TYPES)
    ).reshape(dst_type_count.shape)
    triple = (src_type * len(EDGE_TYPES) + relation) * len(NODE_TYPES) + dst_type
    triple_flat = graph * triple_count.shape[1] + triple
    triple_count += np.bincount(
        triple_flat, minlength=600 * triple_count.shape[1]
    ).reshape(triple_count.shape)


def normalized(values):
    denominator = np.maximum(values.sum(axis=1, keepdims=True), 1)
    return values / denominator


def entropy_rows(values):
    probability = normalized(values.astype(float))
    log_probability = np.zeros_like(probability)
    positive = probability > 0
    log_probability[positive] = np.log(probability[positive])
    return -np.sum(probability * log_probability, axis=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=r"D:\download\all.tar\all.tar")
    parser.add_argument("--out-dir", default="artifacts/tifs_results/StreamSpot/full_streaming_sketch_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    parser.add_argument("--batch-lines", type=int, default=250000)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_count = np.zeros(600, dtype=np.int64)
    relation_count = np.zeros((600, len(EDGE_TYPES)), dtype=np.int64)
    src_type_count = np.zeros((600, len(NODE_TYPES)), dtype=np.int64)
    dst_type_count = np.zeros((600, len(NODE_TYPES)), dtype=np.int64)
    triple_count = np.zeros(
        (600, len(NODE_TYPES) * len(EDGE_TYPES) * len(NODE_TYPES)), dtype=np.int64
    )
    malformed = 0
    raw_edges = 0
    batch = []
    started = time.perf_counter()
    with tarfile.open(args.raw, "r") as archive:
        stream = archive.extractfile("all.tsv")
        if stream is None:
            raise FileNotFoundError("all.tsv is missing")
        for raw in stream:
            fields = raw.rstrip().split(b"\t")
            if len(fields) < 6:
                malformed += 1
                continue
            try:
                graph_id = int(fields[5])
                batch.append((
                    graph_id, NODE_INDEX[fields[1]], NODE_INDEX[fields[3]], EDGE_INDEX[fields[4]],
                ))
            except (ValueError, KeyError):
                malformed += 1
                continue
            raw_edges += 1
            if len(batch) >= args.batch_lines:
                flush_batch(
                    batch, edge_count, relation_count, src_type_count, dst_type_count, triple_count
                )
                batch.clear()
            if raw_edges % 10_000_000 == 0:
                print(
                    f"scanned={raw_edges:,} elapsed_sec={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    flush_batch(batch, edge_count, relation_count, src_type_count, dst_type_count, triple_count)
    np.savez_compressed(
        out_dir / "unlabelled_streaming_counts.npz",
        edge_count=edge_count, relation_count=relation_count,
        src_type_count=src_type_count, dst_type_count=dst_type_count,
        triple_count=triple_count,
    )

    size_features = np.log1p(edge_count)[:, None].astype(np.float32)
    structure_features = np.concatenate([
        normalized(relation_count), normalized(src_type_count), normalized(dst_type_count),
        normalized(triple_count),
        (np.count_nonzero(relation_count, axis=1) / len(EDGE_TYPES))[:, None],
        (entropy_rows(relation_count) / np.log(len(EDGE_TYPES)))[:, None],
        (entropy_rows(src_type_count + dst_type_count) / np.log(len(NODE_TYPES)))[:, None],
    ], axis=1).astype(np.float32)
    feature_variants = {
        "dual_head_full": np.concatenate([size_features, structure_features], axis=1),
        "normalized_structure_only": structure_features,
        "size_only": size_features,
    }

    train_ids = np.asarray([graph_id for graph_id in range(600) if split_for_graph(graph_id) == "train"])
    val_ids = np.asarray([graph_id for graph_id in range(600) if split_for_graph(graph_id) == "val"])
    test_ids = np.asarray([graph_id for graph_id in range(600) if split_for_graph(graph_id) == "test"])
    labels = np.asarray([int(300 <= graph_id <= 399) for graph_id in test_ids])
    results = []
    per_graph = pd.DataFrame({"graph_id": test_ids, "label": labels, "raw_edges": edge_count[test_ids]})
    scan_seconds = time.perf_counter() - started
    state_bytes = int(
        edge_count.nbytes + relation_count.nbytes + src_type_count.nbytes
        + dst_type_count.nbytes + triple_count.nbytes
    )
    for variant, features in feature_variants.items():
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
            "dataset": "StreamSpot", "method": "online_dynamic_full_streaming_sketch_knn",
            "variant": variant, "seed": args.seed, "raw_edges": int(raw_edges),
            "malformed_rows": malformed, "train_graphs": len(train_ids),
            "calibration_graphs": len(val_ids), "test_graphs": len(test_ids),
            "attack_graphs": int(labels.sum()), "feature_dimension": int(features.shape[1]),
            "neighbors": args.neighbors, "calibration_alpha": args.calibration_alpha,
            "threshold": threshold, "reported_graphs": int(predicted.sum()),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "auroc": float(roc_auc_score(labels, test_scores)),
            "average_precision": float(average_precision_score(labels, test_scores)),
            "scan_seconds": scan_seconds, "peak_feature_state_bytes": state_bytes,
            "decision_time": "graph_window_close_after_last_observed_event",
            "label_leakage_control": "all benign scenarios stratified; fit train only; threshold validation only",
        })
        per_graph[f"{variant}_score"] = test_scores
        per_graph[f"{variant}_predicted"] = predicted
    per_graph.to_csv(out_dir / "per_graph_scores.csv", index=False)
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"results": results}, file, indent=2)
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
