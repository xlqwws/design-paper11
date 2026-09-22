"""Fixed-budget causal early-detection experiment on the full StreamSpot stream."""

from __future__ import annotations

import argparse
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

from run_streamspot_full_streaming_sketch import (
    EDGE_INDEX, EDGE_TYPES, NODE_INDEX, NODE_TYPES, entropy_rows,
    finite_threshold, normalized, split_for_graph,
)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection.conformal import conformal_predictions  # noqa: E402


def flush_batch(batch, budgets, states):
    if not batch:
        return
    values = np.asarray(batch, dtype=np.int64)
    graph, position, src_type, dst_type, relation = values.T
    for budget in budgets:
        selected = position < budget
        if not np.any(selected):
            continue
        g = graph[selected]
        s = src_type[selected]
        d = dst_type[selected]
        r = relation[selected]
        state = states[budget]
        state["relation"] += np.bincount(
            g * len(EDGE_TYPES) + r, minlength=600 * len(EDGE_TYPES)
        ).reshape(state["relation"].shape)
        state["src"] += np.bincount(
            g * len(NODE_TYPES) + s, minlength=600 * len(NODE_TYPES)
        ).reshape(state["src"].shape)
        state["dst"] += np.bincount(
            g * len(NODE_TYPES) + d, minlength=600 * len(NODE_TYPES)
        ).reshape(state["dst"].shape)
        triple = (s * len(EDGE_TYPES) + r) * len(NODE_TYPES) + d
        state["triple"] += np.bincount(
            g * state["triple"].shape[1] + triple,
            minlength=600 * state["triple"].shape[1],
        ).reshape(state["triple"].shape)


def feature_matrix(state):
    relation = state["relation"]
    src = state["src"]
    dst = state["dst"]
    return np.concatenate([
        normalized(relation), normalized(src), normalized(dst), normalized(state["triple"]),
        (np.count_nonzero(relation, axis=1) / len(EDGE_TYPES))[:, None],
        (entropy_rows(relation) / np.log(len(EDGE_TYPES)))[:, None],
        (entropy_rows(src + dst) / np.log(len(NODE_TYPES)))[:, None],
    ], axis=1).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=r"D:\download\all.tar\all.tar")
    parser.add_argument("--out-dir", default="artifacts/tifs_results/StreamSpot/full_early_detection_seed0")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 5000, 10000, 25000])
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    parser.add_argument("--batch-lines", type=int, default=250000)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set(args.budgets))
    triple_dim = len(NODE_TYPES) * len(EDGE_TYPES) * len(NODE_TYPES)
    states = {
        budget: {
            "relation": np.zeros((600, len(EDGE_TYPES)), dtype=np.int64),
            "src": np.zeros((600, len(NODE_TYPES)), dtype=np.int64),
            "dst": np.zeros((600, len(NODE_TYPES)), dtype=np.int64),
            "triple": np.zeros((600, triple_dim), dtype=np.int64),
        }
        for budget in budgets
    }
    positions = np.zeros(600, dtype=np.int64)
    batch = []
    raw_edges = 0
    started = time.perf_counter()
    with tarfile.open(args.raw, "r") as archive:
        stream = archive.extractfile("all.tsv")
        if stream is None:
            raise FileNotFoundError("all.tsv is missing")
        for raw in stream:
            fields = raw.rstrip().split(b"\t")
            if len(fields) < 6:
                continue
            graph_id = int(fields[5])
            position = int(positions[graph_id])
            positions[graph_id] += 1
            if position < budgets[-1]:
                batch.append((
                    graph_id, position, NODE_INDEX[fields[1]], NODE_INDEX[fields[3]],
                    EDGE_INDEX[fields[4]],
                ))
            raw_edges += 1
            if len(batch) >= args.batch_lines:
                flush_batch(batch, budgets, states)
                batch.clear()
            if raw_edges % 10_000_000 == 0:
                print(f"scanned={raw_edges:,} elapsed_sec={time.perf_counter() - started:.1f}", flush=True)
    flush_batch(batch, budgets, states)
    cache_payload = {"positions": positions, "budgets": np.asarray(budgets, dtype=np.int64)}
    for budget in budgets:
        for name, values in states[budget].items():
            cache_payload[f"budget_{budget}_{name}"] = values
    np.savez_compressed(out_dir / "unlabelled_prefix_counts.npz", **cache_payload)

    train_ids = np.asarray([index for index in range(600) if split_for_graph(index) == "train"])
    val_ids = np.asarray([index for index in range(600) if split_for_graph(index) == "val"])
    test_ids = np.asarray([index for index in range(600) if split_for_graph(index) == "test"])
    labels = np.asarray([int(300 <= graph_id <= 399) for graph_id in test_ids])
    results = []
    scores_by_budget = {}
    predictions_by_budget = {}
    for budget in budgets:
        features = feature_matrix(states[budget])
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
        scores_by_budget[budget] = test_scores
        predictions_by_budget[budget] = predicted
        results.append({
            "protocol_version": "tifs-fixed-budget-early-detection-v1",
            "dataset": "StreamSpot", "method": "online_dynamic_structure_sketch_knn",
            "event_budget": budget, "raw_edges_scanned": int(raw_edges),
            "train_graphs": len(train_ids), "calibration_graphs": len(val_ids),
            "test_graphs": len(test_ids), "attack_graphs": int(labels.sum()),
            "calibration_alpha": args.calibration_alpha, "threshold": threshold,
            "reported_graphs": int(predicted.sum()),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "auroc": float(roc_auc_score(labels, test_scores)),
            "average_precision": float(average_precision_score(labels, test_scores)),
            "scan_seconds": time.perf_counter() - started,
            "causal_decision": "uses exactly the first event_budget edges of each graph",
        })

    per_graph = pd.DataFrame({"graph_id": test_ids, "label": labels, "raw_edges": positions[test_ids]})
    for budget in budgets:
        per_graph[f"score_at_{budget}"] = scores_by_budget[budget]
        per_graph[f"predicted_at_{budget}"] = predictions_by_budget[budget]
    per_graph["first_detection_budget"] = [
        next((budget for budget in budgets if predictions_by_budget[budget][index]), None)
        for index in range(len(test_ids))
    ]
    per_graph.to_csv(out_dir / "per_graph_early_detection.csv", index=False)
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"results": results}, file, indent=2)
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
