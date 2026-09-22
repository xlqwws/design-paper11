"""Evaluate the validation-calibrated causal structural sketch head."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.causal_graph_sketch import extract_causal_graph_sketch  # noqa: E402
from detection.conformal import conformal_predictions  # noqa: E402


def load_split(root, split, node_types, edge_types):
    paths = sorted((root / split).glob("*.TemporalData"))
    vectors = []
    for path in paths:
        data = torch.load(path, map_location="cpu")
        vectors.append(extract_causal_graph_sketch(data, node_types, edge_types))
    return paths, np.asarray(vectors, dtype=np.float32)


def conformal_threshold(scores, alpha):
    values = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - alpha)))
    return float(values[rank - 1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuilt-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--distance-metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.prebuilt_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    node_types = len(metadata.get("node_types", [])) or int(metadata["node_type_buckets"])
    edge_types = len(metadata.get("edge_types", [])) or int(metadata["edge_type_buckets"])
    train_paths, train = load_split(root, "train", node_types, edge_types)
    val_paths, validation = load_split(root, "val", node_types, edge_types)
    test_paths, test = load_split(root, "test", node_types, edge_types)

    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    validation_scaled = scaler.transform(validation)
    test_scaled = scaler.transform(test)
    model = NearestNeighbors(
        n_neighbors=min(args.neighbors, len(train_scaled)), metric=args.distance_metric,
        algorithm="brute",
    ).fit(train_scaled)
    validation_scores = model.kneighbors(validation_scaled, return_distance=True)[0].mean(axis=1)
    test_scores = model.kneighbors(test_scaled, return_distance=True)[0].mean(axis=1)
    threshold = conformal_threshold(validation_scores, args.calibration_alpha)
    predicted = conformal_predictions(validation_scores, test_scores, args.calibration_alpha)

    truth = pd.read_csv(args.ground_truth).sort_values("time_window")
    if len(truth) != len(test_scores):
        raise ValueError(f"Ground truth has {len(truth)} graphs, test has {len(test_scores)}")
    labels = truth["label"].to_numpy(dtype=int)
    result = {
        "protocol_version": "tifs-causal-graph-sketch-v1",
        "dataset": args.dataset,
        "method": "online_dynamic_dual_head_knn_wl",
        "seed": args.seed,
        "train_graphs": len(train_paths), "calibration_graphs": len(val_paths),
        "test_graphs": len(test_paths), "attack_graphs": int(labels.sum()),
        "feature_dimension": int(train.shape[1]), "neighbors": args.neighbors,
        "distance_metric": args.distance_metric,
        "calibration_alpha": args.calibration_alpha, "threshold": threshold,
        "reported_graphs": int(predicted.sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, test_scores)),
        "average_precision": float(average_precision_score(labels, test_scores)),
        "decision_time": "graph_window_close_after_last_observed_event",
        "label_leakage_control": "fit on train graphs; threshold on validation graphs; test labels used once for metrics",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_graph = truth.copy()
    per_graph["score"] = test_scores
    per_graph["predicted"] = predicted
    per_graph.to_csv(out_dir / "per_graph_scores.csv", index=False)
    pd.DataFrame([result]).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
