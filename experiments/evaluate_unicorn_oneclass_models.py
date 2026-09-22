"""Validation-calibrated one-class model comparison on full Unicorn sketches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from detection.conformal import conformal_predictions  # noqa: E402
from evaluate_cached_full_sketch_repeats import feature_variants, split_ids  # noqa: E402


def scores(method, train, validation, test, seed):
    if method == "knn5":
        model = NearestNeighbors(n_neighbors=5, metric="euclidean", algorithm="brute").fit(train)
        return tuple(model.kneighbors(x, return_distance=True)[0].mean(axis=1) for x in [validation, test])
    if method.startswith("lof"):
        neighbors = int(method[3:])
        model = LocalOutlierFactor(n_neighbors=min(neighbors, len(train) - 1), novelty=True).fit(train)
        return -model.score_samples(validation), -model.score_samples(test)
    if method == "isolation_forest":
        model = IsolationForest(n_estimators=500, max_samples="auto", random_state=seed).fit(train)
        return -model.score_samples(validation), -model.score_samples(test)
    if method == "ocsvm":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(train)
        return -model.score_samples(validation), -model.score_samples(test)
    if method == "pca_reconstruction":
        model = PCA(n_components=min(32, len(train) - 1), random_state=seed).fit(train)
        def error(values):
            reconstructed = model.inverse_transform(model.transform(values))
            return np.mean((values - reconstructed) ** 2, axis=1)
        return error(validation), error(test)
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    features = feature_variants("Unicorn-Wget", np.load(args.cache))["normalized_structure_only"]
    methods = ["knn5", "lof5", "lof10", "lof20", "lof40", "isolation_forest", "ocsvm", "pca_reconstruction"]
    rows = []
    for seed in range(5):
        train_ids, val_ids, test_ids, labels = split_ids("Unicorn-Wget", seed)
        scaler = StandardScaler().fit(features[train_ids])
        train, validation, test = [scaler.transform(features[ids]) for ids in [train_ids, val_ids, test_ids]]
        for method in methods:
            val_scores, test_scores = scores(method, train, validation, test, seed)
            predicted = conformal_predictions(val_scores, test_scores, args.alpha)
            rows.append({
                "dataset": "Unicorn-Wget", "method": method, "split_seed": seed,
                "alpha": args.alpha, "reported_graphs": int(predicted.sum()),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
                "auroc": float(roc_auc_score(labels, test_scores)),
                "average_precision": float(average_precision_score(labels, test_scores)),
            })
    frame = pd.DataFrame(rows)
    summary = frame.groupby("method", as_index=False).agg(
        repetitions=("split_seed", "count"), precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"), f1_mean=("f1", "mean"), f1_min=("f1", "min"),
        auroc_mean=("auroc", "mean"), average_precision_mean=("average_precision", "mean"),
    ).sort_values("f1_mean", ascending=False)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "split_repetitions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "selection": "all preregistered one-class candidates are reported",
        "fit": "benign train only", "calibration": "benign validation only",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
