"""Test full-stream structural sketches under label-agnostic telemetry dropout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from detection.conformal import conformal_predictions  # noqa: E402
from evaluate_cached_full_sketch_repeats import feature_variants, split_ids  # noqa: E402


def thin_cache(cache, test_ids, keep_probability, seed):
    rng = np.random.default_rng(seed)
    arrays = {name: cache[name].copy() for name in cache.files}
    if keep_probability < 1.0:
        for name, values in arrays.items():
            values[test_ids] = rng.binomial(values[test_ids], keep_probability)
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["StreamSpot", "Unicorn-Wget"])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drop-rates", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    cache = np.load(args.cache)
    train_ids, val_ids, test_ids, labels = split_ids(args.dataset, 0)
    clean_features = feature_variants(args.dataset, cache)["normalized_structure_only"]
    scaler = StandardScaler().fit(clean_features[train_ids])
    train = scaler.transform(clean_features[train_ids])
    validation = scaler.transform(clean_features[val_ids])
    model = NearestNeighbors(n_neighbors=5, metric="euclidean", algorithm="brute").fit(train)
    validation_scores = model.kneighbors(validation, return_distance=True)[0].mean(axis=1)
    rows = []
    for rate in args.drop_rates:
        active_seeds = [args.seeds[0]] if rate == 0.0 else args.seeds
        for seed in active_seeds:
            thinned = thin_cache(cache, test_ids, 1.0 - rate, seed)
            features = feature_variants(args.dataset, thinned)["normalized_structure_only"]
            scores = model.kneighbors(
                scaler.transform(features[test_ids]), return_distance=True
            )[0].mean(axis=1)
            predicted = conformal_predictions(validation_scores, scores, args.alpha)
            rows.append({
                "dataset": args.dataset, "variant": "normalized_structure_only",
                "drop_rate": rate, "perturbation_seed": seed,
                "test_graphs": len(test_ids), "attack_graphs": int(labels.sum()),
                "reported_graphs": int(predicted.sum()),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
                "mcc": float(matthews_corrcoef(labels, predicted)),
                "auroc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "fit_calibration": "clean benign train/validation",
                "perturbation": "label-agnostic independent binomial thinning of test count marginals",
            })
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "variant", "drop_rate"], as_index=False).agg(
        repetitions=("perturbation_seed", "count"), precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        mcc_mean=("mcc", "mean"), auroc_mean=("auroc", "mean"),
        average_precision_mean=("average_precision", "mean"),
    ).fillna(0.0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "repetitions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "labels_used_to_apply_dropout": False,
        "calibration_reused_without_test_adaptation": True,
        "limitation": "Marginal count thinning approximates event dropout and does not preserve exact cross-feature covariance.",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
