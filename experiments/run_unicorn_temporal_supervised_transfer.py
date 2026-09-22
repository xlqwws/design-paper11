"""Known-family temporal transfer on full Unicorn-Wget graph sketches.

Early attack graphs 125--149 are divided into train/validation sets. Later
attack graphs 150--174 are always held out for test. Benign graph partitions
are repeated five times. All model and threshold selection uses validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_cached_full_sketch_repeats import feature_variants  # noqa: E402


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def model_factory(name: str, seed: int):
    if name == "logistic_c0.1":
        return LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=seed,
        )
    if name == "logistic_c1":
        return LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=seed,
        )
    if name == "linear_svm_c0.1":
        return LinearSVC(
            C=0.1, class_weight="balanced", dual=True,
            max_iter=20000, random_state=seed,
        )
    if name == "linear_svm_c1":
        return LinearSVC(
            C=1.0, class_weight="balanced", dual=True,
            max_iter=20000, random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
    raise ValueError(name)


def model_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    return np.asarray(model.decision_function(features), dtype=np.float64)


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return max(0.0, mean - half), min(1.0, mean + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=(
            "artifacts/tifs_results/Unicorn-Wget/full_streaming_sketch_seed0/"
            "unlabelled_streaming_counts.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tifs_results/Unicorn-Wget/temporal_supervised_transfer",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    variants = feature_variants("Unicorn-Wget", np.load(args.cache))
    variants = {name: values for name, values in variants.items() if name != "size_only"}
    methods = [
        "logistic_c0.1", "logistic_c1", "linear_svm_c0.1", "linear_svm_c1",
        "extra_trees", "random_forest",
    ]
    rows = []
    selection_rows = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        benign = rng.permutation(np.arange(125))
        early_attack = rng.permutation(np.arange(125, 150))
        train_ids = np.concatenate([benign[:75], early_attack[:15]])
        validation_ids = np.concatenate([benign[75:100], early_attack[15:]])
        test_ids = np.concatenate([benign[100:], np.arange(150, 175)])
        train_labels = np.concatenate([np.zeros(75), np.ones(15)]).astype(np.int8)
        validation_labels = np.concatenate([np.zeros(25), np.ones(10)]).astype(np.int8)

        fitted = {}
        for variant, all_features in variants.items():
            scaler = StandardScaler().fit(all_features[train_ids])
            train = scaler.transform(all_features[train_ids])
            validation = scaler.transform(all_features[validation_ids])
            test = scaler.transform(all_features[test_ids])
            for method in methods:
                model = model_factory(method, seed).fit(train, train_labels)
                validation_scores = model_scores(model, validation)
                threshold, validation_f1 = best_threshold(validation_labels, validation_scores)
                key = f"{variant}__{method}"
                fitted[key] = (model, scaler, test, threshold)
                selection_rows.append({
                    "split_seed": seed, "candidate": key,
                    "validation_f1": validation_f1, "locked_threshold": threshold,
                })

        seed_selection = [row for row in selection_rows if row["split_seed"] == seed]
        selected = sorted(
            seed_selection, key=lambda row: (-row["validation_f1"], row["candidate"])
        )[0]
        model, _, test, threshold = fitted[selected["candidate"]]
        test_scores = model_scores(model, test)
        test_predicted = (test_scores >= threshold).astype(np.int8)

        # Labels for the fixed later-attack test set are instantiated only after
        # predictions and the validation-selected threshold are immutable.
        test_labels = np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int8)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_predicted, average="binary", zero_division=0
        )
        rows.append({
            "protocol_version": "tifs-unicorn-known-family-temporal-transfer-v1",
            "dataset": "Unicorn-Wget", "split_seed": seed,
            "selected_candidate": selected["candidate"],
            "validation_f1": selected["validation_f1"],
            "locked_threshold": threshold,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "mcc": float(matthews_corrcoef(test_labels, test_predicted)),
            "auroc": float(roc_auc_score(test_labels, test_scores)),
            "average_precision": float(average_precision_score(test_labels, test_scores)),
            "reported_graphs": int(test_predicted.sum()),
            "true_reported_graphs": int(((test_predicted == 1) & (test_labels == 1)).sum()),
            "train_benign_graphs": 75, "train_early_attack_graphs": 15,
            "validation_benign_graphs": 25, "validation_early_attack_graphs": 10,
            "test_benign_graphs": 25, "test_later_attack_graphs": 25,
            "repetition_unit": "benign and early-attack partition; later attacks fixed",
            "result_tier": "exploratory_reused_holdout_known_family",
        })

    frame = pd.DataFrame(rows)
    aggregate = {
        "protocol_version": "tifs-unicorn-known-family-temporal-transfer-v1",
        "dataset": "Unicorn-Wget", "row_type": "aggregate",
        "n_repetitions": len(frame),
        "repetition_unit": "benign and early-attack partition; later attacks fixed",
        "result_tier": "exploratory_reused_holdout_known_family",
    }
    for metric in ["precision", "recall", "f1", "mcc", "auroc", "average_precision"]:
        values = frame[metric].to_numpy(float)
        low, high = confidence_interval(values)
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=1))
        aggregate[f"{metric}_ci95_low"] = low
        aggregate[f"{metric}_ci95_high"] = high
        aggregate[f"{metric}_min"] = float(values.min())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "split_repetitions.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(out / "aggregate.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "validation_model_selection.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps({
            "protocol": {
                "early_attack_ids": "125-149", "later_attack_ids": "150-174",
                "selection": "validation-only candidate and threshold selection",
                "claim_scope": "known-family temporal transfer, not zero-day detection",
            },
            "repetitions": json.loads(frame.to_json(orient="records")),
            "aggregate": aggregate,
        }, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)
    print(pd.DataFrame([aggregate]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
