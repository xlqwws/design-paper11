"""Publish the upgraded proposed-method results without training artifacts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from detection.conformal import zero_exceedance_predictions  # noqa: E402
from evaluate_cached_full_sketch_repeats import feature_variants, split_ids  # noqa: E402


def metric_row(dataset, seed, labels, predicted, scores, variant, protocol):
    return {
        "dataset": dataset, "seed": seed, "variant": variant,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "reported": int(predicted.sum()),
        "true_positive": int(((labels == 1) & (predicted == 1)).sum()),
        "false_positive": int(((labels == 0) & (predicted == 1)).sum()),
        "protocol": protocol,
    }


def graph_rows(dataset: str, cache_path: Path) -> list[dict]:
    features = feature_variants(dataset, np.load(cache_path))["dual_head_full"]
    rows = []
    for seed in range(5):
        train_ids, validation_ids, test_ids, labels = split_ids(dataset, seed)
        scaler = StandardScaler().fit(features[train_ids])
        train = scaler.transform(features[train_ids])
        validation = scaler.transform(features[validation_ids])
        test = scaler.transform(features[test_ids])
        detector = NearestNeighbors(
            n_neighbors=5, metric="euclidean", algorithm="brute"
        ).fit(train)
        validation_scores = detector.kneighbors(validation, return_distance=True)[0].mean(axis=1)
        test_scores = detector.kneighbors(test, return_distance=True)[0].mean(axis=1)
        predicted = zero_exceedance_predictions(validation_scores, test_scores)
        rows.append(metric_row(
            dataset, seed, labels, predicted, test_scores,
            "dynamic_full_sketch_knn_zero_validation_exceedance",
            "scenario-stratified benign split; attacks test-only",
        ))
    return rows


def e3_rows() -> list[dict]:
    rows = []
    for seed in range(5):
        path = (
            ROOT / "artifacts/tifs_results/E3-CADETS-Causal"
            / f"proposed_v2_seed{seed}" / "metrics.csv"
        )
        source = pd.read_csv(path).iloc[0].to_dict()
        rows.append({
            "dataset": "E3-CADETS-Causal", "seed": seed,
            "variant": "dynamic_relation_path_context_gate",
            "precision": float(source["precision"]),
            "recall": float(source["recall"]),
            "f1": float(source["f1"]),
            "mcc": float(source["mcc"]),
            "auroc": float(source["auroc"]),
            "average_precision": float(source["average_precision"]),
            "reported": int(source["reported_nodes"]),
            "true_positive": int(source["true_reported_nodes"]),
            "false_positive": int(source["false_reported_nodes"]),
            "protocol": "day 6 supervised fit; day 12 supervised gate selection; frozen day 13 test",
        })
    return rows


def interval(values: np.ndarray, lower=0.0, upper=1.0):
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, max(lower, mean - half), min(upper, mean + half)


def aggregate(rows: list[dict], baseline_frame: pd.DataFrame) -> list[dict]:
    output = []
    for dataset, group in pd.DataFrame(rows).groupby("dataset", sort=False):
        record = {
            "dataset": dataset,
            "method": "online_dynamic_tracking_v2",
            "variant": group.iloc[0]["variant"],
            "n_repetitions": len(group),
            "result_tier": "exploratory_reused_public_holdout",
            "protocol": group.iloc[0]["protocol"],
            "test_labels_used_for_threshold_or_model_selection": False,
        }
        for metric in ["precision", "recall", "f1", "mcc", "auroc", "average_precision"]:
            values = group[metric].to_numpy(float)
            mean, low, high = interval(values, -1.0 if metric == "mcc" else 0.0, 1.0)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95_low"] = low
            record[f"{metric}_ci95_high"] = high
        candidates = baseline_frame[baseline_frame["dataset"] == dataset]
        best = candidates.sort_values("f1_mean", ascending=False).iloc[0]
        record["best_adapted_baseline"] = best["method"]
        record["best_adapted_baseline_f1"] = float(best["f1_mean"])
        record["absolute_f1_margin"] = record["f1_mean"] - record["best_adapted_baseline_f1"]
        record["ours_above_all_adapted_baselines"] = bool(record["absolute_f1_margin"] > 0)
        output.append(record)
    return output


def main() -> None:
    rows = []
    rows.extend(graph_rows(
        "StreamSpot",
        ROOT / "artifacts/tifs_results/StreamSpot/full_streaming_sketch_seed0/unlabelled_streaming_counts.npz",
    ))
    rows.extend(graph_rows(
        "Unicorn-Wget",
        ROOT / "artifacts/tifs_results/Unicorn-Wget/full_streaming_sketch_seed0/unlabelled_streaming_counts.npz",
    ))
    rows.extend(e3_rows())
    baselines = pd.read_csv(ROOT / "results/27_baseline_comparison/final_results.csv")
    aggregates = aggregate(rows, baselines)
    out = ROOT / "results/28_proposed_method_upgrade"
    out.mkdir(parents=True, exist_ok=True)
    for path in out.iterdir():
        if path.is_file() and path.name not in {"final_results.csv", "final_results.json"}:
            raise RuntimeError(f"Non-final file in result directory: {path}")
    pd.DataFrame(aggregates).to_csv(out / "final_results.csv", index=False)
    payload = {
        "metadata": {
            "protocol_version": "tifs-proposed-v2-zero-exceedance-v1",
            "environment_unchanged": True,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "integrity_notice": (
                "No test label is used by model fitting or threshold calibration. Results remain exploratory "
                "because the public holdouts were inspected during prior development."
            ),
        },
        "results": aggregates,
        "final_repetition_results": rows,
    }
    (out / "final_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(aggregates)[[
        "dataset", "f1_mean", "f1_ci95_low", "f1_ci95_high", "best_adapted_baseline",
        "best_adapted_baseline_f1", "absolute_f1_margin", "ours_above_all_adapted_baselines",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
