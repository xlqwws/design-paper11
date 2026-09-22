"""Evaluate frozen-score detection under reproducible stream perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-events", required=True)
    parser.add_argument("--e3-ground-truth", required=True)
    parser.add_argument("--streamspot-events", required=True)
    parser.add_argument("--streamspot-ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--perturbation-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def binary_counts(truth, pred):
    truth = np.asarray(truth, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def conditions():
    yield "clean", "none", 0.0
    for rate in (0.1, 0.3, 0.5):
        yield f"event_deletion_{int(rate * 100)}pct", "deletion", rate
    for sigma in (0.05, 0.10, 0.20):
        yield f"gaussian_score_noise_{int(sigma * 100)}pct", "noise", sigma
    for attenuation in (0.9, 0.8, 0.7):
        yield f"score_attenuation_{round((1 - attenuation) * 100)}pct", "attenuation", attenuation


def perturb(frame, kind, value, rng, threshold):
    output = frame.copy()
    if kind == "deletion":
        output = output.loc[rng.random(len(output)) >= value].copy()
    elif kind == "noise":
        output["loss"] = np.maximum(0.0, output["loss"].to_numpy(float) + rng.normal(0, value * threshold, len(output)))
    elif kind == "attenuation":
        output["loss"] = output["loss"].to_numpy(float) * value
    return output


def e3_rows(events_path, truth_path, seeds):
    events = pd.read_csv(events_path, usecols=["srcnode", "dstnode", "loss", "threshold"])
    events[["srcnode", "dstnode"]] = events[["srcnode", "dstnode"]].astype(str)
    truth = pd.read_csv(truth_path)
    node_column = "node_id" if "node_id" in truth else "nid"
    label_column = next(name for name in ("label", "y_true", "is_malicious") if name in truth)
    malicious = set(truth.loc[truth[label_column].astype(int) == 1, node_column].astype(str))
    all_nodes = sorted(set(events.srcnode) | set(events.dstnode) | malicious)
    threshold = float(events.threshold.iloc[0])
    rows = []
    for condition, kind, value in conditions():
        active_seeds = [seeds[0]] if kind == "none" else seeds
        for seed in active_seeds:
            changed = perturb(events, kind, value, np.random.default_rng(seed), threshold)
            alerts = changed.loc[changed.loss >= threshold]
            selected = set(alerts.srcnode) | set(alerts.dstnode)
            y_true = [int(node in malicious) for node in all_nodes]
            y_pred = [int(node in selected) for node in all_nodes]
            precision, recall, f1 = binary_counts(y_true, y_pred)
            rows.append({
                "dataset": "E3-CADETS-Causal", "condition": condition,
                "perturbation_seed": seed, "model_seed": 0, "status": "pilot",
                "evaluation_scope": "frozen_score_primary_detection",
                "precision": precision, "recall": recall, "f1": f1,
                "reported_units": len(selected), "events_retained": len(changed),
            })
    return rows


def streamspot_rows(events_path, truth_path, seeds):
    events = pd.read_csv(events_path, usecols=["time_window", "loss", "threshold"])
    truth = pd.read_csv(truth_path)[["time_window", "label"]]
    threshold = float(events.threshold.iloc[0])
    rows = []
    for condition, kind, value in conditions():
        active_seeds = [seeds[0]] if kind == "none" else seeds
        for seed in active_seeds:
            changed = perturb(events, kind, value, np.random.default_rng(seed), threshold)
            scores = changed.groupby("time_window").loss.max().rename("score")
            frame = truth.merge(scores, on="time_window", how="left").fillna({"score": 0.0})
            y_true = frame.label.astype(int).to_numpy()
            score = frame.score.astype(float).to_numpy()
            y_pred = (score >= threshold).astype(int)
            precision, recall, f1 = binary_counts(y_true, y_pred)
            rows.append({
                "dataset": "StreamSpot", "condition": condition,
                "perturbation_seed": seed, "model_seed": 0, "status": "pilot",
                "evaluation_scope": "frozen_score_graph_detection",
                "precision": precision, "recall": recall, "f1": f1,
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "auroc": float(roc_auc_score(y_true, score)),
                "average_precision": float(average_precision_score(y_true, score)),
                "reported_units": int(y_pred.sum()), "events_retained": len(changed),
            })
    return rows


def main():
    args = parse_args()
    rows = e3_rows(args.e3_events, args.e3_ground_truth, args.perturbation_seeds)
    rows.extend(streamspot_rows(args.streamspot_events, args.streamspot_ground_truth, args.perturbation_seeds))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "condition", "status", "evaluation_scope"], as_index=False).agg(
        repetitions=("perturbation_seed", "count"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        reported_units_mean=("reported_units", "mean"), events_retained_mean=("events_retained", "mean"),
    ).fillna(0.0)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "final_results.csv", index=False)
    payload = {
        "experiment": "frozen_score_robustness",
        "claim_scope": "Perturbation repetitions are not independent model-training seeds.",
        "protocol": "Test-only random deletion, score attenuation, and zero-mean score noise; the validation threshold remains frozen.",
        "results": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
    }
    with (output / "final_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
