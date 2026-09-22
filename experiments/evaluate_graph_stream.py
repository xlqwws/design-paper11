"""Evaluate graph-labelled online traces such as StreamSpot and Unicorn."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection.conformal import conformal_predictions, upper_tail_pvalues  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="online_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    return parser.parse_args()


def finite_sample_threshold(scores, alpha):
    values = sorted(float(value) for value in scores)
    if not values:
        raise ValueError("Graph-level conformal calibration requires validation graphs")
    if not 0.0 < alpha < 1.0:
        raise ValueError("calibration alpha must be in (0, 1)")
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - alpha)))
    return values[rank - 1]


def load_graph_calibration(directory):
    paths = sorted(Path(directory).glob("*.csv"))
    scores = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["loss"])
        if not frame.empty:
            scores.append(float(frame["loss"].max()))
    return scores


def main():
    args = parse_args()
    events = pd.read_csv(args.events)
    truth = pd.read_csv(args.ground_truth)
    required = {"time_window", "graph_id", "label"}
    if not required.issubset(truth.columns):
        raise ValueError(f"Ground truth requires {sorted(required)}")
    grouped = events.groupby("time_window", as_index=False).agg(
        score=("loss", "max"),
        predicted=("activated", "max"),
        cut=("cut", "max"),
        events=("event_id", "count"),
    )
    frame = truth.merge(grouped, on="time_window", how="left", validate="one_to_one").fillna(0)
    calibration_scores = []
    graph_threshold = None
    if args.calibration_dir:
        calibration_scores = load_graph_calibration(args.calibration_dir)
        graph_threshold = finite_sample_threshold(calibration_scores, args.calibration_alpha)
        frame["conformal_pvalue"] = upper_tail_pvalues(calibration_scores, frame["score"])
        frame["predicted"] = conformal_predictions(
            calibration_scores, frame["score"], args.calibration_alpha
        )
    y_true = frame["label"].astype(int)
    y_pred = frame["predicted"].astype(int)
    score = frame["score"].astype(float)
    result = {
        "protocol_version": "tifs-causal-graph-v1",
        "dataset": args.dataset,
        "method": (
            "online_dynamic_graph_conformal" if args.calibration_dir else args.method
        ),
        "seed": args.seed,
        "graphs": len(frame),
        "attack_graphs": int(y_true.sum()),
        "reported_graphs": int(y_pred.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "graphs_with_cut": int(frame["cut"].astype(int).sum()),
        "calibration_unit": "graph" if args.calibration_dir else "event",
        "calibration_graphs": len(calibration_scores),
        "calibration_alpha": args.calibration_alpha if args.calibration_dir else None,
        "graph_threshold": graph_threshold,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    pd.DataFrame([result]).to_csv(out_dir / "metrics.csv", index=False)
    frame.to_csv(out_dir / "per_graph_scores.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
