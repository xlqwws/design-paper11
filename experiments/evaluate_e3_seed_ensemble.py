"""Robust online ensemble of independently trained E3 causal GNN score streams."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection.conformal import upper_tail_pvalues  # noqa: E402
from detection.tail_calibration import EmpiricalGPDCalibrator  # noqa: E402


KEYS = ["time", "srcnode", "dstnode", "edge_type"]


def load_split(directory: Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(path, usecols=["loss", *KEYS]) for path in sorted(directory.glob("*.csv"))],
        ignore_index=True,
    )


def robust_parameters(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = max(float(1.4826 * np.median(np.abs(values - center))), np.finfo(float).eps)
    return center, scale


def aligned_ensemble(val_dirs: list[Path], test_dirs: list[Path]):
    validations = [load_split(path) for path in val_dirs]
    tests = [load_split(path) for path in test_dirs]
    reference_val = validations[0][KEYS].reset_index(drop=True)
    reference_test = tests[0][KEYS].reset_index(drop=True)
    val_evidence, test_evidence = [], []
    for seed, (validation, test) in enumerate(zip(validations, tests)):
        if not reference_val.equals(validation[KEYS].reset_index(drop=True)):
            raise ValueError(f"Validation stream for seed {seed} is not event-aligned")
        if not reference_test.equals(test[KEYS].reset_index(drop=True)):
            raise ValueError(f"Test stream for seed {seed} is not event-aligned")
        center, scale = robust_parameters(validation["loss"].to_numpy(float))
        val_evidence.append(np.maximum(0.0, (validation["loss"].to_numpy(float) - center) / scale))
        test_evidence.append(np.maximum(0.0, (test["loss"].to_numpy(float) - center) / scale))
    validation = reference_val.copy()
    test = reference_test.copy()
    validation["evidence"] = np.median(np.stack(val_evidence), axis=0)
    test["evidence"] = np.median(np.stack(test_evidence), axis=0)
    return validation, test


def node_scores(
    frame: pd.DataFrame, top_k: int, calibration=None, alpha=0.01, pvalue_model=None
):
    heaps = defaultdict(list)
    scores = defaultdict(float)
    first_alert = {}
    for row in frame.itertuples(index=False):
        for node in {int(row.srcnode), int(row.dstnode)}:
            heap = heaps[node]
            value = float(row.evidence)
            if len(heap) < top_k:
                heapq.heappush(heap, value)
            elif value > heap[0]:
                heapq.heapreplace(heap, value)
            score = float(sum(heap))
            scores[node] = score
            if calibration is not None and node not in first_alert:
                pvalue = (
                    pvalue_model.pvalues(np.asarray([score]))[0]
                    if pvalue_model is not None
                    else upper_tail_pvalues(calibration, np.asarray([score]))[0]
                )
                if pvalue <= alpha:
                    first_alert[node] = int(row.time)
    return scores, first_alert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dirs", nargs="+", required=True)
    parser.add_argument("--test-dirs", nargs="+", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--evt-alpha", type=float, default=0.001)
    args = parser.parse_args()
    if len(args.val_dirs) != len(args.test_dirs) or len(args.val_dirs) < 3:
        parser.error("Provide matching validation/test directories for at least three seeds")

    validation, test = aligned_ensemble(
        [Path(path) for path in args.val_dirs], [Path(path) for path in args.test_dirs]
    )
    val_nodes, _ = node_scores(validation, args.top_k)
    calibration = np.asarray(list(val_nodes.values()), dtype=float)
    evt_calibrator = EmpiricalGPDCalibrator(tail_fraction=0.1, min_tail=20).fit(calibration)
    test_nodes, first_alert = node_scores(test, args.top_k, calibration, args.alpha)
    _, evt_first_alert = node_scores(
        test, args.top_k, calibration, args.evt_alpha, pvalue_model=evt_calibrator
    )
    nodes = np.asarray(sorted(test_nodes), dtype=np.int64)
    scores = np.asarray([test_nodes[int(node)] for node in nodes])
    conformal_pvalues = upper_tail_pvalues(calibration, scores)
    evt_pvalues = evt_calibrator.pvalues(scores)
    truth = set(pd.read_csv(args.ground_truth)["node_id"].astype(int))
    labels = np.asarray([int(int(node) in truth) for node in nodes])
    common = {
        "protocol_version": "tifs-causal-robust-seed-ensemble-v1",
        "dataset": "E3-CADETS-Causal", "method": "median_robust_gnn_node_ensemble",
        "ensemble_members": len(args.val_dirs), "top_k": args.top_k,
        "calibration_nodes": len(calibration), "test_nodes": len(nodes),
        "malicious_nodes_in_test": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "online": True, "fusion": "per-seed validation robust z-score then event-wise median",
        "test_label_access": "metrics only after immutable node decisions",
    }
    metrics = []
    decisions_by_mode = {}
    for mode, alpha, pvalues in [
        ("finite_conformal", args.alpha, conformal_pvalues),
        ("gpd_extreme_tail", args.evt_alpha, evt_pvalues),
    ]:
        predicted = (pvalues <= alpha).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        metrics.append({**common, "calibration_mode": mode, "alpha": alpha,
            "reported_nodes": int(predicted.sum()),
            "true_reported_nodes": int(((labels == 1) & (predicted == 1)).sum()),
            "benign_reported_nodes": int(((labels == 0) & (predicted == 1)).sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "gpd_threshold": evt_calibrator.threshold if mode == "gpd_extreme_tail" else None,
            "gpd_shape": evt_calibrator.shape if mode == "gpd_extreme_tail" else None,
            "gpd_scale": evt_calibrator.scale if mode == "gpd_extreme_tail" else None,
        })
        decisions_by_mode[mode] = (pvalues, predicted)
    decisions = pd.DataFrame({
        "node_id": nodes, "score": scores,
        "conformal_pvalue": conformal_pvalues,
        "conformal_predicted": decisions_by_mode["finite_conformal"][1],
        "evt_pvalue": evt_pvalues,
        "evt_predicted": decisions_by_mode["gpd_extreme_tail"][1],
        "label": labels, "first_alert_time": [first_alert.get(int(node), -1) for node in nodes],
        "evt_first_alert_time": [evt_first_alert.get(int(node), -1) for node in nodes],
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    decisions.to_csv(out / "node_decisions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"results": metrics}, indent=2), encoding="utf-8")
    print(json.dumps({"results": metrics}, indent=2))


if __name__ == "__main__":
    main()
