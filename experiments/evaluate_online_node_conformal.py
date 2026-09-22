"""Evaluate bounded, causal node evidence with validation-only conformal control."""

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
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.conformal import upper_tail_pvalues  # noqa: E402


def read_stream(directory: Path):
    for path in sorted(directory.glob("*.csv")):
        for frame in pd.read_csv(
            path, usecols=["loss", "srcnode", "dstnode", "time"], chunksize=100_000
        ):
            for row in frame.itertuples(index=False):
                yield float(row.loss), int(row.srcnode), int(row.dstnode), int(row.time)


def insert_topk(heap: list[float], value: float, k: int) -> None:
    if len(heap) < k:
        heapq.heappush(heap, value)
    elif value > heap[0]:
        heapq.heapreplace(heap, value)


def aggregate_nodes(
    events, calibration_losses: np.ndarray, k: int, evidence_mode: str = "conformal_surprisal"
) -> tuple[dict[int, float], dict[int, int]]:
    heaps: dict[int, list[float]] = defaultdict(list)
    first_seen: dict[int, int] = {}
    median = float(np.median(calibration_losses))
    mad = float(np.median(np.abs(calibration_losses - median)))
    robust_scale = max(1.4826 * mad, np.finfo(float).eps)
    for loss, src, dst, timestamp in events:
        if evidence_mode == "robust_excess":
            evidence = max(0.0, (loss - median) / robust_scale)
        elif evidence_mode == "conformal_surprisal":
            pvalue = float(upper_tail_pvalues(calibration_losses, np.asarray([loss]))[0])
            evidence = -math.log(max(pvalue, np.finfo(float).tiny))
        else:
            raise ValueError(f"Unknown evidence mode: {evidence_mode}")
        for node in {src, dst}:
            insert_topk(heaps[node], evidence, k)
            first_seen.setdefault(node, timestamp)
    return {node: float(sum(values)) for node, values in heaps.items()}, first_seen


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "reported_nodes": int(y_pred.sum()),
        "true_reported_nodes": int(((y_true == 1) & (y_pred == 1)).sum()),
        "benign_reported_nodes": int(((y_true == 0) & (y_pred == 1)).sum()),
        "evaluated_nodes": int(len(y_true)),
        "malicious_nodes": int(y_true.sum()),
    }


def evaluate(
    val_dir: Path, test_dir: Path, truth_path: Path, alpha: float,
    topks: list[int], evidence_modes: list[str]
):
    calibration_losses = np.concatenate(
        [pd.read_csv(path, usecols=["loss"])["loss"].to_numpy(float) for path in sorted(val_dir.glob("*.csv"))]
    )
    truth = set(
        pd.read_csv(truth_path).loc[lambda frame: frame["label"].astype(int) == 1, "node_id"].astype(int)
    )
    results = []
    node_tables = {}
    for evidence_mode in evidence_modes:
      for k in topks:
        val_scores, _ = aggregate_nodes(read_stream(val_dir), calibration_losses, k, evidence_mode)
        test_scores, first_seen = aggregate_nodes(
            read_stream(test_dir), calibration_losses, k, evidence_mode
        )
        calibration_node_scores = np.asarray(list(val_scores.values()), dtype=float)
        nodes = np.asarray(sorted(test_scores), dtype=np.int64)
        scores = np.asarray([test_scores[int(node)] for node in nodes], dtype=float)
        pvalues = upper_tail_pvalues(calibration_node_scores, scores)
        predictions = (pvalues <= alpha).astype(int)
        labels = np.asarray([int(int(node) in truth) for node in nodes], dtype=int)
        metrics = binary_metrics(labels, predictions, scores)
        metrics.update(
            {
                "variant": f"{evidence_mode}_top{k}_sum",
                "evidence_mode": evidence_mode,
                "top_k": k,
                "alpha": alpha,
                "calibration_events": int(len(calibration_losses)),
                "calibration_nodes": int(len(calibration_node_scores)),
                "finite_sample_rule": "upper_tail_pvalue_le_alpha",
                "causal": True,
                "bounded_values_per_node": k,
            }
        )
        results.append(metrics)
        node_tables[(evidence_mode, k)] = pd.DataFrame(
            {
                "node_id": nodes,
                "score": scores,
                "pvalue": pvalues,
                "predicted": predictions,
                "label": labels,
                "first_seen_time": [first_seen[int(node)] for node in nodes],
            }
        )
    return results, node_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument(
        "--evidence-mode", nargs="+",
        choices=["robust_excess", "conformal_surprisal"],
        default=["robust_excess", "conformal_surprisal"],
    )
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    if any(k <= 0 for k in args.top_k):
        parser.error("--top-k values must be positive")

    results, node_tables = evaluate(
        Path(args.val_dir), Path(args.test_dir), Path(args.ground_truth), args.alpha,
        args.top_k, args.evidence_mode
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    for (evidence_mode, k), table in node_tables.items():
        table.to_csv(out_dir / f"node_decisions_{evidence_mode}_top{k}.csv", index=False)
    with (out_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "calibration": "benign validation nodes only",
                "test_label_access": "metrics join after immutable decisions",
                "online_statistic": "sum of fixed-capacity top-k robust excesses or event surprisals",
                "main_variant": "robust_excess_top3_sum",
                "sensitivity_variants": [
                    f"{mode}_top{k}_sum" for mode in args.evidence_mode for k in args.top_k
                ],
                "alpha": args.alpha,
            },
            file,
            indent=2,
        )
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
