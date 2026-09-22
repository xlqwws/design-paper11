"""Aggregate repeated runs and perform paired, seed-level comparisons."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "event_detection.mcc", "event_detection.average_precision",
    "node_attribution.precision", "node_attribution.recall",
    "tracing.edge_f1", "campaign_detection.attack_detection_rate",
    "prediction.next_stage_accuracy", "containment.attack_event_prevention_rate",
    "containment.benign_collateral_rate", "efficiency.latency_us_p99",
]


def exact_sign_test(differences):
    nonzero = [value for value in differences if value != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, n - positives)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def paired_bootstrap(differences, seed, repetitions=10000):
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(repetitions, len(values)), replace=True), axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_adjust(pvalues):
    """Return Holm-adjusted p-values in their original order."""
    if not pvalues:
        return []
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(pvalues) - rank) * float(pvalues[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Directory containing evaluator metrics.csv files")
    parser.add_argument("--reference", default="full")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = list(Path(args.root).rglob("metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv files found below {args.root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"dataset", "method", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Metrics files require columns {sorted(required)}")

    summaries = []
    comparisons = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        for metric in args.metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            summaries.append({
                "dataset": dataset, "method": method, "metric": metric, "runs": len(values),
                "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()), "max": float(values.max()),
                "relative_std": float(values.std(ddof=1) / abs(values.mean())) if len(values) > 1 and values.mean() != 0 else 0.0,
                "ci95_low": float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float(values.mean()),
                "ci95_high": float(values.mean() + 1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float(values.mean()),
            })

    for dataset, group in frame.groupby("dataset"):
        reference = group[group["method"] == args.reference]
        for method in sorted(set(group["method"]) - {args.reference}):
            candidate = group[group["method"] == method]
            paired = reference.merge(candidate, on=["dataset", "seed"], suffixes=("_ref", "_candidate"))
            for metric in args.metrics:
                ref_column, candidate_column = f"{metric}_ref", f"{metric}_candidate"
                if ref_column not in paired or candidate_column not in paired:
                    continue
                valid = paired[[ref_column, candidate_column]].dropna()
                differences = (valid[ref_column] - valid[candidate_column]).tolist()
                low, high = paired_bootstrap(differences, args.seed)
                comparisons.append({
                    "dataset": dataset, "reference": args.reference, "candidate": method,
                    "metric": metric, "paired_runs": len(differences),
                    "mean_difference": float(np.mean(differences)) if differences else None,
                    "difference_ci95_low": low, "difference_ci95_high": high,
                    "exact_sign_p": exact_sign_test(differences),
                    "paired_effect_size_dz": (
                        float(np.mean(differences) / np.std(differences, ddof=1))
                        if len(differences) > 1 and np.std(differences, ddof=1) > 0 else None
                    ),
                })

    primary_indices = [index for index, row in enumerate(comparisons) if row["metric"] in args.metrics[:4]]
    adjusted = holm_adjust([comparisons[index]["exact_sign_p"] for index in primary_indices])
    for index, pvalue in zip(primary_indices, adjusted):
        comparisons[index]["holm_adjusted_sign_p"] = pvalue

    out_dir = Path(args.out_dir or args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_dir / "aggregate_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "paired_comparisons.csv", index=False)
    with open(out_dir / "aggregate_manifest.json", "w", encoding="utf-8") as file:
        json.dump({"input_files": [str(path) for path in paths], "reference": args.reference,
                   "metrics": args.metrics}, file, indent=2)
    print(f"Aggregated {len(paths)} runs into {out_dir}")


if __name__ == "__main__":
    main()
