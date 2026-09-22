"""Temporal-localized, relation-normalized E3 attack-subgraph tracking.

The April 6 campaign trains the seed detector. April 12 selects all tracking
parameters. April 13 is scored and tracked before its labels are loaded.
This experiment is exploratory because April 13 was consumed by earlier runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
EASTERN = pytz.timezone("America/New_York")


def campaign_nodes(path: Path, uuid_to_node: dict[str, int]) -> set[int]:
    values = set()
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for row in csv.reader(file):
            if row:
                uuid = row[0].strip().upper()
                if uuid in uuid_to_node:
                    values.add(uuid_to_node[uuid])
    return values


def labels_for(nodes: np.ndarray, truth: set[int]) -> np.ndarray:
    return np.fromiter((int(int(node) in truth) for node in nodes), dtype=np.int8)


def percentile_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = (np.arange(len(scores), dtype=np.float32) + 1) / len(scores)
    return ranks


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def partition_paths(test_dir: Path, days: set[int]) -> dict[int, list[Path]]:
    grouped = {day: [] for day in days}
    for path in sorted(test_dir.glob("*.TemporalData")):
        data = torch.load(path)
        day = datetime.fromtimestamp(int(data.t[0]) / 1e9, EASTERN).day
        if day in grouped:
            grouped[day].append(path)
    return grouped


def load_windows(paths: list[Path], nodes: np.ndarray) -> list[dict[str, np.ndarray]]:
    windows = []
    for path in paths:
        data = torch.load(path)
        windows.append({
            "src": np.searchsorted(nodes, data.src.cpu().numpy()).astype(np.int32),
            "dst": np.searchsorted(nodes, data.dst.cpu().numpy()).astype(np.int32),
            "relation": data.edge_type.cpu().numpy().astype(np.int8),
        })
    return windows


def select_windows(
    windows: list[dict[str, np.ndarray]], seed: np.ndarray,
    rank_scores: np.ndarray, count: int, strategy: str,
) -> list[dict[str, np.ndarray]]:
    evidence = []
    for index, window in enumerate(windows):
        active = np.unique(np.concatenate([window["src"], window["dst"]]))
        active_seed = active[seed[active]]
        evidence.append((len(active_seed), float(rank_scores[active_seed].sum()), -index, window))
    if strategy == "top_evidence":
        return [
            item[3]
            for item in sorted(evidence, key=lambda item: item[:3], reverse=True)[:count]
        ]
    if strategy == "contiguous_block":
        if count >= len(evidence):
            return [item[3] for item in evidence]
        candidates = []
        for start in range(len(evidence) - count + 1):
            block = evidence[start:start + count]
            candidates.append((
                sum(item[0] for item in block),
                sum(item[1] for item in block),
                -start,
                [item[3] for item in block],
            ))
        return max(candidates, key=lambda item: item[:3])[3]
    raise ValueError(f"Unknown window selection strategy: {strategy}")


def track(
    windows: list[dict[str, np.ndarray]], seed: np.ndarray, rank_scores: np.ndarray,
    top_windows: int, window_strategy: str, degree_cap: int,
    degree_mode: str, num_relations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    selected = select_windows(windows, seed, rank_scores, top_windows, window_strategy)
    weights = rank_scores.copy()
    tracked = seed.copy()
    updates = 0

    for window in selected:
        relation_groups = range(num_relations) if degree_mode == "per_relation" else (-1,)
        for relation in relation_groups:
            mask = (
                np.ones(len(window["relation"]), dtype=bool)
                if relation < 0 else window["relation"] == relation
            )
            src = window["src"][mask]
            dst = window["dst"][mask]
            if not len(src):
                continue
            degree = np.bincount(np.concatenate([src, dst]), minlength=len(seed))
            eligible = seed & (degree <= degree_cap)

            forward = eligible[src]
            if forward.any():
                targets = dst[forward]
                candidates = weights[src[forward]]
                improving = candidates > weights[targets]
                np.maximum.at(weights, targets, candidates)
                changed = targets[improving]
                tracked[changed] = True
                updates += len(np.unique(changed))

            backward = eligible[dst]
            if backward.any():
                targets = src[backward]
                candidates = weights[dst[backward]]
                improving = candidates > weights[targets]
                np.maximum.at(weights, targets, candidates)
                changed = targets[improving]
                tracked[changed] = True
                updates += len(np.unique(changed))
    return tracked, weights, updates


def classification_metrics(labels: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "reported_nodes": int(predicted.sum()),
        "true_reported_nodes": int(((predicted == 1) & (labels == 1)).sum()),
        "false_reported_nodes": int(((predicted == 1) & (labels == 0)).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="artifacts/raw_temporal_full/E3-CADETS/edge_embeds"
    )
    parser.add_argument(
        "--cache-dir", default="artifacts/feature_cache/E3-CADETS-Causal/cross_campaign_transfer"
    )
    parser.add_argument(
        "--feature-key", default="features",
        help="Array key inside each day cache (for example: features, pooled, all_views).",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=(
            "D:/download/E3/ground-truth-usenix-sec-2025/"
            "ground-truth-usenix-sec-2025/darpa/E3-CADETS"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tifs_results/E3-CADETS-Causal/temporal_localized_tracking",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    cached = {day: np.load(cache_dir / f"day_{day}.npz") for day in (6, 12, 13)}
    nodes = {day: cached[day]["nodes"] for day in cached}
    features = {day: cached[day][args.feature_key] for day in cached}
    grouped = partition_paths(data_dir / "test", {12, 13})
    windows = {day: load_windows(grouped[day], nodes[day]) for day in (12, 13)}

    node_map = json.loads((data_dir / "nodeid2msg.json").read_text(encoding="utf-8"))
    uuid_to_node = {uuid.upper(): int(node) for node, uuid in node_map.items()}
    del node_map
    gt_dir = Path(args.ground_truth_dir)
    train_truth = campaign_nodes(gt_dir / "node_Nginx_Backdoor_06.csv", uuid_to_node)
    validation_truth = campaign_nodes(gt_dir / "node_Nginx_Backdoor_12.csv", uuid_to_node)
    train_labels = labels_for(nodes[6], train_truth)
    validation_labels = labels_for(nodes[12], validation_truth)

    model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=args.seed, n_jobs=-1,
    ).fit(features[6], train_labels)
    validation_raw = model.predict_proba(features[12])[:, 1]
    seed_threshold, seed_validation_f1 = best_threshold(validation_labels, validation_raw)
    validation_seed = validation_raw >= seed_threshold
    validation_rank = percentile_scores(validation_raw)

    grid = []
    for top_windows in (1, 2, 3, 4, 5):
        for window_strategy in ("contiguous_block", "top_evidence"):
            for degree_cap in (1, 2, 3, 5, 10, 20, 50):
                for degree_mode in ("joint", "per_relation"):
                    predicted, propagated, updates = track(
                        windows[12], validation_seed, validation_rank, top_windows,
                        window_strategy, degree_cap, degree_mode, len(metadata["relations"]),
                    )
                    row = {
                        "top_windows": top_windows, "window_strategy": window_strategy,
                        "degree_cap": degree_cap, "degree_mode": degree_mode,
                        "monotone_updates": updates,
                        **classification_metrics(validation_labels, predicted, propagated),
                    }
                    grid.append(row)
    selection = pd.DataFrame(grid).sort_values(
        ["f1", "top_windows", "window_strategy", "degree_cap", "degree_mode"],
        ascending=[False, True, True, True, True],
    )
    chosen = selection.iloc[0]

    test_raw = model.predict_proba(features[13])[:, 1]
    test_seed = test_raw >= seed_threshold
    test_rank = percentile_scores(test_raw)
    test_predicted, test_propagated, test_updates = track(
        windows[13], test_seed, test_rank, int(chosen["top_windows"]),
        str(chosen["window_strategy"]), int(chosen["degree_cap"]),
        str(chosen["degree_mode"]),
        len(metadata["relations"]),
    )

    # Test labels are intentionally accessed only after model selection and all
    # test decisions have been frozen.
    test_truth = campaign_nodes(gt_dir / "node_Nginx_Backdoor_13.csv", uuid_to_node)
    test_labels = labels_for(nodes[13], test_truth)
    seed_result = {
        "protocol_version": "tifs-e3-temporal-localized-tracking-v1",
        "dataset": "E3-CADETS-Causal", "variant": "seed_detector",
        "seed": args.seed, "feature_key": args.feature_key,
        "seed_threshold": seed_threshold,
        "validation_f1": seed_validation_f1,
        **classification_metrics(test_labels, test_seed, test_raw),
    }
    tracked_result = {
        "protocol_version": "tifs-e3-temporal-localized-tracking-v1",
        "dataset": "E3-CADETS-Causal", "variant": "temporal_relation_normalized_tracking",
        "seed": args.seed, "feature_key": args.feature_key,
        "seed_threshold": seed_threshold,
        "top_windows": int(chosen["top_windows"]),
        "window_strategy": str(chosen["window_strategy"]),
        "degree_cap": int(chosen["degree_cap"]),
        "degree_mode": str(chosen["degree_mode"]),
        "validation_f1": float(chosen["f1"]),
        "monotone_updates": test_updates,
        **classification_metrics(test_labels, test_predicted, test_propagated),
    }
    common = {
        "train_campaign": "2018-04-06", "validation_campaign": "2018-04-12",
        "test_campaign": "2018-04-13", "test_nodes": len(test_labels),
        "malicious_nodes": int(test_labels.sum()),
        "label_access_order": "test labels loaded after test decisions were frozen",
        "result_tier": "exploratory_reused_holdout",
        "elapsed_seconds": time.perf_counter() - started,
    }
    rows = [{**seed_result, **common}, {**tracked_result, **common}]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "validation_tracking_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"results": rows, "source_metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    print(selection.head(10).to_string(index=False), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
