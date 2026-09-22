"""Repeated benign-split evaluation from immutable full-stream sketch counts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

from detection.conformal import conformal_predictions  # noqa: E402
from run_streamspot_full_streaming_sketch import entropy_rows, normalized  # noqa: E402


def feature_variants(dataset: str, cache: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    edge_count = cache["edge_count"]
    relation = cache["relation_count"]
    if dataset == "StreamSpot":
        src, dst = cache["src_type_count"], cache["dst_type_count"]
    else:
        src, dst = cache["src_count"], cache["dst_count"]
    triple = cache["triple_count"]
    size = np.log1p(edge_count)[:, None].astype(np.float32)
    structure = np.concatenate([
        normalized(relation), normalized(src), normalized(dst), normalized(triple),
        (np.count_nonzero(relation, axis=1) / relation.shape[1])[:, None],
        (entropy_rows(relation) / np.log(relation.shape[1]))[:, None],
        (entropy_rows(src + dst) / np.log(src.shape[1]))[:, None],
    ], axis=1).astype(np.float32)
    return {
        "dual_head_full": np.concatenate([size, structure], axis=1),
        "normalized_structure_only": structure,
        "size_only": size,
    }


def split_ids(dataset: str, split_seed: int):
    rng = np.random.default_rng(split_seed)
    if dataset == "StreamSpot":
        train, validation, benign_test = [], [], []
        for start in [0, 100, 200, 400, 500]:
            ids = np.arange(start, start + 100)
            if split_seed:
                ids = rng.permutation(ids)
            train.extend(ids[:60])
            validation.extend(ids[60:80])
            benign_test.extend(ids[80:])
        attacks = list(range(300, 400))
    else:
        ids = np.arange(125)
        if split_seed:
            ids = rng.permutation(ids)
        train, validation, benign_test = ids[:75], ids[75:100], ids[100:]
        attacks = list(range(125, 175))
    test = np.asarray(list(benign_test) + attacks, dtype=int)
    labels = np.asarray([0] * len(benign_test) + [1] * len(attacks), dtype=int)
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int), test, labels


def confidence_rows(seed_rows: pd.DataFrame) -> list[dict]:
    rows = []
    metrics = ["precision", "recall", "f1", "mcc", "auroc", "average_precision"]
    for variant, group in seed_rows.groupby("variant", sort=False):
        record = {"row_type": "aggregate", "split_seed": None, "variant": variant,
                  "n_split_repetitions": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            mean = float(values.mean())
            half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95_low"] = max(0.0, mean - half)
            record[f"{metric}_ci95_high"] = min(1.0, mean + half)
        rows.append(record)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["StreamSpot", "Unicorn-Wget"])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    variants = feature_variants(args.dataset, np.load(args.cache))
    rows = []
    for seed in args.split_seeds:
        train_ids, val_ids, test_ids, labels = split_ids(args.dataset, seed)
        for variant, features in variants.items():
            scaler = StandardScaler().fit(features[train_ids])
            train = scaler.transform(features[train_ids])
            validation = scaler.transform(features[val_ids])
            test = scaler.transform(features[test_ids])
            model = NearestNeighbors(
                n_neighbors=min(args.neighbors, len(train)), metric="euclidean", algorithm="brute"
            ).fit(train)
            val_scores = model.kneighbors(validation, return_distance=True)[0].mean(axis=1)
            scores = model.kneighbors(test, return_distance=True)[0].mean(axis=1)
            predicted = conformal_predictions(val_scores, scores, args.alpha)
            rows.append({
                "row_type": "split_repetition", "dataset": args.dataset,
                "variant": variant, "split_seed": seed, "n_split_repetitions": 1,
                "train_graphs": len(train_ids), "calibration_graphs": len(val_ids),
                "test_graphs": len(test_ids), "attack_graphs": int(labels.sum()),
                "reported_graphs": int(predicted.sum()),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
                "mcc": float(matthews_corrcoef(labels, predicted)),
                "auroc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "calibration_alpha": args.alpha,
                "protocol": "scenario-stratified repeated benign split; attacks test-only",
            })
    frame = pd.DataFrame(rows)
    aggregates = confidence_rows(frame)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "split_repetitions.csv", index=False)
    pd.DataFrame(aggregates).to_csv(out / "aggregate.csv", index=False)
    with (out / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump({
            "dataset": args.dataset, "split_seeds": args.split_seeds,
            "labels_used_for_fit_or_calibration": False,
            "repetition_unit": "benign train/calibration/test split, not neural optimization seed",
            "cache": str(Path(args.cache).resolve()),
        }, file, indent=2)
    print(pd.DataFrame(aggregates).to_string(index=False))


if __name__ == "__main__":
    main()
