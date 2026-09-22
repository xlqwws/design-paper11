"""Explore label-free graph-score calibration variants on existing full caches.

This is an exploratory reused-holdout probe.  Candidate definitions do not use
test labels, but the resulting comparison must not be promoted to confirmatory
evidence because these public test attacks have already been inspected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_cached_full_sketch_repeats import feature_variants, split_ids  # noqa: E402


def exceedance(calibration: np.ndarray, scores: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        threshold = np.max(calibration)
    else:
        threshold = np.sort(calibration)[-min(rank, len(calibration))]
    return (scores > threshold).astype(int)


def run(dataset: str, cache_path: Path) -> list[dict]:
    variants = feature_variants(dataset, np.load(cache_path))
    rows = []
    for seed in range(5):
        train_ids, validation_ids, test_ids, labels = split_ids(dataset, seed)
        head_scores = {}
        for name, features in variants.items():
            scaler = StandardScaler().fit(features[train_ids])
            train = scaler.transform(features[train_ids])
            validation = scaler.transform(features[validation_ids])
            test = scaler.transform(features[test_ids])
            knn = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(train)
            head_scores[name] = (
                knn.kneighbors(validation, return_distance=True)[0].mean(axis=1),
                knn.kneighbors(test, return_distance=True)[0].mean(axis=1),
            )

        full = variants["dual_head_full"]
        scaler = StandardScaler().fit(full[train_ids])
        train = scaler.transform(full[train_ids])
        validation = scaler.transform(full[validation_ids])
        test = scaler.transform(full[test_ids])
        isolation = IsolationForest(
            n_estimators=256, max_samples=min(256, len(train)), random_state=seed, n_jobs=-1
        ).fit(train)
        head_scores["isolation"] = (
            -isolation.score_samples(validation), -isolation.score_samples(test)
        )

        candidates = {}
        calibration_agreement = float(spearmanr(
            head_scores["dual_head_full"][0], head_scores["isolation"][0]
        )[0])
        for head, (validation_scores, test_scores) in head_scores.items():
            candidates[f"{head}_zero_exceedance"] = (
                exceedance(validation_scores, test_scores, 0), test_scores
            )
        # A causal alert is retained only when both independent evidence heads
        # exceed every calibration observation.  This is fixed before labels.
        candidates["dynamic_dual_confirm"] = (
            candidates["dual_head_full_zero_exceedance"][0]
            & candidates["isolation_zero_exceedance"][0],
            np.minimum(
                head_scores["dual_head_full"][1], head_scores["isolation"][1]
            ),
        )
        candidates["dynamic_size_confirm"] = (
            candidates["size_only_zero_exceedance"][0]
            & candidates["isolation_zero_exceedance"][0],
            np.minimum(head_scores["size_only"][1], head_scores["isolation"][1]),
        )
        for candidate, (predicted, scores) in candidates.items():
            rows.append({
                "dataset": dataset, "seed": seed, "candidate": candidate,
                "precision": precision_score(labels, predicted, zero_division=0),
                "recall": recall_score(labels, predicted, zero_division=0),
                "f1": f1_score(labels, predicted, zero_division=0),
                "auroc": roc_auc_score(labels, scores),
                "reported": int(predicted.sum()),
                "calibration_agreement": calibration_agreement,
            })
    return rows


def main() -> None:
    rows = []
    rows.extend(run(
        "StreamSpot",
        ROOT / "artifacts/tifs_results/StreamSpot/full_streaming_sketch_seed0/unlabelled_streaming_counts.npz",
    ))
    rows.extend(run(
        "Unicorn-Wget",
        ROOT / "artifacts/tifs_results/Unicorn-Wget/full_streaming_sketch_seed0/unlabelled_streaming_counts.npz",
    ))
    frame = pd.DataFrame(rows)
    print(frame.groupby(["dataset", "candidate"])[[
        "precision", "recall", "f1", "auroc", "calibration_agreement"
    ]].mean().to_string())
    print("\nvalidation-only agreement by split")
    print(frame.drop_duplicates(["dataset", "seed"])[[
        "dataset", "seed", "calibration_agreement"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
