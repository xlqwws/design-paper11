"""Validation-selected cross-day rank ensemble for E3 dynamic graph gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def assert_aligned(left: pd.DataFrame, right: pd.DataFrame, split: str) -> None:
    for column in ["candidate_index", "node_id", "label", "required_localized"]:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"Misaligned {split} column: {column}")


def best_required_threshold(labels, scores, required):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    required = np.asarray(required, dtype=bool)
    tp = int((required & (labels == 1)).sum())
    fp = int((required & (labels == 0)).sum())
    positives = int(labels.sum())
    best_f1 = 2 * tp / max(2 * tp + fp + positives - tp, 1)
    best_threshold = float(np.nextafter(scores.max(), np.inf))
    order = np.argsort(-scores, kind="mergesort")
    index = 0
    while index < len(order):
        end = index + 1
        value = scores[order[index]]
        while end < len(order) and scores[order[end]] == value:
            end += 1
        group = order[index:end]
        added = group[~required[group]]
        tp += int((labels[added] == 1).sum())
        fp += int((labels[added] == 0).sum())
        f1 = 2 * tp / max(2 * tp + fp + positives - tp, 1)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_threshold = float(value)
        index = end
    return best_threshold, float(best_f1)


def stack_features(tree: pd.DataFrame, rgcn: pd.DataFrame, split: str):
    score_name = "gate_oof_score" if split == "validation" else "gate_score"
    rank_name = "gate_oof_percentile" if split == "validation" else "gate_percentile"
    return np.column_stack([
        tree[score_name], rgcn[score_name], tree[rank_name], rgcn[rank_name],
        tree["base_score"],
    ]).astype(np.float64)


def add_score_variants(validation, test, tree_validation, tree_test, rgcn_validation, rgcn_test):
    validation["tree_raw"] = tree_validation["gate_oof_score"].to_numpy(float)
    test["tree_raw"] = tree_test["gate_score"].to_numpy(float)
    validation["tree_rank"] = tree_validation["gate_oof_percentile"].to_numpy(float)
    test["tree_rank"] = tree_test["gate_percentile"].to_numpy(float)
    validation["rgcn_raw"] = rgcn_validation["gate_oof_score"].to_numpy(float)
    test["rgcn_raw"] = rgcn_test["gate_score"].to_numpy(float)
    validation["rgcn_rank"] = rgcn_validation["gate_oof_percentile"].to_numpy(float)
    test["rgcn_rank"] = rgcn_test["gate_percentile"].to_numpy(float)

    tree_rank_validation = validation["tree_rank"]
    tree_rank_test = test["tree_rank"]
    rgcn_rank_validation = validation["rgcn_rank"]
    rgcn_rank_test = test["rgcn_rank"]
    for tree_weight in np.linspace(0.0, 1.0, 21):
        name = f"rank_blend_tree_{tree_weight:.2f}"
        validation[name] = tree_weight * tree_rank_validation + (1 - tree_weight) * rgcn_rank_validation
        test[name] = tree_weight * tree_rank_test + (1 - tree_weight) * rgcn_rank_test

    validation_features = stack_features(tree_validation, rgcn_validation, "validation")
    test_features = stack_features(tree_test, rgcn_test, "test")
    labels = tree_validation["label"].to_numpy(np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for regularization in (0.1, 1.0, 10.0):
        name = f"nested_logistic_c_{regularization:g}"
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization, class_weight="balanced", max_iter=2000,
                random_state=0,
            ),
        )
        validation[name] = cross_val_predict(
            model, validation_features, labels, cv=splitter,
            method="predict_proba", n_jobs=1,
        )[:, 1]
        model.fit(validation_features, labels)
        test[name] = model.predict_proba(test_features)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree-dir",
        default="artifacts/tifs_results/E3-CADETS-Causal/multihop_context_gate_monotone_localized",
    )
    parser.add_argument(
        "--rgcn-dir",
        default="artifacts/tifs_results/E3-CADETS-Causal/rgcn_temporal_trajectory_gate",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tifs_results/E3-CADETS-Causal/rank_ensemble_gate",
    )
    args = parser.parse_args()

    tree_dir = Path(args.tree_dir)
    rgcn_dir = Path(args.rgcn_dir)
    tree_validation = pd.read_csv(tree_dir / "validation_candidate_scores.csv")
    tree_test = pd.read_csv(tree_dir / "test_candidate_scores.csv")
    rgcn_validation = pd.read_csv(rgcn_dir / "validation_candidate_scores.csv")
    rgcn_test = pd.read_csv(rgcn_dir / "test_candidate_scores.csv")
    assert_aligned(tree_validation, rgcn_validation, "validation")
    assert_aligned(tree_test, rgcn_test, "test")

    validation_scores = {}
    test_scores = {}
    add_score_variants(
        validation_scores, test_scores,
        tree_validation, tree_test, rgcn_validation, rgcn_test,
    )
    validation_labels = tree_validation["label"].to_numpy(np.int8)
    validation_required = tree_validation["required_localized"].to_numpy(bool)
    rows = []
    for complexity, name in enumerate(validation_scores):
        scores = validation_scores[name]
        threshold, f1 = best_required_threshold(
            validation_labels, scores, validation_required
        )
        predicted = validation_required | (scores >= threshold)
        precision, recall, _, _ = precision_recall_fscore_support(
            validation_labels, predicted, average="binary", zero_division=0
        )
        rows.append({
            "candidate": name, "validation_f1": f1,
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_average_precision": float(
                average_precision_score(validation_labels, scores)
            ),
            "threshold": threshold, "complexity_order": complexity,
        })
    selection = pd.DataFrame(rows).sort_values(
        ["validation_f1", "validation_average_precision", "complexity_order"],
        ascending=[False, False, True],
    )
    chosen = selection.iloc[0]
    chosen_name = str(chosen["candidate"])
    locked_threshold = float(chosen["threshold"])
    chosen_test_scores = test_scores[chosen_name]
    test_required = tree_test["required_localized"].to_numpy(bool)
    test_predicted = test_required | (chosen_test_scores >= locked_threshold)

    # Reused day-13 labels are accessed only after model, variant and threshold freeze.
    test_labels = tree_test["label"].to_numpy(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_predicted, average="binary", zero_division=0
    )
    outside_candidates = 30893 - len(test_labels)
    full_labels = np.concatenate([test_labels, np.zeros(outside_candidates, dtype=np.int8)])
    full_scores = np.concatenate([chosen_test_scores, np.zeros(outside_candidates)])
    full_predicted = np.concatenate([
        test_predicted, np.zeros(outside_candidates, dtype=bool)
    ])
    result = {
        "protocol_version": "tifs-e3-rank-ensemble-gate-v1",
        "dataset": "E3-CADETS-Causal",
        "variant": "validation_selected_cross_day_rank_ensemble",
        "selected_candidate": chosen_name,
        "validation_f1": float(chosen["validation_f1"]),
        "validation_average_precision": float(chosen["validation_average_precision"]),
        "locked_threshold": locked_threshold,
        "validation_candidates": len(validation_labels),
        "test_candidates": len(test_labels),
        "test_nodes": len(full_labels),
        "malicious_nodes": int(full_labels.sum()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(full_labels, full_predicted)),
        "auroc": float(roc_auc_score(full_labels, full_scores)),
        "average_precision": float(average_precision_score(full_labels, full_scores)),
        "reported_nodes": int(full_predicted.sum()),
        "true_reported_nodes": int((full_predicted & (full_labels == 1)).sum()),
        "false_reported_nodes": int((full_predicted & (full_labels == 0)).sum()),
        "required_localized_nodes": int(test_required.sum()),
        "result_tier": "exploratory_reused_holdout_validation_selected_ensemble",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "validation_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "candidate_index": tree_test["candidate_index"],
        "node_id": tree_test["node_id"],
        "score": chosen_test_scores,
        "required_localized": test_required.astype(np.int8),
        "predicted": test_predicted.astype(np.int8),
        "label": test_labels,
    }).to_csv(out / "test_candidate_decisions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result}, indent=2), encoding="utf-8"
    )
    print(selection.head(12).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
