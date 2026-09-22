"""Role-conditioned E3 gate with validation-only per-role calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_e3_multihop_context_gate import (  # noqa: E402
    best_threshold,
    best_threshold_with_required,
    campaign_nodes,
    context_features,
    labels_for,
    model_factory,
)
from experiments.run_e3_temporal_localized_tracking import (  # noqa: E402
    load_windows, partition_paths, percentile_scores, track,
)


ROLE_NAMES = {0: "subject", 1: "file", 2: "netflow"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="artifacts/raw_temporal_full/E3-CADETS/edge_embeds"
    )
    parser.add_argument(
        "--cache-dir", default="artifacts/feature_cache/E3-CADETS-Causal/aggregated_semantics"
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=(
            "D:/download/E3/ground-truth-usenix-sec-2025/"
            "ground-truth-usenix-sec-2025/darpa/E3-CADETS"
        ),
    )
    parser.add_argument(
        "--out-dir", default="artifacts/tifs_results/E3-CADETS-Causal/role_conditioned_gate"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--include-trajectory", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    relation_count = len(metadata["relations"])
    cached = {day: np.load(cache_dir / f"day_{day}.npz") for day in (6, 12, 13)}
    nodes = {day: cached[day]["nodes"] for day in cached}
    pooled = {day: cached[day]["pooled"] for day in cached}
    grouped = partition_paths(data_dir / "test", {12, 13})
    windows = {day: load_windows(grouped[day], nodes[day]) for day in (12, 13)}

    node_map = json.loads((data_dir / "nodeid2msg.json").read_text(encoding="utf-8"))
    uuid_to_node = {uuid.upper(): int(node) for node, uuid in node_map.items()}
    gt_dir = Path(args.ground_truth_dir)
    labels = {
        day: labels_for(
            nodes[day],
            campaign_nodes(
                gt_dir / f"node_Nginx_Backdoor_{day:02d}.csv", uuid_to_node
            ),
        )
        for day in (6, 12)
    }
    seed_model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=args.seed, n_jobs=-1,
    ).fit(pooled[6], labels[6])
    validation_base = seed_model.predict_proba(pooled[12])[:, 1]
    seed_threshold, _ = best_threshold(labels[12], validation_base)
    validation_seed = validation_base >= seed_threshold
    validation_context = context_features(
        windows[12], validation_seed, validation_base, pooled[12],
        relation_count, args.max_hops, args.include_trajectory,
    )
    validation_candidates, validation_features = validation_context[:2]
    validation_labels = labels[12][validation_candidates]
    validation_roles = np.argmax(
        pooled[12][validation_candidates, 29:32], axis=1
    )
    validation_localized, _, _ = track(
        windows[12], validation_seed, percentile_scores(validation_base),
        3, "contiguous_block", 10, "joint", relation_count,
    )
    validation_required = validation_localized[validation_candidates]

    test_base = seed_model.predict_proba(pooled[13])[:, 1]
    test_seed = test_base >= seed_threshold
    test_context = context_features(
        windows[13], test_seed, test_base, pooled[13],
        relation_count, args.max_hops, args.include_trajectory,
    )
    test_candidates, test_features = test_context[:2]
    test_roles = np.argmax(pooled[13][test_candidates, 29:32], axis=1)
    test_localized, _, _ = track(
        windows[13], test_seed, percentile_scores(test_base),
        3, "contiguous_block", 10, "joint", relation_count,
    )

    candidate_names = [
        "extra_trees_leaf1", "extra_trees_leaf2", "extra_trees_leaf4",
        "random_forest_leaf1",
    ]
    selection_rows = []
    role_choices = {}
    validation_scores = np.zeros(len(validation_candidates), dtype=np.float64)
    validation_predicted = validation_required.copy()
    test_scores = np.zeros(len(test_candidates), dtype=np.float64)
    test_predicted = test_localized[test_candidates].copy()
    for role_id, role_name in ROLE_NAMES.items():
        validation_index = np.flatnonzero(validation_roles == role_id)
        test_index = np.flatnonzero(test_roles == role_id)
        role_labels = validation_labels[validation_index]
        role_required = validation_required[validation_index]
        folds = min(5, int(role_labels.sum()), int((role_labels == 0).sum()))
        if folds < 2:
            raise RuntimeError(f"Insufficient {role_name} examples for role CV")
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=args.seed
        )
        role_rows = []
        role_oof = {}
        for name in candidate_names:
            scores = cross_val_predict(
                model_factory(name, args.seed),
                validation_features[validation_index], role_labels,
                cv=splitter, method="predict_proba", n_jobs=1,
            )[:, 1]
            threshold, f1 = best_threshold_with_required(
                role_labels, scores, role_required
            )
            row = {
                "role": role_name, "role_id": role_id, "candidate": name,
                "folds": folds, "nodes": len(validation_index),
                "positives": int(role_labels.sum()), "oof_f1": f1,
                "average_precision": float(
                    average_precision_score(role_labels, scores)
                ),
                "locked_threshold": threshold,
            }
            selection_rows.append(row)
            role_rows.append(row)
            role_oof[name] = scores
        chosen = sorted(
            role_rows,
            key=lambda row: (
                -row["oof_f1"], -row["average_precision"], row["candidate"]
            ),
        )[0]
        role_choices[role_name] = chosen
        chosen_scores = role_oof[str(chosen["candidate"])]
        validation_scores[validation_index] = chosen_scores
        validation_predicted[validation_index] |= (
            chosen_scores >= float(chosen["locked_threshold"])
        )
        model = model_factory(str(chosen["candidate"]), args.seed).fit(
            validation_features[validation_index], role_labels
        )
        role_test_scores = model.predict_proba(test_features[test_index])[:, 1]
        test_scores[test_index] = role_test_scores
        test_predicted[test_index] |= (
            role_test_scores >= float(chosen["locked_threshold"])
        )

    validation_precision, validation_recall, validation_f1, _ = (
        precision_recall_fscore_support(
            validation_labels, validation_predicted,
            average="binary", zero_division=0,
        )
    )
    # Day-13 labels are joined only after every role model and threshold freezes.
    test_truth = campaign_nodes(
        gt_dir / "node_Nginx_Backdoor_13.csv", uuid_to_node
    )
    full_labels = labels_for(nodes[13], test_truth)
    full_scores = np.zeros(len(nodes[13]), dtype=np.float64)
    full_scores[test_candidates] = test_scores
    full_predicted = np.zeros(len(nodes[13]), dtype=bool)
    full_predicted[test_candidates] = test_predicted
    precision, recall, f1, _ = precision_recall_fscore_support(
        full_labels, full_predicted, average="binary", zero_division=0
    )
    result = {
        "protocol_version": "tifs-e3-role-conditioned-gate-v1",
        "dataset": "E3-CADETS-Causal",
        "variant": "entity_role_conditioned_context_gate",
        "trajectory_features": args.include_trajectory,
        "seed_threshold": seed_threshold,
        "role_choices": role_choices,
        "validation_precision": float(validation_precision),
        "validation_recall": float(validation_recall),
        "validation_f1": float(validation_f1),
        "validation_average_precision": float(
            average_precision_score(validation_labels, validation_scores)
        ),
        "test_nodes": len(full_labels), "malicious_nodes": int(full_labels.sum()),
        "test_candidate_nodes": len(test_candidates),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(full_labels, full_predicted)),
        "auroc": float(roc_auc_score(full_labels, full_scores)),
        "average_precision": float(average_precision_score(full_labels, full_scores)),
        "reported_nodes": int(full_predicted.sum()),
        "true_reported_nodes": int((full_predicted & (full_labels == 1)).sum()),
        "false_reported_nodes": int((full_predicted & (full_labels == 0)).sum()),
        "result_tier": "exploratory_reused_holdout_role_conditioned",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(out / "validation_selection.csv", index=False)
    pd.DataFrame([{
        key: value for key, value in result.items() if key != "role_choices"
    }]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "candidate_index": test_candidates,
        "node_id": nodes[13][test_candidates],
        "role": [ROLE_NAMES[int(role)] for role in test_roles],
        "score": test_scores,
        "required_localized": test_localized[test_candidates].astype(np.int8),
        "predicted": test_predicted.astype(np.int8),
        "label": full_labels[test_candidates],
    }).to_csv(out / "test_candidate_decisions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result}, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(selection_rows).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
