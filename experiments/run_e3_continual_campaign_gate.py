"""Continual E3 context gate using prior-campaign out-of-fold candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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


def fold_predictions(
    name, seed, prior_features, prior_labels, validation_features,
    validation_labels, prior_weight,
):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    output = np.zeros(len(validation_labels), dtype=np.float64)
    for train_index, holdout_index in splitter.split(
        validation_features, validation_labels
    ):
        features = np.concatenate([
            prior_features, validation_features[train_index]
        ])
        labels = np.concatenate([
            prior_labels, validation_labels[train_index]
        ])
        weights = np.concatenate([
            np.full(len(prior_labels), prior_weight),
            np.ones(len(train_index)),
        ])
        model = model_factory(name, seed).fit(
            features, labels, sample_weight=weights
        )
        output[holdout_index] = model.predict_proba(
            validation_features[holdout_index]
        )[:, 1]
    return output


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
        "--out-dir", default="artifacts/tifs_results/E3-CADETS-Causal/continual_campaign_gate"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--include-trajectory", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    relations = len(metadata["relations"])
    cached = {day: np.load(cache_dir / f"day_{day}.npz") for day in (6, 12, 13)}
    nodes = {day: cached[day]["nodes"] for day in cached}
    pooled = {day: cached[day]["pooled"] for day in cached}
    grouped = partition_paths(data_dir / "test", {6, 12, 13})
    windows = {day: load_windows(grouped[day], nodes[day]) for day in (6, 12, 13)}

    node_map = json.loads((data_dir / "nodeid2msg.json").read_text(encoding="utf-8"))
    uuid_to_node = {uuid.upper(): int(node) for node, uuid in node_map.items()}
    gt_dir = Path(args.ground_truth_dir)
    development_labels = {
        day: labels_for(
            nodes[day],
            campaign_nodes(
                gt_dir / f"node_Nginx_Backdoor_{day:02d}.csv", uuid_to_node
            ),
        )
        for day in (6, 12)
    }

    seed_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    prior_base = cross_val_predict(
        ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced", random_state=args.seed, n_jobs=-1,
        ),
        pooled[6], development_labels[6], cv=seed_cv,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    seed_model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=args.seed, n_jobs=-1,
    ).fit(pooled[6], development_labels[6])
    validation_base = seed_model.predict_proba(pooled[12])[:, 1]
    seed_threshold, _ = best_threshold(development_labels[12], validation_base)

    day_base = {6: prior_base, 12: validation_base}
    contexts = {}
    for day in (6, 12):
        seed = day_base[day] >= seed_threshold
        contexts[day] = context_features(
            windows[day], seed, day_base[day], pooled[day], relations,
            args.max_hops, args.include_trajectory,
        )
    prior_candidates, prior_features = contexts[6][:2]
    validation_candidates, validation_features = contexts[12][:2]
    prior_candidate_labels = development_labels[6][prior_candidates]
    validation_candidate_labels = development_labels[12][validation_candidates]
    validation_seed = validation_base >= seed_threshold
    validation_localized, _, _ = track(
        windows[12], validation_seed, percentile_scores(validation_base),
        3, "contiguous_block", 10, "joint", relations,
    )
    validation_required = validation_localized[validation_candidates]

    rows = []
    oof_scores = {}
    model_names = [
        "extra_trees_leaf1", "extra_trees_leaf2", "extra_trees_leaf4",
        "random_forest_leaf1",
    ]
    for name in model_names:
        for prior_weight in (0.25, 0.50, 1.00):
            key = f"{name}_prior_weight_{prior_weight:.2f}"
            scores = fold_predictions(
                name, args.seed, prior_features, prior_candidate_labels,
                validation_features, validation_candidate_labels, prior_weight,
            )
            threshold, f1 = best_threshold_with_required(
                validation_candidate_labels, scores, validation_required
            )
            rows.append({
                "candidate": key, "model": name, "prior_weight": prior_weight,
                "oof_f1": f1, "average_precision": float(
                    average_precision_score(validation_candidate_labels, scores)
                ),
                "locked_threshold": threshold,
            })
            oof_scores[key] = scores
    selection = pd.DataFrame(rows).sort_values(
        ["oof_f1", "average_precision", "prior_weight"],
        ascending=[False, False, True],
    )
    chosen = selection.iloc[0]
    chosen_key = str(chosen["candidate"])
    train_features = np.concatenate([prior_features, validation_features])
    train_labels = np.concatenate([
        prior_candidate_labels, validation_candidate_labels
    ])
    train_weights = np.concatenate([
        np.full(len(prior_candidate_labels), float(chosen["prior_weight"])),
        np.ones(len(validation_candidate_labels)),
    ])
    gate = model_factory(str(chosen["model"]), args.seed).fit(
        train_features, train_labels, sample_weight=train_weights
    )

    test_base = seed_model.predict_proba(pooled[13])[:, 1]
    test_seed = test_base >= seed_threshold
    test_context = context_features(
        windows[13], test_seed, test_base, pooled[13], relations,
        args.max_hops, args.include_trajectory,
    )
    test_candidates, test_features = test_context[:2]
    test_localized, _, _ = track(
        windows[13], test_seed, percentile_scores(test_base),
        3, "contiguous_block", 10, "joint", relations,
    )
    candidate_scores = gate.predict_proba(test_features)[:, 1]
    candidate_predicted = (
        (candidate_scores >= float(chosen["locked_threshold"]))
        | test_localized[test_candidates]
    )
    full_scores = np.zeros(len(nodes[13]), dtype=np.float64)
    full_scores[test_candidates] = candidate_scores
    full_predicted = np.zeros(len(nodes[13]), dtype=bool)
    full_predicted[test_candidates] = candidate_predicted

    # Day-13 labels are joined only after all continual decisions are frozen.
    test_truth = campaign_nodes(
        gt_dir / "node_Nginx_Backdoor_13.csv", uuid_to_node
    )
    test_labels = labels_for(nodes[13], test_truth)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, full_predicted, average="binary", zero_division=0
    )
    result = {
        "protocol_version": "tifs-e3-continual-campaign-gate-v1",
        "dataset": "E3-CADETS-Causal",
        "variant": "prior_campaign_oof_plus_current_campaign_gate",
        "selected_model": str(chosen["model"]),
        "prior_weight": float(chosen["prior_weight"]),
        "gate_oof_f1": float(chosen["oof_f1"]),
        "gate_oof_average_precision": float(chosen["average_precision"]),
        "gate_threshold": float(chosen["locked_threshold"]),
        "seed_threshold": seed_threshold,
        "trajectory_features": args.include_trajectory,
        "prior_candidate_nodes": len(prior_candidates),
        "prior_candidate_malicious_nodes": int(prior_candidate_labels.sum()),
        "validation_candidate_nodes": len(validation_candidates),
        "validation_candidate_malicious_nodes": int(validation_candidate_labels.sum()),
        "test_candidate_nodes": len(test_candidates),
        "test_candidate_malicious_nodes": int(test_labels[test_candidates].sum()),
        "test_nodes": len(test_labels), "malicious_nodes": int(test_labels.sum()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(test_labels, full_predicted)),
        "auroc": float(roc_auc_score(test_labels, full_scores)),
        "average_precision": float(average_precision_score(test_labels, full_scores)),
        "reported_nodes": int(full_predicted.sum()),
        "true_reported_nodes": int((full_predicted & (test_labels == 1)).sum()),
        "false_reported_nodes": int((full_predicted & (test_labels == 0)).sum()),
        "result_tier": "exploratory_reused_holdout_continual_supervision",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "validation_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "candidate_index": test_candidates,
        "node_id": nodes[13][test_candidates],
        "score": candidate_scores,
        "required_localized": test_localized[test_candidates].astype(np.int8),
        "predicted": candidate_predicted.astype(np.int8),
        "label": test_labels[test_candidates],
    }).to_csv(out / "test_candidate_decisions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result}, indent=2), encoding="utf-8"
    )
    print(selection.head(12).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
