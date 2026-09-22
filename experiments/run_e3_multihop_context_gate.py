"""Exploratory relation-path gate over high-recall E3 dynamic candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_e3_temporal_localized_tracking import (  # noqa: E402
    load_windows, partition_paths, percentile_scores, track,
)


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


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def best_threshold_with_required(
    labels: np.ndarray, scores: np.ndarray, required: np.ndarray,
) -> tuple[float, float]:
    """Calibrate expansion while preserving every existing seed decision."""
    thresholds = np.unique(scores)
    best_value = -1.0
    best_cut = float(thresholds[-1])
    for threshold in thresholds:
        predicted = required | (scores >= threshold)
        tp = int(((labels == 1) & predicted).sum())
        fp = int(((labels == 0) & predicted).sum())
        fn = int(((labels == 1) & ~predicted).sum())
        value = 2 * tp / max(2 * tp + fp + fn, 1)
        if value > best_value:
            best_value = value
            best_cut = float(threshold)
    return best_cut, best_value


def select_seed_neighborhood_windows(windows, seed: np.ndarray, radius: int = 1):
    selected = set()
    for index, window in enumerate(windows):
        active = np.unique(np.concatenate([window["src"], window["dst"]]))
        if seed[active].any():
            selected.update(
                range(max(0, index - radius), min(len(windows), index + radius + 1))
            )
    return [windows[index] for index in sorted(selected)]


def temporal_trajectory_features(
    windows, count: int, seed: np.ndarray, num_relations: int,
) -> np.ndarray:
    """Summarize each node's causal trajectory across the selected windows."""
    steps = len(windows)
    activity = np.zeros((steps, count), dtype=np.float32)
    seed_contacts = np.zeros_like(activity)
    relation_presence = np.zeros((count, num_relations * 2), dtype=np.float32)
    for step, window in enumerate(windows):
        src = window["src"]
        dst = window["dst"]
        relation = window["relation"]
        np.add.at(activity[step], src, 1)
        np.add.at(activity[step], dst, 1)
        np.add.at(seed_contacts[step], dst[seed[src]], 1)
        np.add.at(seed_contacts[step], src[seed[dst]], 1)
        forward = src.astype(np.int64) * (num_relations * 2) + relation
        backward = dst.astype(np.int64) * (num_relations * 2) + num_relations + relation
        flat = relation_presence.reshape(-1)
        np.add.at(flat, np.unique(forward), 1)
        np.add.at(flat, np.unique(backward), 1)

    active = activity > 0
    contact_active = seed_contacts > 0
    active_count = active.sum(axis=0)
    contact_count = contact_active.sum(axis=0)
    first = np.argmax(active, axis=0)
    last = steps - 1 - np.argmax(active[::-1], axis=0)
    contact_first = np.argmax(contact_active, axis=0)
    contact_last = steps - 1 - np.argmax(contact_active[::-1], axis=0)
    first[active_count == 0] = 0
    last[active_count == 0] = 0
    contact_first[contact_count == 0] = 0
    contact_last[contact_count == 0] = 0
    total = activity.sum(axis=0)
    contact_total = seed_contacts.sum(axis=0)
    mean = activity.mean(axis=0)
    std = activity.std(axis=0)
    centered_time = np.arange(steps, dtype=np.float32) - (steps - 1) / 2
    trend = centered_time @ activity / max(float((centered_time ** 2).sum()), 1.0)
    quarters = np.array_split(np.arange(steps), 4)
    quarter_mass = np.stack([
        activity[index].sum(axis=0) / np.maximum(total, 1)
        for index in quarters
    ], axis=1)
    scale = max(steps - 1, 1)
    summary = np.stack([
        active_count / max(steps, 1),
        first / scale,
        last / scale,
        np.maximum(last - first + 1, 0) / max(steps, 1),
        np.log1p(total),
        np.log1p(activity.max(axis=0)),
        np.log1p(mean),
        std / np.maximum(mean, 1e-6),
        activity.max(axis=0) / np.maximum(total, 1),
        trend / np.maximum(mean, 1e-6),
        contact_count / max(steps, 1),
        contact_first / scale,
        contact_last / scale,
        np.log1p(contact_total),
        np.log1p(seed_contacts.max(axis=0)),
        contact_total / np.maximum(total, 1),
    ], axis=1).astype(np.float32)
    relation_presence /= max(steps, 1)
    return np.concatenate([summary, quarter_mass, relation_presence], axis=1)


def context_features(
    windows, seed: np.ndarray, base_scores: np.ndarray,
    pooled: np.ndarray, num_relations: int, max_hops: int,
    include_trajectory: bool = False,
):
    selected = select_seed_neighborhood_windows(windows, seed)
    src = np.concatenate([window["src"] for window in selected])
    dst = np.concatenate([window["dst"] for window in selected])
    relation = np.concatenate([window["relation"] for window in selected])
    count = len(seed)

    local_degree = np.zeros((count, num_relations * 2), dtype=np.float32)
    np.add.at(local_degree, (src, relation), 1)
    np.add.at(local_degree, (dst, num_relations + relation), 1)
    seed_degree = np.zeros_like(local_degree)
    forward_seed = seed[src]
    backward_seed = seed[dst]
    np.add.at(seed_degree, (dst[forward_seed], relation[forward_seed]), 1)
    np.add.at(
        seed_degree,
        (src[backward_seed], num_relations + relation[backward_seed]),
        1,
    )

    hop = np.full(count, max_hops + 1, dtype=np.int8)
    hop[seed] = 0
    path_signature = np.zeros(
        (count, max_hops * num_relations * 2), dtype=np.uint8
    )
    reached = seed.copy()
    frontier = seed.copy()
    for level in range(1, max_hops + 1):
        next_frontier = np.zeros(count, dtype=bool)
        offset = (level - 1) * num_relations * 2
        for relation_id in range(num_relations):
            relation_mask = relation == relation_id
            rel_src = src[relation_mask]
            rel_dst = dst[relation_mask]

            forward = frontier[rel_src]
            targets = rel_dst[forward]
            path_signature[targets, offset + relation_id] = 1
            next_frontier[targets] = True

            backward = frontier[rel_dst]
            targets = rel_src[backward]
            path_signature[targets, offset + num_relations + relation_id] = 1
            next_frontier[targets] = True
        next_frontier &= ~reached
        hop[next_frontier] = level
        reached |= next_frontier
        frontier = next_frontier

    candidates = np.flatnonzero(reached)
    hop_one_hot = np.eye(max_hops + 1, dtype=np.float32)[hop[candidates]]
    rank = percentile_scores(base_scores)
    feature_fields = [
        pooled[candidates],
        np.log1p(local_degree[candidates]),
        np.log1p(seed_degree[candidates]),
        path_signature[candidates].astype(np.float32),
        hop_one_hot,
        base_scores[candidates, None].astype(np.float32),
        rank[candidates, None],
    ]
    if include_trajectory:
        trajectories = temporal_trajectory_features(
            selected, count, seed, num_relations
        )
        feature_fields.append(trajectories[candidates])
    features = np.concatenate(feature_fields, axis=1)
    local_index = np.full(count, -1, dtype=np.int32)
    local_index[candidates] = np.arange(len(candidates), dtype=np.int32)
    induced = reached[src] & reached[dst]
    induced_src = local_index[src[induced]]
    induced_dst = local_index[dst[induced]]
    induced_relation = relation[induced].astype(np.int64, copy=False)
    edge_index = np.vstack([
        np.concatenate([induced_src, induced_dst]),
        np.concatenate([induced_dst, induced_src]),
    ]).astype(np.int64, copy=False)
    edge_type = np.concatenate([
        induced_relation, induced_relation + num_relations,
    ]).astype(np.int64, copy=False)
    return candidates, features, len(selected), edge_index, edge_type


def model_factory(name: str, seed: int):
    if name.startswith("extra_trees_leaf"):
        leaf = int(name.rsplit("leaf", 1)[1])
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if name == "random_forest_leaf1":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    if name.startswith("hist_gradient_leaf"):
        leaf = int(name.rsplit("leaf", 1)[1])
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=leaf,
            min_samples_leaf=10, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=seed,
        )
    raise ValueError(name)


def metrics(labels: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict:
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
        "--out-dir", default="artifacts/tifs_results/E3-CADETS-Causal/multihop_context_gate"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--include-trajectory", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    cached = {day: np.load(cache_dir / f"day_{day}.npz") for day in (6, 12, 13)}
    nodes = {day: cached[day]["nodes"] for day in cached}
    pooled = {day: cached[day]["pooled"] for day in cached}
    grouped = partition_paths(data_dir / "test", {12, 13})
    windows = {day: load_windows(grouped[day], nodes[day]) for day in (12, 13)}

    node_map = json.loads((data_dir / "nodeid2msg.json").read_text(encoding="utf-8"))
    uuid_to_node = {uuid.upper(): int(node) for node, uuid in node_map.items()}
    del node_map
    gt_dir = Path(args.ground_truth_dir)
    labels = {
        day: labels_for(
            nodes[day],
            campaign_nodes(gt_dir / f"node_Nginx_Backdoor_{day:02d}.csv", uuid_to_node),
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
    validation_candidates, validation_features, validation_windows, _, _ = context_features(
        windows[12], validation_seed, validation_base, pooled[12],
        len(metadata["relations"]), args.max_hops, args.include_trajectory,
    )
    validation_candidate_labels = labels[12][validation_candidates]
    validation_localized, _, _ = track(
        windows[12], validation_seed, percentile_scores(validation_base),
        3, "contiguous_block", 10, "joint", len(metadata["relations"]),
    )
    validation_required = validation_localized[validation_candidates]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = [
        "extra_trees_leaf1", "extra_trees_leaf2", "extra_trees_leaf4",
        "random_forest_leaf1",
    ]
    if args.include_trajectory:
        candidates.extend(["hist_gradient_leaf15", "hist_gradient_leaf31"])
    selection_rows = []
    oof_by_candidate = {}
    for name in candidates:
        oof_scores = cross_val_predict(
            model_factory(name, args.seed), validation_features,
            validation_candidate_labels, cv=cv, method="predict_proba", n_jobs=1,
        )[:, 1]
        oof_by_candidate[name] = oof_scores
        threshold, f1 = best_threshold_with_required(
            validation_candidate_labels, oof_scores, validation_required
        )
        selection_rows.append({
            "candidate": name, "oof_f1": f1, "locked_threshold": threshold,
        })
    selection = pd.DataFrame(selection_rows).sort_values(
        ["oof_f1", "candidate"], ascending=[False, True]
    )
    chosen = selection.iloc[0]
    chosen_name = str(chosen["candidate"])
    chosen_oof_scores = oof_by_candidate[chosen_name]
    gate = model_factory(chosen_name, args.seed).fit(
        validation_features, validation_candidate_labels
    )

    test_base = seed_model.predict_proba(pooled[13])[:, 1]
    test_seed = test_base >= seed_threshold
    test_candidates, test_features, test_windows, _, _ = context_features(
        windows[13], test_seed, test_base, pooled[13],
        len(metadata["relations"]), args.max_hops, args.include_trajectory,
    )
    test_localized, _, _ = track(
        windows[13], test_seed, percentile_scores(test_base),
        3, "contiguous_block", 10, "joint", len(metadata["relations"]),
    )
    candidate_scores = gate.predict_proba(test_features)[:, 1]
    candidate_predicted = (
        (candidate_scores >= float(chosen["locked_threshold"]))
        | test_localized[test_candidates]
    )
    test_scores = np.zeros(len(nodes[13]), dtype=np.float64)
    test_scores[test_candidates] = candidate_scores
    test_predicted = np.zeros(len(nodes[13]), dtype=np.int8)
    test_predicted[test_candidates] = candidate_predicted.astype(np.int8)

    # Join the reused holdout labels only after all decisions are frozen.
    test_truth = campaign_nodes(
        gt_dir / "node_Nginx_Backdoor_13.csv", uuid_to_node
    )
    test_labels = labels_for(nodes[13], test_truth)
    result = {
        "protocol_version": "tifs-e3-multihop-context-gate-v2",
        "dataset": "E3-CADETS-Causal", "variant": (
            "relation_path_temporal_trajectory_gate" if args.include_trajectory
            else "relation_path_context_gate"
        ),
        "seed": args.seed, "seed_threshold": seed_threshold,
        "selected_gate": str(chosen["candidate"]),
        "gate_oof_f1": float(chosen["oof_f1"]),
        "gate_threshold": float(chosen["locked_threshold"]),
        "max_hops": args.max_hops,
        "trajectory_features": args.include_trajectory,
        "validation_candidate_nodes": len(validation_candidates),
        "validation_candidate_malicious_nodes": int(validation_candidate_labels.sum()),
        "validation_selected_windows": validation_windows,
        "test_candidate_nodes": len(test_candidates),
        "test_candidate_malicious_nodes": int(test_labels[test_candidates].sum()),
        "test_selected_windows": test_windows,
        "required_localized_nodes": int(test_localized.sum()),
        "test_nodes": len(test_labels), "malicious_nodes": int(test_labels.sum()),
        "result_tier": "exploratory_reused_holdout_continual_gate",
        **metrics(test_labels, test_predicted, test_scores),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "validation_gate_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "candidate_index": validation_candidates,
        "node_id": nodes[12][validation_candidates],
        "label": validation_candidate_labels,
        "base_score": validation_base[validation_candidates],
        "gate_oof_score": chosen_oof_scores,
        "gate_oof_percentile": percentile_scores(chosen_oof_scores),
        "required_localized": validation_required.astype(np.int8),
    }).to_csv(out / "validation_candidate_scores.csv", index=False)
    pd.DataFrame({
        "candidate_index": test_candidates,
        "node_id": nodes[13][test_candidates],
        "label": test_labels[test_candidates],
        "base_score": test_base[test_candidates],
        "gate_score": candidate_scores,
        "gate_percentile": percentile_scores(candidate_scores),
        "required_localized": test_localized[test_candidates].astype(np.int8),
        "predicted": candidate_predicted.astype(np.int8),
    }).to_csv(out / "test_candidate_scores.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result, "source_metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    print(selection.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
