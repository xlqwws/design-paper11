"""Chronological E3 cross-campaign node attribution without test-label tuning.

The detector learns from the April 6 campaign, selects its feature/model/threshold
on April 12, and evaluates once on the untouched April 13 campaign.  Features
are computed from the dynamic graph prefix for each day and never use labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EASTERN = pytz.timezone("America/New_York")


def entropy(values: np.ndarray) -> np.ndarray:
    totals = np.maximum(values.sum(axis=1, keepdims=True), 1)
    probability = values / totals
    logs = np.zeros_like(probability, dtype=np.float32)
    positive = probability > 0
    logs[positive] = np.log(probability[positive])
    return -np.sum(probability * logs, axis=1, keepdims=True)


def day_for(path: Path) -> int:
    data = torch.load(path)
    return datetime.fromtimestamp(int(data.t[0]) / 1e9, EASTERN).day


def partition_paths(test_dir: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for path in sorted(test_dir.glob("*.TemporalData")):
        grouped.setdefault(day_for(path), []).append(path)
    return grouped


def extract_day_features(
    paths: list[Path], num_relations: int, embedding_dim: int, day: int,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        print(f"day={day} loaded feature cache", flush=True)
        return cached["nodes"], cached["features"], cached["src"], cached["dst"]

    endpoint_chunks = []
    for path in paths:
        data = torch.load(path)
        endpoint_chunks.extend((data.src.cpu().numpy(), data.dst.cpu().numpy()))
    nodes = np.unique(np.concatenate(endpoint_chunks).astype(np.int64, copy=False))
    del endpoint_chunks

    count = len(nodes)
    relations = np.zeros((count, num_relations * 2), dtype=np.uint32)
    counterpart_types = np.zeros((count, 6), dtype=np.uint32)
    node_types = np.full(count, 3, dtype=np.uint8)
    event_counts = np.zeros(count, dtype=np.uint32)
    active_windows = np.zeros(count, dtype=np.uint16)
    peak_window_count = np.zeros(count, dtype=np.uint32)
    first_window = np.full(count, len(paths), dtype=np.uint16)
    last_window = np.zeros(count, dtype=np.uint16)
    semantic = np.zeros((count, embedding_dim), dtype=np.float16)
    graph_src = []
    graph_dst = []

    for window, path in enumerate(paths):
        data = torch.load(path)
        src = data.src.cpu().numpy().astype(np.int64, copy=False)
        dst = data.dst.cpu().numpy().astype(np.int64, copy=False)
        src_local = np.searchsorted(nodes, src)
        dst_local = np.searchsorted(nodes, dst)
        graph_src.append(src_local.astype(np.int32, copy=False))
        graph_dst.append(dst_local.astype(np.int32, copy=False))
        relation = data.edge_type.cpu().numpy().astype(np.int64, copy=False)
        msg = data.msg.cpu().numpy()
        src_type = np.argmax(msg[:, :3], axis=1)
        dst_type_start = msg.shape[1] - embedding_dim - 3
        dst_type = np.argmax(msg[:, dst_type_start:dst_type_start + 3], axis=1)

        np.add.at(relations, (src_local, relation), 1)
        np.add.at(relations, (dst_local, num_relations + relation), 1)
        np.add.at(counterpart_types, (src_local, dst_type), 1)
        np.add.at(counterpart_types, (dst_local, 3 + src_type), 1)
        np.add.at(event_counts, src_local, 1)
        np.add.at(event_counts, dst_local, 1)
        node_types[src_local] = src_type
        node_types[dst_local] = dst_type
        semantic[src_local] = msg[:, 3:3 + embedding_dim]
        semantic[dst_local] = msg[:, dst_type_start + 3:]

        local, local_counts = np.unique(
            np.concatenate([src_local, dst_local]), return_counts=True
        )
        active_windows[local] += 1
        peak_window_count[local] = np.maximum(peak_window_count[local], local_counts)
        first_window[local] = np.minimum(first_window[local], window)
        last_window[local] = window

    rel = relations.astype(np.float32)
    counterpart = counterpart_types.astype(np.float32)
    rel_norm = rel / np.maximum(rel.sum(axis=1, keepdims=True), 1)
    counterpart_norm = counterpart / np.maximum(counterpart.sum(axis=1, keepdims=True), 1)
    source_fraction = (
        rel[:, :num_relations].sum(axis=1, keepdims=True)
        / np.maximum(rel.sum(axis=1, keepdims=True), 1)
    )
    node_type = np.eye(4, dtype=np.float32)[node_types][:, :3]
    structure = np.concatenate([
        rel_norm,
        counterpart_norm,
        source_fraction,
        entropy(rel[:, :num_relations]) / math.log(num_relations),
        entropy(rel[:, num_relations:]) / math.log(num_relations),
        node_type,
    ], axis=1)
    temporal = np.column_stack([
        np.log1p(event_counts),
        np.log1p(active_windows),
        np.log1p(peak_window_count),
        first_window / max(len(paths) - 1, 1),
        last_window / max(len(paths) - 1, 1),
        (last_window - first_window) / max(len(paths) - 1, 1),
    ]).astype(np.float32)
    features = np.concatenate([structure, temporal, semantic.astype(np.float32)], axis=1)
    edge_src = np.concatenate(graph_src)
    edge_dst = np.concatenate(graph_dst)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, nodes=nodes, features=features, src=edge_src, dst=edge_dst
    )
    print(
        f"day={day} windows={len(paths)} nodes={len(nodes):,} "
        f"edges={int(event_counts.sum() // 2):,} features={features.shape[1]}",
        flush=True,
    )
    return nodes, features, edge_src, edge_dst


def campaign_nodes(path: Path, uuid_to_node: dict[str, int]) -> set[int]:
    uuids = set()
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for row in csv.reader(file):
            if row:
                uuids.add(row[0].strip().upper())
    return {uuid_to_node[value] for value in uuids if value in uuid_to_node}


def labels_for(nodes: np.ndarray, truth: set[int]) -> np.ndarray:
    return np.fromiter((int(int(node) in truth) for node in nodes), dtype=np.int8)


def candidate_factories(seed: int):
    factories = {}
    for c_value in (0.01, 0.1, 1.0, 10.0):
        factories[f"logistic_c{c_value:g}"] = lambda c=c_value: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight="balanced", solver="liblinear",
                max_iter=2000, random_state=seed,
            ),
        )
        factories[f"linear_svm_c{c_value:g}"] = lambda c=c_value: make_pipeline(
            StandardScaler(),
            LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=10000),
        )
    for leaf in (1, 2, 4):
        factories[f"extra_trees_leaf{leaf}"] = lambda leaf_size=leaf: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=leaf_size, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    return factories


def model_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float64)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def percentile_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = (np.arange(len(scores), dtype=np.float32) + 1) / len(scores)
    return ranks


def monotone_graph_propagation(
    base: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray,
    decay: float, hops: int,
) -> np.ndarray:
    """Max-product updates: a lower candidate can never reduce node state."""
    current = np.asarray(base, dtype=np.float32).copy()
    for _ in range(hops):
        neighbor = np.zeros_like(current)
        np.maximum.at(neighbor, edge_src, current[edge_dst])
        np.maximum.at(neighbor, edge_dst, current[edge_src])
        current = np.maximum(current, decay * neighbor)
    return current


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
        "true_reported_nodes": int(((labels == 1) & (predicted == 1)).sum()),
        "malicious_nodes": int(labels.sum()),
        "test_nodes": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="artifacts/raw_temporal_full/E3-CADETS/edge_embeds"
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
        default="artifacts/tifs_results/E3-CADETS-Causal/cross_campaign_transfer",
    )
    parser.add_argument(
        "--cache-dir",
        default="artifacts/feature_cache/E3-CADETS-Causal/cross_campaign_transfer",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    grouped = partition_paths(data_dir / "test")
    required_days = {6, 12, 13}
    if not required_days.issubset(grouped):
        raise RuntimeError(f"Missing E3 test days: {sorted(required_days - set(grouped))}")

    node_map = json.loads((data_dir / "nodeid2msg.json").read_text(encoding="utf-8"))
    uuid_to_node = {value.upper(): int(node) for node, value in node_map.items()}
    del node_map
    gt_dir = Path(args.ground_truth_dir)
    truth_train = campaign_nodes(gt_dir / "node_Nginx_Backdoor_06.csv", uuid_to_node)
    truth_validation = campaign_nodes(gt_dir / "node_Nginx_Backdoor_12.csv", uuid_to_node)

    extracted = {}
    for day in sorted(required_days):
        extracted[day] = extract_day_features(
            grouped[day], len(metadata["relations"]), int(metadata["embedding_dim"]), day,
            Path(args.cache_dir) / f"day_{day}.npz",
        )
    train_nodes, train_features, _, _ = extracted[6]
    validation_nodes, validation_features, validation_src, validation_dst = extracted[12]
    test_nodes, test_features, test_src, test_dst = extracted[13]
    train_labels = labels_for(train_nodes, truth_train)
    validation_labels = labels_for(validation_nodes, truth_validation)
    if train_labels.sum() != 8 or validation_labels.sum() != 41:
        raise RuntimeError(
            f"Campaign join mismatch: train={train_labels.sum()} validation={validation_labels.sum()}"
        )

    feature_slices = {
        "structure_temporal": slice(0, 38),
        "semantic": slice(38, None),
        "joint": slice(None),
    }
    factories = candidate_factories(args.seed)
    development_rows = []
    fitted = {}
    for feature_name, feature_slice in feature_slices.items():
        for model_name, factory in factories.items():
            model = factory()
            model.fit(train_features[:, feature_slice], train_labels)
            raw_scores = model_scores(model, validation_features[:, feature_slice])
            rank_scores = percentile_scores(raw_scores)
            score_variants = [("raw", raw_scores, 0.0, 0), ("rank", rank_scores, 0.0, 0)]
            for decay in (0.5, 0.7, 0.9):
                propagated = rank_scores
                for hops in (1, 2, 3):
                    propagated = monotone_graph_propagation(
                        rank_scores, validation_src, validation_dst, decay, hops
                    )
                    score_variants.append((f"dynamic_d{decay:g}_h{hops}", propagated, decay, hops))
            for score_name, validation_scores, decay, hops in score_variants:
                threshold, validation_f1 = best_threshold(validation_labels, validation_scores)
                key = f"{feature_name}__{model_name}__{score_name}"
                fitted[key] = (model, feature_slice, threshold, decay, hops, score_name)
                development_rows.append({
                    "candidate": key, "validation_f1": validation_f1,
                    "locked_threshold": threshold, "decay": decay, "hops": hops,
                })
            best_for_model = max(row["validation_f1"] for row in development_rows
                                 if row["candidate"].startswith(f"{feature_name}__{model_name}__"))
            print(
                f"candidate={feature_name}__{model_name} "
                f"best_validation_f1={best_for_model:.6f}", flush=True,
            )

    development = pd.DataFrame(development_rows).sort_values(
        ["validation_f1", "candidate"], ascending=[False, True]
    )
    selected = str(development.iloc[0]["candidate"])
    selected_model, selected_slice, locked_threshold, decay, hops, score_name = fitted[selected]
    test_raw_scores = model_scores(selected_model, test_features[:, selected_slice])
    if score_name == "raw":
        test_scores = test_raw_scores
    else:
        test_scores = percentile_scores(test_raw_scores)
        if hops:
            test_scores = monotone_graph_propagation(
                test_scores, test_src, test_dst, decay, hops
            )
    test_predicted = (test_scores >= locked_threshold).astype(np.int8)

    # The final campaign labels are deliberately loaded only after every test
    # score and decision has been frozen.
    truth_test = campaign_nodes(gt_dir / "node_Nginx_Backdoor_13.csv", uuid_to_node)
    test_labels = labels_for(test_nodes, truth_test)
    if test_labels.sum() != 22:
        raise RuntimeError(f"Final campaign join mismatch: test={test_labels.sum()}")

    result = {
        "protocol_version": "tifs-e3-cross-campaign-transfer-v1",
        "dataset": "E3-CADETS-Causal",
        "variant": "chronological_supervised_transfer",
        "seed": args.seed,
        "train_campaign": "2018-04-06",
        "validation_campaign": "2018-04-12",
        "untouched_test_campaign": "2018-04-13",
        "selected_candidate": selected,
        "locked_validation_threshold": float(locked_threshold),
        "dynamic_decay": float(decay),
        "dynamic_hops": int(hops),
        "validation_f1": float(development.iloc[0]["validation_f1"]),
        **metrics(test_labels, test_predicted, test_scores),
        "elapsed_seconds": time.perf_counter() - started,
        "label_access_order": "test labels loaded after test scores and decisions were frozen",
        "claim_scope": "known-family supervised transfer; not zero-day unsupervised detection",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    development.to_csv(out / "validation_model_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result, "source_metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
