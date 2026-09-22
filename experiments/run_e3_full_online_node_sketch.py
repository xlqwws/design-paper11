"""Full E3 online node-sketch detector with anytime validation calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    average_precision_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection.conformal import upper_tail_pvalues  # noqa: E402


def entropy(values: np.ndarray) -> np.ndarray:
    totals = np.maximum(values.sum(axis=1, keepdims=True), 1)
    probability = values / totals
    logs = np.zeros_like(probability, dtype=np.float32)
    positive = probability > 0
    logs[positive] = np.log(probability[positive])
    return -np.sum(probability * logs, axis=1, keepdims=True)


class NodeState:
    def __init__(self, num_nodes: int, num_relations: int):
        self.relations = np.zeros((num_nodes, num_relations * 2), dtype=np.uint32)
        self.counterpart_types = np.zeros((num_nodes, 6), dtype=np.uint32)
        self.node_types = np.full(num_nodes, 3, dtype=np.uint8)
        self.counts = np.zeros(num_nodes, dtype=np.uint32)

    def update(self, data, embedding_dim: int) -> np.ndarray:
        src = data.src.cpu().numpy().astype(np.int64, copy=False)
        dst = data.dst.cpu().numpy().astype(np.int64, copy=False)
        rel = data.edge_type.cpu().numpy().astype(np.int64, copy=False)
        msg = data.msg.cpu().numpy()
        src_type = np.argmax(msg[:, :3], axis=1)
        dst_start = msg.shape[1] - embedding_dim - 3
        dst_type = np.argmax(msg[:, dst_start:dst_start + 3], axis=1)
        relation_width = self.relations.shape[1] // 2
        np.add.at(self.relations, (src, rel), 1)
        np.add.at(self.relations, (dst, relation_width + rel), 1)
        np.add.at(self.counterpart_types, (src, dst_type), 1)
        np.add.at(self.counterpart_types, (dst, 3 + src_type), 1)
        np.add.at(self.counts, src, 1)
        np.add.at(self.counts, dst, 1)
        self.node_types[src] = src_type
        self.node_types[dst] = dst_type
        return np.unique(np.concatenate([src, dst]))

    def features(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = self.relations[nodes].astype(np.float32)
        counterpart = self.counterpart_types[nodes].astype(np.float32)
        rel_norm = rel / np.maximum(rel.sum(axis=1, keepdims=True), 1)
        counterpart_norm = counterpart / np.maximum(counterpart.sum(axis=1, keepdims=True), 1)
        half = rel.shape[1] // 2
        source_fraction = (
            rel[:, :half].sum(axis=1, keepdims=True)
            / np.maximum(rel.sum(axis=1, keepdims=True), 1)
        )
        node_type = np.eye(4, dtype=np.float32)[self.node_types[nodes]][:, :3]
        structure = np.concatenate([
            rel_norm, counterpart_norm, source_fraction,
            entropy(rel[:, :half]) / math.log(half),
            entropy(rel[:, half:]) / math.log(half), node_type,
        ], axis=1)
        size = np.log1p(self.counts[nodes]).astype(np.float32)[:, None]
        return structure, size


def paths(split_dir: Path):
    return sorted(split_dir.glob("*.TemporalData"))


def fit_models(state: NodeState, nodes: np.ndarray, seed: int, clusters: int):
    structure, size = state.features(nodes)
    variants = {
        "dual_head_full": np.concatenate([structure, size], axis=1),
        "normalized_structure_only": structure,
        "size_only": size,
    }
    models = {}
    for name, features in variants.items():
        scaler = StandardScaler().fit(features)
        transformed = scaler.transform(features)
        cluster_count = min(8 if name == "size_only" else clusters, len(transformed))
        model = MiniBatchKMeans(
            n_clusters=cluster_count, batch_size=8192, random_state=seed,
            n_init=3, max_iter=100,
        ).fit(transformed)
        models[name] = (scaler, model)
    return models


def scores_for(models, structure: np.ndarray, size: np.ndarray):
    raw = {
        "dual_head_full": np.concatenate([structure, size], axis=1),
        "normalized_structure_only": structure,
        "size_only": size,
    }
    return {
        name: model.transform(scaler.transform(raw[name])).min(axis=1)
        for name, (scaler, model) in models.items()
    }


def replay_split(
    split_dir: Path, num_nodes: int, num_relations: int, embedding_dim: int,
    models, calibration=None, alpha=0.001,
):
    state = NodeState(num_nodes, num_relations)
    maxima = {name: np.full(num_nodes, -np.inf, dtype=np.float32) for name in models}
    first_alert = {name: np.full(num_nodes, -1, dtype=np.int32) for name in models}
    seen = np.zeros(num_nodes, dtype=bool)
    event_count = 0
    for window, path in enumerate(paths(split_dir)):
        data = torch.load(path)
        touched = state.update(data, embedding_dim)
        event_count += len(data.src)
        seen[touched] = True
        structure, size = state.features(touched)
        for name, values in scores_for(models, structure, size).items():
            maxima[name][touched] = np.maximum(maxima[name][touched], values)
            if calibration is not None:
                pvalues = upper_tail_pvalues(calibration[name], maxima[name][touched])
                newly_alerted = touched[(pvalues <= alpha) & (first_alert[name][touched] < 0)]
                first_alert[name][newly_alerted] = window
        if (window + 1) % 50 == 0:
            print(f"{split_dir.name}: windows={window + 1} edges={event_count:,}", flush=True)
    nodes = np.flatnonzero(seen)
    return state, nodes, {name: values[nodes] for name, values in maxima.items()}, first_alert, event_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/raw_temporal_full/E3-CADETS/edge_embeds")
    parser.add_argument("--out-dir", default="artifacts/tifs_results/E3-CADETS-Causal/full_online_node_sketch_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.001)
    args = parser.parse_args()
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    num_nodes = int(metadata["nodes"])
    num_relations = len(metadata["relations"])
    embedding_dim = int(metadata["embedding_dim"])

    train_state = NodeState(num_nodes, num_relations)
    train_seen = np.zeros(num_nodes, dtype=bool)
    train_edges = 0
    for index, path in enumerate(paths(data_dir / "train")):
        data = torch.load(path)
        train_seen[train_state.update(data, embedding_dim)] = True
        train_edges += len(data.src)
        if (index + 1) % 50 == 0:
            print(f"train: windows={index + 1} edges={train_edges:,}", flush=True)
    train_nodes = np.flatnonzero(train_seen)
    models = fit_models(train_state, train_nodes, args.seed, args.clusters)
    del train_state, train_seen

    _, val_nodes, val_scores, _, val_edges = replay_split(
        data_dir / "val", num_nodes, num_relations, embedding_dim, models
    )
    calibration = {name: np.asarray(values) for name, values in val_scores.items()}
    _, test_nodes, test_scores, first_alert, test_edges = replay_split(
        data_dir / "test", num_nodes, num_relations, embedding_dim, models,
        calibration=calibration, alpha=args.alpha,
    )
    truth = set(pd.read_csv(data_dir / "ground_truth_nodes.csv")["node_id"].astype(int))
    labels = np.asarray([int(int(node) in truth) for node in test_nodes], dtype=int)
    results = []
    node_output = pd.DataFrame({"node_id": test_nodes, "label": labels})
    for name, values in test_scores.items():
        pvalues = upper_tail_pvalues(calibration[name], values)
        predicted = (pvalues <= args.alpha).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        malicious_alert_windows = first_alert[name][test_nodes][(labels == 1) & (predicted == 1)]
        results.append({
            "protocol_version": "tifs-e3-full-anytime-node-sketch-v1",
            "dataset": "E3-CADETS-Causal", "variant": name, "seed": args.seed,
            "train_edges": train_edges, "validation_edges": val_edges, "test_edges": test_edges,
            "train_nodes": len(train_nodes), "calibration_nodes": len(val_nodes),
            "test_nodes": len(test_nodes), "malicious_nodes": int(labels.sum()),
            "alpha": args.alpha, "reported_nodes": int(predicted.sum()),
            "true_reported_nodes": int(((labels == 1) & (predicted == 1)).sum()),
            "benign_reported_nodes": int(((labels == 0) & (predicted == 1)).sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "auroc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
            "median_alert_window_for_detected_malicious_nodes": (
                float(np.median(malicious_alert_windows)) if len(malicious_alert_windows) else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "decision_time": "each 15-minute window close",
            "calibration": "validation-node maximum prefix score; test labels joined after decisions",
        })
        node_output[f"{name}_score"] = values
        node_output[f"{name}_pvalue"] = pvalues
        node_output[f"{name}_predicted"] = predicted
        node_output[f"{name}_first_alert_window"] = first_alert[name][test_nodes]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    node_output.to_csv(out / "node_decisions.csv", index=False)
    with (out / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"results": results, "source_metadata": metadata}, file, indent=2)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
