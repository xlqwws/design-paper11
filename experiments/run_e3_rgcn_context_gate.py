"""Five-fold relation-aware GPU context gate for E3 dynamic candidates."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import RGCNConv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_e3_multihop_context_gate import (
    best_threshold,
    best_threshold_with_required,
    campaign_nodes,
    context_features,
    labels_for,
)
from experiments.run_e3_temporal_localized_tracking import (
    load_windows,
    partition_paths,
    percentile_scores,
    track,
)


class RelationContextGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(self, features, edge_index, edge_type):
        base = functional.relu(self.input_projection(features))
        hidden = self.conv1(base, edge_index, edge_type)
        hidden = functional.relu(hidden)
        hidden = functional.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.conv2(hidden, edge_index, edge_type)
        hidden = functional.relu(hidden)
        return self.classifier(torch.cat([base, hidden], dim=1)).squeeze(1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_graph(features, edge_index, edge_type, device):
    return (
        torch.from_numpy(features.astype(np.float32, copy=False)).to(device),
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_type).long().to(device),
    )


def train_fold(
    fold: int, train_indices: np.ndarray, validation_indices: np.ndarray,
    train_graph, test_graph, labels: np.ndarray, args, device,
):
    seed_everything(args.seed + fold)
    features, edge_index, edge_type = train_graph
    model = RelationContextGate(
        features.shape[1], args.hidden_dim, args.num_relations, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    label_tensor = torch.from_numpy(labels.astype(np.float32)).to(device)
    train_index = torch.from_numpy(train_indices).long().to(device)
    positives = max(int(labels[train_indices].sum()), 1)
    negatives = max(len(train_indices) - positives, 1)
    positive_weight = torch.tensor(
        [np.sqrt(negatives / positives)], dtype=torch.float32, device=device
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, edge_index, edge_type)
        loss = functional.binary_cross_entropy_with_logits(
            logits[train_index], label_tensor[train_index], pos_weight=positive_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (epoch + 1) % args.eval_every:
            continue
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(features, edge_index, edge_type)).cpu().numpy()
        score = average_precision_score(labels[validation_indices], probabilities[validation_indices])
        if score > best_ap + 1e-6:
            best_ap = float(score)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(
            model(features, edge_index, edge_type)
        ).cpu().numpy()[validation_indices]
        test_probabilities = torch.sigmoid(model(*test_graph)).cpu().numpy()
    return validation_probabilities, test_probabilities, {
        "fold": fold, "best_epoch": best_epoch, "validation_average_precision": best_ap,
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
        "--out-dir", default="artifacts/tifs_results/E3-CADETS-Causal/rgcn_context_gate"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--include-trajectory", action="store_true")
    args = parser.parse_args()
    args.num_relations = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    original_relations = len(metadata["relations"])
    args.num_relations = original_relations * 2
    cached = {
        day: np.load(Path(args.cache_dir) / f"day_{day}.npz")
        for day in (6, 12, 13)
    }
    nodes = {day: cached[day]["nodes"] for day in cached}
    pooled = {day: cached[day]["pooled"] for day in cached}
    grouped = partition_paths(data_dir / "test", {12, 13})
    windows = {day: load_windows(grouped[day], nodes[day]) for day in (12, 13)}

    node_map = json.loads((data_dir / "nodeid2msg.json").read_text(encoding="utf-8"))
    uuid_to_node = {uuid.upper(): int(node) for node, uuid in node_map.items()}
    del node_map
    gt_dir = Path(args.ground_truth_dir)
    development_labels = {
        day: labels_for(
            nodes[day],
            campaign_nodes(gt_dir / f"node_Nginx_Backdoor_{day:02d}.csv", uuid_to_node),
        )
        for day in (6, 12)
    }

    seed_model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=args.seed, n_jobs=-1,
    ).fit(pooled[6], development_labels[6])
    validation_base = seed_model.predict_proba(pooled[12])[:, 1]
    seed_threshold, _ = best_threshold(development_labels[12], validation_base)
    validation_seed = validation_base >= seed_threshold
    validation_context = context_features(
        windows[12], validation_seed, validation_base, pooled[12],
        original_relations, args.max_hops, args.include_trajectory,
    )
    validation_candidates, validation_features, _, validation_edges, validation_types = validation_context
    candidate_labels = development_labels[12][validation_candidates]
    validation_localized, _, _ = track(
        windows[12], validation_seed, percentile_scores(validation_base),
        3, "contiguous_block", 10, "joint", original_relations,
    )
    required_validation = validation_localized[validation_candidates]

    test_base = seed_model.predict_proba(pooled[13])[:, 1]
    test_seed = test_base >= seed_threshold
    test_context = context_features(
        windows[13], test_seed, test_base, pooled[13], original_relations,
        args.max_hops, args.include_trajectory,
    )
    test_candidates, test_features, _, test_edges, test_types = test_context
    test_localized, _, _ = track(
        windows[13], test_seed, percentile_scores(test_base),
        3, "contiguous_block", 10, "joint", original_relations,
    )

    scaler = StandardScaler().fit(validation_features)
    validation_features = scaler.transform(validation_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)
    validation_graph = tensor_graph(
        validation_features, validation_edges, validation_types, device
    )
    test_graph = tensor_graph(test_features, test_edges, test_types, device)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    out_of_fold = np.zeros(len(candidate_labels), dtype=np.float64)
    test_fold_scores = []
    fold_rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(validation_features, candidate_labels)
    ):
        validation_scores, test_scores, fold_row = train_fold(
            fold, train_indices, validation_indices, validation_graph,
            test_graph, candidate_labels, args, device,
        )
        out_of_fold[validation_indices] = validation_scores
        test_fold_scores.append(test_scores)
        fold_rows.append(fold_row)
        print(f"fold={fold} best_epoch={fold_row['best_epoch']} ap={fold_row['validation_average_precision']:.6f}", flush=True)

    gate_threshold, gate_oof_f1 = best_threshold_with_required(
        candidate_labels, out_of_fold, required_validation
    )
    candidate_test_scores = np.mean(np.stack(test_fold_scores), axis=0)
    candidate_test_predicted = (
        (candidate_test_scores >= gate_threshold)
        | test_localized[test_candidates]
    )
    full_scores = np.zeros(len(nodes[13]), dtype=np.float64)
    full_scores[test_candidates] = candidate_test_scores
    full_predicted = np.zeros(len(nodes[13]), dtype=np.int8)
    full_predicted[test_candidates] = candidate_test_predicted.astype(np.int8)

    # The reused test labels are joined only after the GPU ensemble decisions.
    test_truth = campaign_nodes(gt_dir / "node_Nginx_Backdoor_13.csv", uuid_to_node)
    test_labels = labels_for(nodes[13], test_truth)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, full_predicted, average="binary", zero_division=0
    )
    result = {
        "protocol_version": "tifs-e3-rgcn-context-gate-v2",
        "dataset": "E3-CADETS-Causal", "variant": (
            "five_fold_rgcn_temporal_trajectory_gate" if args.include_trajectory
            else "five_fold_rgcn_context_gate"
        ),
        "seed": args.seed, "device": str(device), "folds": args.folds,
        "hidden_dim": args.hidden_dim, "max_hops": args.max_hops,
        "trajectory_features": args.include_trajectory,
        "seed_threshold": seed_threshold, "gate_threshold": gate_threshold,
        "gate_oof_f1": gate_oof_f1,
        "validation_candidate_nodes": len(validation_candidates),
        "test_candidate_nodes": len(test_candidates),
        "test_candidate_malicious_nodes": int(test_labels[test_candidates].sum()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(test_labels, full_predicted)),
        "auroc": float(roc_auc_score(test_labels, full_scores)),
        "average_precision": float(average_precision_score(test_labels, full_scores)),
        "reported_nodes": int(full_predicted.sum()),
        "true_reported_nodes": int(((full_predicted == 1) & (test_labels == 1)).sum()),
        "false_reported_nodes": int(((full_predicted == 1) & (test_labels == 0)).sum()),
        "result_tier": "exploratory_reused_holdout_transductive_validation",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out / "fold_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "candidate_index": validation_candidates,
        "node_id": nodes[12][validation_candidates],
        "label": candidate_labels,
        "base_score": validation_base[validation_candidates],
        "gate_oof_score": out_of_fold,
        "gate_oof_percentile": percentile_scores(out_of_fold),
        "required_localized": required_validation.astype(np.int8),
    }).to_csv(out / "validation_candidate_scores.csv", index=False)
    pd.DataFrame({
        "candidate_index": test_candidates,
        "node_id": nodes[13][test_candidates],
        "label": test_labels[test_candidates],
        "base_score": test_base[test_candidates],
        "gate_score": candidate_test_scores,
        "gate_percentile": percentile_scores(candidate_test_scores),
        "required_localized": test_localized[test_candidates].astype(np.int8),
        "predicted": candidate_test_predicted.astype(np.int8),
    }).to_csv(out / "test_candidate_scores.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result, "folds": fold_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
