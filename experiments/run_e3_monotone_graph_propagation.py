"""Validation-selected monotone score propagation on E3 dynamic subgraphs."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_e3_rank_ensemble_gate import (  # noqa: E402
    best_required_threshold,
)
from experiments.run_e3_multihop_context_gate import (  # noqa: E402
    select_seed_neighborhood_windows,
)
from experiments.run_e3_temporal_localized_tracking import (  # noqa: E402
    load_windows, partition_paths,
)


def induced_graph(windows, seed, max_hops, num_relations):
    selected = select_seed_neighborhood_windows(windows, seed)
    src = np.concatenate([window["src"] for window in selected]).astype(np.int64)
    dst = np.concatenate([window["dst"] for window in selected]).astype(np.int64)
    relation = np.concatenate([window["relation"] for window in selected]).astype(np.int64)
    reached = seed.copy()
    frontier = seed.copy()
    for _ in range(max_hops):
        next_frontier = np.zeros_like(seed)
        np.logical_or.at(next_frontier, dst, frontier[src])
        np.logical_or.at(next_frontier, src, frontier[dst])
        next_frontier &= ~reached
        reached |= next_frontier
        frontier = next_frontier
    candidates = np.flatnonzero(reached)
    local = np.full(len(seed), -1, dtype=np.int32)
    local[candidates] = np.arange(len(candidates), dtype=np.int32)
    mask = reached[src] & reached[dst]
    forward_src = local[src[mask]].astype(np.int64)
    forward_dst = local[dst[mask]].astype(np.int64)
    forward_relation = relation[mask]
    edge_src = np.concatenate([forward_src, forward_dst])
    edge_dst = np.concatenate([forward_dst, forward_src])
    edge_type = np.concatenate([forward_relation, forward_relation + num_relations])
    count = len(candidates)
    packed = (edge_src * count + edge_dst) * (num_relations * 2) + edge_type
    packed = np.unique(packed)
    edge_type = packed % (num_relations * 2)
    pair = packed // (num_relations * 2)
    edge_src = pair // count
    edge_dst = pair % count
    degree = np.bincount(edge_src, minlength=count)
    return candidates, edge_src, edge_dst, edge_type, degree, len(selected)


def relation_policies(num_relations):
    forward = np.arange(num_relations)
    reverse = forward + num_relations
    process = np.asarray([1, 2, 9])
    data = np.asarray([3, 8])
    network = np.asarray([0, 4, 5, 6, 7])
    return {
        "bidirectional_all": np.concatenate([forward, reverse]),
        "causal_forward_all": forward,
        "causal_reverse_all": reverse,
        "bidirectional_process": np.concatenate([process, process + num_relations]),
        "causal_forward_process": process,
        "bidirectional_data": np.concatenate([data, data + num_relations]),
        "bidirectional_network": np.concatenate([network, network + num_relations]),
    }


def propagate(
    scores, edge_src, edge_dst, edge_type, gamma, hops, degree_cap,
    allowed_relations,
):
    output = np.asarray(scores, dtype=np.float64).copy()
    relation_mask = np.isin(edge_type, allowed_relations)
    relation_src = edge_src[relation_mask]
    relation_dst = edge_dst[relation_mask]
    degree = np.bincount(relation_src, minlength=len(output))
    degree_mask = degree[relation_src] <= degree_cap
    source = relation_src[degree_mask]
    target = relation_dst[degree_mask]
    for _ in range(hops):
        updated = output.copy()
        np.maximum.at(updated, target, gamma * output[source])
        output = updated
    return output


def metrics(labels, predicted, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "reported_nodes": int(predicted.sum()),
        "true_reported_nodes": int((predicted & (labels == 1)).sum()),
        "false_reported_nodes": int((predicted & (labels == 0)).sum()),
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
        "--tree-dir", default="artifacts/tifs_results/E3-CADETS-Causal/multihop_context_gate_monotone_localized"
    )
    parser.add_argument(
        "--rgcn-dir", default="artifacts/tifs_results/E3-CADETS-Causal/rgcn_temporal_trajectory_gate"
    )
    parser.add_argument(
        "--out-dir", default="artifacts/tifs_results/E3-CADETS-Causal/monotone_graph_propagation"
    )
    parser.add_argument("--max-hops", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    relation_count = len(metadata["relations"])
    cache_dir = Path(args.cache_dir)
    cached = {day: np.load(cache_dir / f"day_{day}.npz") for day in (12, 13)}
    nodes = {day: cached[day]["nodes"] for day in cached}
    grouped = partition_paths(data_dir / "test", {12, 13})
    windows = {day: load_windows(grouped[day], nodes[day]) for day in (12, 13)}

    tree_dir = Path(args.tree_dir)
    rgcn_dir = Path(args.rgcn_dir)
    tree = {
        12: pd.read_csv(tree_dir / "validation_candidate_scores.csv"),
        13: pd.read_csv(tree_dir / "test_candidate_scores.csv"),
    }
    rgcn = {
        12: pd.read_csv(rgcn_dir / "validation_candidate_scores.csv"),
        13: pd.read_csv(rgcn_dir / "test_candidate_scores.csv"),
    }
    graphs = {}
    for day in (12, 13):
        frame = tree[day]
        full_seed = np.zeros(len(nodes[day]), dtype=bool)
        seed_rows = frame["base_score"].to_numpy(float) >= 0.11
        full_seed[frame.loc[seed_rows, "candidate_index"].to_numpy(np.int64)] = True
        graph = induced_graph(
            windows[day], full_seed, args.max_hops, relation_count
        )
        if not np.array_equal(graph[0], frame["candidate_index"].to_numpy(np.int64)):
            raise RuntimeError(f"Candidate reconstruction mismatch on day {day}")
        if not np.array_equal(frame["node_id"], rgcn[day]["node_id"]):
            raise RuntimeError(f"Tree/RGCN node mismatch on day {day}")
        graphs[day] = graph

    sources = {}
    for name, tree_weight in [
        ("tree_rank", 1.0), ("blend_tree_050", 0.50), ("rgcn_rank", 0.0),
    ]:
        sources[name] = {}
        for day, score_column in [(12, "gate_oof_percentile"), (13, "gate_percentile")]:
            sources[name][day] = (
                tree_weight * tree[day][score_column].to_numpy(float)
                + (1 - tree_weight) * rgcn[day][score_column].to_numpy(float)
            )

    validation_labels = tree[12]["label"].to_numpy(np.int8)
    validation_required = tree[12]["required_localized"].to_numpy(bool)
    selection_rows = []
    policies = relation_policies(relation_count)
    for source_name, source_scores in sources.items():
        for policy_name, allowed_relations in policies.items():
            graph = graphs[12]
            relation_mask = np.isin(graph[3], allowed_relations)
            relation_src = graph[1][relation_mask]
            relation_dst = graph[2][relation_mask]
            relation_degree = np.bincount(
                relation_src, minlength=len(source_scores[12])
            )
            for degree_cap in (5, 10, 25, 50):
                degree_mask = relation_degree[relation_src] <= degree_cap
                edge_src = relation_src[degree_mask]
                edge_dst = relation_dst[degree_mask]
                for gamma in (0.85, 1.00):
                    current = source_scores[12].copy()
                    for hops in range(1, min(args.max_hops, 3) + 1):
                        updated = current.copy()
                        np.maximum.at(updated, edge_dst, gamma * current[edge_src])
                        current = updated
                        name = (
                            f"{source_name}_{policy_name}_cap{degree_cap}_"
                            f"gamma{gamma:.2f}_h{hops}"
                        )
                        threshold, f1 = best_required_threshold(
                            validation_labels, current, validation_required
                        )
                        predicted = validation_required | (current >= threshold)
                        precision, recall, _, _ = precision_recall_fscore_support(
                            validation_labels, predicted, average="binary", zero_division=0
                        )
                        selection_rows.append({
                            "candidate": name, "source": source_name,
                            "relation_policy": policy_name,
                            "degree_cap": degree_cap, "gamma": gamma, "hops": hops,
                            "threshold": threshold, "validation_f1": f1,
                            "validation_precision": float(precision),
                            "validation_recall": float(recall),
                            "validation_average_precision": float(
                                average_precision_score(validation_labels, current)
                            ),
                        })

    selection = pd.DataFrame(selection_rows).sort_values(
        ["validation_f1", "validation_average_precision", "hops", "degree_cap"],
        ascending=[False, False, True, True],
    )
    chosen = selection.iloc[0]
    source_name = str(chosen["source"])
    test_graph = graphs[13]
    test_scores = propagate(
        sources[source_name][13], test_graph[1], test_graph[2], test_graph[3],
        float(chosen["gamma"]), int(chosen["hops"]), int(chosen["degree_cap"]),
        policies[str(chosen["relation_policy"])],
    )
    test_required = tree[13]["required_localized"].to_numpy(bool)
    test_predicted = test_required | (test_scores >= float(chosen["threshold"]))

    # Day-13 labels are joined only after propagation policy and threshold freeze.
    candidate_labels = tree[13]["label"].to_numpy(np.int8)
    outside_count = 30893 - len(candidate_labels)
    full_labels = np.concatenate([candidate_labels, np.zeros(outside_count, dtype=np.int8)])
    full_scores = np.concatenate([test_scores, np.zeros(outside_count)])
    full_predicted = np.concatenate([test_predicted, np.zeros(outside_count, dtype=bool)])
    result = {
        "protocol_version": "tifs-e3-monotone-graph-propagation-v1",
        "dataset": "E3-CADETS-Causal",
        "variant": "validation_selected_monotone_neighbor_propagation",
        "selected_candidate": str(chosen["candidate"]),
        "source": source_name,
        "relation_policy": str(chosen["relation_policy"]),
        "degree_cap": int(chosen["degree_cap"]),
        "gamma": float(chosen["gamma"]), "propagation_hops": int(chosen["hops"]),
        "locked_threshold": float(chosen["threshold"]),
        "validation_f1": float(chosen["validation_f1"]),
        "validation_average_precision": float(chosen["validation_average_precision"]),
        "validation_edges": len(graphs[12][1]),
        "test_edges": len(graphs[13][1]),
        "validation_selected_windows": graphs[12][5],
        "test_selected_windows": graphs[13][5],
        "test_nodes": len(full_labels), "malicious_nodes": int(full_labels.sum()),
        **metrics(full_labels, full_predicted, full_scores),
        "result_tier": "exploratory_reused_holdout_validation_selected_propagation",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "validation_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "candidate_index": tree[13]["candidate_index"],
        "node_id": tree[13]["node_id"],
        "score": test_scores,
        "required_localized": test_required.astype(np.int8),
        "predicted": test_predicted.astype(np.int8),
        "label": candidate_labels,
    }).to_csv(out / "test_candidate_decisions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result}, indent=2), encoding="utf-8"
    )
    print(selection.head(12).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
