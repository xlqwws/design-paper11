"""Budget-matched online versus post-hoc tracing evaluation.

This evaluator uses a frozen event-score stream.  It never changes the detector
or threshold and deliberately labels full-test-graph traversals as post-hoc.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import deque
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from attack_reconstruction.online_dynamic_tracing import OnlineDynamicAttackTracer  # noqa: E402
from experiments.evaluate_ablation_final import make_cfg  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="E3-CADETS-Causal")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def node_metrics(selected, malicious, all_nodes):
    selected = set(selected)
    tp = len(selected & malicious)
    fp = len(selected - malicious)
    fn = len(malicious - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "reported_nodes": len(selected),
        "true_reported_nodes": tp,
        "benign_reported_nodes": fp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "workload_reduction": 1.0 - (len(selected) / len(all_nodes) if all_nodes else 0.0),
    }


def multi_source_distances(graph, seeds, max_hops, directed=False):
    distances = {node: 0 for node in seeds if node in graph}
    queue = deque(distances)
    while queue:
        node = queue.popleft()
        if distances[node] >= max_hops:
            continue
        if directed:
            neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
        else:
            neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def budgeted_nodes(distances, node_scores, budget):
    ranked = sorted(
        distances,
        key=lambda node: (distances[node], -node_scores.get(node, 0.0), str(node)),
    )
    return ranked[:budget]


def main():
    args = parse_args()
    events = pd.read_csv(args.events)
    truth = pd.read_csv(args.ground_truth)
    node_column = "node_id" if "node_id" in truth else "nid"
    label_column = next(name for name in ("label", "y_true", "is_malicious") if name in truth)
    malicious = set(truth.loc[truth[label_column].astype(int) == 1, node_column].astype(str))

    events["srcnode"] = events["srcnode"].astype(str)
    events["dstnode"] = events["dstnode"].astype(str)
    all_nodes = set(events["srcnode"]) | set(events["dstnode"])
    threshold = float(events.threshold.iloc[0])
    positive_pvalues = events.loc[events.conformal_pvalue > 0, "conformal_pvalue"]
    calibration_size = max(1, round(1.0 / float(positive_pvalues.min())) - 1)
    with tempfile.TemporaryDirectory() as directory:
        tracer = OnlineDynamicAttackTracer(
            make_cfg(directory, {}), "frozen_epoch_6", "test", threshold,
            calibration_scores=[0.0] * calibration_size,
        )
        online_nodes = set()
        for event in events.to_dict(orient="records"):
            decision = tracer.observe_edge(event, int(event["time_window"]))
            if decision["activated"]:
                online_nodes.update([str(event["srcnode"]), str(event["dstnode"])])
        tracer.close()
    seed_nodes = set(events.loc[events["primary_activation"].astype(int) == 1, "srcnode"]) | set(
        events.loc[events["primary_activation"].astype(int) == 1, "dstnode"]
    )
    budget = len(online_nodes)

    directed_graph = nx.DiGraph()
    node_scores = {}
    for row in events.itertuples(index=False):
        score = float(row.loss)
        directed_graph.add_edge(str(row.srcnode), str(row.dstnode))
        node_scores[str(row.srcnode)] = max(node_scores.get(str(row.srcnode), float("-inf")), score)
        node_scores[str(row.dstnode)] = max(node_scores.get(str(row.dstnode), float("-inf")), score)
    undirected_graph = directed_graph.to_undirected()

    methods = [("online_dynamic_retained_evidence", online_nodes, False, None)]
    for hops in (1, 2, 3):
        distances = multi_source_distances(undirected_graph, seed_nodes, hops)
        methods.append((
            f"posthoc_{hops}hop", budgeted_nodes(distances, node_scores, budget), True, hops
        ))
    distances = multi_source_distances(directed_graph, seed_nodes, 3, directed=True)
    methods.append((
        "posthoc_bidirectional_3hop", budgeted_nodes(distances, node_scores, budget), True, 3
    ))

    rows = []
    for method, nodes, uses_future_graph, hops in methods:
        row = {
            "dataset": args.dataset,
            "method": method,
            "model_seed": args.model_seed,
            "status": "pilot" if args.pilot else "main",
            "same_frozen_scores": True,
            "node_budget": budget,
            "seed_alert_nodes": len(seed_nodes),
            "uses_complete_test_graph": uses_future_graph,
            "maximum_hops": hops,
        }
        row.update(node_metrics(nodes, malicious, all_nodes))
        rows.append(row)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "final_results.csv", index=False)
    payload = {
        "experiment": "online_vs_posthoc_budget_matched",
        "claim_scope": "node attribution only; node-level ground truth cannot support edge-story claims",
        "protocol": "Frozen detector scores and equal maximum reported-node budget. Post-hoc methods may inspect the complete test graph.",
        "results": rows,
    }
    with (output / "final_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
