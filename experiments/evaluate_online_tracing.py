"""Evaluate causal online detection, tracing, prediction, and containment.

The evaluator never tunes a threshold on test labels. Ground truth is used only
after the event log has been produced. It supports event-level labels or the
conservative provenance node-level ground truth.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


POSITIVE_VALUES = {"1", "true", "yes", "malicious", "attack", "anomalous"}


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def as_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) > 0).astype(int)
    return series.fillna("").astype(str).str.strip().str.lower().isin(POSITIVE_VALUES).astype(int)


def binary_metrics(y_true, y_pred, scores=None):
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    result = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "mcc": float(matthews_corrcoef(truth, pred)) if len(np.unique(truth)) == 2 else 0.0,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_alerts_per_million": safe_div(fp * 1_000_000, fp + tn),
    }
    if scores is not None and len(np.unique(truth)) == 2:
        result["average_precision"] = float(average_precision_score(truth, scores))
        result["roc_auc"] = float(roc_auc_score(truth, scores))
    else:
        result["average_precision"] = None
        result["roc_auc"] = None
    return result


def load_events(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_event_log"] = str(path)
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    node_decisions_available = {
        "src_node_activated", "dst_node_activated"
    }.issubset(events.columns)
    required = {"time", "srcnode", "dstnode", "edge_type", "activated", "would_block", "cut"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Event logs are missing required columns: {missing}")
    for column in [
        "activated", "src_node_activated", "dst_node_activated", "context_activation",
        "would_block", "truncated", "cut",
    ]:
        if column not in events:
            events[column] = 0
        events[column] = as_binary(events[column])
    events["srcnode"] = events["srcnode"].astype(str)
    events["dstnode"] = events["dstnode"].astype(str)
    events.attrs["node_decisions_available"] = node_decisions_available
    return events


def attach_ground_truth(events, ground_truth_path):
    ground_truth = pd.read_csv(ground_truth_path)
    label_column = next((name for name in ["label", "y_true", "is_malicious"] if name in ground_truth), None)
    if label_column is None:
        raise ValueError("Ground truth requires label, y_true, or is_malicious")
    ground_truth["_label"] = as_binary(ground_truth[label_column])
    node_column = next((name for name in ["node_id", "nid"] if name in ground_truth), None)
    if node_column is not None and not {"srcnode", "dstnode"}.issubset(ground_truth.columns):
        malicious_nodes = set(ground_truth.loc[ground_truth["_label"] == 1, node_column].astype(str))
        events["y_true"] = (
            events["srcnode"].isin(malicious_nodes) | events["dstnode"].isin(malicious_nodes)
        ).astype(int)
        events.attrs["ground_truth_mode"] = "node_endpoint_projection"
        events.attrs["malicious_nodes"] = malicious_nodes
        return events

    join_keys = None
    if "event_id" in ground_truth and "event_id" in events:
        join_keys = ["event_id"]
    else:
        candidate = ["time", "srcnode", "dstnode", "edge_type"]
        if all(name in ground_truth for name in candidate):
            join_keys = candidate
    if join_keys is None:
        raise ValueError("Event ground truth needs event_id or time/srcnode/dstnode/edge_type keys")
    for column in ["srcnode", "dstnode"]:
        if column in ground_truth:
            ground_truth[column] = ground_truth[column].astype(str)
    payload = join_keys + ["_label"]
    for optional in ["campaign_id", "gt_stage", "stage"]:
        if optional in ground_truth and optional not in payload:
            payload.append(optional)
    truth = ground_truth[payload].drop_duplicates(join_keys, keep="last")
    events = events.merge(truth, how="left", on=join_keys, validate="many_to_one")
    match_rate = float(events["_label"].notna().mean())
    if match_rate < 0.95:
        raise ValueError(f"Only {match_rate:.2%} of events matched ground truth; check identity keys")
    events["y_true"] = events["_label"].fillna(0).astype(int)
    events.attrs["ground_truth_mode"] = "event"
    events.attrs["ground_truth_match_rate"] = match_rate
    return events


def node_metrics(events):
    all_nodes = set(events["srcnode"]) | set(events["dstnode"])
    if events.attrs.get("node_decisions_available", False):
        predicted = set(events.loc[events["src_node_activated"] == 1, "srcnode"]) | set(
            events.loc[events["dst_node_activated"] == 1, "dstnode"]
        )
    else:
        predicted = set(events.loc[events["activated"] == 1, "srcnode"]) | set(
            events.loc[events["activated"] == 1, "dstnode"]
        )
    malicious_events = events[events["y_true"] == 1]
    truth = events.attrs.get("malicious_nodes") or (
        set(malicious_events["srcnode"]) | set(malicious_events["dstnode"])
    )
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "total_nodes": len(all_nodes), "reported_nodes": len(predicted),
        "malicious_nodes": len(truth), "true_reported_nodes": tp,
        "benign_reported_nodes": fp, "precision": precision, "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "analyst_workload_reduction": 1.0 - safe_div(len(predicted), len(all_nodes)),
    }


def context_node_metrics(events):
    all_nodes = set(events["srcnode"]) | set(events["dstnode"])
    context = set(events.loc[events["activated"] == 1, "srcnode"]) | set(
        events.loc[events["activated"] == 1, "dstnode"]
    )
    malicious_events = events[events["y_true"] == 1]
    truth = events.attrs.get("malicious_nodes") or (
        set(malicious_events["srcnode"]) | set(malicious_events["dstnode"])
    )
    tp = len(context & truth)
    precision = safe_div(tp, len(context))
    recall = safe_div(tp, len(truth))
    return {
        "context_nodes": len(context), "covered_malicious_nodes": tp,
        "malicious_nodes": len(truth), "precision": precision, "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "workload_reduction": 1.0 - safe_div(len(context), len(all_nodes)),
    }


def tracing_metrics(events):
    selected = events[events["activated"] == 1]
    if events.attrs.get("ground_truth_mode") != "event":
        return {
            "available": False,
            "reason": "event-level ground truth is required for edge/path tracing metrics",
            "reported_edges": int(selected[["srcnode", "dstnode", "edge_type"]].drop_duplicates().shape[0]),
        }
    true_edges = set(map(tuple, events.loc[events["y_true"] == 1, ["srcnode", "dstnode", "edge_type"]].values))
    traced_edges = set(map(tuple, selected[["srcnode", "dstnode", "edge_type"]].values))
    precision = safe_div(len(true_edges & traced_edges), len(traced_edges))
    recall = safe_div(len(true_edges & traced_edges), len(true_edges))
    graph = nx.Graph()
    graph.add_edges_from(selected[["srcnode", "dstnode"]].itertuples(index=False, name=None))
    components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    return {
        "available": True,
        "reported_edges": len(traced_edges), "malicious_edges": len(true_edges),
        "edge_precision": precision, "edge_recall": recall,
        "edge_f1": safe_div(2 * precision * recall, precision + recall),
        "attack_subgraph_components": components,
        "largest_component_fraction": safe_div(
            max((len(c) for c in nx.connected_components(graph)), default=0), graph.number_of_nodes()
        ),
    }


def campaign_metrics(events):
    if "campaign_id" not in events:
        return {"available": False}
    malicious = events[(events["y_true"] == 1) & events["campaign_id"].notna()].copy()
    rows = []
    for campaign, group in malicious.groupby("campaign_id"):
        start = int(group["time"].min())
        detections = group[group["activated"] == 1]
        cuts = group[group["would_block"] == 1]
        rows.append({
            "campaign_id": str(campaign), "detected": int(not detections.empty),
            "detection_delay": None if detections.empty else int(detections["time"].min()) - start,
            "contained": int(not cuts.empty),
            "containment_delay": None if cuts.empty else int(cuts["time"].min()) - start,
        })
    detected_delays = [row["detection_delay"] for row in rows if row["detection_delay"] is not None]
    cut_delays = [row["containment_delay"] for row in rows if row["containment_delay"] is not None]
    return {
        "available": True, "campaigns": len(rows),
        "attack_detection_rate": safe_div(sum(row["detected"] for row in rows), len(rows)),
        "campaign_containment_rate": safe_div(sum(row["contained"] for row in rows), len(rows)),
        "median_detection_delay": float(np.median(detected_delays)) if detected_delays else None,
        "median_containment_delay": float(np.median(cut_delays)) if cut_delays else None,
        "per_campaign": rows,
    }


def attack_detection_precision(events):
    """Threshold-free ADP approximation on a 1001-point precision grid."""
    if "campaign_id" not in events or "loss" not in events:
        return {"available": False}
    labelled = events[(events["y_true"] == 1) & events["campaign_id"].notna()]
    campaigns = sorted(labelled["campaign_id"].astype(str).unique())
    if not campaigns:
        return {"available": False}
    node_scores = pd.concat([
        events[["srcnode", "loss"]].rename(columns={"srcnode": "node"}),
        events[["dstnode", "loss"]].rename(columns={"dstnode": "node"}),
    ]).groupby("node")["loss"].max().sort_values(ascending=False)
    node_truth = {}
    node_campaigns = defaultdict(set)
    for row in events.itertuples(index=False):
        for node in (str(row.srcnode), str(row.dstnode)):
            node_truth[node] = max(node_truth.get(node, 0), int(row.y_true))
            campaign = getattr(row, "campaign_id", None)
            if int(row.y_true) == 1 and pd.notna(campaign):
                node_campaigns[node].add(str(campaign))
    true_reported = 0
    detected_campaigns = set()
    operating_points = []
    for rank, node in enumerate(node_scores.index.astype(str), start=1):
        true_reported += int(node_truth.get(node, 0))
        detected_campaigns.update(node_campaigns.get(node, set()))
        operating_points.append((safe_div(true_reported, rank), safe_div(len(detected_campaigns), len(campaigns))))
    grid = np.linspace(0.0, 1.0, 1001)
    envelope = [max((detected for precision, detected in operating_points if precision >= p), default=0.0) for p in grid]
    return {
        "available": True, "adp": float(np.trapz(envelope, grid)),
        "attacks": len(campaigns),
        "attack_detection_at_precision_100": max((d for p, d in operating_points if p >= 1.0), default=0.0),
        "attack_detection_at_precision_90": max((d for p, d in operating_points if p >= 0.9), default=0.0),
    }


def prediction_metrics(events):
    stage_column = "gt_stage" if "gt_stage" in events else ("stage_y" if "stage_y" in events else None)
    if stage_column is None or "campaign_id" not in events:
        return {"available": False}
    targets = []
    predictions = []
    confidences = []
    lead_times = []
    malicious = events[(events["y_true"] == 1) & events["campaign_id"].notna()].sort_values("time")
    for _, group in malicious.groupby("campaign_id"):
        stages = group[stage_column].fillna("").astype(str).tolist()
        predicted_stages = group["predicted_next_stage"].fillna("").astype(str).tolist()
        times = pd.to_numeric(group["time"], errors="coerce").tolist()
        confidence_series = (
            group["prediction_confidence"]
            if "prediction_confidence" in group
            else pd.Series(0.0, index=group.index)
        )
        confidence_values = pd.to_numeric(confidence_series, errors="coerce").fillna(0.0).tolist()
        for index in range(len(group) - 1):
            next_index = next((offset for offset in range(index + 1, len(stages))
                               if stages[offset] and stages[offset] != stages[index]), None)
            if next_index is not None and predicted_stages[index] not in {"", "disabled", "unknown"}:
                targets.append(stages[next_index])
                predictions.append(predicted_stages[index])
                confidences.append(float(confidence_values[index]))
                lead_times.append(float(times[next_index] - times[index]))
    correct = np.asarray([int(pred == target) for pred, target in zip(predictions, targets)])
    confidence_array = np.asarray(confidences, dtype=float)
    ece = 0.0
    if len(correct):
        for low in np.linspace(0.0, 0.9, 10):
            mask = (confidence_array >= low) & (confidence_array < low + 0.1)
            if np.any(mask):
                ece += float(np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(confidence_array[mask])))
    return {
        "available": True, "evaluated_transitions": len(targets),
        "next_stage_accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        "next_stage_macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)) if targets else 0.0,
        "expected_calibration_error": ece,
        "median_prediction_lead_time": float(np.median(lead_times)) if lead_times else None,
    }


def containment_metrics(events):
    if events.attrs.get("ground_truth_mode") != "event":
        return {
            "available": False,
            "reason": "event-level ground truth is required to separate prevention from benign collateral",
            "unique_cut_nodes": int(events.loc[events["cut"] == 1, "cut_node"].dropna().astype(str).nunique()),
        }
    blocked = events[events["would_block"] == 1]
    malicious_total = int((events["y_true"] == 1).sum())
    malicious_blocked = int((blocked["y_true"] == 1).sum())
    benign_blocked = int((blocked["y_true"] == 0).sum())
    benign_total = int((events["y_true"] == 0).sum())
    return {
        "available": True,
        "counterfactual_blocked_events": len(blocked),
        "prevented_malicious_events": malicious_blocked,
        "attack_event_prevention_rate": safe_div(malicious_blocked, malicious_total),
        "benign_collateral_events": benign_blocked,
        "benign_collateral_rate": safe_div(benign_blocked, benign_total),
        "containment_precision": safe_div(malicious_blocked, len(blocked)),
        "net_prevention_utility": safe_div(malicious_blocked, malicious_total) - safe_div(benign_blocked, benign_total),
        "unique_cut_nodes": int(events.loc[events["cut"] == 1, "cut_node"].dropna().astype(str).nunique()),
    }


def efficiency_metrics(events):
    if "trace_latency_us" not in events:
        return {"available": False}
    latency = pd.to_numeric(events["trace_latency_us"], errors="coerce").dropna().to_numpy()
    result = {
        "available": True, "events": len(events),
        "latency_us_mean": float(np.mean(latency)) if len(latency) else None,
        "latency_us_p50": float(np.percentile(latency, 50)) if len(latency) else None,
        "latency_us_p95": float(np.percentile(latency, 95)) if len(latency) else None,
        "latency_us_p99": float(np.percentile(latency, 99)) if len(latency) else None,
        "tracing_throughput_events_per_second": safe_div(1_000_000, float(np.mean(latency))) if len(latency) else None,
    }
    for source, prefix in [("model_latency_us", "model"), ("end_to_end_latency_us", "end_to_end")]:
        if source not in events:
            continue
        values = pd.to_numeric(events[source], errors="coerce").dropna().to_numpy()
        if not len(values):
            continue
        result[f"{prefix}_latency_us_mean"] = float(np.mean(values))
        result[f"{prefix}_latency_us_p95"] = float(np.percentile(values, 95))
        result[f"{prefix}_latency_us_p99"] = float(np.percentile(values, 99))
        result[f"{prefix}_throughput_events_per_second"] = safe_div(1_000_000, float(np.mean(values)))
    return result


def state_resource_metrics(event_paths, explicit_paths):
    paths = [Path(path) for path in explicit_paths]
    if not paths:
        paths = [Path(path).parent / "dynamic_state.json" for path in event_paths]
    states = []
    for path in paths:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as file:
                states.append(json.load(file))
    if not states:
        return {"available": False}
    return {
        "available": True,
        "peak_process_rss_bytes": max(int(state.get("peak_process_rss_bytes", 0)) for state in states),
        "peak_cuda_allocated_bytes": max(int(state.get("peak_cuda_allocated_bytes", 0)) for state in states),
        "dynamic_nodes": sum(int(state.get("num_nodes", 0)) for state in states),
        "dynamic_typed_edges": sum(int(state.get("num_edges", 0)) for state in states),
        "attack_nodes": sum(int(state.get("num_attack_nodes", 0)) for state in states),
        "attack_typed_edges": sum(int(state.get("num_attack_edges", 0)) for state in states),
        "dropped_dynamic_edges": sum(int(state.get("dropped_dynamic_edges", 0)) for state in states),
        "dropped_attack_edges": sum(int(state.get("dropped_attack_edges", 0)) for state in states),
    }


def block_bootstrap_ci(events, repetitions, seed):
    if repetitions <= 0:
        return {}
    block_column = "time_window" if "time_window" in events else None
    if block_column is None:
        events = events.copy()
        events["_block"] = np.arange(len(events)) // 10000
        block_column = "_block"
    aggregates = []
    for block, group in events.groupby(block_column, dropna=False):
        truth = group["y_true"].to_numpy(dtype=int)
        pred = group["activated"].to_numpy(dtype=int)
        aggregates.append((block, int(((truth == 1) & (pred == 1)).sum()),
                           int(((truth == 0) & (pred == 1)).sum()),
                           int(((truth == 0) & (pred == 0)).sum()),
                           int(((truth == 1) & (pred == 0)).sum())))
    rng = np.random.default_rng(seed)
    samples = {"precision": [], "recall": [], "f1": [], "mcc": []}
    for _ in range(repetitions):
        picked = rng.integers(0, len(aggregates), len(aggregates))
        tp = sum(aggregates[index][1] for index in picked)
        fp = sum(aggregates[index][2] for index in picked)
        tn = sum(aggregates[index][3] for index in picked)
        fn = sum(aggregates[index][4] for index in picked)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        samples["precision"].append(precision)
        samples["recall"].append(recall)
        samples["f1"].append(safe_div(2 * precision * recall, precision + recall))
        samples["mcc"].append(safe_div(tp * tn - fp * fn, denominator))
    return {
        metric: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}
        for metric, values in samples.items()
    }


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "per_campaign":
                continue
            flatten(f"{prefix}.{key}" if prefix else key, child, output)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value


def evaluate(args):
    events = attach_ground_truth(load_events(args.events), args.ground_truth)
    if events.attrs.get("ground_truth_mode") == "event":
        event_detection = binary_metrics(events["y_true"], events["activated"], events.get("loss"))
        confidence_intervals = block_bootstrap_ci(events, args.bootstrap, args.seed)
    else:
        event_detection = {
            "available": False,
            "reason": "node labels cannot be projected into unbiased event-level detection labels",
        }
        confidence_intervals = {}
    results = {
        "protocol_version": "tifs-causal-online-v1", "dataset": args.dataset,
        "method": args.method, "seed": args.seed,
        "ground_truth_mode": events.attrs.get("ground_truth_mode"),
        "event_detection": event_detection,
        "node_attribution": node_metrics(events),
        "context_node_coverage": context_node_metrics(events),
        "tracing": tracing_metrics(events),
        "campaign_detection": campaign_metrics(events),
        "attack_detection_precision": attack_detection_precision(events),
        "prediction": prediction_metrics(events),
        "containment": containment_metrics(events), "efficiency": efficiency_metrics(events),
        "state_resources": state_resource_metrics(args.events, args.states),
        "confidence_intervals_95": confidence_intervals,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    flat = {}
    flatten("", results, flat)
    pd.DataFrame([flat]).to_csv(out_dir / "metrics.csv", index=False)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="+", required=True, help="online_trace_events.csv files")
    parser.add_argument("--states", nargs="*", default=[], help="optional dynamic_state.json files")
    parser.add_argument("--ground-truth", required=True, help="event- or node-level CSV ground truth")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="online_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    evaluated = evaluate(parsed_args)
    if not parsed_args.quiet:
        print(json.dumps(evaluated, indent=2, ensure_ascii=False))
