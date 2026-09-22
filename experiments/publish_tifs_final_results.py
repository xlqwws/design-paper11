"""Publish paper-facing final tables without copying training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--artifacts-root", default=str(ROOT / "artifacts"))
    return parser.parse_args()


def write_final(directory, experiment, protocol, rows, limitations):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "final_results.csv", index=False)
    payload = {
        "experiment": experiment,
        "protocol": protocol,
        "limitations": limitations,
        "results": frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
    }
    with (directory / "final_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)


def one_row(path):
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one metrics row in {path}")
    return frame.iloc[0]


def value(row, name, default=None):
    result = row.get(name, default)
    return default if pd.isna(result) else result


def full_sources(artifacts):
    e3 = artifacts / "tifs_results" / "E3-CADETS-Causal" / "pilot_semantic_ablation_full_seed0" / "metrics.csv"
    streamspot = artifacts / "tifs_results" / "StreamSpot" / "pilot_seed0_relation_fixed" / "metrics.csv"
    if not e3.is_file() or not streamspot.is_file():
        raise FileNotFoundError("Validated E3 and StreamSpot pilot metrics are required")
    return e3, streamspot


def design_rows():
    datasets = "E3-CADETS;E5-CADETS;StreamSpot;Unicorn-Wget"
    return [
        {"id": "E1", "folder": "01_main_effectiveness", "question": "Detection and attribution effectiveness", "datasets": datasets, "primary_metrics": "AP;MCC;F1;node_precision;node_recall;false_alerts_per_million", "required_model_seeds": 5},
        {"id": "E2", "folder": "02_online_vs_posthoc", "question": "Online tracing versus budget-matched post-hoc traversal", "datasets": "E3-CADETS;E5-CADETS", "primary_metrics": "node_edge_F1;stage_coverage;workload;components", "required_model_seeds": 5},
        {"id": "E3", "folder": "03_component_ablation", "question": "Contribution of each dynamic mechanism", "datasets": "E3-CADETS;E5-CADETS", "primary_metrics": "paired_delta_primary_outcomes", "required_model_seeds": 5},
        {"id": "E4", "folder": "04_prediction_containment", "question": "Prospective prediction and counterfactual containment", "datasets": "E3-CADETS;E5-CADETS", "primary_metrics": "next_stage_macro_F1;lead_time;prevention;collateral;utility", "required_model_seeds": 5},
        {"id": "E5", "folder": "05_system_efficiency", "question": "Strict-online latency, throughput and bounded state", "datasets": datasets, "primary_metrics": "p50_p95_p99;events_per_second;RAM;VRAM;state_growth", "required_model_seeds": 5},
        {"id": "E6", "folder": "06_robustness", "question": "Missing events, score noise, evasion and temporal shift", "datasets": datasets, "primary_metrics": "absolute_metric;degradation_from_clean", "required_model_seeds": 5},
        {"id": "E7", "folder": "07_statistical_readiness", "question": "Hierarchical uncertainty, paired effects and acceptance gates", "datasets": datasets, "primary_metrics": "hierarchical_CI;Holm_p;effect_size", "required_model_seeds": 5},
    ]


def publish_design(results):
    write_final(
        results / "00_experiment_design", "preregistered_tifs_matrix",
        "Strict chronological split; benign-only validation calibration; test labels are joined only after immutable decisions; five full training seeds; attack/graph-level hierarchical bootstrap; Holm correction for four preregistered outcomes.",
        design_rows(),
        "Five seeds alone do not support a two-sided exact sign-test claim at p<0.05. Primary uncertainty must resample model seed and independent attack campaign or graph without treating events as independent replicates.",
    )


def publish_main(results, e3_path, streamspot_path):
    ablations = pd.read_csv(results / "03_component_ablation" / "final_results.csv")
    e3 = ablations.loc[ablations.variant == "full"].iloc[0]
    stream = one_row(streamspot_path)
    rows = [
        {
            "dataset": "E3-CADETS-Causal", "method": "online_dynamic", "model_seed": 0,
            "result_tier": "pilot_sampled", "evaluation_unit": "node",
            "precision": value(e3, "node_precision"), "recall": value(e3, "node_recall"),
            "f1": value(e3, "node_f1"), "mcc": None, "auroc": None,
            "average_precision": None, "reported_units": value(e3, "reported_nodes"),
            "workload_reduction": value(e3, "workload_reduction"),
        },
        {
            "dataset": "StreamSpot", "method": "online_dynamic_detection_only", "model_seed": 0,
            "result_tier": "pilot_50_edges_per_graph", "evaluation_unit": "graph",
            "precision": value(stream, "precision"), "recall": value(stream, "recall"),
            "f1": value(stream, "f1"), "mcc": value(stream, "mcc"), "auroc": value(stream, "auroc"),
            "average_precision": value(stream, "average_precision"), "reported_units": value(stream, "reported_graphs"),
            "workload_reduction": None,
        },
        {"dataset": "E5-CADETS-Causal", "method": "online_dynamic", "model_seed": None, "result_tier": "not_run", "evaluation_unit": "node", "precision": None, "recall": None, "f1": None, "mcc": None, "auroc": None, "average_precision": None, "reported_units": None, "workload_reduction": None},
        {"dataset": "Unicorn-Wget", "method": "online_dynamic_detection_only", "model_seed": None, "result_tier": "not_run", "evaluation_unit": "graph", "precision": None, "recall": None, "f1": None, "mcc": None, "auroc": None, "average_precision": None, "reported_units": None, "workload_reduction": None},
    ]
    write_final(
        results / "01_main_effectiveness", "main_effectiveness",
        "All reported values come from immutable online decisions with validation-only calibration.", rows,
        "E3 and StreamSpot are sampled pilots with one model seed. E5 and Unicorn-Wget have no final run and therefore contain explicit null metrics.",
    )


def publish_ablation(results, artifacts):
    rows = []
    root = artifacts / "tifs_results" / "E3-CADETS-Causal"
    for path in sorted(root.glob("pilot_semantic_ablation_*_seed0/metrics.csv")):
        metric = one_row(path)
        rows.append({
            "dataset": "E3-CADETS-Causal", "variant": value(metric, "method"), "model_seed": 0,
            "result_tier": "pilot_sampled", "node_precision": value(metric, "node_attribution.precision"),
            "node_recall": value(metric, "node_attribution.recall"), "node_f1": value(metric, "node_attribution.f1"),
            "reported_nodes": value(metric, "node_attribution.reported_nodes"),
            "workload_reduction": value(metric, "node_attribution.analyst_workload_reduction"),
            "unique_cut_nodes": value(metric, "containment.unique_cut_nodes"),
            "trace_p99_us": value(metric, "efficiency.latency_us_p99"),
        })
    write_final(
        results / "03_component_ablation", "component_ablation",
        "Downstream variants replay the identical frozen detector score stream. Detector weights and threshold are unchanged.", rows,
        "One sampled E3 campaign and one model seed cannot establish statistical superiority. Node-only labels do not evaluate prediction or prevention efficacy.",
    )


def publish_prediction_containment(results, e3_path, e3_events):
    ablations = pd.read_csv(results / "03_component_ablation" / "final_results.csv")
    metric = ablations.loc[ablations.variant == "full"].iloc[0]
    rows = [
        {
            "task": "prospective_next_stage_prediction", "dataset": "E3-CADETS-Causal",
            "available": False, "result_tier": "not_evaluable",
            "primary_metric": "next_stage_macro_f1", "value": None,
            "reason": "No independently annotated campaign-stage ground truth is available.",
        },
        {
            "task": "counterfactual_containment_efficacy", "dataset": "E3-CADETS-Causal",
            "available": False, "result_tier": "operational_audit_only",
            "primary_metric": "attack_event_prevention_rate", "value": None,
            "reason": "Node-level labels cannot distinguish prevented malicious events from benign collateral.",
        },
        {
            "task": "cut_decision_audit", "dataset": "E3-CADETS-Causal",
            "available": True, "result_tier": "pilot_sampled",
            "primary_metric": "unique_cut_nodes", "value": value(metric, "unique_cut_nodes"),
            "reason": f"{int(value(metric, 'would_block_events', 0))} later endpoint-touching events were marked would_block; this is not a prevention claim.",
        },
    ]
    write_final(
        results / "04_prediction_containment", "prediction_and_containment",
        "Predictions must be logged before the next event update. Containment is counterfactual and must retain blocked events for scoring.", rows,
        "Independent stage and event-level attack labels are mandatory before accuracy, prevention, collateral, or utility can be reported.",
    )


def latency_row(dataset, events_path, tier):
    columns = ["model_latency_us", "trace_latency_us", "end_to_end_latency_us"]
    events = pd.read_csv(events_path, usecols=columns)
    row = {"dataset": dataset, "model_seed": 0, "result_tier": tier, "events": len(events)}
    for column in columns:
        values = pd.to_numeric(events[column], errors="coerce").dropna().to_numpy()
        prefix = column[:-len("_latency_us")] if column.endswith("_latency_us") else column
        row[f"{prefix}_mean_us"] = float(np.mean(values))
        row[f"{prefix}_p50_us"] = float(np.percentile(values, 50))
        row[f"{prefix}_p95_us"] = float(np.percentile(values, 95))
        row[f"{prefix}_p99_us"] = float(np.percentile(values, 99))
        row[f"{prefix}_throughput_eps"] = float(1_000_000 / np.mean(values)) if np.mean(values) else None
    return row


def publish_efficiency(results, e3_events, stream_events):
    rows = [
        latency_row("E3-CADETS-Causal", e3_events, "pilot_sampled"),
        latency_row("StreamSpot", stream_events, "pilot_50_edges_per_graph"),
    ]
    write_final(
        results / "05_system_efficiency", "strict_online_efficiency",
        "Causal batch size 1. Model, tracer, and end-to-end latency are measured separately on each event.", rows,
        "Pilot rates are not a full-dataset peak-rate sustainability demonstration; warm-up, repeated runs, energy, and 24-hour state growth remain required.",
    )


def publish_readiness(results):
    rows = []
    completed = {"E3-CADETS-Causal": 1, "E5-CADETS-Causal": 0, "StreamSpot": 1, "Unicorn-Wget": 0}
    for dataset, seeds in completed.items():
        rows.append({
            "dataset": dataset, "completed_full_model_seeds": seeds, "required_full_model_seeds": 5,
            "full_raw_dataset": False, "independent_attack_units_available": dataset == "StreamSpot",
            "holm_corrected_primary_tests_available": False,
            "readiness": "pilot_only",
        })
    write_final(
        results / "07_statistical_readiness", "statistical_and_acceptance_gate_audit",
        "Use paired model seeds and hierarchical campaign/graph block bootstrap. Correct four primary comparisons with Holm and report paired effect sizes plus confidence intervals.", rows,
        "No current dataset satisfies the complete main-paper gate. Pilot results must not be presented as TIFS-level evidence.",
    )


def main():
    args = parse_args()
    results = Path(args.results_root)
    artifacts = Path(args.artifacts_root)
    e3_path, stream_path = full_sources(artifacts)
    e3_events = artifacts / "attack_reconstruction" / "tracing" / "534b00e1ff2c0dbd42cb298c7a37da837fe7615aa37992d983b1da8a14c9d017" / "E3-CADETS-Causal" / "online_dynamic" / "model_epoch_6" / "test" / "online_trace_events.csv"
    stream_events = artifacts / "attack_reconstruction" / "tracing" / "e0a1096ef1bf6e07e37c90e9d341f080099fff3d75b88c4b0aa13ec09fd485e1" / "StreamSpot" / "online_dynamic" / "model_epoch_6" / "test" / "online_trace_events.csv"
    publish_design(results)
    publish_main(results, e3_path, stream_path)
    publish_prediction_containment(results, e3_path, e3_events)
    publish_readiness(results)
    print(f"Published final-only experiment tables to {results}")


if __name__ == "__main__":
    main()
