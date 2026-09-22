"""Publish final-only multi-seed experiment tables from immutable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
T_CRITICAL_95_DF4 = 2.7764451051977987

E3_METRICS = {
    0: "E3-CADETS-Causal/pilot_semantic_ablation_full_seed0/metrics.csv",
    1: "E3-CADETS-Causal/pilot_multiseed_full_seed1/metrics.csv",
    2: "E3-CADETS-Causal/pilot_multiseed_full_seed2/metrics.csv",
    3: "E3-CADETS-Causal/pilot_multiseed_full_seed3/metrics.csv",
    4: "E3-CADETS-Causal/pilot_multiseed_full_seed4/metrics.csv",
}
GRAPH_METRICS = {
    "StreamSpot": {seed: f"StreamSpot/pilot_multiseed_full_seed{seed}/metrics.csv" for seed in range(5)},
    "Unicorn-Wget": {seed: f"Unicorn-Wget/pilot_seed{seed}/metrics.csv" for seed in range(5)},
}
TRACE_HASHES = {
    "E3-CADETS-Causal": [
        "534b00e1ff2c0dbd42cb298c7a37da837fe7615aa37992d983b1da8a14c9d017",
        "f4225c26b299f0829c6847a3f196759044645a2a1afeb94d30d20b9e557d95f5",
        "783dbeacc713bd5484fb577f22f7f644d7db358cd0b14242c36549948eebed43",
        "9f32fb07954497a1e10fa223190194a6284b99a541101a9c839a287eabaa4e43",
        "dbcd7b72eebd44f93156ab152cee224e69ecaca1b5eeb8b9969599865e33bdd2",
    ],
    "StreamSpot": [
        "3983e722945451fdb33a611a4a9db4295b6327187e8894e838d8fb26ac7733c5",
        "878034914d1a08f84946ac9c646c2c53ed994f758b1e2c146f5af9771fc200d5",
        "92207e4a98e0660c7c5d070158dff608b8d017d754556f4076aa0ce2f3285f4d",
        "0d3451527ef495e7139f07d9012c1bf177c18662998dd9a700c8c24b8a39ac3c",
        "2ecb2d7e33c129af771fb85bdf9ee9ba9e0d0fc10ec188533dda2f9f3f1e7f73",
    ],
    "Unicorn-Wget": [
        "895339a805827b25d7a09032ea7d7b066a49c045959b420cfbab5861da7aed39",
        "2fc6199d19567fc9b3e8487a4538d604634813d393b13dee1604ba513220b669",
        "bb8bdf27d251355c46818f9bf4fad692faab0241744361c3a7925cf4d0ba4570",
        "9a10f6c54080ea03d480c368eca8352fb3bc56e48864977cc5453420fc3f779e",
        "e8bb6a7e2b262f3fef9f58a396839603b08c4dafe1bc5c40162b52cd1bb96c73",
    ],
}


def read_one(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(frame)}")
    return frame.iloc[0]


def scalar(row: pd.Series, key: str):
    value = row.get(key)
    return None if value is None or pd.isna(value) else value


def write_final(folder: str, experiment: str, protocol: str, limitations: str, rows: list[dict]) -> None:
    target = RESULTS / folder
    target.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in target.iterdir() if path.name not in {"final_results.csv", "final_results.json"}]
    if unexpected:
        raise RuntimeError(f"Non-final files found in {target}: {unexpected}")
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "final_results.csv", index=False)
    records = json.loads(frame.to_json(orient="records", double_precision=15))
    payload = {
        "experiment": experiment,
        "protocol": protocol,
        "limitations": limitations,
        "results": records,
    }
    with (target / "final_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)


def aggregate_rows(rows: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    grouped = frame.groupby(group_keys, dropna=False, sort=True)
    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(group_keys, group_values))
        n = int(group["model_seed"].nunique())
        for statistic in ["mean", "std", "ci95_low", "ci95_high"]:
            row = {**base, "row_type": "aggregate", "statistic": statistic, "model_seed": None, "n_model_seeds": n}
            for metric in metrics:
                values = pd.to_numeric(group.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    row[metric] = None
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                half = T_CRITICAL_95_DF4 * std / np.sqrt(len(values)) if len(values) == 5 else None
                if statistic == "mean":
                    row[metric] = mean
                elif statistic == "std":
                    row[metric] = std
                elif statistic == "ci95_low":
                    row[metric] = mean - half if half is not None else None
                else:
                    row[metric] = mean + half if half is not None else None
            output.append(row)
    return output


def publish_design() -> None:
    rows = [
        {"id": "E1", "folder": "01_main_effectiveness", "question": "Multi-seed detection and attribution", "datasets": "E3-CADETS;E5-CADETS;StreamSpot;Unicorn-Wget", "required_model_seeds": 5, "completed_scope": "E3/StreamSpot/Unicorn five-seed pilots; E5 blocked by incomplete raw corpus"},
        {"id": "E2", "folder": "02_online_vs_posthoc", "question": "Budget-matched online versus post-hoc tracing", "datasets": "E3-CADETS;E5-CADETS", "required_model_seeds": 5, "completed_scope": "E3 five seeds"},
        {"id": "E3", "folder": "03_component_ablation", "question": "Dynamic mechanism contribution", "datasets": "E3-CADETS;E5-CADETS", "required_model_seeds": 5, "completed_scope": "E3 five seeds, nine variants"},
        {"id": "E4", "folder": "04_prediction_containment", "question": "Prospective prediction and counterfactual containment", "datasets": "E3-CADETS;E5-CADETS", "required_model_seeds": 5, "completed_scope": "Decision audit only; required event/stage labels unavailable"},
        {"id": "E5", "folder": "05_system_efficiency", "question": "Strict-online event latency", "datasets": "E3-CADETS;StreamSpot;Unicorn-Wget", "required_model_seeds": 5, "completed_scope": "Fifteen GPU runs"},
        {"id": "E6", "folder": "06_robustness", "question": "Missing events, score noise and attenuation", "datasets": "E3-CADETS;StreamSpot", "required_model_seeds": 5, "completed_scope": "Frozen-score pilot; perturbation repeats are not model seeds"},
        {"id": "E7", "folder": "07_statistical_readiness", "question": "Acceptance-gate audit", "datasets": "all", "required_model_seeds": 5, "completed_scope": "Seed uncertainty available for three sampled pilots"},
    ]
    write_final(
        "00_experiment_design", "tifs_experiment_matrix",
        "Strict chronological online decisions, validation-only calibration, five model seeds, and final-only publication tables.",
        "This matrix describes completed local evidence, not a guarantee of editorial acceptance.", rows,
    )


def main_seed_rows() -> list[dict]:
    rows = []
    metric_root = ARTIFACTS / "tifs_results"
    for seed, relative in E3_METRICS.items():
        metric = read_one(metric_root / relative)
        rows.append({
            "dataset": "E3-CADETS-Causal", "method": "online_dynamic", "row_type": "seed",
            "statistic": "observed", "model_seed": seed, "n_model_seeds": 1,
            "status": "completed", "result_tier": "pilot_sampled", "evaluation_unit": "node",
            "precision": scalar(metric, "node_attribution.precision"),
            "recall": scalar(metric, "node_attribution.recall"),
            "f1": scalar(metric, "node_attribution.f1"), "mcc": None, "auroc": None,
            "average_precision": None,
            "reported_units": scalar(metric, "node_attribution.reported_nodes"),
            "workload_reduction": scalar(metric, "node_attribution.analyst_workload_reduction"),
        })
    for dataset, seed_paths in GRAPH_METRICS.items():
        tier = "pilot_50_edges_per_graph" if dataset == "StreamSpot" else "pilot_200_edges_per_graph"
        for seed, relative in seed_paths.items():
            metric = read_one(metric_root / relative)
            rows.append({
                "dataset": dataset, "method": "online_dynamic_detection_only", "row_type": "seed",
                "statistic": "observed", "model_seed": seed, "n_model_seeds": 1,
                "status": "completed", "result_tier": tier, "evaluation_unit": "graph",
                "precision": scalar(metric, "precision"), "recall": scalar(metric, "recall"),
                "f1": scalar(metric, "f1"), "mcc": scalar(metric, "mcc"),
                "auroc": scalar(metric, "auroc"), "average_precision": scalar(metric, "average_precision"),
                "reported_units": scalar(metric, "reported_graphs"), "workload_reduction": None,
            })
    return rows


def publish_main() -> None:
    rows = main_seed_rows()
    metrics = ["precision", "recall", "f1", "mcc", "auroc", "average_precision", "reported_units", "workload_reduction"]
    rows.extend(aggregate_rows(rows, ["dataset", "method", "result_tier", "evaluation_unit"], metrics))
    rows.append({
        "dataset": "E5-CADETS-Causal", "method": "online_dynamic", "row_type": "status",
        "statistic": "not_computed", "model_seed": None, "n_model_seeds": 0,
        "status": "blocked_incomplete_raw_data", "result_tier": "not_evaluable", "evaluation_unit": "node",
        "reason": "Local manifest lists 122 E5 gzip files; only numbered shards 1-10 are present and attack-period shards are missing.",
    })
    write_final(
        "01_main_effectiveness", "multi_seed_main_effectiveness",
        "Seed rows are immutable online decisions. Aggregate 95% intervals are two-sided t intervals over five independently trained model seeds.",
        "All completed runs are sampled pilots. E3 has one attack campaign and node-only labels. E5 cannot be evaluated from the incomplete local corpus.", rows,
    )


def load_ablation_seed_rows() -> list[dict]:
    rows = []
    for seed in range(5):
        path = ARTIFACTS / "tifs_results" / "E3-CADETS-Causal" / f"pilot_multiseed_ablation_seed{seed}" / "final_results.csv"
        frame = pd.read_csv(path)
        if len(frame) != 9:
            raise ValueError(f"Expected nine ablations for seed {seed}")
        full = frame.loc[frame["variant"] == "full"].iloc[0]
        for record in frame.to_dict(orient="records"):
            record.update({"row_type": "seed", "statistic": "observed", "n_model_seeds": 1})
            record["full_minus_variant_node_f1"] = float(full["node_f1"] - record["node_f1"])
            record["full_minus_variant_node_recall"] = float(full["node_recall"] - record["node_recall"])
            record["variant_minus_full_trace_p99_us"] = float(record["trace_p99_us"] - full["trace_p99_us"])
            rows.append(record)
    return rows


def publish_ablation() -> list[dict]:
    rows = load_ablation_seed_rows()
    metrics = [
        "primary_activations", "context_activations", "retained_context_activations",
        "unique_cut_nodes", "would_block_events", "trace_p99_us", "node_precision",
        "node_recall", "node_f1", "reported_nodes", "true_reported_nodes", "workload_reduction",
        "full_minus_variant_node_f1", "full_minus_variant_node_recall",
        "variant_minus_full_trace_p99_us",
    ]
    rows.extend(aggregate_rows(rows, ["dataset", "variant", "result_tier"], metrics))
    write_final(
        "03_component_ablation", "five_seed_component_ablation",
        "All nine variants replay the identical frozen detector score stream within each seed; positive full-minus-variant deltas favor the full method.",
        "One sampled E3 campaign limits attack-level inference. Node labels cannot validate stage prediction or event prevention.", rows,
    )
    return rows


def publish_posthoc() -> None:
    rows = []
    for seed in range(5):
        path = ARTIFACTS / "tifs_results" / "E3-CADETS-Causal" / f"pilot_multiseed_posthoc_seed{seed}" / "final_results.csv"
        frame = pd.read_csv(path)
        if len(frame) != 5:
            raise ValueError(f"Expected five tracing methods for seed {seed}")
        for record in frame.to_dict(orient="records"):
            record.update({"row_type": "seed", "statistic": "observed", "n_model_seeds": 1})
            rows.append(record)
    metrics = ["node_budget", "seed_alert_nodes", "reported_nodes", "true_reported_nodes", "benign_reported_nodes", "precision", "recall", "f1", "workload_reduction"]
    rows.extend(aggregate_rows(rows, ["dataset", "method", "status"], metrics))
    rows.append({
        "dataset": "E5-CADETS-Causal", "method": "all", "status": "blocked_incomplete_raw_data",
        "row_type": "status", "statistic": "not_computed", "model_seed": None, "n_model_seeds": 0,
    })
    write_final(
        "02_online_vs_posthoc", "budget_matched_online_vs_posthoc",
        "Each post-hoc traversal receives exactly the online method's reported-node budget and the same frozen score stream, but may inspect the completed test graph.",
        "The comparison is a five-seed sampled E3 pilot with one labeled campaign; E5 is unavailable locally.", rows,
    )


def publish_prediction(ablation_rows: list[dict]) -> None:
    seed_rows = []
    for row in ablation_rows:
        if row.get("row_type") == "seed" and row.get("variant") == "full":
            seed_rows.append({
                "dataset": "E3-CADETS-Causal", "task": "cut_decision_audit", "row_type": "seed",
                "statistic": "observed", "model_seed": row["model_seed"], "n_model_seeds": 1,
                "status": "operational_audit_only", "prediction_accuracy_available": False,
                "containment_efficacy_available": False, "unique_cut_nodes": row["unique_cut_nodes"],
                "would_block_events": row["would_block_events"],
            })
    rows = list(seed_rows)
    rows.extend(aggregate_rows(seed_rows, ["dataset", "task", "status"], ["unique_cut_nodes", "would_block_events"]))
    rows.extend([
        {"dataset": "E5-CADETS-Causal", "task": "prediction_and_containment", "row_type": "status", "statistic": "not_computed", "n_model_seeds": 0, "status": "blocked_incomplete_raw_data"},
        {"dataset": "StreamSpot", "task": "prediction_and_containment", "row_type": "status", "statistic": "not_evaluable", "n_model_seeds": 5, "status": "anonymous_relations_no_stage_labels"},
        {"dataset": "Unicorn-Wget", "task": "prediction_and_containment", "row_type": "status", "statistic": "not_evaluable", "n_model_seeds": 5, "status": "anonymous_relations_no_stage_labels"},
    ])
    write_final(
        "04_prediction_containment", "prospective_prediction_and_containment_audit",
        "Predictions are emitted before the next update and cuts retain later events as would-block counterfactuals.",
        "No available dataset supplies independent event-level attack and campaign-stage labels, so prediction accuracy and prevention efficacy are not claimed.", rows,
    )


def trace_path(dataset: str, seed: int) -> Path:
    digest = TRACE_HASHES[dataset][seed]
    return ARTIFACTS / "attack_reconstruction" / "tracing" / digest / dataset / "online_dynamic" / "model_epoch_6" / "test" / "online_trace_events.csv"


def publish_efficiency() -> None:
    rows = []
    metric_root = ARTIFACTS / "tifs_results"
    for dataset in TRACE_HASHES:
        tier = {"E3-CADETS-Causal": "pilot_sampled", "StreamSpot": "pilot_50_edges_per_graph", "Unicorn-Wget": "pilot_200_edges_per_graph"}[dataset]
        for seed in range(5):
            frame = pd.read_csv(trace_path(dataset, seed), usecols=["model_latency_us", "trace_latency_us", "end_to_end_latency_us"])
            row = {
                "dataset": dataset, "row_type": "seed", "statistic": "observed", "model_seed": seed,
                "n_model_seeds": 1, "status": "completed", "result_tier": tier, "events": len(frame),
            }
            for column in ["model_latency_us", "trace_latency_us", "end_to_end_latency_us"]:
                values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
                prefix = column[:-len("_latency_us")]
                row[f"{prefix}_mean_us"] = float(np.mean(values))
                row[f"{prefix}_p50_us"] = float(np.percentile(values, 50))
                row[f"{prefix}_p95_us"] = float(np.percentile(values, 95))
                row[f"{prefix}_p99_us"] = float(np.percentile(values, 99))
                row[f"{prefix}_throughput_eps"] = float(1_000_000 / np.mean(values))
            if dataset == "E3-CADETS-Causal":
                metric = read_one(metric_root / E3_METRICS[seed])
                row["peak_process_rss_bytes"] = scalar(metric, "state_resources.peak_process_rss_bytes")
                row["peak_cuda_allocated_bytes"] = scalar(metric, "state_resources.peak_cuda_allocated_bytes")
            rows.append(row)
    metrics = [key for key in rows[0] if key.endswith(("_us", "_eps", "_bytes"))] + ["events"]
    rows.extend(aggregate_rows(rows, ["dataset", "status", "result_tier"], metrics))
    rows.append({
        "dataset": "E5-CADETS-Causal", "row_type": "status", "statistic": "not_computed",
        "model_seed": None, "n_model_seeds": 0, "status": "blocked_incomplete_raw_data",
    })
    write_final(
        "05_system_efficiency", "five_seed_strict_online_efficiency",
        "Causal batch size is one. Model, tracer and end-to-end event latency are measured separately; intervals summarize five GPU runs.",
        "Pilot rates are not full-ingestion peak-rate or 24-hour bounded-state demonstrations. Graph datasets lack process/GPU resource snapshots.", rows,
    )


def publish_readiness() -> None:
    rows = [
        {"dataset": "E3-CADETS-Causal", "completed_model_seeds": 5, "required_model_seeds": 5, "raw_data_status": "sampled_pilot", "independent_labeled_attack_units": 1, "seed_ci_available": True, "holm_tests_available": False, "readiness": "pilot_only"},
        {"dataset": "E5-CADETS-Causal", "completed_model_seeds": 0, "required_model_seeds": 5, "raw_data_status": "incomplete_10_of_122_manifest_files", "independent_labeled_attack_units": 0, "seed_ci_available": False, "holm_tests_available": False, "readiness": "blocked"},
        {"dataset": "StreamSpot", "completed_model_seeds": 5, "required_model_seeds": 5, "raw_data_status": "pilot_50_edges_per_graph", "independent_labeled_attack_units": 100, "seed_ci_available": True, "holm_tests_available": False, "readiness": "pilot_only"},
        {"dataset": "Unicorn-Wget", "completed_model_seeds": 5, "required_model_seeds": 5, "raw_data_status": "pilot_200_edges_per_graph", "independent_labeled_attack_units": 50, "seed_ci_available": True, "holm_tests_available": False, "readiness": "pilot_only"},
    ]
    write_final(
        "07_statistical_readiness", "statistical_acceptance_gate_audit",
        "Acceptance gate requires full raw corpora, five paired model seeds, multiple attack units, hierarchical confidence intervals and Holm-corrected preregistered tests.",
        "No current dataset clears the complete gate; seed intervals quantify optimization instability but do not replace campaign-level replication.", rows,
    )


def main() -> None:
    publish_design()
    publish_main()
    publish_posthoc()
    ablation_rows = publish_ablation()
    publish_prediction(ablation_rows)
    publish_efficiency()
    publish_readiness()
    print(f"Published completed multi-seed experiments to {RESULTS}")


if __name__ == "__main__":
    main()
