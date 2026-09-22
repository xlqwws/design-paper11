"""Publish the improved full-stream and online-node experiments as final-only tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "tifs_results"
RESULTS = ROOT / "results"
T95_DF4 = 2.7764451051977987


def binomial_exact_interval(successes: int, trials: int, confidence: float = 0.95):
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - tail, successes + 1, trials - successes))
    return low, high


def write_final(folder: str, experiment: str, protocol: str, limitations: str, rows) -> None:
    target = RESULTS / folder
    target.mkdir(parents=True, exist_ok=True)
    for path in target.iterdir():
        if path.name not in {"final_results.csv", "final_results.json"}:
            raise RuntimeError(f"Non-final artifact in {target}: {path.name}")
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "final_results.csv", index=False)
    payload = {
        "experiment": experiment, "protocol": protocol, "limitations": limitations,
        "results": json.loads(frame.to_json(orient="records", double_precision=15)),
    }
    (target / "final_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def mean_ci_rows(frame: pd.DataFrame, method: str, repetition: str):
    rows = []
    metrics = ["precision", "recall", "f1", "mcc", "auroc", "average_precision", "reported_nodes"]
    for statistic in ["mean", "std", "ci95_low", "ci95_high"]:
        row = {"dataset": "E3-CADETS-Causal", "method": method, "row_type": "aggregate",
               "statistic": statistic, "repetition_unit": repetition, "n_repetitions": len(frame)}
        for metric in metrics:
            if metric not in frame:
                continue
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
            mean = values.mean()
            std = values.std(ddof=1)
            half = T95_DF4 * std / np.sqrt(len(values))
            low, high = mean - half, mean + half
            if metric != "reported_nodes":
                low, high = max(0.0, low), min(1.0, high)
            row[metric] = {"mean": mean, "std": std, "ci95_low": low,
                           "ci95_high": high}[statistic]
        rows.append(row)
    return rows


def publish_design():
    rows = [
        {"id": "E1", "folder": "01_main_effectiveness", "question": "Full-stream detection and node attribution", "status": "completed_except_E5"},
        {"id": "E2", "folder": "02_online_vs_posthoc", "question": "Budget-matched online versus post-hoc tracing", "status": "completed_E3_pilot"},
        {"id": "E3", "folder": "03_component_ablation", "question": "Semantic, structure and size contribution", "status": "completed_three_datasets"},
        {"id": "E4", "folder": "04_prediction_containment", "question": "Prospective prediction and cut audit", "status": "decision_audit_only"},
        {"id": "E5", "folder": "05_system_efficiency", "question": "GPU online latency and full-stream sketch throughput", "status": "completed_three_datasets"},
        {"id": "E6", "folder": "06_robustness", "question": "Missing events and score perturbation", "status": "pilot_only"},
        {"id": "E7", "folder": "07_statistical_readiness", "question": "Evidence and acceptance-gate audit", "status": "partial"},
        {"id": "E8", "folder": "08_early_detection", "question": "Fixed-prefix attack detection", "status": "completed_StreamSpot"},
        {"id": "E9", "folder": "09_campaign_detection", "question": "Independent attack-campaign detection", "status": "completed_E3"},
        {"id": "E10", "folder": "10_cross_campaign_transfer", "question": "Chronological supervised transfer with monotone graph tracking", "status": "completed_E3"},
        {"id": "E11", "folder": "11_f1_target_audit", "question": "Per-dataset confirmatory F1 >= 0.95 gate", "status": "one_of_four_passed"},
        {"id": "E12", "folder": "12_temporal_localized_tracking", "question": "Temporal-localized monotone subgraph tracking", "status": "completed_E3_exploratory"},
        {"id": "E13", "folder": "13_unicorn_temporal_transfer", "question": "Known-family early-to-later attack transfer", "status": "completed_Unicorn_exploratory"},
        {"id": "E14", "folder": "14_e3_multihop_context_gate", "question": "Relation-path multi-hop candidate filtering", "status": "completed_E3_exploratory"},
        {"id": "E15", "folder": "15_e5_local_coverage", "question": "Local E5 attack-period sufficiency gate", "status": "blocked_missing_attack_period"},
        {"id": "E16", "folder": "16_e3_gpu_trajectory_ablation", "question": "GPU relation-GNN temporal-trajectory ablation", "status": "completed_E3_exploratory"},
        {"id": "E17", "folder": "17_e5_local_benign_stability", "question": "Causal adaptation under local E5 benign drift", "status": "completed_incomplete_raw_diagnostic"},
        {"id": "E18", "folder": "18_e3_cross_day_generalization", "question": "Cross-day calibration, propagation and continual-learning comparison", "status": "completed_E3_exploratory"},
        {"id": "E19", "folder": "19_e3_role_error_analysis", "question": "Entity-role attribution error decomposition", "status": "completed_posthoc_diagnostic"},
    ]
    write_final("00_experiment_design", "improved_tifs_experiment_matrix",
                "Chronological or scenario-stratified splits; train-only fitting; validation-only calibration; immutable online test decisions.",
                "A strong experimental package improves review readiness but cannot guarantee acceptance.", rows)


def e3_neural_rows():
    rows = []
    for seed in range(5):
        metric = pd.read_csv(ARTIFACTS / "E3-CADETS-Causal" / f"integrated_node_gate_seed{seed}" / "metrics.csv").iloc[0]
        rows.append({
            "dataset": "E3-CADETS-Causal", "method": "online_GNN_node_gate",
            "row_type": "repetition", "statistic": "observed", "repetition_unit": "training_seed",
            "seed": seed, "n_repetitions": 1, "result_tier": "sampled_pilot",
            "evaluation_unit": "priority_node", "precision": metric["node_attribution.precision"],
            "recall": metric["node_attribution.recall"], "f1": metric["node_attribution.f1"],
            "reported_nodes": metric["node_attribution.reported_nodes"],
            "context_nodes": metric["context_node_coverage.context_nodes"],
            "context_recall": metric["context_node_coverage.recall"],
            "context_workload_reduction": metric["context_node_coverage.workload_reduction"],
        })
    return rows


def publish_main():
    rows = e3_neural_rows()
    rows.extend(mean_ci_rows(pd.DataFrame(rows), "online_GNN_node_gate", "training_seed"))
    e3_full = pd.read_csv(ARTIFACTS / "E3-CADETS-Causal" / "full_online_node_sketch_seed0" / "metrics.csv")
    e3_decisions = pd.read_csv(
        ARTIFACTS / "E3-CADETS-Causal" / "full_online_node_sketch_seed0" / "node_decisions.csv"
    )
    for record in e3_full.to_dict(orient="records"):
        score_column = f"{record['variant']}_score"
        record["average_precision"] = average_precision_score(
            e3_decisions["label"], e3_decisions[score_column]
        )
        record["analyst_workload_reduction"] = 1.0 - (
            float(record["reported_nodes"]) / float(record["test_nodes"])
        )
        if record["variant"] == "normalized_structure_only":
            record["campaign_detection_rate"] = 1.0
            record["detected_campaigns"] = 3
            record["campaigns"] = 3
        rows.append({**record, "method": "online_full_stream_node_sketch",
                     "row_type": "full_stream", "statistic": "observed",
                     "result_tier": "publication_full", "evaluation_unit": "node"})
    ensemble = pd.read_csv(ARTIFACTS / "E3-CADETS-Causal" / "robust_seed_ensemble" / "metrics.csv")
    for record in ensemble.to_dict(orient="records"):
        rows.append({**record, "row_type": "ensemble", "statistic": "observed",
                     "result_tier": "sampled_pilot", "evaluation_unit": "node"})
    for dataset in ["StreamSpot", "Unicorn-Wget"]:
        root = ARTIFACTS / dataset / "full_sketch_repeated_splits"
        repeats = pd.read_csv(root / "split_repetitions.csv")
        aggregate = pd.read_csv(root / "aggregate.csv")
        repeats["method"] = "online_full_stream_graph_sketch"
        repeats["result_tier"] = "publication_full"
        repeats["evaluation_unit"] = "graph"
        aggregate["dataset"] = dataset
        aggregate["method"] = "online_full_stream_graph_sketch"
        aggregate["result_tier"] = "publication_full"
        aggregate["evaluation_unit"] = "graph"
        rows.extend(repeats.to_dict(orient="records"))
        rows.extend(aggregate.to_dict(orient="records"))
    rows.append({"dataset": "E5-CADETS-Causal", "row_type": "status", "statistic": "not_computed",
                 "result_tier": "blocked", "reason": "112 of 122 manifest shards are missing locally, including attack-period data"})
    write_final("01_main_effectiveness", "improved_full_stream_effectiveness",
                "E3 decisions are chronological; graph datasets use all raw edges with attack graphs held out and benign-only calibration.",
                "E3 full-stream node F1 remains low under extreme class imbalance; StreamSpot has a strong graph-size shortcut; E5 is blocked by incomplete raw data.", rows)


def publish_ablation():
    rows = []
    for dataset, path in [
        ("E3-CADETS-Causal", ARTIFACTS / "E3-CADETS-Causal" / "full_online_node_sketch_seed0" / "metrics.csv"),
        ("StreamSpot", ARTIFACTS / "StreamSpot" / "full_streaming_sketch_seed0" / "metrics.csv"),
        ("Unicorn-Wget", ARTIFACTS / "Unicorn-Wget" / "full_streaming_sketch_seed0" / "metrics.csv"),
    ]:
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            record.update({"dataset": dataset, "row_type": "full_stream_ablation", "statistic": "observed"})
            rows.append(record)
    write_final("03_component_ablation", "full_stream_dual_head_ablation",
                "Identical full raw streams, splits and calibration are used for dual-head, normalized-structure-only and size-only variants.",
                "StreamSpot is separable by size alone; Unicorn-Wget and E3 provide clearer evidence for structural contribution.", rows)


def publish_prediction():
    rows = []
    for seed in range(5):
        metric = pd.read_csv(ARTIFACTS / "E3-CADETS-Causal" / f"integrated_node_gate_seed{seed}" / "metrics.csv").iloc[0]
        rows.append({"dataset": "E3-CADETS-Causal", "seed": seed, "row_type": "training_seed",
                     "status": "counterfactual_decision_audit", "prediction_accuracy_available": False,
                     "prevention_efficacy_available": False,
                     "unique_cut_nodes": metric.get("containment.unique_cut_nodes"),
                     "would_block_events": metric.get("containment.counterfactual_blocked_events"),
                     "priority_nodes": metric["node_attribution.reported_nodes"],
                     "context_recall": metric["context_node_coverage.recall"]})
    write_final("04_prediction_containment", "online_prediction_and_cut_audit",
                "Prediction and cut decisions are emitted during stream processing; later events are retained only as would-block counterfactuals.",
                "Available node labels cannot establish next-stage prediction accuracy or causal prevention efficacy.", rows)


def publish_efficiency():
    rows = []
    for seed in range(5):
        metric = pd.read_csv(ARTIFACTS / "E3-CADETS-Causal" / f"integrated_node_gate_seed{seed}" / "metrics.csv").iloc[0]
        rows.append({"dataset": "E3-CADETS-Causal", "seed": seed, "mode": "GPU_GNN_plus_node_gate",
                     "events": metric["efficiency.events"], "trace_mean_us": metric["efficiency.latency_us_mean"],
                     "trace_p95_us": metric["efficiency.latency_us_p95"], "trace_p99_us": metric["efficiency.latency_us_p99"],
                     "trace_throughput_eps": metric["efficiency.tracing_throughput_events_per_second"],
                     "result_tier": "sampled_pilot"})
    for dataset, path in [
        ("StreamSpot", ARTIFACTS / "StreamSpot" / "full_streaming_sketch_seed0" / "metrics.csv"),
        ("Unicorn-Wget", ARTIFACTS / "Unicorn-Wget" / "full_streaming_sketch_seed0" / "metrics.csv"),
    ]:
        metric = pd.read_csv(path).iloc[0]
        seconds = metric["scan_seconds"]
        rows.append({"dataset": dataset, "seed": 0, "mode": "bounded_full_stream_sketch",
                     "events": metric["raw_edges"], "scan_seconds": seconds,
                     "throughput_edges_per_second": metric["raw_edges"] / seconds,
                     "bounded_state_bytes": metric.get("peak_feature_state_bytes", metric.get("state_bytes")),
                     "result_tier": "publication_full"})
    write_final("05_system_efficiency", "online_and_full_stream_efficiency",
                "GNN tracing uses causal batch size one on GPU; sketch throughput scans every raw edge with bounded count state.",
                "Offline archive decompression throughput is not equivalent to deployed end-to-end sensor throughput.", rows)


def publish_readiness():
    rows = [
        {"dataset": "E3-CADETS-Causal", "raw_data": "full_44.4M_records", "training_seeds": 5,
         "split_repetitions": 0, "attack_nodes": 72, "readiness": "partial_full",
         "remaining_gap": "low full-stream node AP/F1 and no event-stage labels"},
        {"dataset": "E5-CADETS-Causal", "raw_data": "incomplete_10_of_122_shards", "training_seeds": 0,
         "split_repetitions": 0, "readiness": "blocked", "remaining_gap": "obtain complete corpus"},
        {"dataset": "StreamSpot", "raw_data": "full_89.77M_edges", "training_seeds": 0,
         "split_repetitions": 5, "attack_graphs": 100, "readiness": "full_data_repeated_split",
         "remaining_gap": "size shortcut and no optimization-seed variance"},
        {"dataset": "Unicorn-Wget", "raw_data": "full_26.75M_edges", "training_seeds": 0,
         "split_repetitions": 5, "attack_graphs": 50, "readiness": "full_data_repeated_split",
         "remaining_gap": "add external temporal-shift validation"},
    ]
    write_final("07_statistical_readiness", "updated_acceptance_gate_audit",
                "Evidence tiers distinguish full raw streams, optimization seeds and benign split repetitions.",
                "The package is materially stronger but does not yet satisfy every strong-accept gate.", rows)


def publish_robustness():
    rows = []
    for dataset in ["StreamSpot", "Unicorn-Wget"]:
        root = ARTIFACTS / dataset / "full_structure_dropout"
        repetitions = pd.read_csv(root / "repetitions.csv")
        summary = pd.read_csv(root / "summary.csv")
        repetitions["row_type"] = "perturbation_repetition"
        summary["row_type"] = "aggregate"
        rows.extend(repetitions.to_dict(orient="records"))
        rows.extend(summary.to_dict(orient="records"))
    write_final("06_robustness", "full_stream_telemetry_dropout",
                "The clean benign train/validation model is frozen; test count marginals are thinned without using graph labels; five perturbation seeds are used for nonzero rates.",
                "Independent marginal thinning approximates missing events but does not preserve exact cross-feature covariance.", rows)


def publish_early_detection():
    rows = pd.read_csv(ARTIFACTS / "StreamSpot" / "full_early_detection_seed0" / "metrics.csv").to_dict(orient="records")
    write_final("08_early_detection", "fixed_prefix_early_detection",
                "Every graph is truncated to the same first-B-event budget before train fitting, validation calibration and immutable test prediction.",
                "StreamSpot attacks become reliably separable near 25k events; results do not support a claim at 1k or 5k events.", rows)


def publish_campaign_detection():
    root = ARTIFACTS / "E3-CADETS-Causal" / "full_campaign_detection"
    rows = pd.read_csv(root / "per_campaign.csv").to_dict(orient="records")
    summary = pd.read_csv(root / "summary.csv").iloc[0].to_dict()
    rows.append({**summary, "row_type": "aggregate"})
    write_final("09_campaign_detection", "e3_full_campaign_detection",
                "The frozen full-stream node decisions are joined afterward to the three separate Nginx Backdoor ground-truth files.",
                "Three campaigns are too few for a narrow confidence interval, and campaign detection does not imply complete malicious-node recovery.", rows)


def publish_cross_campaign_transfer():
    root = ARTIFACTS / "E3-CADETS-Causal" / "cross_campaign_transfer"
    rows = pd.read_csv(root / "metrics.csv").to_dict(orient="records")
    write_final(
        "10_cross_campaign_transfer",
        "e3_chronological_cross_campaign_transfer",
        "April 6 labels train the model; April 12 selects model, propagation and threshold; April 13 labels are loaded only after decisions are frozen.",
        "This is a known-family supervised-transfer setting with one campaign per split. It does not establish zero-day performance, and repeated development has consumed April 13 as an untouched holdout for future changes.",
        rows,
    )


def publish_f1_target_audit():
    stream = pd.read_csv(
        ARTIFACTS / "StreamSpot" / "full_sketch_repeated_splits" / "aggregate.csv"
    ).query("variant == 'normalized_structure_only'").iloc[0]
    unicorn = pd.read_csv(
        ARTIFACTS / "Unicorn-Wget" / "full_sketch_repeated_splits" / "aggregate.csv"
    ).query("variant == 'normalized_structure_only'").iloc[0]
    e3 = pd.read_csv(
        ARTIFACTS / "E3-CADETS-Causal" / "cross_campaign_transfer" / "metrics.csv"
    ).iloc[0]
    e3_exploratory = pd.read_csv(
        ARTIFACTS / "E3-CADETS-Causal" / "multihop_context_gate_monotone_localized" / "metrics.csv"
    ).iloc[0]
    unicorn_transfer = pd.read_csv(
        ARTIFACTS / "Unicorn-Wget" / "temporal_supervised_transfer" / "aggregate.csv"
    ).iloc[0]
    rows = [
        {"dataset": "StreamSpot", "evaluation_unit": "graph", "protocol": "five scenario-stratified repeated splits", "f1": float(stream["f1_mean"]), "target": 0.95, "passed": bool(stream["f1_mean"] >= 0.95), "status": "computed"},
        {"dataset": "Unicorn-Wget", "evaluation_unit": "graph", "protocol": "five benign-split repetitions; benign-only calibration", "f1": float(unicorn["f1_mean"]), "exploratory_known_family_f1": float(unicorn_transfer["f1_mean"]), "target": 0.95, "passed": bool(unicorn["f1_mean"] >= 0.95), "status": "computed; known-family transfer holdout reused"},
        {"dataset": "E3-CADETS-Causal", "evaluation_unit": "node", "protocol": "April 6 train / April 12 validation / April 13 test", "f1": float(e3["f1"]), "exploratory_best_f1": float(e3_exploratory["f1"]), "target": 0.95, "passed": bool(e3["f1"] >= 0.95), "status": "computed_known_family_transfer; exploratory holdout reused"},
        {"dataset": "E5-CADETS-Causal", "evaluation_unit": "node", "protocol": "strict chronological", "f1": None, "target": 0.95, "passed": False, "status": "blocked_missing_112_of_122_shards"},
    ]
    write_final(
        "11_f1_target_audit",
        "confirmatory_per_dataset_f1_gate",
        "A dataset passes only when its declared, leakage-controlled confirmatory F1 is at least 0.95; unavailable results fail the readiness gate.",
        "F1 values use different natural units across graph and provenance datasets and must not be pooled. The 0.95 target is an engineering goal, not a valid basis for test-set tuning.",
        rows,
    )


def publish_temporal_localized_tracking():
    rows = []
    for feature_name, folder in [
        ("latest", "temporal_localized_tracking"),
        ("pooled", "temporal_localized_tracking_pooled"),
        ("all_views", "temporal_localized_tracking_all_views"),
    ]:
        frame = pd.read_csv(ARTIFACTS / "E3-CADETS-Causal" / folder / "metrics.csv")
        for record in frame.to_dict(orient="records"):
            record["semantic_view"] = feature_name
            rows.append(record)
    write_final(
        "12_temporal_localized_tracking",
        "e3_temporal_localized_monotone_tracking",
        "April 6 trains the seed detector; April 12 selects seed threshold, number of high-evidence windows, degree cap and relation normalization; April 13 decisions are frozen before label join.",
        "April 13 was already consumed by preceding development and is therefore exploratory, not a new confirmatory holdout. Pooled and all-view equality is resolved in favor of the lower-dimensional pooled representation.",
        rows,
    )


def publish_unicorn_temporal_transfer():
    root = ARTIFACTS / "Unicorn-Wget" / "temporal_supervised_transfer"
    repetitions = pd.read_csv(root / "split_repetitions.csv")
    repetitions["row_type"] = "repetition"
    aggregate = pd.read_csv(root / "aggregate.csv")
    rows = repetitions.to_dict(orient="records") + aggregate.to_dict(orient="records")
    write_final(
        "13_unicorn_temporal_transfer",
        "unicorn_known_family_temporal_transfer",
        "Early attack graphs 125-149 are partitioned into train/validation; later attack graphs 150-174 remain fixed test graphs; model and threshold use validation only.",
        "This is known-family supervised transfer, not zero-day detection. The later attack graphs were included in earlier aggregate evaluations, so this result is exploratory and needs a new external holdout for confirmation.",
        rows,
    )


def publish_e3_multihop_context_gate():
    root = ARTIFACTS / "E3-CADETS-Causal" / "multihop_context_gate_monotone_localized"
    rows = pd.read_csv(root / "metrics.csv").to_dict(orient="records")
    write_final(
        "14_e3_multihop_context_gate",
        "e3_relation_path_multihop_context_gate",
        "A day-6 pooled-semantic seed detector generates candidates; day-12 labels train and cross-validate a four-hop relation-path gate; day-13 decisions are frozen before label join.",
        "The day-13 holdout was consumed by prior development, so this is exploratory. Random node-level cross-validation within one day may understate graph dependence and requires campaign-blocked external confirmation.",
        rows,
    )


def publish_e5_local_coverage():
    manifest = Path(r"D:\download\E5\bins.md5sum")
    data_dir = manifest.parent
    expected = [line.split()[-1] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = [name for name in expected if (data_dir / name).exists()]
    rows = [{
        "dataset": "E5-CADETS-Causal", "expected_shards": len(expected),
        "available_shards": len(available), "missing_shards": len(expected) - len(available),
        "shard_coverage": len(available) / len(expected),
        "available_protocol_days": "8", "required_attack_days": "16,17",
        "attack_period_available": False, "f1_available": False,
        "status": "blocked_by_missing_attack_period_under_no_download_constraint",
    }]
    write_final(
        "15_e5_local_coverage",
        "e5_local_attack_period_sufficiency_gate",
        "The official 122-entry checksum manifest is joined to local files; an effectiveness score is permitted only when attack-period shards are present.",
        "Only the first ten shards are local and they contain day 8, while E5 attacks occur on days 16 and 17. Computing F1 from this subset would be invalid.",
        rows,
    )


def publish_e3_gpu_trajectory_ablation():
    rows = []
    for model_family, device, feature_view, folder in [
        ("ExtraTrees", "cpu", "static_relation_path", "multihop_context_gate_monotone_localized"),
        ("ExtraTrees", "cpu", "temporal_trajectory", "multihop_temporal_trajectory_gate"),
        ("RGCN", "cuda", "static_relation_path", "rgcn_context_gate"),
        ("RGCN", "cuda", "temporal_trajectory", "rgcn_temporal_trajectory_gate"),
    ]:
        metric = pd.read_csv(
            ARTIFACTS / "E3-CADETS-Causal" / folder / "metrics.csv"
        ).iloc[0].to_dict()
        rows.append({
            **metric, "model_family": model_family, "execution_device": device,
            "feature_view": feature_view,
            "row_type": "trajectory_ablation", "statistic": "observed",
        })
    write_final(
        "16_e3_gpu_trajectory_ablation",
        "e3_gpu_rgcn_temporal_trajectory_ablation",
        "All variants use the same day-6 seed model, day-12 out-of-fold threshold selection, induced relation graph, and frozen day-13 evaluation; RGCN training executes on CUDA.",
        "Day 13 is a reused exploratory holdout. Node-wise folds may understate graph dependence; trajectory features improve validation F1 and RGCN test AP but do not exceed the static ExtraTrees gate in test F1.",
        rows,
    )


def publish_e5_local_benign_stability():
    metric = pd.read_csv(
        ARTIFACTS / "E5-CADETS-Causal" / "local_day8_benign_stability" / "metrics.csv"
    ).iloc[0]
    rows = []
    policies = [
        ("frozen", "frozen"),
        ("causal_rolling_model", "rolling"),
        ("causal_rolling_model_and_calibration", "adaptive"),
    ]
    for policy, prefix in policies:
        for alpha_key, alpha in [("0_05", 0.05), ("0_1", 0.10)]:
            false_alerts = int(metric[f"{prefix}_late_false_alerts_alpha_{alpha_key}"])
            late_windows = int(metric["late_holdout_windows"])
            ci_low, ci_high = binomial_exact_interval(false_alerts, late_windows)
            rows.append({
                "dataset": "E5-CADETS-Causal",
                "policy": policy,
                "nominal_alpha": alpha,
                "late_false_alerts": false_alerts,
                "late_windows": late_windows,
                "late_false_alert_rate": float(metric[f"{prefix}_late_false_alert_rate_alpha_{alpha_key}"]),
                "late_false_alert_rate_ci95_low": ci_low,
                "late_false_alert_rate_ci95_high": ci_high,
                "attack_f1": None,
                "attack_period_available": False,
                "result_tier": "incomplete_raw_diagnostic",
                "supported_day8_events": int(metric["fused_edges"]),
                "raw_scan_throughput_events_per_second": float(metric["raw_scan_throughput_events_per_second"]),
                "score_ks_statistic": (
                    float(metric["frozen_score_ks_statistic"]) if prefix == "frozen"
                    else float(metric["rolling_score_ks_statistic"])
                ),
            })
    write_final(
        "17_e5_local_benign_stability",
        "e5_local_day8_causal_benign_stability",
        "All 31,121,179 supported events in the ten local day-8 shards are aggregated into 15-minute windows; early/middle/late windows are split 40/40/20 and every adaptive decision uses only prior windows.",
        "Only 10 of 122 shards are local and E5 attack days 16/17 are absent, so no attack F1 is available. The late holdout has only 12 windows and therefore wide exact confidence intervals. Updating from all unlabeled prior windows can absorb a real attack and needs an attack-aware update guard when complete data become available.",
        rows,
    )


def publish_e3_cross_day_generalization():
    configurations = [
        ("static_extra_trees", "multihop_context_gate_monotone_localized"),
        ("gpu_rgcn_trajectory", "rgcn_temporal_trajectory_gate"),
        ("rank_ensemble", "rank_ensemble_gate"),
        ("relation_directed_propagation", "monotone_graph_propagation"),
        ("continual_static", "continual_campaign_gate"),
        ("continual_trajectory", "continual_campaign_trajectory_gate"),
        ("role_conditioned_static", "role_conditioned_gate"),
        ("role_conditioned_trajectory", "role_conditioned_trajectory_gate"),
    ]
    rows = []
    for approach, folder in configurations:
        metric = pd.read_csv(
            ARTIFACTS / "E3-CADETS-Causal" / folder / "metrics.csv"
        ).iloc[0].to_dict()
        rows.append({
            "dataset": "E3-CADETS-Causal", "approach": approach,
            "precision": metric.get("precision"), "recall": metric.get("recall"),
            "f1": metric.get("f1"), "mcc": metric.get("mcc"),
            "auroc": metric.get("auroc"),
            "average_precision": metric.get("average_precision"),
            "reported_nodes": metric.get("reported_nodes"),
            "true_reported_nodes": metric.get("true_reported_nodes"),
            "false_reported_nodes": metric.get("false_reported_nodes"),
            "validation_f1": metric.get("gate_oof_f1", metric.get("validation_f1")),
            "result_tier": metric.get("result_tier"),
            "selected_variant": metric.get(
                "selected_candidate", metric.get("selected_model", metric.get("selected_gate"))
            ),
        })
    write_final(
        "18_e3_cross_day_generalization",
        "e3_cross_day_generalization_comparison",
        "All approaches use day 6 as prior training, day 12 for model/threshold selection, and frozen day-13 decisions. Rank ensemble, relation-directed monotone propagation, and prior-campaign continual gates are selected without day-13 labels.",
        "Day 13 has been reused across exploratory development and is not a fresh confirmatory holdout. Continual static training improves AP while preserving the best F1, but none of the tested approaches raises test F1 above 0.6111; this negative result is retained to expose cross-day generalization failure.",
        rows,
    )


def publish_e3_role_error_analysis():
    root = ARTIFACTS / "E3-CADETS-Causal"
    role_reference = pd.read_csv(
        root / "role_conditioned_gate" / "test_candidate_decisions.csv"
    )[["node_id", "role"]]
    sources = [
        (
            "static_extra_trees",
            pd.read_csv(
                root / "multihop_context_gate_monotone_localized"
                / "test_candidate_scores.csv"
            ).merge(role_reference, on="node_id", how="left"),
        ),
        (
            "role_conditioned_static",
            pd.read_csv(root / "role_conditioned_gate" / "test_candidate_decisions.csv"),
        ),
        (
            "role_conditioned_trajectory",
            pd.read_csv(
                root / "role_conditioned_trajectory_gate"
                / "test_candidate_decisions.csv"
            ),
        ),
    ]
    rows = []
    for method, frame in sources:
        for role in ["subject", "file", "netflow"]:
            subset = frame[frame["role"] == role]
            labels = subset["label"].astype(bool).to_numpy()
            predicted = subset["predicted"].astype(bool).to_numpy()
            tp = int((labels & predicted).sum())
            fp = int((~labels & predicted).sum())
            fn = int((labels & ~predicted).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            rows.append({
                "dataset": "E3-CADETS-Causal", "method": method,
                "entity_role": role, "true_positives": tp,
                "false_positives": fp, "false_negatives": fn,
                "malicious_nodes": tp + fn, "precision": precision,
                "recall": recall, "f1": f1,
                "result_tier": "posthoc_reused_holdout_error_analysis",
            })
    write_final(
        "19_e3_role_error_analysis",
        "e3_entity_role_error_decomposition",
        "Frozen day-13 candidate decisions are decomposed by the label-free entity type encoded in the dynamic graph features; no role result is used to retune a model.",
        "This is post-hoc analysis on a repeatedly used exploratory holdout. It identifies subject/file recall as the dominant failure mode but is not additional confirmatory evidence.",
        rows,
    )


def main():
    publish_design()
    publish_main()
    publish_ablation()
    publish_prediction()
    publish_efficiency()
    publish_robustness()
    publish_readiness()
    publish_early_detection()
    publish_campaign_detection()
    publish_cross_campaign_transfer()
    publish_f1_target_audit()
    publish_temporal_localized_tracking()
    publish_unicorn_temporal_transfer()
    publish_e3_multihop_context_gate()
    publish_e5_local_coverage()
    publish_e3_gpu_trajectory_ablation()
    publish_e5_local_benign_stability()
    publish_e3_cross_day_generalization()
    publish_e3_role_error_analysis()
    print(f"Published improved final-only results to {RESULTS}")


if __name__ == "__main__":
    main()
