"""Replay downstream ablations in temporary storage and publish only final metrics."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attack_reconstruction.online_dynamic_tracing import OnlineDynamicAttackTracer  # noqa: E402


VARIANTS = {
    "full": {},
    "latest_update": {"update_rule": "latest"},
    "ema_update": {"update_rule": "ema"},
    "no_context": {"context_enabled": False},
    "no_retained_context": {"retained_evidence_margin": 1e12},
    "no_prediction": {"prediction_enabled": False},
    "no_cutoff": {"cutoff_enabled": False},
    "weighted_policy": {"decision_policy": "weighted_risk"},
    "destination_cut": {"cut_target_policy": "destination"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    return parser.parse_args()


def make_cfg(output, overrides):
    settings = {
        "threshold_method": "conformal", "calibration_alpha": 0.001,
        "context_alpha": 0.01, "cutoff_pvalue": 0.0001,
        "activation_margin": 1.0, "context_margin": 0.65,
        "retained_evidence_margin": 1.0, "retained_event_floor": 0.2,
        "context_horizon_windows": 4, "cutoff_risk_threshold": 0.78,
        "cut_override_margin": 1.5, "decision_policy": "conformal_risk",
        "update_rule": "max", "ema_alpha": 0.2, "context_enabled": True,
        "prediction_enabled": True, "cutoff_enabled": True,
        "containment_scope": "incident", "cut_target_policy": "causal_actor",
        "min_attack_support": 2, "prediction_prior_strength": 0.25,
        "min_prediction_confidence": 0.55,
        "cutoff_stages": "execution,persistence_or_lateral,c2_or_exfiltration",
        "max_logged_events": 0, "max_dynamic_edges": 0,
        "max_attack_nodes": 200000, "max_attack_edges": 300000,
        "log_all_events": True,
    }
    settings.update(overrides)
    dynamic = SimpleNamespace(**settings)
    tracing = SimpleNamespace(
        used_method="online_dynamic", online_dynamic=dynamic,
        _dynamic_tracing_dir=str(output), _tracing_graph_dir=str(output),
    )
    return SimpleNamespace(attack_reconstruction=SimpleNamespace(tracing=tracing))


def node_metrics(selected, malicious, all_nodes):
    tp = len(selected & malicious)
    fp = len(selected - malicious)
    fn = len(malicious - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "node_precision": precision, "node_recall": recall,
        "node_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "reported_nodes": len(selected), "true_reported_nodes": tp,
        "workload_reduction": 1.0 - len(selected) / len(all_nodes),
    }


def main():
    args = parse_args()
    events = pd.read_csv(args.events)
    truth = pd.read_csv(args.ground_truth)
    node_column = "node_id" if "node_id" in truth else "nid"
    label_column = next(name for name in ("label", "y_true", "is_malicious") if name in truth)
    malicious = set(truth.loc[truth[label_column].astype(int) == 1, node_column].astype(str))
    events["srcnode"] = events.srcnode.astype(str)
    events["dstnode"] = events.dstnode.astype(str)
    all_nodes = set(events.srcnode) | set(events.dstnode)
    threshold = float(events.threshold.iloc[0])
    positive_pvalues = events.loc[events.conformal_pvalue > 0, "conformal_pvalue"]
    calibration_size = max(1, round(1.0 / float(positive_pvalues.min())) - 1)
    calibration_placeholder = [0.0] * calibration_size
    rows = []
    for variant, overrides in VARIANTS.items():
        with tempfile.TemporaryDirectory() as directory:
            tracer = OnlineDynamicAttackTracer(
                make_cfg(directory, overrides), "frozen_epoch_6", "test", threshold,
                calibration_scores=calibration_placeholder,
            )
            selected = set()
            for event in events.to_dict(orient="records"):
                decision = tracer.observe_edge(event, int(event["time_window"]))
                if decision["activated"]:
                    selected.update([str(event["srcnode"]), str(event["dstnode"])])
            tracer.close()
            row = {
                "dataset": "E3-CADETS-Causal", "variant": variant, "model_seed": args.model_seed,
                "result_tier": "pilot_sampled", "same_frozen_scores": True,
                "primary_activations": tracer.activated_events,
                "context_activations": tracer.context_events,
                "retained_context_activations": tracer.retained_context_events,
                "unique_cut_nodes": len(tracer.cut_nodes),
                "would_block_events": tracer.would_block_events,
                "trace_p99_us": float(np.percentile(tracer.trace_latency_us, 99)),
                "out_of_order_events": tracer.out_of_order_events,
            }
            row.update(node_metrics(selected, malicious, all_nodes))
            rows.append(row)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "final_results.csv", index=False)
    payload = {
        "experiment": "component_ablation",
        "protocol": "Every variant causally replays the same frozen epoch-6 score stream in temporary storage.",
        "limitations": "One sampled E3 campaign and one model seed; prediction and prevention efficacy are unavailable without stage/event labels.",
        "results": rows,
    }
    with (output / "final_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
