"""Replay the current tracer and publish strict-online latency summaries only."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from attack_reconstruction.online_dynamic_tracing import OnlineDynamicAttackTracer  # noqa: E402
from experiments.evaluate_ablation_final import make_cfg  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-events", required=True)
    parser.add_argument("--streamspot-events", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def summarize(dataset, path, tier, overrides):
    events = pd.read_csv(path)
    threshold = float(events.threshold.iloc[0])
    positive = events.loc[events.conformal_pvalue > 0, "conformal_pvalue"]
    calibration_size = max(1, round(1.0 / float(positive.min())) - 1)
    model_latency = []
    trace_latency = []
    end_to_end_latency = []
    with tempfile.TemporaryDirectory() as directory:
        tracer = OnlineDynamicAttackTracer(
            make_cfg(directory, overrides), "frozen_epoch_6", "test", threshold,
            calibration_scores=[0.0] * calibration_size,
        )
        for event in events.to_dict(orient="records"):
            decision = tracer.observe_edge(event, int(event["time_window"]))
            model_latency.append(float(decision["model_latency_us"]))
            trace_latency.append(float(decision["trace_latency_us"]))
            end_to_end_latency.append(float(decision["end_to_end_latency_us"]))
        state_nodes = tracer.graph.number_of_nodes()
        state_edges = tracer.graph.number_of_edges()
        attack_nodes = tracer.attack_graph.number_of_nodes()
        attack_edges = tracer.attack_graph.number_of_edges()
        out_of_order = tracer.out_of_order_events
        tracer.close()
    row = {
        "dataset": dataset, "model_seed": 0, "result_tier": tier,
        "events": len(events), "causal_batch_size": 1,
        "dynamic_nodes": state_nodes, "dynamic_edges": state_edges,
        "attack_nodes": attack_nodes, "attack_edges": attack_edges,
        "out_of_order_events": out_of_order,
    }
    for prefix, values in (
        ("model", model_latency), ("trace", trace_latency), ("end_to_end", end_to_end_latency)
    ):
        array = np.asarray(values, dtype=float)
        row[f"{prefix}_mean_us"] = float(np.mean(array))
        row[f"{prefix}_p50_us"] = float(np.percentile(array, 50))
        row[f"{prefix}_p95_us"] = float(np.percentile(array, 95))
        row[f"{prefix}_p99_us"] = float(np.percentile(array, 99))
        row[f"{prefix}_throughput_eps"] = float(1_000_000 / np.mean(array)) if np.mean(array) else None
    return row


def main():
    args = parse_args()
    rows = [
        summarize("E3-CADETS-Causal", args.e3_events, "pilot_sampled", {}),
        summarize(
            "StreamSpot", args.streamspot_events, "pilot_50_edges_per_graph",
            {"context_enabled": False, "prediction_enabled": False, "cutoff_enabled": False},
        ),
    ]
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "final_results.csv", index=False)
    payload = {
        "experiment": "strict_online_efficiency",
        "protocol": "Causal batch size 1; current tracer replay; recorded GPU model latency plus current tracer overhead.",
        "limitations": "Sampled pilots do not establish full-stream peak-rate sustainability or 24-hour state bounds.",
        "results": rows,
    }
    with (output / "final_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
