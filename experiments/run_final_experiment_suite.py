"""Build all currently supportable final-only TIFS experiment tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    return parser.parse_args()


def run(command):
    print(subprocess.list2cmdline([str(part) for part in command]))
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def main():
    args = parse_args()
    results = Path(args.results_root)
    e3_events = ARTIFACTS / "attack_reconstruction" / "tracing" / "534b00e1ff2c0dbd42cb298c7a37da837fe7615aa37992d983b1da8a14c9d017" / "E3-CADETS-Causal" / "online_dynamic" / "model_epoch_6" / "test" / "online_trace_events.csv"
    stream_events = ARTIFACTS / "attack_reconstruction" / "tracing" / "e0a1096ef1bf6e07e37c90e9d341f080099fff3d75b88c4b0aa13ec09fd485e1" / "StreamSpot" / "online_dynamic" / "model_epoch_6" / "test" / "online_trace_events.csv"
    e3_truth = ARTIFACTS / "raw_temporal_pilot_semantic" / "E3-CADETS" / "edge_embeds" / "ground_truth_nodes.csv"
    stream_truth = ARTIFACTS / "raw_temporal_pilot" / "StreamSpot" / "edge_embeds" / "ground_truth_graphs.csv"
    required = [e3_events, stream_events, e3_truth, stream_truth]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing validated inputs:\n" + "\n".join(missing))

    run([
        args.python, "experiments/evaluate_tracing_baselines.py",
        "--events", e3_events, "--ground-truth", e3_truth,
        "--out-dir", results / "02_online_vs_posthoc", "--pilot",
    ])
    run([
        args.python, "experiments/evaluate_score_robustness.py",
        "--e3-events", e3_events, "--e3-ground-truth", e3_truth,
        "--streamspot-events", stream_events, "--streamspot-ground-truth", stream_truth,
        "--out-dir", results / "06_robustness",
    ])
    run([
        args.python, "experiments/evaluate_ablation_final.py",
        "--events", e3_events, "--ground-truth", e3_truth,
        "--out-dir", results / "03_component_ablation",
    ])
    run([
        args.python, "experiments/evaluate_efficiency_final.py",
        "--e3-events", e3_events, "--streamspot-events", stream_events,
        "--out-dir", results / "05_system_efficiency",
    ])
    run([
        args.python, "experiments/publish_tifs_final_results.py",
        "--results-root", results, "--artifacts-root", ARTIFACTS,
    ])
    expected_names = {"final_results.csv", "final_results.json"}
    expected_directories = {f"{index:02d}_{name}" for index, name in [
        (0, "experiment_design"), (1, "main_effectiveness"), (2, "online_vs_posthoc"),
        (3, "component_ablation"), (4, "prediction_containment"),
        (5, "system_efficiency"), (6, "robustness"), (7, "statistical_readiness"),
    ]}
    unexpected = []
    for directory in results.iterdir():
        if not directory.is_dir() or directory.name not in expected_directories:
            unexpected.append(str(directory))
            continue
        names = {path.name for path in directory.iterdir() if path.is_file()}
        if names != expected_names:
            unexpected.append(f"{directory}: expected {sorted(expected_names)}, got {sorted(names)}")
    if unexpected:
        raise RuntimeError("Non-final files found in results tree:\n" + "\n".join(unexpected))
    print("Final-only suite complete: 8 experiment folders, no training-process artifacts.")


if __name__ == "__main__":
    main()
