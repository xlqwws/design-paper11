"""Causally replay a frozen anomaly-score stream for downstream ablations."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attack_reconstruction.online_dynamic_tracing import (  # noqa: E402
    OnlineDynamicAttackTracer,
    load_calibration_scores,
    load_calibration_node_scores,
    resolve_online_threshold,
)
from config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from detection.evaluation_utils import calculate_threshold  # noqa: E402


VARIANTS = {
    "full": {},
    "latest_update": {"attack_reconstruction.tracing.online_dynamic.update_rule": "latest"},
    "ema_update": {"attack_reconstruction.tracing.online_dynamic.update_rule": "ema"},
    "no_context": {"attack_reconstruction.tracing.online_dynamic.context_enabled": "False"},
    "no_retained_context": {"attack_reconstruction.tracing.online_dynamic.retained_evidence_margin": "1000000000000"},
    "no_prediction": {"attack_reconstruction.tracing.online_dynamic.prediction_enabled": "False"},
    "no_cutoff": {"attack_reconstruction.tracing.online_dynamic.cutoff_enabled": "False"},
    "weighted_policy": {"attack_reconstruction.tracing.online_dynamic.decision_policy": "weighted_risk"},
    "destination_cut": {"attack_reconstruction.tracing.online_dynamic.cut_target_policy": "destination"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", default="")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--prebuilt-dir", default="")
    parser.add_argument("--feature-mode", default="only_type")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Repeatable dotted config override used to reproduce the frozen score path or sensitivity setting.",
    )
    args = parser.parse_args()

    cfg_argv = [args.dataset, "--seed", str(args.seed)]
    if args.prebuilt_dir:
        cfg_argv.extend([
            f"--edge_featurization.embed_edges.prebuilt_dir={args.prebuilt_dir}",
            f"--edge_featurization.embed_nodes.used_method={args.feature_mode}",
            f"--edge_featurization.embed_nodes.emb_dim={args.embedding_dim}",
        ])
    cfg_argv.extend(f"--{key}={value}" for key, value in VARIANTS[args.variant].items())
    for override in args.override:
        if "=" not in override:
            parser.error(f"Invalid --override value: {override}")
        cfg_argv.append(f"--{override}")
    runtime_args = get_runtime_required_args(args=cfg_argv)
    cfg = get_yml_cfg(runtime_args)

    test_root = Path(cfg.detection.gnn_testing._edge_losses_dir) / "test"
    epochs = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    epoch = args.epoch or (epochs[-1] if epochs else "")
    if not epoch:
        raise FileNotFoundError(f"No frozen test score stream found under {test_root}")
    val_dir = Path(cfg.detection.gnn_testing._edge_losses_dir) / "val" / epoch
    test_dir = test_root / epoch
    calibration_scores = load_calibration_scores(str(val_dir))
    calibration_node_scores = load_calibration_node_scores(
        str(val_dir), calibration_scores,
        cfg.attack_reconstruction.tracing.online_dynamic.node_gate_top_k,
    )
    threshold = resolve_online_threshold(
        calculate_threshold(str(val_dir)), cfg, calibration_scores=calibration_scores
    )
    tracer = OnlineDynamicAttackTracer(
        cfg=cfg, model_epoch_file=epoch, split="test", anomaly_threshold=threshold,
        calibration_scores=calibration_scores,
        calibration_node_scores=calibration_node_scores,
    )
    for window, path in enumerate(sorted(test_dir.glob("*.csv"))):
        for frame in pd.read_csv(path, chunksize=args.chunk_size):
            for event in frame.to_dict(orient="records"):
                tracer.observe_edge(event, window)
        tracer.close_time_window(window, path.stem)
    summary = tracer.flush()
    with (Path(summary.out_dir) / "replay_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"dataset": args.dataset, "seed": args.seed, "variant": args.variant,
             "feature_mode": args.feature_mode, "overrides": args.override},
            file, indent=2,
        )
    print(f"Replayed {tracer.total_events} events to {summary.out_dir}")


if __name__ == "__main__":
    main()
