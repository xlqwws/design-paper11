"""Generate or execute the preregistered main/ablation/latency experiment matrix."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


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


def command_for(python, dataset, seed, variant, overrides, batch_size, preprocessed, prebuilt_dir):
    if variant != "full":
        command = [
            python, "experiments/replay_online_tracing.py", dataset,
            f"--seed={seed}", f"--variant={variant}",
        ]
        if prebuilt_dir:
            command.append(f"--prebuilt-dir={prebuilt_dir}")
        return command
    command = [
        python, "src/dynamic_trace.py", dataset, f"--seed={seed}",
        f"--exp=tifs_{variant}_seed{seed}",
        f"--detection.gnn_testing.causal_batch_size={batch_size}",
    ]
    if preprocessed or prebuilt_dir:
        command.append("--run_from_training")
    if prebuilt_dir:
        command.extend([
            f"--edge_featurization.embed_edges.prebuilt_dir={prebuilt_dir}",
            "--edge_featurization.embed_nodes.used_method=only_type",
            "--skip_offline_evaluation",
        ])
    command.extend(f"--{key}={value}" for key, value in overrides.items())
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["E3-CADETS", "E5-CADETS", "StreamSpot", "Unicorn-Wget"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-dir", default="artifacts/tifs_experiment_manifest")
    parser.add_argument("--preprocessed", action="store_true", help="Skip graph construction and featurization")
    parser.add_argument("--prebuilt", nargs="*", default=[], help="DATASET=TemporalData_directory entries")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    prebuilt = {}
    for item in args.prebuilt:
        if "=" not in item:
            parser.error(f"Invalid --prebuilt entry: {item}")
        dataset, path = item.split("=", 1)
        prebuilt[dataset] = path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                for batch_size in args.batch_sizes:
                    if variant != "full" and batch_size != args.batch_sizes[0]:
                        continue
                    prebuilt_dir = prebuilt.get(dataset, "")
                    command = command_for(
                        args.python, dataset, seed, variant, VARIANTS[variant], batch_size,
                        args.preprocessed, prebuilt_dir,
                    )
                    record = {"dataset": dataset, "seed": seed, "variant": variant,
                              "causal_batch_size": batch_size, "command": subprocess.list2cmdline(command),
                              "status": "planned"}
                    if dataset in {"StreamSpot", "Unicorn-Wget"} and not prebuilt_dir:
                        record["status"] = "requires_prebuilt_temporal_data"
                        if args.execute:
                            raise ValueError(f"{dataset} requires --prebuilt {dataset}=<directory>")
                    if args.execute:
                        completed = subprocess.run(command, check=False)
                        record["return_code"] = completed.returncode
                        record["status"] = "completed" if completed.returncode == 0 else "failed"
                    records.append(record)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} experiment records to {out_dir}")


if __name__ == "__main__":
    main()
