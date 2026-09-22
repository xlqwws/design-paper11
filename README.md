# DynamicTrace

DynamicTrace is a causal, online provenance-graph intrusion detection and attack
tracking research prototype. It updates node evidence while events arrive,
retains the strongest bounded historical evidence, expands relation-aware local
context, predicts likely next attack behavior, and emits counterfactual cutoff
decisions without requiring a post-processing traversal of the completed graph.

## Main Contributions

- Causal dynamic graph construction from chronological provenance events.
- Monotone node-evidence updates with bounded retained context.
- Relation-aware temporal attack-subgraph tracking.
- Validation-only finite-sample alert calibration.
- Prospective behavior prediction and counterfactual containment decisions.
- Full-stream evaluation for E3-CADETS, StreamSpot, and Unicorn-Wget.

## Entry Point

```shell
python src/dynamic_trace.py E3-CADETS-Causal --seed=0
```

To reuse preprocessed graph and edge features:

```shell
python src/dynamic_trace.py E3-CADETS-Causal --run_from_training --seed=0
```

The default configuration is [config/dynamic_trace.yml](config/dynamic_trace.yml).

## Experiments

The TIFS-oriented experiment protocol is documented in
[TIFS_EXPERIMENT_PROTOCOL.md](TIFS_EXPERIMENT_PROTOCOL.md). Paper-facing output
is stored under `results/`; training traces and intermediate artifacts remain
under `artifacts/`.

Useful commands:

```shell
python experiments/run_tifs_experiments.py --help
python experiments/run_tifs_baseline_suite.py --help
python experiments/publish_proposed_v2_results.py
python -m unittest discover -s tests -v
```

## Reproducibility Scope

The public test sets have been inspected during iterative development. Results
labelled `exploratory_reused_public_holdout` must not be presented as fresh
confirmatory evidence. Baseline adaptations are identified as adaptations and
must not be described as official upstream outputs.

## License And Attribution

This repository contains substantial modifications to a third-party
provenance-IDS codebase. The Apache-2.0 license is retained in [LICENSE](LICENSE),
and upstream attribution is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Renaming project modules does
not remove those license or citation obligations.
