# TIFS Experiment Status

Updated: 2026-09-15

## Evidence Policy

- Only outputs derived from raw audit logs are eligible for paper tables.
- Pilot, sampled, synthetic, or label-contaminated resources are never promoted to main results.
- E3/E5 node labels support entity attribution metrics. They do not support unbiased event/path prevention metrics.
- StreamSpot and Unicorn-Wget provide graph/session labels; node/path claims require a separately released annotation.
- All main results use chronological train/calibration/test streams, five fixed seeds, and causal batch size 1.

## Resource Audit

The prebuilt `native_full_training_resource_20260701` and
`native_cadets_streamspot_resource_20260701` resources are rejected for main-paper use. Their generator shuffles rows and derives `edge_type` from the attack label, which leaks ground truth into model inputs.

Raw-data audit findings:

| Dataset | Raw source available | Label granularity | Current status |
|---|---:|---|---|
| E3-CADETS | Yes | Node UUID | Leakage-free pilot completed |
| E5-CADETS | Yes | Node UUID | Streaming CDM20 Avro builder smoke-tested across all 10 shards |
| StreamSpot | Yes, 89,770,902 edges | Graph | Leakage-free reservoir pilot and GPU run completed |
| Unicorn-Wget | Yes | Session/graph | Full stream ingestion still required |

## Completed GPU Pilot

The completed E3 pilot uses raw CDM JSON, Eastern-time chronological splits,
15-minute causal edge fusion, days 2-4 for training, day 5 for calibration, and
the day-6 attack window for test. A cap of 10,000 non-attack supported events
per day is active, so these numbers are diagnostic and must not appear in the
main comparison table.

Data audit:

- Train: 10,122 fused events.
- Calibration: 3,310 fused events.
- Test: 19,129 fused events and 13 labelled malicious entities.
- Event ordering violations: 0.
- Ground truth used in model features: no.

GPU run (`RTX 3060 Laptop`, seed 0, six epochs, causal batch size 1):

| Quantity | Result |
|---|---:|
| Final training loss | 2.2591 |
| Peak training CUDA memory | 1.27 GB |
| Training time | 4.28 s |
| Online validation plus test time | 115.62 s |
| Model throughput | 206.0 events/s |
| End-to-end p99 latency | 6.15 ms/event |
| Tracing-only p99 latency | 0.11 ms/event |
| Malicious-node recall | 0.5385 |
| Malicious-node precision | 0.0258 |
| Analyst workload reduction | 0.8521 |
| Cut decisions | 30 |
| Subsequent events touching cut nodes | 1,054 |

The semantic representation improves precision over the type-only pilot
(0.0258 versus 0.0182) at the same 0.5385 recall, but absolute precision is
not publication-ready. This is a negative diagnostic result, not evidence of
TIFS-level effectiveness.

## Completed Ablations

Frozen-score causal replay is complete for `max`, `latest`, `EMA`, no context,
no prediction, no cutoff, weighted-risk policy, and destination-only cutoff.
With only one campaign and one seed, `max/latest/EMA` have identical endpoint
metrics; no statistical superiority claim is permitted. The no-cutoff variant
produces zero cut decisions, weighted risk produces 5, and the finite-sample
conformal policy produces 30.

## Required Main Experiments

1. Build full, unsampled E3/E5 chronological streams and cover every documented campaign.
2. Implement scalable StreamSpot ingestion without dense one-hot materialization, and evaluate graph detection only.
3. Implement Unicorn-Wget session-stream ingestion and evaluate session detection, transfer, and throughput only.
4. Run five seeds for the full method, all ablations, AttributionGNN post-processing, KAIROS, MAGIC, ThreaTrace, ProvFusion, and ProvX under matched splits.
5. Report AUROC/AUPRC/MCC/F1, node precision/recall, time-to-detect, workload, p50/p95/p99 latency, throughput, RAM/VRAM, and state growth.
6. Add expert event/stage annotations before claiming path F1, next-stage accuracy, attack-event prevention, or benign collateral rate.
7. Use paired seed-level bootstrap intervals, exact paired tests, and Holm correction across primary comparisons.

No acceptance outcome can be guaranteed. Strong-receive viability depends on
the full experiments materially improving attribution precision and showing
consistent gains over matched online baselines.

## StreamSpot GPU Pilot

The StreamSpot pilot reads all 89,770,902 official edges and uses deterministic
per-graph reservoir sampling with 50 retained edges per graph. Graphs 0-239 are
training, 240-299 calibration, and 300-599 test. Official attack graph IDs
300-399 are used only by the graph-level evaluator. Anonymous relations are
preserved as `A-H/i-z`; stage prediction and containment are disabled because
these symbols do not provide defensible ATT&CK-stage semantics.

| Quantity | Result |
|---|---:|
| Test graphs / attack graphs | 300 / 100 |
| Graph precision | 0.7500 |
| Graph recall | 0.1200 |
| Graph F1 | 0.2069 |
| Graph MCC | 0.2098 |
| Graph AUROC | 0.4882 |
| Graph AUPRC | 0.4848 |
| Model p99 latency | 6.02 ms/event |
| Tracing p99 latency | 0.077 ms/event |
| End-to-end p99 latency | 6.08 ms/event |

This run validates graph-labelled online evaluation but is not a main result:
50-edge sampling yields poor recall and near-random ranking. The full protocol
requires sparse categorical storage so all edges can be retained.

## E5 Ingestion

`experiments/cdm_avro.py` now provides a dependency-free streaming CDM Avro
reader with projected-field decoding. Projection matches full decoding on the
first 100 official records. `experiments/build_cadets_e5_temporal.py` was smoke
tested across all ten shards and recovered CDM20 events, UUIDs, relation types,
timestamps, paths, and process properties. A full E5 pilot has not yet been run
because the official shards contain large event blocks and require a long CPU
preprocessing pass before GPU training.

## Unified Seven-Baseline Run

The fixed-environment adaptation suite completed 105 final evaluations: seven
methods, three datasets, and five repetitions. Results are in
`results/20_baseline_provx` through `results/27_baseline_comparison`; every
directory contains only `final_results.csv` and `final_results.json`.

Mean F1 ranges by dataset:

- StreamSpot: 0.9692-0.9805 versus ours 0.9766. ProvFusion (0.9796) and FLASH
  (0.9805) are slightly higher in point estimate, with overlapping 95% CIs.
- Unicorn-Wget: 0.0871-0.8799 versus ours 0.8997 under the same attack-test-only
  protocol. All seven adapted baselines are lower, but several intervals are
  wide because only 125 benign sessions exist.
- E3-CADETS-Causal: 0.0360-0.0998 versus ours 0.6111. All adapted baselines are
  lower under the fixed day-6/day-12/day-13 node protocol.

Thus 19 of 21 method-dataset point estimates are below ours. The evidence does
not support a claim that ours beats every baseline on every dataset. A TIFS
submission should report the two StreamSpot exceptions and emphasize the
online tracing, prospective prediction, and intervention contribution instead
of claiming universal detection dominance.

## Proposed Method V2

`results/28_proposed_method_upgrade` contains the five-repetition evaluation of
the revised detector. Graph-level detection now uses strict zero-exceedance
calibration: an accumulated anomaly score must be greater than every validation
score. E3 retains the relation-path dynamic context gate and is rerun for seeds
0-4 instead of reporting only the best seed.

| Dataset | Proposed F1 mean | Best adapted baseline | Baseline F1 mean |
|---|---:|---|---:|
| StreamSpot | 0.9921 | FLASH | 0.9805 |
| Unicorn-Wget | 0.8997 | AttributionGNN | 0.8799 |
| E3-CADETS-Causal | 0.5794 | AttributionGNN | 0.0998 |

All three point estimates are higher. These remain exploratory reused-public-
holdout results: the test sets were inspected in earlier development, and the
E3 proposed method has a supervised cross-day information budget while most
adapted E3 baselines are unsupervised. Neither limitation may be hidden in a
paper claim.
