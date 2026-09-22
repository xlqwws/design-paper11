# TIFS Experimental Protocol for Causal Online Attack Tracing

## 1. Claim Boundary

The paper should claim a causal online tracing and risk-guided containment layer
for provenance IDSs. It should not claim that heuristic stage names alone are a
learned ATT&CK predictor, nor that offline replay proves real-world prevention.
The defensible contribution is:

> A validation-calibrated, prequential dynamic provenance method that retains
> strongest evidence, predicts the next behavioral state from past-only attack
> context, and emits auditable containment decisions under a false-intervention
> budget.

The key comparison is not merely against weaker detectors. The strongest test
uses identical attribution-GNN anomaly scores and changes only tracing: post-hoc
DepImpact versus causal online tracing. This isolates the value of the proposed
mechanism.

## 2. Research Questions

| RQ | Question | Primary evidence |
|---|---|---|
| RQ1 | Does online tracing preserve detection and improve attribution quality? | ADP, MCC, node/edge precision, attack coverage, inspected nodes |
| RQ2 | Does it beat post-hoc tracing under equal anomaly scores and budgets? | attack-story F1, stage coverage, components, latency, workload |
| RQ3 | Is next-state prediction genuinely prospective? | prequential next-stage accuracy, macro-F1, lead time; no future updates |
| RQ4 | Is containment useful and safe? | prevented malicious events, benign collateral rate, time-to-cut, utility |
| RQ5 | Is the system deployable online? | end-to-end p50/p95/p99 latency, events/s, CPU/GPU/RAM, state growth |
| RQ6 | Which mechanisms matter and are results stable? | controlled ablations, five full-pipeline seeds, paired confidence intervals |
| RQ7 | Does it remain reliable under shift and adaptive evasion? | cross-dataset transfer, temporal drift, mimicry, delay, missing-event tests |

## 3. Dataset Roles

E3-CADETS and E5-CADETS are the primary datasets for fine-grained node
attribution, attack-story reconstruction, prediction, and counterfactual
containment. Use the conservative provenance ground truth and add campaign IDs and
stage labels from the DARPA scenario descriptions. Two annotators should label
stages independently; report Cohen's kappa and resolve disagreements before
evaluation.

StreamSpot and Unicorn-Wget have graph/batch labels in their standard public
form. Use them for attack detection, transfer, stream throughput, and drift.
Do not report node-level tracing precision, stage prediction, or containment
precision on those labels. Fine-grained claims require releasing new event-level
annotations.

All splits are fixed before experimentation. Text/token embeddings are trained
only on training entities. Validation is benign-only calibration and tuning.
Test labels are opened only by the evaluation script after decisions are saved.

Report both the published benchmark split and a strict chronological sensitivity
split. Use `E3-CADETS-Causal` (days 2-4 train, day 5 validation, days 6-13 test)
and `E5-CADETS-Causal` (days 8-11 train, day 12 validation, days 13-17 test).
Train these models from scratch; standard-split pretrained weights are invalid.

## 4. Baselines and Fairness

Detection baselines: VELOX, AttributionGNN, KAIROS, MAGIC, FLASH, ThreaTrace, and the
native StreamSpot/UNICORN methods on their corresponding datasets.

Tracing baselines: AttributionGNN+DepImpact, KAIROS reconstruction, unweighted 1/2/3-hop
causal traversal, backward/forward reachability with the same seed alerts, and
an oracle-budget traversal upper bound. SLEUTH may be reported separately as a
knowledge-based reference, not mixed with attack-agnostic systems.

Each baseline receives the same train/validation/test events, text fields,
ground truth granularity, seed alerts, and analyst node/edge budget. Tune every
method with the same validation compute budget. Report failures and timeouts.

## 5. Main Experiments

### E1 Detection and Quality of Attribution

Report per dataset and five seeds: threshold-free ADP and AP; deployed MCC,
precision, recall and false alerts per million events; campaign detection rate;
and the exact number of benign and malicious nodes sent to analysts. Accuracy
and ROC-AUC are secondary due to extreme imbalance.

### E2 Online Versus Post-Hoc Tracing

Freeze one anomaly-score stream. Compare full online tracing with DepImpact and
fixed-hop traversal. Report node/edge precision, recall and F1, attack-stage
coverage, disconnected components, attack-story recovery rate, and nodes/edges
required for 50/80/90/100% stage coverage. Plot stage recall versus inspected
nodes as the analyst-workload Pareto curve.

### E3 Prospective Prediction

At event t, save the prediction before updating transition counts with event
t+1. Evaluate the next different ground-truth stage within each campaign:
accuracy, macro-F1, top-2 recall, expected calibration error, and median lead
time before the next stage. Compare static stage prior, prequential Markov,
frozen validation model, and shuffled-time negative control. The shuffled
control should collapse performance; otherwise the task likely leaks labels.

### E4 Counterfactual Containment

Replay each campaign chronologically. When a cut decision occurs, mark later
incident events as `would_block`; do not delete them from the evaluation log.
Report malicious-event prevention rate, benign collateral rate, containment
precision, campaign containment rate, time-to-cut, and net utility over a sweep
of false-intervention costs (1, 5, 10, 100). Also report an oracle earliest-safe
cut upper bound and random-node budget-matched control.

### E5 Online Systems Performance

Strict-online main results use causal batch size 1. Sweep 1/32/256/1024 only as
a latency-throughput tradeoff study and explicitly label larger batches as
micro-batched. Measure model inference and tracer overhead separately, plus
end-to-end p50/p95/p99 latency, peak RAM/VRAM, events/s, graph state size, and
24-hour projected storage. Verify throughput against each dataset's peak input
rate, not only its average.

### E6 Robustness and Generalization

Use train-on-E3/test-on-E5 and train-on-E5/test-on-E3 where feature semantics
permit. Add chronological benign drift before attacks. Stress attacks with
10-50% critical-event deletion, 1-10 benign mimicry events between attack steps,
timestamp stretching, event reordering, and score perturbation. Report both
absolute metrics and degradation from clean replay.

## 6. Required Ablations

The executable matrix in `experiments/run_tifs_experiments.py` includes:

| Variant | Removed or replaced mechanism | What it tests |
|---|---|---|
| full | none | proposed system |
| latest_update | max becomes latest score | monotonic evidence retention |
| ema_update | max becomes EMA | robustness versus smoothing |
| no_context | contextual expansion disabled | attack continuity benefit |
| no_retained_context | historical strongest evidence disabled | whether monotonic evidence changes future causal tracking |
| no_prediction | prospective state disabled | prediction contribution |
| no_cutoff | containment disabled | attribution versus intervention |
| weighted_policy | conformal decision becomes manual weighted risk | calibration and false-cut control |
| destination_cut | relation-aware actor becomes destination node | containment target semantics |

Add post-hoc DepImpact and 1/2/3-hop traversal outside the online switch. For
every ablation, keep detector weights and anomaly-score stream fixed whenever
the ablated component is downstream of detection.

## 7. Statistical Protocol

Run the complete featurization-training-inference pipeline with seeds 0-4, not
only the final tracer. Report mean, standard deviation, min/max, 95% temporal
block-bootstrap intervals, and relative ADP standard deviation. Use paired
seed-level differences for full versus each ablation/baseline. A two-sided exact
sign test over only five seeds cannot attain p < 0.05, so it is diagnostic rather
than the primary significance test. Primary uncertainty uses a hierarchical
paired bootstrap that first samples model seeds and then independent campaigns
or graphs within seed. Events from one campaign are never treated as independent
replicates. Correct the four preregistered primary-hypothesis p-values with
Holm's method and report paired effect sizes and confidence intervals.

The four preregistered primary outcomes should be: ADP, node precision at full
campaign coverage, attack-story edge F1, and prevention utility at a fixed
benign-collateral budget. Treat all other sweeps as secondary analyses.

## 8. Acceptance Gates

Before framing the work as TIFS-ready, require all of the following:

1. Full method improves attack-story or workload outcomes over post-hoc tracing
   under identical anomaly scores on both CADETS datasets.
2. All attacks are considered separately; no result relies on inflated 2-hop or
   whole-window malicious labels.
3. Prediction beats static and shuffled controls with positive lead time.
4. Containment gains remain under a declared benign collateral budget.
5. Strict-online p99 latency sustains the measured peak event rate.
6. Gains survive five full-pipeline seeds and at least one cross-version test.
7. Every table is generated from saved decisions and a manifest; no hand-entered
   result or test-set threshold tuning is used.

Meeting these gates does not guarantee acceptance, but failing any of gates 1-4
would make the central claim difficult to defend.

## 9. Final-Result Storage Policy

Paper-facing outputs are published below `results/`, one numbered directory per
experiment. Each directory contains only `final_results.csv` and
`final_results.json`. Training losses, epoch logs, checkpoints, event-level
decision streams, graphs, and temporary perturbations remain outside this tree.
The JSON records protocol scope and limitations; unavailable metrics remain
null and are never replaced by zero. `experiments/run_final_experiment_suite.py`
enforces this layout after every publication run.

## 10. Cross-Repository Baseline Adaptation Protocol

`experiments/run_tifs_baseline_suite.py` evaluates ProvX, ProvFusion, AttributionGNN,
THREATRACE, MAGIC, Kairos, and FLASH on the immutable full-data StreamSpot and
Unicorn-Wget sketches and the E3 day-6/day-12/day-13 node split. Five final
evaluations are run per method and dataset. StreamSpot and Unicorn-Wget use
normal training and benign validation graphs. E3 uses unlabeled chronological
training and validation streams, which may contain campaign entities; their
labels are not accessed by the adapted baselines. Test labels are joined only
after decisions are frozen.

These runs are labelled **method-faithful current-environment adaptations**.
They are not official upstream results: no upstream repository can execute
unchanged across all three datasets in the fixed PyTorch 1.13 environment.
Neural adaptations use CUDA; fusion, clustering, and isolation-forest paths are
reported as CPU statistical estimators in the same CUDA-enabled environment.
The result manifest records the upstream limitation for every method.

Baseline values are never clipped or selected to make the proposed method win.
The `baseline_below_ours` field is computed only after test metrics are fixed.
Any baseline win must remain in the table and be discussed as a negative result.
