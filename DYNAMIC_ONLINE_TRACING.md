# Online Dynamic Attack Tracing

This fork provides a causal online tracing path that runs as each scored event
arrives. DepImpact remains available as the controlled post-hoc baseline.

## Core Rule

For every streamed edge event `(src, dst, edge_type, loss, time)`:

1. The dynamic multi-relational graph is updated online.
2. A node score is replaced only when the new score is larger than the current
   score.
3. An edge weight is replaced only when the new edge score is larger than the
   current weight.
4. Lower scores never erase stronger earlier evidence.

This implements monotonic evidence retention for both node and typed-edge
attribution. A `MultiDiGraph` prevents WRITE/READ/EXECUTE relations between the
same endpoints from overwriting one another.

The main-paper protocol uses causal inference batches of one event and restores
the training checkpoint between validation and test. Validation scores calibrate
a split-conformal threshold; no test score or test label is used for threshold
selection. Word2Vec is trained on training nodes only, with deterministic OOV
handling at validation/test time.

## Attack Subgraph and Cutoff

The online tracer maintains a separate attack subgraph. Edges enter the attack
subgraph when their loss exceeds the validation-derived anomaly threshold, or
when they provide strong context around an already active attack node.

For each active edge, the tracer estimates an attack stage from the edge type
and local message context:

- initial access
- payload write
- execution
- persistence or lateral movement
- C2 or exfiltration

The current attack subgraph, causal stage-transition history, validation
conformal p-value, edge severity, and local support are used by the containment
policy. The default policy requires both extreme validation-calibrated evidence
and graph support. Later expansion through a cut node is truncated, while
stronger new evidence may override the cut. Every later event that would be
blocked is logged, including benign events, enabling counterfactual prevention
and collateral-damage evaluation.

## Outputs

For each tested model epoch, artifacts are saved under:

`artifacts/attack_reconstruction/tracing/<hash>/<dataset>/online_dynamic/<model_epoch>/test/`

Important files:

- `dynamic_graph.pkl`: online monotonic dynamic graph
- `attack_subgraph.pkl`: current traced attack subgraph
- `dynamic_state.json`: summary, top nodes/edges, stage transitions, cut nodes
- `online_trace_events.csv`: complete streamed decisions, prediction, potential
  block outcome, and tracing latency
- `results.pth`: compatibility view keyed by time window

## Datasets

The configuration now accepts:

- `CADETS_E3`
- `CADETS_E5`
- `E3-CADETS`
- `E5-CADETS`
- `E3-CADETS-Causal`
- `E5-CADETS-Causal`
- `StreamSpot`
- `Unicorn-Wget`

`StreamSpot` and `Unicorn-Wget` are configured as external temporal datasets.
Use prebuilt `TemporalData` splits via:

```shell
python src/dynamic_trace.py StreamSpot --run_from_training --edge_featurization.embed_edges.prebuilt_dir=<path>
python src/dynamic_trace.py Unicorn-Wget --run_from_training --edge_featurization.embed_edges.prebuilt_dir=<path>
```

Optional external labels can be provided with CSV files using:

- `dataset.node_labels_path`: node-level labels with `node_id,label`
- `dataset.tw_labels_path`: time-window labels with `time_window,node_id,count`

Their public labels are graph/batch level, so they are valid for graph-level
detection, transfer, and throughput experiments. They must not be presented as
fine-grained tracing or containment ground truth without additional event/node
annotations. E3-CADETS and E5-CADETS provide the primary attribution results.
The `*-Causal` aliases provide strict chronological sensitivity splits; pretrained
weights from the standard split must not be used for these runs.

## Publication Evaluation

Generate the preregistered 5-seed ablation matrix:

```shell
python experiments/run_tifs_experiments.py \
  --prebuilt StreamSpot=<streamspot-temporal-dir> Unicorn-Wget=<wget-temporal-dir>
```

Add `--preprocessed` only when CADETS features already exist for every requested
seed, and add `--execute` only after all dataset paths are configured. Evaluate one run:

```shell
python experiments/evaluate_online_tracing.py \
  --events <online_trace_events.csv> \
  --ground-truth <event_or_node_ground_truth.csv> \
  --dataset E3-CADETS --method full --seed 0 --out-dir <output>
```

Aggregate paired seed-level comparisons:

```shell
python experiments/aggregate_tifs_results.py <evaluation-root> --reference full
```
