"""Diagnose chronological score stability on the locally available E5 day 8.

This experiment deliberately does not report attack effectiveness: the local
corpus does not contain the documented E5 attack days.  It fits only an early
day-8 baseline, calibrates thresholds on the middle period, and audits false
alerts and distribution drift on the late period.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import RobustScaler


RELATIONS = 10
NODE_TYPES = 3
WINDOW_NS = 15 * 60 * 1_000_000_000


def entropy(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0]
    return float(-(values * np.log(values)).sum()) if len(values) else 0.0


def approximate_cardinality(bitmap: np.ndarray) -> float:
    empty = int((~bitmap).sum())
    if empty == 0:
        return float(len(bitmap) * math.log(len(bitmap)))
    return float(-len(bitmap) * math.log(empty / len(bitmap)))


def finalize_aggregates(aggregates: dict[int, dict]):
    records = []
    previous_relation = None
    previous_log_edges = None
    previous_nodes = None
    for key in sorted(aggregates):
        item = aggregates[key]
        count = max(item["edge_count"], 1)
        relation = item["relations"] / count
        src_types = item["src_types"] / count
        dst_types = item["dst_types"] / count
        cardinality = approximate_cardinality(item["nodes"])
        log_edges = math.log1p(item["edge_count"])
        relation_change = 0.0 if previous_relation is None else float(np.abs(relation - previous_relation).sum())
        edge_change = 0.0 if previous_log_edges is None else log_edges - previous_log_edges
        retention = 0.0
        if previous_nodes is not None:
            union = np.logical_or(item["nodes"], previous_nodes).sum()
            retention = float(np.logical_and(item["nodes"], previous_nodes).sum() / max(union, 1))
        hour = ((key * 15) % (24 * 60)) / 60.0
        values = [
            log_edges, math.log1p(cardinality), cardinality / count,
            entropy(relation), entropy(src_types), entropy(dst_types),
            relation_change, edge_change, retention,
            math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            float(item["parts"]),
            *relation.tolist(), *src_types.tolist(), *dst_types.tolist(),
        ]
        records.append({"window_key": key, "edge_count": item["edge_count"], "features": values})
        previous_relation = relation
        previous_log_edges = log_edges
        previous_nodes = item["nodes"]
    return records


def extract_windows(data_dir: Path, bitmap_size: int = 32768):
    aggregates: dict[int, dict] = {}
    edges = 0
    started = time.perf_counter()
    paths = sorted((data_dir / "train").glob("*.TemporalData"))
    for path in paths:
        data = torch.load(path, map_location="cpu")
        timestamp = data.t.detach().cpu().numpy().astype(np.int64, copy=False)
        if not len(timestamp):
            continue
        key = int(timestamp.min() // WINDOW_NS)
        row = aggregates.setdefault(key, {
            "edge_count": 0,
            "relations": np.zeros(RELATIONS, dtype=np.int64),
            "src_types": np.zeros(NODE_TYPES, dtype=np.int64),
            "dst_types": np.zeros(NODE_TYPES, dtype=np.int64),
            "nodes": np.zeros(bitmap_size, dtype=bool),
            "parts": 0,
        })
        relation = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
        message = data.msg.detach().cpu().numpy()
        src = data.src.detach().cpu().numpy().astype(np.uint64, copy=False)
        dst = data.dst.detach().cpu().numpy().astype(np.uint64, copy=False)
        count = len(relation)
        row["edge_count"] += count
        row["relations"] += np.bincount(relation, minlength=RELATIONS)[:RELATIONS]
        row["src_types"] += message[:, :NODE_TYPES].sum(axis=0).astype(np.int64)
        row["dst_types"] += message[:, -NODE_TYPES:].sum(axis=0).astype(np.int64)
        src_hash = ((src * np.uint64(11400714819323198485)) >> np.uint64(32)) % bitmap_size
        dst_hash = ((dst * np.uint64(11400714819323198485)) >> np.uint64(32)) % bitmap_size
        row["nodes"][src_hash.astype(np.int64)] = True
        row["nodes"][dst_hash.astype(np.int64)] = True
        row["parts"] += 1
        edges += count
    records = finalize_aggregates(aggregates)
    elapsed = time.perf_counter() - started
    return records, edges, elapsed, len(paths)


def extract_sketch(sketch_dir: Path):
    started = time.perf_counter()
    sketch = np.load(sketch_dir / "window_sketch.npz")
    aggregates = {}
    for index, key in enumerate(sketch["window_keys"]):
        aggregates[int(key)] = {
            "edge_count": int(sketch["edge_counts"][index]),
            "relations": sketch["relation_counts"][index],
            "src_types": sketch["src_type_counts"][index],
            "dst_types": sketch["dst_type_counts"][index],
            "nodes": sketch["node_bitmaps"][index],
            "parts": 1,
        }
    records = finalize_aggregates(aggregates)
    return records, int(sketch["edge_counts"].sum()), time.perf_counter() - started


def conformal_pvalues(calibration: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.asarray([
        (1 + int((calibration >= score).sum())) / (len(calibration) + 1)
        for score in scores
    ], dtype=np.float64)


def causal_adaptive_pvalues(
    calibration: np.ndarray, scores: np.ndarray, window_size: int,
) -> np.ndarray:
    bank = list(calibration[-window_size:].astype(float))
    output = []
    for score in scores:
        output.append((1 + sum(value >= score for value in bank)) / (len(bank) + 1))
        bank.append(float(score))
        bank = bank[-window_size:]
    return np.asarray(output, dtype=np.float64)


def causal_rolling_scores(features: np.ndarray, start: int, history_size: int) -> np.ndarray:
    """Score each window using only the immediately preceding windows."""
    scores = []
    for index in range(start, len(features)):
        history = features[max(0, index - history_size):index]
        scaler = RobustScaler(quantile_range=(10, 90)).fit(history)
        history_scaled = scaler.transform(history)
        detector = LedoitWolf().fit(history_scaled)
        current = scaler.transform(features[index:index + 1])
        scores.append(float(detector.mahalanobis(current)[0]))
    return np.asarray(scores, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="artifacts/raw_temporal_local_full/E5-CADETS/edge_embeds"
    )
    parser.add_argument(
        "--sketch-dir", default="artifacts/feature_cache/E5-CADETS-Causal/local_day8_window_sketch"
    )
    parser.add_argument(
        "--out-dir", default="artifacts/tifs_results/E5-CADETS-Causal/local_day8_benign_stability"
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    sketch_dir = Path(args.sketch_dir)
    use_sketch = (sketch_dir / "window_sketch.npz").is_file()
    metadata_path = sketch_dir / "metadata.json" if use_sketch else data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "incomplete_raw_diagnostic":
        raise RuntimeError("This diagnostic requires explicitly marked incomplete local data")
    if metadata.get("malicious_nodes_in_stream", 0) or metadata.get("attack_period_available"):
        raise RuntimeError("Attack-labelled nodes are present; benign-only diagnostic is invalid")

    if use_sketch:
        records, edge_count, feature_seconds = extract_sketch(sketch_dir)
        file_count = int(metadata.get("manifest_present_files", 0))
    else:
        records, edge_count, feature_seconds, file_count = extract_windows(data_dir)
    if len(records) < 20:
        raise RuntimeError(f"Need at least 20 chronological windows, found {len(records)}")
    features = np.asarray([row["features"] for row in records], dtype=np.float64)
    n_train = max(8, int(len(features) * 0.40))
    n_calibration = max(19, int(len(features) * 0.40))
    if n_train + n_calibration >= len(features):
        n_calibration = len(features) - n_train - 1
    train = features[:n_train]
    calibration = features[n_train:n_train + n_calibration]
    holdout = features[n_train + n_calibration:]

    scaler = RobustScaler(quantile_range=(10, 90)).fit(train)
    train_scaled = scaler.transform(train)
    calibration_scaled = scaler.transform(calibration)
    holdout_scaled = scaler.transform(holdout)
    detector = LedoitWolf().fit(train_scaled)
    calibration_scores = detector.mahalanobis(calibration_scaled)
    score_started = time.perf_counter()
    holdout_scores = detector.mahalanobis(holdout_scaled)
    scoring_seconds = time.perf_counter() - score_started
    frozen_pvalues = conformal_pvalues(calibration_scores, holdout_scores)

    rolling_scores = causal_rolling_scores(features, n_train, n_train)
    rolling_calibration_scores = rolling_scores[:n_calibration]
    rolling_holdout_scores = rolling_scores[n_calibration:]
    rolling_pvalues = conformal_pvalues(
        rolling_calibration_scores, rolling_holdout_scores
    )
    adaptive_pvalues = causal_adaptive_pvalues(
        rolling_calibration_scores, rolling_holdout_scores, n_calibration
    )

    ks = ks_2samp(calibration_scores, holdout_scores, method="auto")
    rolling_ks = ks_2samp(
        rolling_calibration_scores, rolling_holdout_scores, method="auto"
    )
    result = {
        "protocol_version": "tifs-e5-local-benign-stability-v1",
        "dataset": "E5-CADETS-Causal",
        "evaluation_scope": "locally_available_day8_benign_only",
        "result_tier": "incomplete_raw_diagnostic",
        "effectiveness_f1_available": False,
        "attack_period_available": False,
        "expected_raw_shards": int(metadata.get("manifest_expected_files", 0)),
        "available_raw_shards": int(metadata.get("manifest_present_files", 0)),
        "source_files": file_count,
        "chronological_windows": len(features),
        "early_train_windows": len(train),
        "middle_calibration_windows": len(calibration),
        "late_holdout_windows": len(holdout),
        "fused_edges": edge_count,
        "feature_extraction_seconds": feature_seconds,
        "raw_scan_seconds": metadata.get("scan_seconds"),
        "raw_scan_throughput_events_per_second": metadata.get("scan_throughput_events_per_second"),
        "feature_throughput_edges_per_second": edge_count / max(feature_seconds, 1e-12),
        "scoring_throughput_windows_per_second": len(holdout) / max(scoring_seconds, 1e-12),
        "conformal_pvalue_resolution": 1.0 / (len(calibration) + 1),
        "frozen_calibration_score_mean": float(calibration_scores.mean()),
        "frozen_holdout_score_mean": float(holdout_scores.mean()),
        "frozen_score_ks_statistic": float(ks.statistic),
        "frozen_score_ks_pvalue": float(ks.pvalue),
        "rolling_calibration_score_mean": float(rolling_calibration_scores.mean()),
        "rolling_holdout_score_mean": float(rolling_holdout_scores.mean()),
        "rolling_score_ks_statistic": float(rolling_ks.statistic),
        "rolling_score_ks_pvalue": float(rolling_ks.pvalue),
        "rolling_history_windows": n_train,
        "rolling_update_policy": "causal_unlabeled_all_prior_windows",
        "adaptive_calibration_policy": "causal_sliding_all_prior_scores",
    }
    for alpha in (0.05, 0.10):
        frozen_alarms = frozen_pvalues <= alpha
        rolling_alarms = rolling_pvalues <= alpha
        adaptive_alarms = adaptive_pvalues <= alpha
        key = str(alpha).replace(".", "_")
        result[f"nominal_alpha_{key}"] = alpha
        result[f"frozen_late_false_alerts_alpha_{key}"] = int(frozen_alarms.sum())
        result[f"frozen_late_false_alert_rate_alpha_{key}"] = float(frozen_alarms.mean())
        result[f"rolling_late_false_alerts_alpha_{key}"] = int(rolling_alarms.sum())
        result[f"rolling_late_false_alert_rate_alpha_{key}"] = float(rolling_alarms.mean())
        result[f"adaptive_late_false_alerts_alpha_{key}"] = int(adaptive_alarms.sum())
        result[f"adaptive_late_false_alert_rate_alpha_{key}"] = float(adaptive_alarms.mean())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "window_key": [row["window_key"] for row in records[n_train + n_calibration:]],
        "edge_count": [row["edge_count"] for row in records[n_train + n_calibration:]],
        "frozen_anomaly_score": holdout_scores,
        "frozen_conformal_pvalue": frozen_pvalues,
        "rolling_anomaly_score": rolling_holdout_scores,
        "rolling_conformal_pvalue": rolling_pvalues,
        "adaptive_conformal_pvalue": adaptive_pvalues,
    }).to_csv(out / "late_window_decisions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps({"result": result, "source_metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
