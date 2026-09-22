from __future__ import annotations

import bisect
import csv
import heapq
import json
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import pandas as pd
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional deployment metric
    psutil = None

from provnet_utils import log


ONLINE_DYNAMIC_METHODS = {"online_dynamic", "dynamic", "dynamic_kg"}


@dataclass
class OnlineDynamicSummary:
    out_dir: str
    graph_path: str
    attack_graph_path: str
    state_path: str
    event_log_path: str
    results_path: str
    num_nodes: int
    num_edges: int
    num_attack_nodes: int
    num_attack_edges: int
    num_cut_events: int


def is_online_dynamic_enabled(cfg) -> bool:
    tracing_cfg = getattr(getattr(cfg, "attack_reconstruction", None), "tracing", None)
    if tracing_cfg is None:
        return False
    method = str(getattr(tracing_cfg, "used_method", "")).strip()
    dyn_cfg = getattr(tracing_cfg, "online_dynamic", None)
    explicit_enabled = bool(getattr(dyn_cfg, "enabled", False)) if dyn_cfg is not None else False
    if method in ONLINE_DYNAMIC_METHODS:
        return True
    if method == "depimpact":
        return False
    return explicit_enabled


def _get_dynamic_cfg(cfg):
    return getattr(cfg.attack_reconstruction.tracing, "online_dynamic", None)


def _cfg_value(cfg, name: str, default):
    dyn_cfg = _get_dynamic_cfg(cfg)
    if dyn_cfg is None:
        return default
    value = getattr(dyn_cfg, name, default)
    return default if value is None else value


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _node_key(value) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _edge_key(u: str, v: str, edge_type: str) -> str:
    return f"{u}->{v}:{edge_type}"


def _parse_csv_set(text: str) -> set[str]:
    if text is None:
        return set()
    return {item.strip() for item in str(text).split(",") if item.strip()}


def load_calibration_scores(val_tw_dir: str) -> list[float]:
    """Load validation losses only; test scores never participate in calibration."""
    scores: list[float] = []
    if not os.path.isdir(val_tw_dir):
        return scores
    for name in sorted(os.listdir(val_tw_dir)):
        path = os.path.join(val_tw_dir, name)
        if not os.path.isfile(path) or not name.lower().endswith(".csv"):
            continue
        frame = pd.read_csv(path, usecols=["loss"])
        scores.extend(float(value) for value in frame["loss"].dropna().tolist())
    return scores


def _robust_location_scale(scores: Iterable[float]) -> tuple[float, float]:
    ordered = sorted(float(value) for value in scores)
    if not ordered:
        raise ValueError("Robust evidence calibration requires validation scores")
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    deviations = sorted(abs(value - median) for value in ordered)
    mad = deviations[middle] if len(deviations) % 2 else (deviations[middle - 1] + deviations[middle]) / 2.0
    return median, max(1.4826 * mad, 1e-12)


def _insert_topk(heap: list[float], value: float, k: int) -> None:
    if len(heap) < k:
        heapq.heappush(heap, value)
    elif value > heap[0]:
        heapq.heapreplace(heap, value)


def load_calibration_node_scores(
    val_tw_dir: str, calibration_scores: Iterable[float], top_k: int
) -> list[float]:
    """Build the exact bounded node statistic used later in the test stream."""
    center, scale = _robust_location_scale(calibration_scores)
    heaps: Dict[str, list[float]] = defaultdict(list)
    if not os.path.isdir(val_tw_dir):
        return []
    for name in sorted(os.listdir(val_tw_dir)):
        path = os.path.join(val_tw_dir, name)
        if not os.path.isfile(path) or not name.lower().endswith(".csv"):
            continue
        for frame in pd.read_csv(path, usecols=["loss", "srcnode", "dstnode"], chunksize=100000):
            for row in frame.itertuples(index=False):
                evidence = max(0.0, (float(row.loss) - center) / scale)
                for node in {_node_key(row.srcnode), _node_key(row.dstnode)}:
                    _insert_topk(heaps[node], evidence, top_k)
    return [float(sum(values)) for values in heaps.values()]


def conformal_threshold(calibration_scores: Iterable[float], alpha: float) -> float:
    """Finite-sample split-conformal upper quantile with the conservative correction."""
    scores = sorted(float(value) for value in calibration_scores)
    if not scores:
        raise ValueError("Conformal calibration requires at least one validation score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("calibration_alpha must be in (0, 1)")
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1.0 - alpha)))
    return scores[max(rank - 1, 0)]


def resolve_online_threshold(
    val_thresholds: Dict[str, float],
    cfg,
    calibration_scores: Optional[Iterable[float]] = None,
) -> float:
    method = str(_cfg_value(
        cfg,
        "threshold_method",
        getattr(cfg.detection.evaluation.node_evaluation, "threshold_method", "max_val_loss"),
    )).strip().lower()
    if method in {"conformal", "split_conformal"}:
        return conformal_threshold(
            calibration_scores or [],
            float(_cfg_value(cfg, "calibration_alpha", 0.001)),
        )
    if method == "max_val_loss":
        return float(val_thresholds["max"])
    if method == "mean_val_loss":
        return float(val_thresholds["mean"])
    if method in {"percentile_90", "90_percent_val_loss"}:
        return float(val_thresholds.get("percentile_90", val_thresholds["max"]))
    try:
        return float(method)
    except ValueError as exc:
        raise ValueError(f"Invalid online dynamic threshold method `{method}`") from exc


class OnlineDynamicAttackTracer:
    """Causal, prequential attack tracing over an event stream.

    Decisions for event t use validation calibration and stream state strictly
    preceding t. Node and relation scores use the configured update rule; the
    publication default is monotonic max evidence retention.
    """

    STATIC_NEXT_STAGE = {
        "initial_access": "payload_write",
        "payload_write": "execution",
        "execution": "persistence_or_lateral",
        "persistence_or_lateral": "c2_or_exfiltration",
        "c2_or_exfiltration": "c2_or_exfiltration",
        "benign_or_context": "payload_write",
    }
    STAGE_SEVERITY = {
        "benign_or_context": 0.05,
        "initial_access": 0.35,
        "payload_write": 0.55,
        "execution": 0.85,
        "persistence_or_lateral": 0.75,
        "c2_or_exfiltration": 0.90,
    }
    EVENT_LOG_FIELDS = [
        "event_id", "model_epoch", "split", "time_window", "time", "srcnode", "dstnode",
        "edge_type", "loss", "threshold", "conformal_pvalue", "normalized_score",
        "node_evidence", "src_node_pvalue", "dst_node_pvalue",
        "src_node_activated", "dst_node_activated", "primary_activation", "activated",
        "context_activation", "retained_context_activation",
        "would_block", "truncated", "stage",
        "predicted_next_stage", "prediction_confidence", "risk", "cut", "cut_node",
        "reason", "model_latency_us", "trace_latency_us", "end_to_end_latency_us",
    ]

    def __init__(
        self,
        cfg,
        model_epoch_file: str,
        split: str,
        anomaly_threshold: float,
        nodeid2msg: Optional[Dict[Any, str]] = None,
        calibration_scores: Optional[Iterable[float]] = None,
        calibration_node_scores: Optional[Iterable[float]] = None,
    ):
        self.cfg = cfg
        self.model_epoch_file = model_epoch_file
        self.split = split
        self.anomaly_threshold = max(float(anomaly_threshold), 1e-12)
        self.nodeid2msg = nodeid2msg or {}
        self.calibration_scores = sorted(float(value) for value in (calibration_scores or []))
        self.calibration_node_scores = sorted(
            float(value) for value in (calibration_node_scores or [])
        )

        base_dir = getattr(
            cfg.attack_reconstruction.tracing,
            "_dynamic_tracing_dir",
            os.path.join(cfg.attack_reconstruction.tracing._tracing_graph_dir, "online_dynamic"),
        )
        self.out_dir = os.path.join(base_dir, model_epoch_file, split)
        os.makedirs(self.out_dir, exist_ok=True)

        self.threshold_method = str(_cfg_value(cfg, "threshold_method", "max_val_loss")).lower()
        self.calibration_alpha = float(_cfg_value(cfg, "calibration_alpha", 0.001))
        self.node_gate_enabled = bool(_cfg_value(cfg, "node_gate_enabled", False))
        self.node_gate_alpha = float(_cfg_value(cfg, "node_gate_alpha", 0.01))
        self.node_gate_top_k = int(_cfg_value(cfg, "node_gate_top_k", 3))
        if self.node_gate_enabled and not self.calibration_node_scores:
            raise ValueError("node_gate_enabled requires validation node calibration scores")
        if self.node_gate_top_k <= 0:
            raise ValueError("node_gate_top_k must be positive")
        self.node_evidence_center, self.node_evidence_scale = _robust_location_scale(
            self.calibration_scores or [0.0]
        )
        self.node_evidence_heaps: Dict[str, list[float]] = defaultdict(list)
        self.context_alpha = float(_cfg_value(cfg, "context_alpha", 0.01))
        self.cutoff_pvalue = float(_cfg_value(cfg, "cutoff_pvalue", 0.0001))
        self.minimum_conformal_pvalue = 1.0 / (len(self.calibration_scores) + 1.0)
        self.minimum_node_pvalue = 1.0 / (len(self.calibration_node_scores) + 1.0)
        self.effective_cutoff_pvalue = max(
            self.cutoff_pvalue, self.minimum_conformal_pvalue,
            self.minimum_node_pvalue if self.node_gate_enabled else 0.0,
        )
        self.activation_margin = float(_cfg_value(cfg, "activation_margin", 1.0))
        self.context_margin = float(_cfg_value(cfg, "context_margin", 0.65))
        self.retained_evidence_margin = float(_cfg_value(cfg, "retained_evidence_margin", 1.0))
        self.retained_event_floor = float(_cfg_value(cfg, "retained_event_floor", 0.2))
        self.context_horizon_windows = int(_cfg_value(cfg, "context_horizon_windows", 4))
        self.cutoff_risk_threshold = float(_cfg_value(cfg, "cutoff_risk_threshold", 0.78))
        self.cut_override_margin = float(_cfg_value(cfg, "cut_override_margin", 1.5))
        self.min_attack_support = int(_cfg_value(cfg, "min_attack_support", 2))
        self.prediction_prior_strength = float(_cfg_value(cfg, "prediction_prior_strength", 0.25))
        self.min_prediction_confidence = float(_cfg_value(cfg, "min_prediction_confidence", 0.55))
        self.decision_policy = str(_cfg_value(cfg, "decision_policy", "conformal_risk")).lower()
        self.update_rule = str(_cfg_value(cfg, "update_rule", "max")).lower()
        self.ema_alpha = float(_cfg_value(cfg, "ema_alpha", 0.2))
        self.context_enabled = bool(_cfg_value(cfg, "context_enabled", True))
        self.prediction_enabled = bool(_cfg_value(cfg, "prediction_enabled", True))
        self.cutoff_enabled = bool(_cfg_value(cfg, "cutoff_enabled", True))
        self.containment_scope = str(_cfg_value(cfg, "containment_scope", "incident")).lower()
        self.cut_target_policy = str(_cfg_value(cfg, "cut_target_policy", "causal_actor")).lower()
        self.max_logged_events = int(_cfg_value(cfg, "max_logged_events", 0))
        self.max_dynamic_edges = int(_cfg_value(cfg, "max_dynamic_edges", 0))
        self.max_attack_nodes = int(_cfg_value(cfg, "max_attack_nodes", 200000))
        self.max_attack_edges = int(_cfg_value(cfg, "max_attack_edges", 300000))
        self.log_all_events = bool(_cfg_value(cfg, "log_all_events", True))
        self.cutoff_stages = _parse_csv_set(
            _cfg_value(cfg, "cutoff_stages", "execution,persistence_or_lateral,c2_or_exfiltration")
        )
        if self.update_rule not in {"max", "latest", "ema"}:
            raise ValueError("update_rule must be one of: max, latest, ema")

        # MultiDiGraph preserves distinct typed relations between the same endpoints.
        self.graph = nx.MultiDiGraph()
        self.attack_graph = nx.MultiDiGraph()
        self.node_scores: Dict[str, float] = defaultdict(float)
        self.edge_scores: Dict[str, float] = defaultdict(float)
        self.active_nodes: set[str] = set()
        self.active_window_by_node: Dict[str, int] = {}
        self.cut_nodes: set[str] = set()
        self.cut_edges: set[str] = set()
        self.cut_time_by_node: Dict[str, int] = {}
        self.time_window_names: Dict[int, str] = {}
        self.tw_to_nodes: Dict[int, set[str]] = defaultdict(set)
        self.tw_to_cut_nodes: Dict[int, set[str]] = defaultdict(set)
        self.stage_counts: Counter[str] = Counter()
        self.stage_transitions: Counter[tuple[str, str]] = Counter()
        self.last_stage_by_node: Dict[str, str] = {}

        self.total_events = 0
        self.activated_events = 0
        self.context_events = 0
        self.retained_context_events = 0
        self.would_block_events = 0
        self.truncated_events = 0
        self.cut_events = 0
        self.out_of_order_events = 0
        self.dropped_dynamic_edges = 0
        self.dropped_attack_edges = 0
        self.last_event_time: Optional[int] = None
        self.trace_latency_us: list[float] = []
        self.peak_rss_bytes = 0
        self.peak_cuda_bytes = 0

        self.event_log_path = os.path.join(self.out_dir, "online_trace_events.csv")
        self._event_log_file = open(self.event_log_path, "w", newline="", encoding="utf-8")
        self._event_writer = csv.DictWriter(self._event_log_file, fieldnames=self.EVENT_LOG_FIELDS)
        self._event_writer.writeheader()
        self._logged_events = 0
        self._sample_memory()

    def observe_edge(self, event: Dict[str, Any], time_window: int) -> Dict[str, Any]:
        started_ns = time.perf_counter_ns()
        self.total_events += 1
        src = _node_key(event.get("srcnode"))
        dst = _node_key(event.get("dstnode"))
        edge_type = str(event.get("edge_type", "unknown"))
        loss = _as_float(event.get("loss"), 0.0)
        event_time = _as_int(event.get("time"), 0)
        srcmsg = str(event.get("srcmsg", self.nodeid2msg.get(src, "")))
        dstmsg = str(event.get("dstmsg", self.nodeid2msg.get(dst, "")))
        normalized_score = loss / self.anomaly_threshold
        pvalue = _as_float(event.get("conformal_pvalue"), self._conformal_pvalue(loss))
        node_evidence, src_node_pvalue, dst_node_pvalue = self._update_node_evidence(src, dst, loss)

        if self.last_event_time is not None and event_time < self.last_event_time:
            self.out_of_order_events += 1
        self.last_event_time = max(self.last_event_time or event_time, event_time)

        would_block = self._would_block(src, dst)
        self.would_block_events += int(would_block)
        retained_score = max(
            self.node_scores.get(src, 0.0), self.node_scores.get(dst, 0.0),
            self.edge_scores.get(_edge_key(src, dst, edge_type), 0.0),
        )
        recently_active = any(
            node in self.active_window_by_node
            and int(time_window) - self.active_window_by_node[node] <= self.context_horizon_windows
            for node in (src, dst)
        )
        self._update_dynamic_graph(
            src, dst, edge_type, loss, normalized_score, event_time, srcmsg, dstmsg, time_window
        )

        if self.threshold_method in {"conformal", "split_conformal"} and self.calibration_scores:
            primary_activation = pvalue <= self.calibration_alpha
            current_context_evidence = pvalue <= self.context_alpha
        else:
            primary_activation = loss >= self.anomaly_threshold * self.activation_margin
            current_context_evidence = loss >= self.anomaly_threshold * self.context_margin
        if self.node_gate_enabled:
            primary_activation = min(src_node_pvalue, dst_node_pvalue) <= self.node_gate_alpha
            current_context_evidence = min(src_node_pvalue, dst_node_pvalue) <= self.context_alpha
        src_node_activated = bool(
            primary_activation and (
                not self.node_gate_enabled or src_node_pvalue <= self.node_gate_alpha
            )
        )
        dst_node_activated = bool(
            primary_activation and (
                not self.node_gate_enabled or dst_node_pvalue <= self.node_gate_alpha
            )
        )
        retained_context_activation = bool(
            self.context_enabled and recently_active
            and retained_score / self.anomaly_threshold + 1e-12 >= self.retained_evidence_margin
            and normalized_score + 1e-12 >= self.retained_event_floor
            and not primary_activation
        )
        context_activation = (
            self.context_enabled and recently_active
            and (current_context_evidence or retained_context_activation)
        )
        context_activation = bool(context_activation and not primary_activation)
        activated = primary_activation or context_activation
        truncated = bool(activated and would_block and loss < self.anomaly_threshold * self.cut_override_margin)
        stage = self._stage(edge_type, srcmsg, dstmsg)
        predicted_next_stage, prediction_confidence = self._predict_next_stage(stage)
        prediction = {
            "stage": stage,
            "predicted_next_stage": predicted_next_stage,
            "prediction_confidence": prediction_confidence,
            "risk": 0.0,
            "cut": False,
            "cut_node": "",
            "reason": "",
        }

        if activated and not truncated:
            self._update_attack_subgraph(
                src, dst, edge_type, loss, normalized_score, event_time, srcmsg, dstmsg,
                time_window, context_activation,
                {node for node, selected in ((src, src_node_activated), (dst, dst_node_activated)) if selected},
            )
            prediction = self._predict_and_maybe_cut(
                src, dst, edge_type, loss, normalized_score,
                min(pvalue, src_node_pvalue, dst_node_pvalue) if self.node_gate_enabled else pvalue,
                event_time, time_window, stage, predicted_next_stage, prediction_confidence,
                src_node_pvalue, dst_node_pvalue,
            )
            self._record_stage_transition(dst, stage)
            self.activated_events += int(primary_activation)
            self.context_events += int(context_activation)
            self.retained_context_events += int(retained_context_activation)
        elif truncated:
            self.truncated_events += 1
            self.tw_to_cut_nodes[int(time_window)].update([src, dst])

        decision = {
            "event_id": self.total_events - 1,
            "model_epoch": self.model_epoch_file,
            "split": self.split,
            "time_window": int(time_window),
            "time": int(event_time),
            "srcnode": src,
            "dstnode": dst,
            "edge_type": edge_type,
            "loss": float(loss),
            "threshold": float(self.anomaly_threshold),
            "conformal_pvalue": float(pvalue),
            "node_evidence": float(node_evidence),
            "src_node_pvalue": float(src_node_pvalue),
            "dst_node_pvalue": float(dst_node_pvalue),
            "src_node_activated": int(src_node_activated),
            "dst_node_activated": int(dst_node_activated),
            "normalized_score": float(normalized_score),
            "primary_activation": int(primary_activation),
            "activated": int(activated),
            "context_activation": int(context_activation),
            "retained_context_activation": int(retained_context_activation),
            "would_block": int(would_block),
            "truncated": int(truncated),
            "stage": prediction["stage"],
            "predicted_next_stage": prediction["predicted_next_stage"],
            "prediction_confidence": float(prediction["prediction_confidence"]),
            "risk": float(prediction["risk"]),
            "cut": int(prediction["cut"]),
            "cut_node": prediction["cut_node"],
            "reason": prediction["reason"],
            "model_latency_us": _as_float(event.get("model_latency_us"), 0.0),
        }
        decision["trace_latency_us"] = (time.perf_counter_ns() - started_ns) / 1000.0
        decision["end_to_end_latency_us"] = decision["model_latency_us"] + decision["trace_latency_us"]
        self.trace_latency_us.append(decision["trace_latency_us"])
        self._maybe_log_event(decision)
        return decision

    def _update_node_evidence(self, src: str, dst: str, loss: float) -> tuple[float, float, float]:
        evidence = max(0.0, (loss - self.node_evidence_center) / self.node_evidence_scale)
        pvalues = {}
        for node in {src, dst}:
            heap = self.node_evidence_heaps[node]
            _insert_topk(heap, evidence, self.node_gate_top_k)
            score = float(sum(heap))
            if self.calibration_node_scores:
                tail = len(self.calibration_node_scores) - bisect.bisect_left(
                    self.calibration_node_scores, score
                )
                pvalues[node] = (tail + 1.0) / (len(self.calibration_node_scores) + 1.0)
            else:
                pvalues[node] = 1.0
        return max(sum(self.node_evidence_heaps[src]), sum(self.node_evidence_heaps[dst])), pvalues[src], pvalues[dst]

    def close_time_window(self, time_window: int, time_interval: str) -> None:
        self.time_window_names[int(time_window)] = str(time_interval)
        self._event_log_file.flush()
        self._sample_memory()

    def close(self) -> None:
        if not self._event_log_file.closed:
            self._event_log_file.flush()
            self._event_log_file.close()

    def flush(self) -> OnlineDynamicSummary:
        self._sample_memory()
        self.close()
        graph_path = os.path.join(self.out_dir, "dynamic_graph.pkl")
        attack_graph_path = os.path.join(self.out_dir, "attack_subgraph.pkl")
        state_path = os.path.join(self.out_dir, "dynamic_state.json")
        results_path = os.path.join(self.out_dir, "results.pth")
        with open(graph_path, "wb") as file:
            pickle.dump(self.graph, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(attack_graph_path, "wb") as file:
            pickle.dump(self.attack_graph, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(state_path, "w", encoding="utf-8") as file:
            json.dump(self._state_dict(), file, indent=2, ensure_ascii=False)
        torch.save(self._legacy_tw_results(), results_path)
        summary = OnlineDynamicSummary(
            self.out_dir, graph_path, attack_graph_path, state_path, self.event_log_path,
            results_path, self.graph.number_of_nodes(), self.graph.number_of_edges(),
            self.attack_graph.number_of_nodes(), self.attack_graph.number_of_edges(), self.cut_events
        )
        log(
            "[OnlineDynamicTracing] Saved causal online trace artifacts "
            f"to {self.out_dir} (attack_nodes={summary.num_attack_nodes}, cuts={summary.num_cut_events})"
        )
        return summary

    def _conformal_pvalue(self, score: float) -> float:
        if not self.calibration_scores:
            return 1.0 if score < self.anomaly_threshold else 0.0
        first_ge = bisect.bisect_left(self.calibration_scores, float(score))
        num_ge = len(self.calibration_scores) - first_ge
        return (num_ge + 1.0) / (len(self.calibration_scores) + 1.0)

    def _sample_memory(self) -> None:
        if psutil is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes, psutil.Process(os.getpid()).memory_info().rss)
        if torch.cuda.is_available():
            self.peak_cuda_bytes = max(self.peak_cuda_bytes, int(torch.cuda.max_memory_allocated()))

    def _would_block(self, src: str, dst: str) -> bool:
        if self.containment_scope == "outgoing":
            return src in self.cut_nodes
        if self.containment_scope == "destination":
            return dst in self.cut_nodes
        return src in self.cut_nodes or dst in self.cut_nodes

    def _updated_score(self, current: float, new: float) -> float:
        if self.update_rule == "max":
            return max(current, new)
        if self.update_rule == "ema":
            return ((1.0 - self.ema_alpha) * current) + (self.ema_alpha * new)
        return new

    def _update_dynamic_graph(self, src, dst, edge_type, score, normalized_score, event_time,
                              srcmsg, dstmsg, time_window) -> None:
        self._update_node(self.graph, src, score, normalized_score, event_time, srcmsg, time_window)
        self._update_node(self.graph, dst, score, normalized_score, event_time, dstmsg, time_window)
        if (self.max_dynamic_edges <= 0 or self.graph.number_of_edges() < self.max_dynamic_edges
                or self.graph.has_edge(src, dst, key=edge_type)):
            self._update_edge(self.graph, src, dst, edge_type, score, normalized_score, event_time, time_window)
        else:
            self.dropped_dynamic_edges += 1

    def _update_attack_subgraph(self, src, dst, edge_type, score, normalized_score, event_time,
                                srcmsg, dstmsg, time_window, context_activation,
                                qualified_nodes: set[str]) -> None:
        missing_nodes = len({node for node in (src, dst) if node not in self.attack_graph})
        edge_exists = self.attack_graph.has_edge(src, dst, key=edge_type)
        node_budget_ok = self.max_attack_nodes <= 0 or self.attack_graph.number_of_nodes() + missing_nodes <= self.max_attack_nodes
        edge_budget_ok = self.max_attack_edges <= 0 or edge_exists or self.attack_graph.number_of_edges() < self.max_attack_edges
        if not node_budget_ok or not edge_budget_ok:
            self.dropped_attack_edges += 1
            return
        self._update_node(self.attack_graph, src, score, normalized_score, event_time, srcmsg, time_window)
        self._update_node(self.attack_graph, dst, score, normalized_score, event_time, dstmsg, time_window)
        self._update_edge(self.attack_graph, src, dst, edge_type, score, normalized_score, event_time, time_window)
        self.active_nodes.update(qualified_nodes)
        for node in qualified_nodes:
            self.active_window_by_node[node] = int(time_window)
            self.attack_graph.nodes[node]["attack_candidate"] = True
        self.tw_to_nodes[int(time_window)].update(qualified_nodes)
        stage = self._stage(edge_type, srcmsg, dstmsg)
        self.stage_counts[stage] += 1
        if context_activation:
            self.attack_graph.nodes[src]["contextual"] = True
            self.attack_graph.nodes[dst]["contextual"] = True

    def _update_node(self, graph, node, score, normalized_score, event_time, msg, time_window) -> None:
        if node not in graph:
            graph.add_node(node, score=float(score), normalized_score=float(normalized_score),
                           first_seen=int(event_time), last_seen=int(event_time), score_time=int(event_time),
                           score_tw=int(time_window), msg=msg, observations=1)
        else:
            attrs = graph.nodes[node]
            attrs["last_seen"] = max(_as_int(attrs.get("last_seen"), event_time), int(event_time))
            attrs["observations"] = _as_int(attrs.get("observations"), 0) + 1
            current = _as_float(attrs.get("score"), 0.0)
            updated = self._updated_score(current, score)
            if updated != current:
                attrs["score"] = float(updated)
                attrs["normalized_score"] = float(updated / self.anomaly_threshold)
                attrs["score_time"] = int(event_time)
                attrs["score_tw"] = int(time_window)
                if msg:
                    attrs["msg"] = msg
        self.node_scores[node] = self._updated_score(self.node_scores[node], float(score))

    def _update_edge(self, graph, src, dst, edge_type, score, normalized_score, event_time, time_window) -> None:
        edge_id = _edge_key(src, dst, edge_type)
        if graph.has_edge(src, dst, key=edge_type):
            attrs = graph[src][dst][edge_type]
            attrs["last_seen"] = max(_as_int(attrs.get("last_seen"), event_time), int(event_time))
            attrs["observations"] = _as_int(attrs.get("observations"), 0) + 1
            current = _as_float(attrs.get("weight"), 0.0)
            updated = self._updated_score(current, score)
            if updated != current:
                attrs.update(weight=float(updated), normalized_weight=float(updated / self.anomaly_threshold),
                             score_time=int(event_time), score_tw=int(time_window))
        else:
            graph.add_edge(src, dst, key=edge_type, edge_type=edge_type, weight=float(score),
                           normalized_weight=float(normalized_score), first_seen=int(event_time),
                           last_seen=int(event_time), score_time=int(event_time), score_tw=int(time_window),
                           observations=1)
        self.edge_scores[edge_id] = self._updated_score(self.edge_scores[edge_id], float(score))

    def _predict_and_maybe_cut(self, src, dst, edge_type, score, normalized_score, pvalue,
                               event_time, time_window, stage, predicted_next_stage,
                               prediction_confidence, src_node_pvalue=1.0,
                               dst_node_pvalue=1.0) -> Dict[str, Any]:
        local_degree = self.attack_graph.degree(src) + self.attack_graph.degree(dst)
        score_strength = min(normalized_score / max(self.cut_override_margin, 1e-6), 1.0)
        graph_support = min(local_degree / 8.0, 1.0)
        stage_severity = self.STAGE_SEVERITY.get(stage, 0.2)
        evidence_confidence = 1.0 - pvalue
        risk = (0.35 * score_strength) + (0.25 * evidence_confidence) + (0.20 * graph_support) + \
               (0.10 * stage_severity) + (0.10 * prediction_confidence)
        dangerous = stage in self.cutoff_stages or (
            predicted_next_stage in self.cutoff_stages
            and prediction_confidence >= self.min_prediction_confidence
        )
        if self.decision_policy == "conformal_risk":
            cut = (
                pvalue <= self.effective_cutoff_pvalue
                and local_degree >= self.min_attack_support
                and dangerous
            )
        else:
            cut = risk >= self.cutoff_risk_threshold and dangerous
        candidate_cut_node = (
            src if self.node_gate_enabled and src_node_pvalue < dst_node_pvalue
            else dst if self.node_gate_enabled and dst_node_pvalue < src_node_pvalue
            else self._select_cut_node(src, dst, edge_type, stage)
        )
        cut = bool(self.cutoff_enabled and cut and candidate_cut_node not in self.cut_nodes)
        cut_node = candidate_cut_node if cut else ""
        reason = ""
        if cut:
            criterion = (
                f"p<={self.effective_cutoff_pvalue:.6g}; support={local_degree}"
                if self.decision_policy == "conformal_risk"
                else f"risk>={self.cutoff_risk_threshold:.6g}"
            )
            reason = (
                f"policy={self.decision_policy}; {criterion}; target={self.cut_target_policy}; "
                f"predicted_next={predicted_next_stage}"
            )
            self.cut_nodes.add(cut_node)
            self.cut_time_by_node[cut_node] = int(event_time)
            self.cut_edges.add(_edge_key(src, dst, edge_type))
            self.tw_to_cut_nodes[int(time_window)].add(cut_node)
            self.cut_events += 1
            edge_attrs = self.attack_graph[src][dst][edge_type]
            edge_attrs.update(cut=True, cut_reason=reason, cut_time=int(event_time))
            self.attack_graph.nodes[cut_node].update(cut=True, cut_reason=reason, cut_time=int(event_time))
        return {"stage": stage, "predicted_next_stage": predicted_next_stage,
                "prediction_confidence": prediction_confidence, "risk": float(risk),
                "cut": cut, "cut_node": cut_node, "reason": reason}

    def _select_cut_node(self, src: str, dst: str, edge_type: str, stage: str) -> str:
        if self.cut_target_policy == "destination":
            return dst
        if self.cut_target_policy == "source":
            return src
        relation = edge_type.lower()
        if any(token in relation for token in ["read", "recv", "open", "mmap", "lseek", "execute", "clone"]):
            return dst
        if stage in {"payload_write", "c2_or_exfiltration", "persistence_or_lateral"}:
            return src
        return dst

    def _predict_next_stage(self, current_stage: str) -> tuple[str, float]:
        if not self.prediction_enabled:
            return "disabled", 0.0
        outgoing = {dst: count for (src, dst), count in self.stage_transitions.items() if src == current_stage}
        prior_stage = self.STATIC_NEXT_STAGE.get(current_stage, "unknown")
        candidates = set(self.STAGE_SEVERITY) | set(outgoing) | {prior_stage}
        smoothed = {
            stage: outgoing.get(stage, 0) + self.prediction_prior_strength + (0.75 if stage == prior_stage else 0.0)
            for stage in candidates
        }
        total = sum(smoothed.values())
        predicted, count = max(smoothed.items(), key=lambda item: (item[1], item[0]))
        return predicted, count / max(total, 1)

    def _record_stage_transition(self, anchor_node: str, current_stage: str) -> None:
        previous = self.last_stage_by_node.get(anchor_node)
        if previous is not None:
            self.stage_transitions[(previous, current_stage)] += 1
        self.last_stage_by_node[anchor_node] = current_stage

    def _stage(self, edge_type: str, srcmsg: str, dstmsg: str) -> str:
        text = f"{edge_type} {srcmsg} {dstmsg}".lower()
        if any(token in text for token in ["execute", "clone", "fork", "exec"]):
            return "execution"
        if any(token in text for token in ["connect", "send", "recv", "socket", "netflow"]):
            return "c2_or_exfiltration"
        if any(token in text for token in ["chmod", "load", "module", "setuid", "inject"]):
            return "persistence_or_lateral"
        if any(token in text for token in ["write", "rename", "unlink", "create"]):
            return "payload_write"
        if any(token in text for token in ["open", "read", "mmap", "lseek"]):
            return "initial_access"
        return "benign_or_context"

    def _maybe_log_event(self, decision: Dict[str, Any]) -> None:
        if self.max_logged_events > 0 and self._logged_events >= self.max_logged_events:
            return
        selected = self.log_all_events or any(
            decision[key] for key in ("activated", "context_activation", "would_block", "truncated", "cut")
        )
        if selected:
            self._event_writer.writerow({field: decision.get(field, "") for field in self.EVENT_LOG_FIELDS})
            self._logged_events += 1

    def _legacy_tw_results(self) -> Dict[int, Dict[str, Any]]:
        windows = set(self.time_window_names) | set(self.tw_to_nodes)
        return {
            int(tw): {"graph_dir": "", "subgraph_nodes": set(self.tw_to_nodes.get(tw, set())),
                      "cut_nodes": set(self.tw_to_cut_nodes.get(tw, set())),
                      "time_interval": self.time_window_names.get(int(tw), "")}
            for tw in windows
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
        return float(ordered[index])

    def _state_dict(self) -> Dict[str, Any]:
        top_nodes = sorted(self.node_scores.items(), key=lambda item: item[1], reverse=True)[:100]
        top_edges = sorted(self.edge_scores.items(), key=lambda item: item[1], reverse=True)[:100]
        return {
            "protocol_version": "causal-online-v2", "model_epoch": self.model_epoch_file,
            "split": self.split, "threshold_method": self.threshold_method,
            "threshold": float(self.anomaly_threshold), "calibration_size": len(self.calibration_scores),
            "node_gate_enabled": self.node_gate_enabled,
            "node_gate_alpha": self.node_gate_alpha,
            "node_gate_top_k": self.node_gate_top_k,
            "node_calibration_size": len(self.calibration_node_scores),
            "calibration_alpha": self.calibration_alpha, "context_alpha": self.context_alpha,
            "cutoff_pvalue": self.cutoff_pvalue, "decision_policy": self.decision_policy,
            "update_rule": self.update_rule, "containment_scope": self.containment_scope,
            "cut_target_policy": self.cut_target_policy,
            "prediction_prior_strength": self.prediction_prior_strength,
            "min_prediction_confidence": self.min_prediction_confidence,
            "total_events": self.total_events, "activated_events": self.activated_events,
            "context_events": self.context_events,
            "retained_context_events": self.retained_context_events,
            "would_block_events": self.would_block_events,
            "truncated_events": self.truncated_events, "cut_events": self.cut_events,
            "out_of_order_events": self.out_of_order_events,
            "dropped_dynamic_edges": self.dropped_dynamic_edges,
            "dropped_attack_edges": self.dropped_attack_edges,
            "num_nodes": self.graph.number_of_nodes(), "num_edges": self.graph.number_of_edges(),
            "num_attack_nodes": self.attack_graph.number_of_nodes(),
            "num_attack_edges": self.attack_graph.number_of_edges(),
            "active_nodes": len(self.active_nodes), "cut_nodes": sorted(self.cut_nodes),
            "cut_time_by_node": self.cut_time_by_node, "cut_edges": sorted(self.cut_edges),
            "stage_counts": dict(self.stage_counts),
            "stage_transitions": {f"{src}->{dst}": count for (src, dst), count in self.stage_transitions.items()},
            "trace_latency_us": {
                "mean": sum(self.trace_latency_us) / max(len(self.trace_latency_us), 1),
                "p50": self._percentile(self.trace_latency_us, 0.50),
                "p95": self._percentile(self.trace_latency_us, 0.95),
                "p99": self._percentile(self.trace_latency_us, 0.99),
            },
            "peak_process_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_bytes,
            "logged_events": self._logged_events, "time_windows": self.time_window_names,
            "configured_cutoff_pvalue": self.cutoff_pvalue,
            "minimum_conformal_pvalue": self.minimum_conformal_pvalue,
            "minimum_node_pvalue": self.minimum_node_pvalue,
            "effective_cutoff_pvalue": self.effective_cutoff_pvalue,
            "top_nodes": [{"node": node, "score": score} for node, score in top_nodes],
            "top_edges": [{"edge": edge, "score": score} for edge, score in top_edges],
        }


def summarize(cfg) -> Optional[Dict[str, Any]]:
    base_dir = getattr(cfg.attack_reconstruction.tracing, "_dynamic_tracing_dir",
                       os.path.join(cfg.attack_reconstruction.tracing._tracing_graph_dir, "online_dynamic"))
    if not os.path.isdir(base_dir):
        log(f"[OnlineDynamicTracing] No online dynamic artifacts found at {base_dir}")
        return None
    states = []
    for root, _, files in os.walk(base_dir):
        if "dynamic_state.json" in files:
            path = os.path.join(root, "dynamic_state.json")
            with open(path, "r", encoding="utf-8") as file:
                state = json.load(file)
            state["state_path"] = path
            states.append(state)
    if not states:
        log(f"[OnlineDynamicTracing] No dynamic_state.json files found at {base_dir}")
        return None
    aggregate = {
        "runs": len(states),
        "total_events": sum(int(state.get("total_events", 0)) for state in states),
        "activated_events": sum(int(state.get("activated_events", 0)) for state in states),
        "would_block_events": sum(int(state.get("would_block_events", 0)) for state in states),
        "truncated_events": sum(int(state.get("truncated_events", 0)) for state in states),
        "cut_events": sum(int(state.get("cut_events", 0)) for state in states),
        "max_attack_nodes": max(int(state.get("num_attack_nodes", 0)) for state in states),
        "max_attack_edges": max(int(state.get("num_attack_edges", 0)) for state in states),
    }
    out_path = os.path.join(base_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2, ensure_ascii=False)
    log(f"[OnlineDynamicTracing] Summary saved to {out_path}: {aggregate}")
    return aggregate
