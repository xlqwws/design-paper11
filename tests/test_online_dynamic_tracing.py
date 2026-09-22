import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attack_reconstruction.online_dynamic_tracing import (  # noqa: E402
    OnlineDynamicAttackTracer,
    conformal_threshold,
)


def make_cfg(root):
    dynamic = SimpleNamespace(
        threshold_method="conformal", calibration_alpha=0.3, context_alpha=1.0,
        cutoff_pvalue=0.3, activation_margin=1.0, context_margin=0.65,
        retained_evidence_margin=1.0, retained_event_floor=0.2, context_horizon_windows=4,
        cutoff_risk_threshold=0.0, cut_override_margin=1.5,
        decision_policy="conformal_risk", update_rule="max", ema_alpha=0.2,
        context_enabled=True, prediction_enabled=True, cutoff_enabled=True,
        containment_scope="incident", cut_target_policy="causal_actor", min_attack_support=0,
        prediction_prior_strength=0.25, min_prediction_confidence=0.0,
        cutoff_stages="execution,persistence_or_lateral,c2_or_exfiltration",
        max_logged_events=0, max_dynamic_edges=0, max_attack_nodes=100,
        max_attack_edges=100, log_all_events=True,
    )
    tracing = SimpleNamespace(
        used_method="online_dynamic", online_dynamic=dynamic,
        _dynamic_tracing_dir=str(root), _tracing_graph_dir=str(root),
    )
    return SimpleNamespace(attack_reconstruction=SimpleNamespace(tracing=tracing))


class OnlineDynamicTracerTests(unittest.TestCase):
    def test_conformal_threshold_uses_finite_sample_rank(self):
        self.assertEqual(conformal_threshold(range(1, 101), 0.1), 91.0)

    def test_monotonic_multirelation_and_counterfactual_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            tracer = OnlineDynamicAttackTracer(
                make_cfg(directory), "epoch", "test", anomaly_threshold=3.0,
                calibration_scores=[1.0, 2.0, 3.0],
            )
            first = tracer.observe_edge({
                "srcnode": 1, "dstnode": 2, "edge_type": "WRITE",
                "loss": 10.0, "time": 1, "srcmsg": "process", "dstmsg": "payload",
            }, 0)
            second = tracer.observe_edge({
                "srcnode": 1, "dstnode": 2, "edge_type": "READ",
                "loss": 3.2, "time": 2, "srcmsg": "process", "dstmsg": "payload",
            }, 0)
            summary = tracer.flush()

            self.assertEqual(first["cut"], 1)
            self.assertEqual(second["would_block"], 1)
            self.assertEqual(second["truncated"], 1)
            self.assertEqual(tracer.graph.nodes["1"]["score"], 10.0)
            self.assertTrue(tracer.graph.has_edge("1", "2", key="WRITE"))
            self.assertTrue(tracer.graph.has_edge("1", "2", key="READ"))
            self.assertEqual(tracer.cut_events, 1)
            with open(summary.event_log_path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertIn("trace_latency_us", rows[0])

    def test_max_retention_changes_future_context_but_latest_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            max_cfg = make_cfg(Path(directory) / "max")
            latest_cfg = make_cfg(Path(directory) / "latest")
            max_cfg.attack_reconstruction.tracing.online_dynamic.context_alpha = 0.2
            latest_cfg.attack_reconstruction.tracing.online_dynamic.context_alpha = 0.2
            max_cfg.attack_reconstruction.tracing.online_dynamic.cutoff_enabled = False
            latest_cfg.attack_reconstruction.tracing.online_dynamic.cutoff_enabled = False
            latest_cfg.attack_reconstruction.tracing.online_dynamic.update_rule = "latest"
            max_tracer = OnlineDynamicAttackTracer(
                max_cfg, "epoch", "test", 3.0, calibration_scores=[1, 2, 3]
            )
            latest_tracer = OnlineDynamicAttackTracer(
                latest_cfg, "epoch", "test", 3.0, calibration_scores=[1, 2, 3]
            )
            stream = [
                {"srcnode": 1, "dstnode": 2, "edge_type": "WRITE", "loss": 10.0, "time": 1},
                {"srcnode": 1, "dstnode": 3, "edge_type": "READ", "loss": 0.7, "time": 2},
                {"srcnode": 1, "dstnode": 4, "edge_type": "READ", "loss": 0.6, "time": 3},
            ]
            max_decisions = [max_tracer.observe_edge(event, 0) for event in stream]
            latest_decisions = [latest_tracer.observe_edge(event, 0) for event in stream]
            max_tracer.close()
            latest_tracer.close()
            self.assertEqual(max_decisions[-1]["retained_context_activation"], 1)
            self.assertEqual(latest_decisions[-1]["retained_context_activation"], 0)
            self.assertEqual(max_decisions[-1]["activated"], 1)
            self.assertEqual(latest_decisions[-1]["activated"], 0)

    def test_node_gate_marks_runtime_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_cfg(directory)
            dynamic = cfg.attack_reconstruction.tracing.online_dynamic
            dynamic.node_gate_enabled = True
            dynamic.node_gate_alpha = 0.3
            dynamic.node_gate_top_k = 3
            dynamic.cutoff_enabled = False
            tracer = OnlineDynamicAttackTracer(
                cfg, "epoch", "test", 3.0,
                calibration_scores=[0.0, 1.0, 2.0, 3.0],
                calibration_node_scores=[1.0, 2.0, 3.0],
            )
            decision = tracer.observe_edge(
                {"srcnode": 1, "dstnode": 2, "edge_type": "WRITE", "loss": 10.0, "time": 1}, 0
            )
            tracer.close()
            self.assertEqual(decision["src_node_activated"], 1)
            self.assertEqual(decision["dst_node_activated"], 1)
            self.assertEqual(tracer.active_nodes, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
