import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate_online_tracing import (  # noqa: E402
    attack_detection_precision,
    attach_ground_truth,
    containment_metrics,
    load_events,
    node_metrics,
)


class OnlineEvaluationTests(unittest.TestCase):
    def test_node_ground_truth_and_collateral_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_path = root / "events.csv"
            truth_path = root / "truth.csv"
            pd.DataFrame([
                {"event_id": 0, "time": 1, "srcnode": 1, "dstnode": 2, "edge_type": "READ", "loss": 1.0, "activated": 0, "would_block": 0, "cut": 0},
                {"event_id": 1, "time": 2, "srcnode": 2, "dstnode": 3, "edge_type": "EXEC", "loss": 9.0, "activated": 1, "would_block": 0, "cut": 1, "cut_node": 3},
                {"event_id": 2, "time": 3, "srcnode": 3, "dstnode": 4, "edge_type": "SEND", "loss": 8.0, "activated": 1, "would_block": 1, "cut": 0},
                {"event_id": 3, "time": 4, "srcnode": 3, "dstnode": 5, "edge_type": "READ", "loss": 0.2, "activated": 0, "would_block": 1, "cut": 0},
            ]).to_csv(events_path, index=False)
            pd.DataFrame([
                {"node_id": 3, "label": 1}, {"node_id": 1, "label": 0},
                {"node_id": 2, "label": 0}, {"node_id": 4, "label": 0},
                {"node_id": 5, "label": 0},
            ]).to_csv(truth_path, index=False)

            events = attach_ground_truth(load_events([events_path]), truth_path)
            attribution = node_metrics(events)
            containment = containment_metrics(events)
            self.assertEqual(attribution["true_reported_nodes"], 1)
            self.assertFalse(containment["available"])
            self.assertIn("event-level ground truth", containment["reason"])

    def test_event_ground_truth_enables_adp_and_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_path = root / "events.csv"
            truth_path = root / "truth.csv"
            pd.DataFrame([
                {"event_id": 0, "time": 1, "srcnode": 1, "dstnode": 2, "edge_type": "READ", "loss": 0.1, "activated": 0, "would_block": 0, "cut": 0, "predicted_next_stage": "execution"},
                {"event_id": 1, "time": 2, "srcnode": 2, "dstnode": 3, "edge_type": "EXEC", "loss": 9.0, "activated": 1, "would_block": 0, "cut": 1, "cut_node": 3, "predicted_next_stage": "c2_or_exfiltration"},
                {"event_id": 2, "time": 3, "srcnode": 3, "dstnode": 4, "edge_type": "SEND", "loss": 8.0, "activated": 1, "would_block": 1, "cut": 0, "predicted_next_stage": "c2_or_exfiltration"},
            ]).to_csv(events_path, index=False)
            pd.DataFrame([
                {"event_id": 0, "label": 0, "campaign_id": "", "gt_stage": ""},
                {"event_id": 1, "label": 1, "campaign_id": "A", "gt_stage": "execution"},
                {"event_id": 2, "label": 1, "campaign_id": "A", "gt_stage": "c2_or_exfiltration"},
            ]).to_csv(truth_path, index=False)

            events = attach_ground_truth(load_events([events_path]), truth_path)
            self.assertTrue(containment_metrics(events)["available"])
            self.assertTrue(attack_detection_precision(events)["available"])


if __name__ == "__main__":
    unittest.main()
