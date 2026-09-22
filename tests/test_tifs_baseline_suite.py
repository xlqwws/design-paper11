import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baseline_suite", ROOT / "experiments" / "run_tifs_baseline_suite.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BaselineSuiteTests(unittest.TestCase):
    def test_conformal_predictions_are_monotone(self):
        calibration = np.asarray([1.0, 2.0, 3.0, 4.0])
        scores = np.asarray([0.0, 2.5, 5.0])
        predictions = MODULE.conformal_predictions(calibration, scores, 0.21)
        self.assertEqual(predictions.tolist(), [0, 0, 1])

    def test_graph_split_has_attack_test_only(self):
        train, validation, test, labels = MODULE.graph_split("StreamSpot", 3)
        self.assertEqual(len(set(train) & set(validation)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(int(labels.sum()), 100)
        self.assertTrue(np.all(test[labels == 1] >= 300))
        self.assertTrue(np.all(test[labels == 1] <= 399))

    def test_cached_feature_shapes(self):
        streamspot, streamspot_views = MODULE.graph_features("StreamSpot")
        unicorn, unicorn_views = MODULE.graph_features("Unicorn-Wget")
        self.assertEqual(streamspot.shape[0], 600)
        self.assertEqual(unicorn.shape[0], 175)
        self.assertEqual(streamspot_views[-1][1], streamspot.shape[1])
        self.assertEqual(unicorn_views[-1][1], unicorn.shape[1])


if __name__ == "__main__":
    unittest.main()
