import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_graph_stream import finite_sample_threshold  # noqa: E402
from detection.conformal import (  # noqa: E402
    conformal_predictions, upper_tail_pvalues, zero_exceedance_predictions,
)


class GraphConformalTests(unittest.TestCase):
    def test_zero_exceedance_is_strict_at_calibration_maximum(self):
        predicted = zero_exceedance_predictions([1.0, 2.0, 3.0], [2.9, 3.0, 3.1])
        self.assertEqual(predicted.tolist(), [0, 0, 1])

    def test_finite_sample_graph_rank(self):
        self.assertEqual(finite_sample_threshold(range(1, 61), 0.05), 58.0)

    def test_rejects_empty_calibration(self):
        with self.assertRaises(ValueError):
            finite_sample_threshold([], 0.05)

    def test_ties_do_not_trigger_every_graph(self):
        pvalues = upper_tail_pvalues([0.0] * 100, [0.0, 1.0])
        self.assertEqual(pvalues.tolist(), [1.0, 1.0 / 101.0])
        self.assertEqual(conformal_predictions([0.0] * 100, [0.0, 1.0], 0.05).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
