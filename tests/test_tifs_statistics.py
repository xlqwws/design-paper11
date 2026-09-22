import unittest

from experiments.aggregate_tifs_results import holm_adjust


class TifsStatisticsTests(unittest.TestCase):
    def test_holm_adjustment_preserves_order_and_monotonicity(self):
        adjusted = holm_adjust([0.04, 0.01, 0.03, 0.20])
        self.assertEqual(len(adjusted), 4)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.09)
        self.assertAlmostEqual(adjusted[0], 0.09)
        self.assertAlmostEqual(adjusted[3], 0.20)


if __name__ == "__main__":
    unittest.main()
