import unittest

import numpy as np

from experiments.evaluate_online_node_conformal import aggregate_nodes, insert_topk


class OnlineNodeConformalTests(unittest.TestCase):
    def test_topk_is_bounded_and_monotone(self):
        heap = []
        totals = []
        for value in [1.0, 3.0, 2.0, 0.5, 4.0]:
            insert_topk(heap, value, 3)
            totals.append(sum(heap))
        self.assertLessEqual(len(heap), 3)
        self.assertEqual(sorted(heap), [2.0, 3.0, 4.0])
        self.assertTrue(all(right >= left for left, right in zip(totals, totals[1:])))

    def test_aggregation_uses_both_endpoints_once(self):
        events = [(4.0, 1, 2, 10), (3.0, 1, 1, 11)]
        scores, first_seen = aggregate_nodes(events, np.asarray([0.0, 1.0, 2.0]), 2)
        self.assertGreater(scores[1], scores[2])
        self.assertEqual(first_seen, {1: 10, 2: 10})

    def test_robust_excess_preserves_out_of_range_magnitude(self):
        calibration = np.asarray([0.0, 1.0, 2.0, 3.0])
        low, _ = aggregate_nodes([(4.0, 1, 2, 10)], calibration, 1, "robust_excess")
        high, _ = aggregate_nodes([(8.0, 1, 2, 10)], calibration, 1, "robust_excess")
        self.assertGreater(high[1], low[1])


if __name__ == "__main__":
    unittest.main()
