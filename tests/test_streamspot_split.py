import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_streamspot_temporal import split_for_graph  # noqa: E402


class StreamSpotSplitTests(unittest.TestCase):
    def test_every_benign_scenario_contributes_to_each_split(self):
        for start in [0, 100, 200, 400, 500]:
            self.assertEqual(split_for_graph(start), "train")
            self.assertEqual(split_for_graph(start + 60), "val")
            self.assertEqual(split_for_graph(start + 80), "test")

    def test_attack_scenario_is_test_only(self):
        self.assertTrue(all(split_for_graph(graph_id) == "test" for graph_id in range(300, 400)))


if __name__ == "__main__":
    unittest.main()
