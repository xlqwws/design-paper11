import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import TemporalData


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.causal_graph_sketch import extract_causal_graph_sketch  # noqa: E402


class CausalGraphSketchTests(unittest.TestCase):
    def test_prefix_does_not_use_future_event(self):
        data = TemporalData(
            src=torch.tensor([0, 1]), dst=torch.tensor([1, 2]), t=torch.tensor([1, 2]),
            edge_type=torch.tensor([0, 1]),
            msg=torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0]], dtype=torch.float),
        )
        prefix = extract_causal_graph_sketch(data, 2, 2, max_events=1)
        first_only = TemporalData(
            src=data.src[:1], dst=data.dst[:1], t=data.t[:1],
            edge_type=data.edge_type[:1], msg=data.msg[:1],
        )
        expected = extract_causal_graph_sketch(first_only, 2, 2)
        self.assertTrue(torch.equal(torch.from_numpy(prefix), torch.from_numpy(expected)))


if __name__ == "__main__":
    unittest.main()
