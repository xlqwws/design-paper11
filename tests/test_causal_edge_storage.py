import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import TemporalData


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import build_causal_full_data  # noqa: E402


def graph(value):
    return TemporalData(
        src=torch.tensor([0]), dst=torch.tensor([1]), t=torch.tensor([value]),
        msg=torch.tensor([[float(value)]]), edge_type=torch.tensor([[float(value)]]),
    )


class CausalEdgeStorageTests(unittest.TestCase):
    def test_test_storage_skips_validation_features(self):
        train = [graph(10), graph(11)]
        validation = [graph(20)]
        test = [graph(30), graph(31)]

        val_storage = build_causal_full_data(train, validation)
        test_storage = build_causal_full_data(train, test)

        self.assertEqual(val_storage.t.tolist(), [10, 11, 20])
        self.assertEqual(test_storage.t.tolist(), [10, 11, 30, 31])
        self.assertNotIn(20, test_storage.t.tolist())


if __name__ == "__main__":
    unittest.main()
