import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.tail_calibration import EmpiricalGPDCalibrator


class TailCalibrationTests(unittest.TestCase):
    def test_tail_pvalues_preserve_extreme_order(self):
        calibration = np.linspace(0.0, 10.0, 200)
        model = EmpiricalGPDCalibrator(tail_fraction=0.1).fit(calibration)
        pvalues = model.pvalues([11.0, 15.0])
        self.assertGreater(pvalues[0], pvalues[1])
        self.assertTrue(np.all((pvalues > 0.0) & (pvalues <= 1.0)))


if __name__ == "__main__":
    unittest.main()
