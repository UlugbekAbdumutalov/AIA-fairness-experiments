"""CPU-only metrics tests using fabricated labels/scores."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import score_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_ranking_and_zero_gap(self):
        # Female label is 0, male label is 1.
        y = [0, 0, 1, 1]
        logits = np.array([[3, 0], [2, 0], [0, 3], [0, 2]])
        m = score_metrics(y, logits)
        self.assertAlmostEqual(m['auc'], 1)
        self.assertAlmostEqual(m['male_accuracy'], 1)
        self.assertAlmostEqual(m['female_accuracy'], 1)
        self.assertAlmostEqual(m['signed_gap'], 0)

    def test_fixed_male_prediction_big_gap(self):
        y = [0, 0, 1, 1]
        logits = np.array([[0, 2], [0, 2], [0, 2], [0, 2]])
        m = score_metrics(y, logits)
        self.assertAlmostEqual(m['auc'], .5)
        self.assertAlmostEqual(m['male_accuracy'], 1)
        self.assertAlmostEqual(m['female_accuracy'], 0)
        self.assertAlmostEqual(m['signed_gap'], 1)


if __name__ == '__main__':
    unittest.main()
