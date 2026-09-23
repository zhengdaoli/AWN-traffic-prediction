import unittest

import torch

from util import metrics


class MetricsTest(unittest.TestCase):
    def test_unmasked_metrics(self):
        result = metrics(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(result["mae"].item(), 1.5)
        self.assertAlmostEqual(result["rmse"].item(), (2.5) ** 0.5)
        self.assertAlmostEqual(result["mape"].item(), 200.0)

    def test_masked_metrics(self):
        result = metrics(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]), 0.0)
        self.assertAlmostEqual(result["mae"].item(), 2.0)
        self.assertAlmostEqual(result["rmse"].item(), 2.0)
        self.assertAlmostEqual(result["mape"].item(), 200.0)


if __name__ == "__main__":
    unittest.main()
