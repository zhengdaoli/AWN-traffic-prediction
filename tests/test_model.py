import unittest

import numpy as np
import torch

from model import AWN


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    def test_contiguous_forward_and_attention(self):
        model = AWN(self.adjacency, history_length=4, prediction_length=3, scales=[0.1, 0.3], rank=2, graph_layers=2)
        inputs = torch.randn(2, 4, 3, 1)
        output, attention = model(inputs, return_attention=True)
        self.assertEqual(tuple(output.shape), (2, 3, 3, 1))
        self.assertEqual(tuple(attention.shape), (2, 2, 4, 3, 2))
        self.assertTrue(torch.allclose(attention.sum(dim=-1), torch.ones(2, 2, 4, 3), atol=1e-6))
        output.mean().backward()
        self.assertIsNotNone(model.low_rank_left.grad)
        self.assertIsNotNone(model.low_rank_right.grad)

    def test_shared_low_rank_correction(self):
        model = AWN(self.adjacency, history_length=4, prediction_length=3, scales=[0.1, 0.3], rank=2, graph_layers=2)
        correction = model.low_rank_left @ model.low_rank_right
        self.assertLessEqual(torch.linalg.matrix_rank(correction).item(), 2)
        self.assertFalse(any("krandw" in name for name, _ in model.named_parameters()))

    def test_directed_forward(self):
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        model = AWN(adjacency, history_length=4, prediction_length=3, scales=[0.1], rank=2, graph_layers=1, directed=True)
        output = model(torch.randn(2, 4, 3, 1))
        self.assertEqual(tuple(output.shape), (2, 3, 3, 1))


if __name__ == "__main__":
    unittest.main()
