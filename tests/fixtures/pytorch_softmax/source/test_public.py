"""Public reproduction: run `python test_public.py -v` in the workspace."""

import unittest

import torch

from operator_impl import stable_softmax


class PublicSoftmaxTests(unittest.TestCase):
    def test_normal_input(self):
        value = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        actual = stable_softmax(value)
        torch.testing.assert_close(actual, torch.softmax(value, dim=-1))
        self.assertEqual(actual.shape, value.shape)
        self.assertEqual(actual.dtype, value.dtype)
        self.assertEqual(actual.device, value.device)

    def test_large_finite_input(self):
        value = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float32)
        actual = stable_softmax(value, dim=-1)
        self.assertTrue(
            torch.isfinite(actual).all().item(),
            f"Finite inputs must produce finite probabilities; got {actual}",
        )
        torch.testing.assert_close(actual, torch.softmax(value, dim=-1))
        torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
