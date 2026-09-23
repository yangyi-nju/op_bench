"""Evaluator-side behavior tests; the source workspace is supplied explicitly."""

import importlib
from pathlib import Path
import sys
import unittest

import torch


class SoftmaxTests(unittest.TestCase):
    def assert_softmax_contract(self, value, dim=-1):
        before = value.detach().clone()
        actual = stable_softmax(value, dim=dim)
        expected = torch.softmax(value, dim=dim)
        self.assertEqual(actual.shape, value.shape)
        self.assertEqual(actual.dtype, value.dtype)
        self.assertEqual(actual.device, value.device)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())
        tolerance = 1e-12 if value.dtype == torch.float64 else 2e-6
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            actual.sum(dim=dim),
            torch.ones_like(expected.sum(dim=dim)),
            rtol=tolerance,
            atol=tolerance,
        )
        torch.testing.assert_close(value.detach(), before, rtol=0, atol=0)
        return actual

    def assert_gradient_contract(self, value, weights, dim):
        actual = self.assert_softmax_contract(value, dim)
        gradient = torch.autograd.grad((actual * weights).sum(), value)[0]
        expected = torch.softmax(value.detach(), dim=dim)
        # Analytic vector-Jacobian product, independent of the candidate's graph.
        expected_gradient = expected * (weights - (weights * expected).sum(dim=dim, keepdim=True))
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertEqual(gradient.shape, value.shape)
        self.assertEqual(gradient.dtype, value.dtype)
        self.assertEqual(gradient.device, value.device)
        rtol, atol = (1e-10, 1e-12) if value.dtype == torch.float64 else (2e-6, 2e-6)
        torch.testing.assert_close(gradient, expected_gradient, rtol=rtol, atol=atol)

    def test_large_values(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                value = torch.tensor(
                    [[10000.0, 10001.0, 9999.0], [-10000.0, -9999.0, -10001.0]],
                    dtype=dtype,
                )
                self.assert_softmax_contract(value)

    def test_large_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                value = torch.tensor(
                    [[1000.0, 1002.0, 999.0], [2000.0, 2001.0, 1998.0]],
                    dtype=dtype,
                    requires_grad=True,
                )
                weights = torch.tensor([[0.5, -2.0, 3.0], [2.0, 0.25, -1.0]], dtype=dtype)
                self.assert_gradient_contract(value, weights, dim=-1)

    def test_large_nonlast_axis(self):
        value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1000.0
        self.assert_softmax_contract(value, dim=0)
        self.assert_softmax_contract(value.transpose(0, 2), dim=-2)

    def test_normal_values(self):
        value = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.0], dtype=torch.float32)
        self.assert_softmax_contract(value)

    def test_two_dimensional_nonlast_axis(self):
        value = torch.tensor([[0.2, -0.1, 1.0], [1.2, 0.3, 0.0]], dtype=torch.float32)
        self.assert_softmax_contract(value, dim=0)
        self.assert_softmax_contract(value.t(), dim=1)

    def test_float64_precision(self):
        value = torch.tensor([[0.0, 1e-8, 2e-8], [0.3, -0.7, 1.1]], dtype=torch.float64)
        self.assert_softmax_contract(value, dim=1)

    def test_legal_dimensions(self):
        value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 2) / 10.0
        for dim in (0, 1, 2, -1, -2, -3):
            with self.subTest(dim=dim):
                self.assert_softmax_contract(value, dim=dim)

    def test_normal_gradient(self):
        value = torch.tensor([[0.1, -0.3, 0.7], [0.2, 0.4, -0.8]], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([[0.4, -1.0, 2.0], [-0.5, 3.0, 1.0]], dtype=torch.float64)
        self.assert_gradient_contract(value, weights, dim=0)


def load_workspace_operator(workspace):
    workspace = Path(workspace).resolve()
    sys.path.insert(0, str(workspace))
    module = importlib.import_module("operator_impl")
    if not Path(module.__file__).resolve().is_relative_to(workspace):
        raise RuntimeError("operator_impl was not imported from the evaluated workspace")
    return module.stable_softmax


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: test_softmax.py WORKSPACE [SoftmaxTests.test_name] [-v]")
    stable_softmax = load_workspace_operator(sys.argv.pop(1))
    unittest.main()
