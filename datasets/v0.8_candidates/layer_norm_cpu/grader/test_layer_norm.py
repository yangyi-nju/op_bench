"""Behavioral candidate grader against the complete rebuilt CPU Torch package.

The loader is public image infrastructure. No reference patch, source matching,
function extraction, wheel overlay or repair-path restriction is used here.
"""

from pathlib import Path
import sys
import unittest


WORKSPACE = Path(sys.argv[1]).resolve()
SELECTOR = sys.argv[2]
sys.argv = [sys.argv[0], SELECTOR]
sys.path.insert(0, "/opt/opbench-layernorm")
from source_torch import load

torch = load(WORKSPACE)
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode

def eager_and_fake(x, shape, weight, bias, eps):
    eager = torch.ops.aten.native_layer_norm.default(x, list(shape), weight, bias, eps)
    with enable_python_dispatcher(), FakeTensorMode() as mode:
        fake = torch.ops.aten.native_layer_norm.default(
            mode.from_tensor(x), list(shape),
            mode.from_tensor(weight) if weight is not None else None,
            mode.from_tensor(bias) if bias is not None else None, eps,
        )
    return eager, fake


class CpuBFloat16Metadata(unittest.TestCase):
    def test_public_dtype_and_shape_contract(self):
        torch.manual_seed(123)
        for input_shape, normalized_shape in (((1, 2, 3), (1, 2, 3)),
                                               ((2, 3, 4), (4,)),
                                               ((2, 3, 4), (3, 4))):
            for affine in ("both", "weight", "bias", "none"):
                with self.subTest(input_shape=input_shape, normalized_shape=normalized_shape, affine=affine):
                    x = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
                    weight = torch.randn(normalized_shape, dtype=torch.bfloat16, requires_grad=True) if affine in {"both", "weight"} else None
                    bias = torch.randn(normalized_shape, dtype=torch.bfloat16, requires_grad=True) if affine in {"both", "bias"} else None
                    eager, fake = eager_and_fake(x, normalized_shape, weight, bias, 0.5)
                    self.assertEqual(3, len(fake))
                    expected_stat_shape = input_shape[:-len(normalized_shape)] + (1,) * len(normalized_shape)
                    for index, (real, symbolic) in enumerate(zip(eager, fake)):
                        self.assertEqual(torch.bfloat16, real.dtype)
                        self.assertEqual(real.dtype, symbolic.dtype)
                        self.assertEqual(input_shape if index == 0 else expected_stat_shape, tuple(symbolic.shape))
                        self.assertEqual(tuple(real.shape), tuple(symbolic.shape))
                        self.assertEqual("cpu", symbolic.device.type)
        x = torch.randn(2, 3, 4, dtype=torch.bfloat16).transpose(0, 1)
        eager, fake = eager_and_fake(x, (4,), None, None, 0.125)
        for real, symbolic in zip(eager, fake):
            self.assertEqual(torch.bfloat16, real.dtype)
            self.assertEqual(real.dtype, symbolic.dtype)
            self.assertEqual(tuple(real.shape), tuple(symbolic.shape))
            self.assertEqual("cpu", symbolic.device.type)


class LayerNormRegressions(unittest.TestCase):
    def test_float32_fake_metadata(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        eager, fake = eager_and_fake(x, (3, 4), None, None, 0.25)
        for real, symbolic in zip(eager, fake):
            self.assertEqual(torch.float32, real.dtype)
            self.assertEqual(real.dtype, symbolic.dtype)
            self.assertEqual(tuple(real.shape), tuple(symbolic.shape))
            self.assertEqual("cpu", symbolic.device.type)

    def test_eager_numerical_formula(self):
        # A double precision formula is an independent numerical oracle. Bfloat16
        # tolerance permits normal quantization and float32 accumulation error.
        for dtype, rtol, atol in ((torch.bfloat16, 0.03125, 0.03125),
                                  (torch.float32, 2e-5, 2e-6),
                                  (torch.float64, 1e-10, 1e-12)):
            for normalized_shape in ((4,), (3, 4)):
                for affine in ("both", "weight", "bias", "none"):
                    for eps in (1e-5, 0.2):
                        with self.subTest(dtype=dtype, normalized_shape=normalized_shape, affine=affine, eps=eps):
                            torch.manual_seed(47)
                            x = torch.randn(2, 3, 4).clamp(-3, 3).to(dtype)
                            weight = torch.randn(normalized_shape).clamp(-2, 2).to(dtype) if affine in {"both", "weight"} else None
                            bias = torch.randn(normalized_shape).clamp(-2, 2).to(dtype) if affine in {"both", "bias"} else None
                            actual = torch.ops.aten.native_layer_norm.default(x, list(normalized_shape), weight, bias, eps)
                            dims = tuple(range(-len(normalized_shape), 0))
                            precise = x.double()
                            mean = precise.mean(dims, keepdim=True)
                            rstd = ((precise - mean).square().mean(dims, keepdim=True) + eps).rsqrt()
                            out = (precise - mean) * rstd
                            if weight is not None:
                                out = out * weight.double()
                            if bias is not None:
                                out = out + bias.double()
                            for observed, expected in zip(actual, (out, mean, rstd)):
                                self.assertEqual(dtype, observed.dtype)
                                torch.testing.assert_close(observed, expected.to(dtype), rtol=rtol, atol=atol)

    def test_float64_input_and_affine_gradients(self):
        torch.manual_seed(53)
        x = torch.randn(2, 3, 4, dtype=torch.float64).clamp(-3, 3).requires_grad_()
        weight = torch.randn(3, 4, dtype=torch.float64).clamp(-2, 2).requires_grad_()
        bias = torch.randn(3, 4, dtype=torch.float64).clamp(-2, 2).requires_grad_()
        out, _, _ = torch.ops.aten.native_layer_norm.default(x, [3, 4], weight, bias, 0.2)
        mean = x.mean((-2, -1), keepdim=True)
        rstd = ((x - mean).square().mean((-2, -1), keepdim=True) + 0.2).rsqrt()
        expected_out = (x - mean) * rstd * weight + bias
        actual_grads = torch.autograd.grad(out.square().sum(), (x, weight, bias), retain_graph=True)
        expected_grads = torch.autograd.grad(expected_out.square().sum(), (x, weight, bias))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
