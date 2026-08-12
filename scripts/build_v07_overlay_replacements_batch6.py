#!/usr/bin/env python3
"""Build two exact-Base, offline Python-overlay replacements for v0.7."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for selected in (SRC, SCRIPTS):
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))

import build_v07_replacement_tasks as replacement_builder  # noqa: E402
from op_bench.runtime.canonical import canonical_json  # noqa: E402


def _new_file_patch(path: str, body: str) -> str:
    normalized = body.strip("\n") + "\n"
    lines = normalized.splitlines()
    additions = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{additions}\n"
    )


GOLD_186993 = r"""diff --git a/torch/export/_unlift.py b/torch/export/_unlift.py
--- a/torch/export/_unlift.py
+++ b/torch/export/_unlift.py
@@ -199,7 +199,15 @@ def _convert_guards_code_to_fn(
         # printing guards code may potentially introduce redundant parens;
         # we can normalize them out for readability by parsing/unparsing
         # NOTE: this is not necessary for correctness, just deemed desirable
-        _shadow = ast.unparse(ast.parse(shadow, mode="eval"))
+        try:
+            _shadow = ast.unparse(ast.parse(shadow, mode="eval"))
+        except RecursionError as e:
+            # A deeply nested guard expression (e.g. a sum over many symbolic
+            # sizes) can exceed the recursion limit in ast.parse/ast.unparse.
+            # This normalization only affects the assert error message, so fall
+            # back to the un-normalized guard string instead of crashing.
+            warnings.warn(f"ast.unparse failed for guard expression: {e}", stacklevel=2)
+            _shadow = shadow
         # actual code and shadow error message
         code_str += f'  torch._assert({actual}, "Guard failed: {_shadow}")\n'
     code_str += "  return\n"
"""


HIDDEN_186993_BODY = r'''
import unittest
from unittest.mock import patch
import warnings

from torch.export._unlift import _convert_guards_code_to_fn


class LargeGuardMaterializationTests(unittest.TestCase):
    def test_codegen_survives_pretty_print_recursion(self):
        with patch("ast.unparse", side_effect=RecursionError("deep expression")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                guards = _convert_guards_code_to_fn(["args[0] > 0"], [])

        self.assertIsNotNone(guards)
        guards(3)
        self.assertTrue(any("guard expression" in str(item.message) for item in caught))

    def test_multiple_runtime_guards_survive_pretty_print_recursion(self):
        expressions = ["args[0] > 0", "args[1] < 10"]
        with patch("ast.unparse", side_effect=RecursionError("deep expression")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                guards = _convert_guards_code_to_fn(expressions, [])

        guards(2, 7)
        with self.assertRaisesRegex(AssertionError, "args\\[1\\] < 10"):
            guards(2, 12)

    def test_normal_guard_keeps_validation_semantics(self):
        guards = _convert_guards_code_to_fn(["args[0] % 2 == 0"], [])
        guards(4)
        with self.assertRaisesRegex(AssertionError, "Guard failed"):
            guards(3)

    def test_empty_guard_set_remains_callable(self):
        guards = _convert_guards_code_to_fn([], [])
        self.assertIsNone(guards("unused"))


if __name__ == "__main__":
    unittest.main()
'''


GOLD_188278 = """diff --git a/torch/fx/experimental/proxy_tensor.py b/torch/fx/experimental/proxy_tensor.py
--- a/torch/fx/experimental/proxy_tensor.py
+++ b/torch/fx/experimental/proxy_tensor.py
@@ -573,6 +573,13 @@ def _sympy_handlers() -> dict[type[sympy.Expr], Callable[..., Any]]:
             handlers[k] = _nary_sym_max
         elif v == "minimum":
             handlers[k] = _nary_sym_min
+        # sympy.Pow / PowByNatural map to the interp name "pow_by_natural",
+        # which has no operator.* equivalent. sympy canonicalizes x * x into
+        # Pow(x, 2), so without this _build_proxy_for_sym_expr cannot rebuild
+        # any repeated-symbol product (e.g. a reduction numel s0 * s1**2 when
+        # two equal dims duck-share a symbol).
+        elif v == "pow_by_natural":
+            handlers[k] = operator.pow
 
     # sympy.Add is n-ary (e.g. Add(a, b, c)) but operator.add is binary.
     # torch.sym_sum handles n-ary integer addition and accepts both
"""


HIDDEN_188278_BODY = r'''
import unittest

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import (
    _build_proxy_for_sym_expr,
    _SympyExprTrackerValue,
    PythonKeyTracer,
    set_meta,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._thunk import Thunk


class SharedDynamicExtentProxyTests(unittest.TestCase):
    @staticmethod
    def symbols():
        shape_env = ShapeEnv()
        return shape_env.create_unbacked_symint(), shape_env.create_unbacked_symint()

    @staticmethod
    def tracker_without_graph(*symbols):
        tracer = PythonKeyTracer()
        for symbol in symbols:
            tracer.sympy_expr_tracker[symbol.node.expr] = _SympyExprTrackerValue(
                proxy=symbol, value=symbol
            )
        return tracer

    @staticmethod
    def tracker_with_graph(*symbols):
        tracer = PythonKeyTracer()
        tracer.root = torch.nn.Module()
        tracer.graph = fx.Graph(tracer_cls=PythonKeyTracer)
        for index, symbol in enumerate(symbols):
            node = tracer.graph.placeholder(f"u{index}")
            proxy = fx.Proxy(node, tracer)
            set_meta(proxy, symbol)
            tracer.sympy_expr_tracker[symbol.node.expr] = _SympyExprTrackerValue(
                proxy=proxy, value=symbol
            )
            tracer.symnode_tracker[symbol] = Thunk(lambda p=proxy: p)
        return tracer

    def test_square_extent_is_rebuilt_without_existing_output(self):
        u0, u1 = self.symbols()
        expression = u0 * u1**2
        rebuilt = _build_proxy_for_sym_expr(
            self.tracker_without_graph(u0, u1), expression.node.expr
        )
        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt.node.expr, expression.node.expr)

    def test_cube_extent_registers_an_executable_graph(self):
        u0, u1 = self.symbols()
        expression = u0 * u1**3
        tracer = self.tracker_with_graph(u0, u1)
        _build_proxy_for_sym_expr(tracer, expression.node.expr, out=expression)
        output = tracer.symnode_tracker[expression].force()
        tracer.graph.output(output.node)
        graph_module = fx.GraphModule(tracer.root, tracer.graph)
        self.assertEqual(graph_module(2, 3), 54)

    def test_distinct_extent_product_remains_rebuildable(self):
        u0, u1 = self.symbols()
        expression = u0 * u1
        rebuilt = _build_proxy_for_sym_expr(
            self.tracker_without_graph(u0, u1), expression.node.expr
        )
        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt.node.expr, expression.node.expr)

    def test_affine_extent_remains_rebuildable(self):
        u0, u1 = self.symbols()
        expression = u0 + 3 * u1
        rebuilt = _build_proxy_for_sym_expr(
            self.tracker_without_graph(u0, u1), expression.node.expr
        )
        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt.node.expr, expression.node.expr)


if __name__ == "__main__":
    unittest.main()
'''


TASKS = (
    {
        "retired_task_id": "pytorch__181469__dimension_annotation_guards",
        "task_dir": "tasks/pytorch/186993_large_guard_materialization",
        "task_id": "pytorch__186993__large_guard_materialization",
        "public_task_id": "opbench-v07-t0062",
        "pr_number": 186993,
        "screening_index": 85,
        "base_commit": "a4097e577fe5d1e21dfe2fa8c36af3fdf8854e34",
        "merge_commit": "083e2617b8ee20b66def9e05dfcee5be84af623a",
        "source_ref": "pytorch-a4097e57-large-guard-overlay",
        "source_snapshot_commit": "1120136f428a20ec6dc2ce40f9957c33a47d968d",
        "tracked_file_count": 21410,
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/export/_unlift.py"],
        "gold": GOLD_186993,
        "hidden": _new_file_patch(
            "test/op_bench/test_large_guard_materialization.py", HIDDEN_186993_BODY
        ),
        "test_file": "test/op_bench/test_large_guard_materialization.py",
        "f2p": [
            "LargeGuardMaterializationTests.test_codegen_survives_pretty_print_recursion",
            "LargeGuardMaterializationTests.test_multiple_runtime_guards_survive_pretty_print_recursion",
        ],
        "p2p": [
            "LargeGuardMaterializationTests.test_normal_guard_keeps_validation_semantics",
            "LargeGuardMaterializationTests.test_empty_guard_set_remains_callable",
        ],
        "statement": {
            "title": "Export reconstruction crashes on a very large shape guard",
            "body": (
                "A CPU model with many related dynamic dimensions exports successfully, but materializing "
                "the exported program as a runnable module fails when its accumulated shape guard becomes "
                "deeply nested. Repair module materialization so valid inputs can execute even at that scale. "
                "The generated runtime checks must still reject invalid inputs, and ordinary small guards "
                "must retain their current behavior."
            ),
            "labels": ["module: export", "module: dynamic shapes", "bug"],
        },
        "known_constraints": [
            "The failure occurs after export while constructing the runnable module's shape checks.",
            "Large valid guard expressions must not prevent module materialization.",
            "Runtime validation, including failure messages for invalid inputs, must remain effective.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "torch.export module materialization",
            "component": "torch.export",
            "problem_type": "large-symbolic-guard-materialization",
            "tags": ["export", "dynamic-shape", "guard", "module", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "api_behavior",
            "contract_detail_tags": ["shape", "exception", "compatibility"],
            "trigger_tags": ["extreme_value_or_size", "dynamic_shape"],
            "execution_context": {
                "devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False,
            },
            "failure_type": "unexpected_error",
            "root_cause_tags": ["incorrect_validation"],
            "component_tags": ["dynamo"],
        },
        "dimension_evidence": {
            "localization": (
                "The Prompt identifies only the public export-to-runnable-module transition and scale trigger. "
                "It exposes no private test, source path, helper, formatting operation, exception boundary, or fix."
            ),
            "diagnosis": (
                "Diagnosis must separate executable guard semantics from construction of its diagnostic text, "
                "then determine why only structurally deep expressions abort materialization before inference."
            ),
            "repair_regression": (
                "The repair must recover from two overflow scenarios while preserving multiple runtime checks, "
                "invalid-input rejection, normal guard generation, and the empty-guard control."
            ),
        },
        "behavior_tokens": [
            "export_materialization", "deep_shape_guard", "runtime_validation", "diagnostic_generation", "cpu"
        ],
        "risk_signals": ["gold_patch_single_file"],
        "estimated_runtime_min": 16,
    },
    {
        "retired_task_id": "pytorch__182083__autograd_worker_teardown",
        "task_dir": "tasks/pytorch/188278_shared_dynamic_extent_proxy",
        "task_id": "pytorch__188278__shared_dynamic_extent_proxy",
        "public_task_id": "opbench-v07-t0046",
        "pr_number": 188278,
        "screening_index": 90,
        "base_commit": "db23bc386f9a6c12170415efbf429762bfcd3285",
        "merge_commit": "4e5d6a25fdd898a3500ec0a9f75c09522b64ed4f",
        "source_ref": "pytorch-db23bc38-shared-extent-overlay",
        "source_snapshot_commit": "f29247484416f893cab6e87f62c377f6b38c64cc",
        "tracked_file_count": 21374,
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260707-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/fx/experimental/proxy_tensor.py"],
        "gold": GOLD_188278,
        "hidden": _new_file_patch(
            "test/op_bench/test_shared_dynamic_extent_proxy.py", HIDDEN_188278_BODY
        ),
        "test_file": "test/op_bench/test_shared_dynamic_extent_proxy.py",
        "f2p": [
            "SharedDynamicExtentProxyTests.test_square_extent_is_rebuilt_without_existing_output",
            "SharedDynamicExtentProxyTests.test_cube_extent_registers_an_executable_graph",
        ],
        "p2p": [
            "SharedDynamicExtentProxyTests.test_distinct_extent_product_remains_rebuildable",
            "SharedDynamicExtentProxyTests.test_affine_extent_remains_rebuildable",
        ],
        "statement": {
            "title": "Dynamic full reduction fails when dimensions share one extent",
            "body": (
                "A CPU full-graph workload reduces tensors whose two equal, non-unit dynamic dimensions are "
                "tracked as one shared extent. Eager execution succeeds, but compilation cannot reconstruct "
                "the derived element count. Repair compilation for shared dynamic extents, including square "
                "and cube relationships. Products of distinct extents and affine shape expressions must remain stable."
            ),
            "labels": ["module: dynamic shapes", "module: fx", "bug"],
        },
        "known_constraints": [
            "At least two non-unit dynamic dimensions share the same symbolic extent.",
            "The derived element count contains the same dynamic factor more than once and must remain symbolic.",
            "Distinct-factor products and affine symbolic expressions are regression controls.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "dynamic full reduction",
            "component": "FX proxy tracing",
            "problem_type": "shared-dynamic-extent-reconstruction",
            "tags": ["dynamic-shape", "reduction", "proxy", "symbolic", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "shape", "compatibility"],
            "trigger_tags": ["scalar_or_low_rank", "dynamic_shape"],
            "execution_context": {
                "devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False,
            },
            "failure_type": "unexpected_error",
            "root_cause_tags": ["incorrect_lowering"],
            "component_tags": ["dynamo"],
        },
        "dimension_evidence": {
            "localization": (
                "The Prompt gives the public dynamic-shape trigger and failed compiler behavior but no source path, "
                "symbolic expression class, handler table, internal tracer API, callable mapping, or patch location."
            ),
            "diagnosis": (
                "Diagnosis requires following a shared dynamic dimension through shape canonicalization and proxy "
                "reconstruction, then explaining why shared-factor powers fail although distinct products still work."
            ),
            "repair_regression": (
                "The repair must handle two natural exponents, both reconstruction modes, and executable graph "
                "registration while preserving distinct-factor multiplication and affine expressions."
            ),
        },
        "behavior_tokens": [
            "dynamic_reduction", "shared_extent", "repeated_factor", "proxy_reconstruction", "symbolic_numel"
        ],
        "risk_signals": ["gold_patch_single_file"],
        "estimated_runtime_min": 18,
    },
)


def main() -> int:
    replacement_builder.TASKS = TASKS
    replacement_builder.BASE_SOURCE_REGISTRY = ROOT / "sources/staging_v07_replacements.json"
    for spec in TASKS:
        replacement_builder._build_task(spec)
    replacement_builder._register_sources()
    replacement_builder._update_review_packets()
    print(canonical_json({"built": [spec["task_id"] for spec in TASKS]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
