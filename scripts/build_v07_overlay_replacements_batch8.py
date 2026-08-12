#!/usr/bin/env python3
"""Build three additional exact-commit Python-overlay replacements for v0.7."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for selected in (SRC, SCRIPTS):
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))

import build_v07_replacement_tasks as replacement_builder  # noqa: E402
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.source_materialization import materialize_frozen_git_revision  # noqa: E402


LOCAL_HISTORY = ROOT / ".op_bench_cache/manual_repos/pytorch"


def _candidate(pr_number: int) -> dict[str, object]:
    path = ROOT / f"factory/v0.7/p8/screening/candidates/pr-{pr_number}.json"
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected object")
    return value


def _git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _source_fields(pr_number: int) -> dict[str, object]:
    candidate = _candidate(pr_number)
    return {
        "pr_number": pr_number,
        "base_commit": candidate["base_commit"],
        "merge_commit": candidate["merge_commit"],
    }


def _gold(pr_number: int, paths: list[str]) -> str:
    source = _source_fields(pr_number)
    return _git_text(
        LOCAL_HISTORY,
        "diff",
        "--no-ext-diff",
        "--no-renames",
        str(source["base_commit"]),
        str(source["merge_commit"]),
        "--",
        *paths,
    )


def _new_file_patch(path: str, body: str) -> str:
    normalized = body.strip("\n") + "\n"
    lines = normalized.splitlines()
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        + "\n".join(f"+{line}" for line in lines)
        + "\n"
    )


HIDDEN_179708 = r'''
import unittest

import sympy
from torch.utils._sympy.printers import CppPrinter, ExprPrinter, PythonPrinter


class NonFiniteExpressionPrinterTests(unittest.TestCase):
    def test_python_nan_is_executable(self):
        self.assertEqual(PythonPrinter().doprint(sympy.nan), "math.nan")

    def test_cpp_nan_is_executable(self):
        self.assertEqual(
            CppPrinter().doprint(sympy.nan),
            "std::numeric_limits<double>::quiet_NaN()",
        )

    def test_base_printer_requires_explicit_nan_policy(self):
        with self.assertRaises(NotImplementedError):
            ExprPrinter().doprint(sympy.nan)

    def test_infinity_rendering_remains_stable(self):
        self.assertEqual(PythonPrinter().doprint(sympy.oo), "math.inf")
        self.assertEqual(PythonPrinter().doprint(-sympy.oo), "-math.inf")

    def test_cpp_infinity_rendering_remains_stable(self):
        self.assertEqual(
            CppPrinter().doprint(sympy.oo),
            "std::numeric_limits<double>::infinity()",
        )
        self.assertEqual(
            CppPrinter().doprint(-sympy.oo),
            "-std::numeric_limits<double>::infinity()",
        )


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_182101 = r'''
import unittest

from torch._export.serde.schema import Argument, ArgumentKind, NamedArgument
from torch._export.serde.serialize import GraphModuleDeserializer


class NamedHigherOrderArgumentTests(unittest.TestCase):
    @staticmethod
    def integer(name, value, kind):
        return NamedArgument(
            name=name,
            arg=Argument.create(as_int=value),
            kind=kind,
        )

    def test_named_positional_arguments_keep_order(self):
        inputs = [
            self.integer("tensor_slot", 3, ArgumentKind.POSITIONAL),
            self.integer("dynamic_count", 9, ArgumentKind.POSITIONAL),
        ]
        args, kwargs = GraphModuleDeserializer().deserialize_hoo_inputs(inputs)
        self.assertEqual(args, (3, 9))
        self.assertEqual(kwargs, {})

    def test_named_positional_and_launch_metadata_stay_separate(self):
        inputs = [
            self.integer("dynamic_count", 11, ArgumentKind.POSITIONAL),
            self.integer("launch_grid", 4, ArgumentKind.KEYWORD),
        ]
        args, kwargs = GraphModuleDeserializer().deserialize_hoo_inputs(inputs)
        self.assertEqual(args, (11,))
        self.assertEqual(kwargs, {"launch_grid": 4})

    def test_legacy_unnamed_positional_argument_remains_supported(self):
        inputs = [self.integer("", 7, ArgumentKind.UNKNOWN)]
        args, kwargs = GraphModuleDeserializer().deserialize_hoo_inputs(inputs)
        self.assertEqual(args, (7,))
        self.assertEqual(kwargs, {})

    def test_keyword_only_metadata_remains_supported(self):
        inputs = [self.integer("num_warps", 8, ArgumentKind.KEYWORD)]
        args, kwargs = GraphModuleDeserializer().deserialize_hoo_inputs(inputs)
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"num_warps": 8})


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_183973 = r'''
import unittest

import torch
from torch._inductor import config


class UnrealizedAddmmBiasTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        torch.manual_seed(0)

    def tearDown(self):
        torch._dynamo.reset()

    @staticmethod
    def model(source, left, right):
        bias = (source * 2.0).transpose(0, 1)
        return torch.addmm(bias, left, right)

    def assert_compiled_view_bias(self, dynamic):
        source = torch.randn(8, 8, device="cuda")
        left = torch.randn(8, 16, device="cuda")
        right = torch.randn(16, 8, device="cuda")
        expected = self.model(source, left, right)
        with config.patch({"max_autotune": True}):
            actual = torch.compile(self.model, dynamic=dynamic, fullgraph=True)(
                source, left, right
            )
        torch.testing.assert_close(actual, expected)

    def test_static_view_bias_compiles_with_autotuning(self):
        self.assert_compiled_view_bias(False)

    def test_dynamic_view_bias_compiles_with_autotuning(self):
        self.assert_compiled_view_bias(True)

    def test_realized_bias_compilation_remains_stable(self):
        bias = torch.randn(8, 8, device="cuda")
        left = torch.randn(8, 16, device="cuda")
        right = torch.randn(16, 8, device="cuda")

        def function(value, lhs, rhs):
            return torch.addmm(value, lhs, rhs)

        with config.patch({"max_autotune": True}):
            actual = torch.compile(function, fullgraph=True)(bias, left, right)
        torch.testing.assert_close(actual, function(bias, left, right))

    def test_eager_view_bias_remains_stable(self):
        source = torch.randn(4, 4, device="cuda")
        left = torch.randn(4, 6, device="cuda")
        right = torch.randn(6, 4, device="cuda")
        expected = (source * 2.0).transpose(0, 1) + left @ right
        torch.testing.assert_close(self.model(source, left, right), expected)


if __name__ == "__main__":
    unittest.main()
'''


TASKS = (
    {
        **_source_fields(179708),
        "retired_task_id": "pytorch__179028__integer_range_lowering_registry",
        "task_dir": "tasks/pytorch/179708_nonfinite_expression_codegen",
        "task_id": "pytorch__179708__nonfinite_expression_codegen",
        "public_task_id": "opbench-v07-t0039",
        "screening_index": 30,
        "source_ref": "pytorch-e2584b25-nonfinite-printer-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260423-torch2.13.0dev-cpu-py311",
        "overlay_paths": ["torch/utils/_sympy/printers.py"],
        "gold": _gold(179708, ["torch/utils/_sympy/printers.py"]),
        "hidden": _new_file_patch("test/op_bench/test_nonfinite_expression_codegen.py", HIDDEN_179708),
        "test_file": "test/op_bench/test_nonfinite_expression_codegen.py",
        "f2p": [
            "NonFiniteExpressionPrinterTests.test_python_nan_is_executable",
            "NonFiniteExpressionPrinterTests.test_cpp_nan_is_executable",
            "NonFiniteExpressionPrinterTests.test_base_printer_requires_explicit_nan_policy",
        ],
        "p2p": [
            "NonFiniteExpressionPrinterTests.test_infinity_rendering_remains_stable",
            "NonFiniteExpressionPrinterTests.test_cpp_infinity_rendering_remains_stable",
        ],
        "statement": {
            "title": "Generated comparison guards contain an undefined non-finite literal",
            "body": (
                "Compiled comparisons with NaN inputs can emit guard expressions that are not executable in "
                "either generated Python or generated C++ code. Repair code generation so the non-finite value "
                "uses valid language-specific syntax and the generic expression path rejects ambiguous output. "
                "Positive and negative unbounded-value rendering must remain unchanged."
            ),
            "labels": ["module: inductor", "module: codegen", "bug"],
        },
        "known_constraints": [
            "The failure is triggered while emitting comparison guards containing a NaN constant.",
            "Both Python and C++ generated forms must be executable and preserve non-finite semantics.",
            "Positive and negative unbounded constants are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "compiled comparison guards", "component": "symbolic code generation",
            "problem_type": "nonfinite-literal-emission", "tags": ["comparison", "nan", "codegen", "python", "cpp"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["numerical", "exception", "compatibility"],
            "trigger_tags": ["extreme_value_or_size"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt describes generated-language failures and non-finite behavior but names no source path, printer class, dispatch method, emitted literal, helper, or repair site.",
            "diagnosis": "Diagnosis must trace a symbolic NaN through generic and language-specific code generation and distinguish executable constants from bare identifiers in two target languages.",
            "repair_regression": "The repair must implement two language policies plus an explicit generic failure while preserving both signs of infinity across five focused selectors.",
        },
        "behavior_tokens": ["nan_guard", "language_codegen", "python_expression", "cpp_expression", "infinity_control"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 16,
    },
    {
        **_source_fields(182101),
        "retired_task_id": "pytorch__178076__dynamic_isin_decomposition",
        "task_dir": "tasks/pytorch/182101_named_hop_argument_roundtrip",
        "task_id": "pytorch__182101__named_hop_argument_roundtrip",
        "public_task_id": "opbench-v07-t0043",
        "screening_index": 59,
        "source_ref": "pytorch-64ef0a26-named-hop-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/_export/serde/serialize.py"],
        "gold": _gold(182101, ["torch/_export/serde/serialize.py"]),
        "hidden": _new_file_patch("test/op_bench/test_named_hop_argument_roundtrip.py", HIDDEN_182101),
        "test_file": "test/op_bench/test_named_hop_argument_roundtrip.py",
        "f2p": [
            "NamedHigherOrderArgumentTests.test_named_positional_arguments_keep_order",
            "NamedHigherOrderArgumentTests.test_named_positional_and_launch_metadata_stay_separate",
        ],
        "p2p": [
            "NamedHigherOrderArgumentTests.test_legacy_unnamed_positional_argument_remains_supported",
            "NamedHigherOrderArgumentTests.test_keyword_only_metadata_remains_supported",
        ],
        "statement": {
            "title": "Export round-trip reclassifies named kernel operands as metadata",
            "body": (
                "An exported higher-order kernel mixes tensor operands, dynamic scalar operands, and launch "
                "metadata. After serialization, positional operands that carry parameter names are reconstructed "
                "as keyword metadata, changing ABI order. Repair the round trip so named positional values keep "
                "their order and launch metadata stays separate. Legacy unnamed operands must remain compatible."
            ),
            "labels": ["module: export", "module: serialization", "bug"],
        },
        "known_constraints": [
            "Runtime operands can be positional while still carrying stable kernel parameter names.",
            "Launch configuration remains keyword metadata and must not enter the positional ABI.",
            "Legacy unnamed positional values and keyword-only metadata are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "exported higher-order kernel", "component": "torch.export serialization",
            "problem_type": "named-positional-argument-roundtrip", "tags": ["export", "serialization", "kernel", "arguments", "abi", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["serialization", "schema", "compatibility"],
            "trigger_tags": ["scalar_or_low_rank", "dynamic_shape"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "wrong_result", "root_cause_tags": ["wrong_dispatch"], "component_tags": ["dispatcher", "dynamo"],
        },
        "dimension_evidence": {
            "localization": "The Prompt exposes the public argument categories and round-trip failure but no schema class, deserializer branch, input-kind field, serializer helper, source path, or fix.",
            "diagnosis": "Diagnosis must distinguish semantic parameter names from calling convention, preserve mixed ABI order, and follow the metadata through both serialization and reconstruction.",
            "repair_regression": "The repair must separate two named positional cases from keyword launch data while retaining legacy empty-name and keyword-only compatibility across four controls.",
        },
        "behavior_tokens": ["hop_roundtrip", "named_positional", "dynamic_scalar", "launch_metadata", "abi_order"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 18,
    },
    {
        **_source_fields(183973),
        "retired_task_id": "pytorch__181119__benchmark_alias_storage",
        "task_dir": "tasks/pytorch/183973_unrealized_addmm_bias",
        "task_id": "pytorch__183973__unrealized_addmm_bias",
        "public_task_id": "opbench-v07-t0050",
        "screening_index": 70,
        "source_ref": "pytorch-5d72a01a-unrealized-addmm-overlay",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/kernel/mm.py"],
        "gold": _gold(183973, ["torch/_inductor/kernel/mm.py"]),
        "hidden": _new_file_patch("test/op_bench/test_unrealized_addmm_bias.py", HIDDEN_183973),
        "test_file": "test/op_bench/test_unrealized_addmm_bias.py",
        "f2p": [
            "UnrealizedAddmmBiasTests.test_static_view_bias_compiles_with_autotuning",
            "UnrealizedAddmmBiasTests.test_dynamic_view_bias_compiles_with_autotuning",
        ],
        "p2p": [
            "UnrealizedAddmmBiasTests.test_realized_bias_compilation_remains_stable",
            "UnrealizedAddmmBiasTests.test_eager_view_bias_remains_stable",
        ],
        "statement": {
            "title": "Autotuned addmm rejects a computed axis-swapped bias",
            "body": (
                "A CUDA addmm graph derives its bias from a pointwise expression followed by an axis swap. Eager "
                "execution succeeds, but max-autotune compilation fails before selecting a kernel because the "
                "bias has not yet acquired concrete layout metadata. Repair static and dynamic compilation. "
                "Already-materialized bias tensors and eager view semantics must remain unchanged."
            ),
            "labels": ["module: inductor", "module: autotuning", "bug"],
        },
        "known_constraints": [
            "The bias is a dimension-swapped view over a fused pointwise result rather than a plain input tensor.",
            "Both static and dynamic max-autotune compilation must match eager values.",
            "Materialized bias compilation and eager view behavior are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "torch.addmm", "component": "TorchInductor autotuning",
            "problem_type": "unrealized-view-bias-layout", "tags": ["addmm", "bias", "view", "autotune", "cuda"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "shape", "layout", "compatibility"],
            "trigger_tags": ["noncontiguous_or_special_layout", "dynamic_shape", "device_specific"],
            "execution_context": {"devices": ["cuda"], "modes": ["eager", "compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_validation"], "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": "The Prompt identifies addmm, the computed-view trigger, and autotuning phase but no source path, internal IR type, layout accessor, realization helper, call site, or repair line.",
            "diagnosis": "Diagnosis requires following a fused pointwise transpose through matmul argument normalization and identifying a premature stride query on an unrealized compiler value.",
            "repair_regression": "Although Gold is one line, the repair boundary is validated by static and dynamic failures plus materialized-bias and eager controls; disabling autotuning or views cannot pass.",
        },
        "behavior_tokens": ["addmm_bias", "unrealized_view", "max_autotune", "layout_metadata", "dynamic_compile"],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file"], "estimated_runtime_min": 24,
    },
)


def _materialize(spec: dict[str, object]) -> None:
    commit = str(spec["base_commit"])
    destination = ROOT / ".op_bench_cache/sources/pytorch/pytorch" / commit / "source"
    if not destination.exists():
        materialize_frozen_git_revision(
            source_directory=LOCAL_HISTORY,
            revision=commit,
            workspace=destination,
        )
    spec["source_snapshot_commit"] = _git_text(destination, "rev-parse", "HEAD").strip()
    spec["tracked_file_count"] = len(_git_text(destination, "ls-files", "--deduplicate").splitlines())


def main() -> int:
    for spec in TASKS:
        _materialize(spec)
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
