#!/usr/bin/env python3
"""Build four offline exact-commit Python-overlay replacements for v0.7."""

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
from op_bench.runtime.source_materialization import (  # noqa: E402
    materialize_frozen_git_revision,
)


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


def _gold(pr_number: int, paths: list[str]) -> str:
    candidate = _candidate(pr_number)
    return _git_text(
        LOCAL_HISTORY,
        "diff",
        "--no-ext-diff",
        "--no-renames",
        str(candidate["base_commit"]),
        str(candidate["merge_commit"]),
        "--",
        *paths,
    )


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


HIDDEN_179028 = r'''
import unittest
from unittest.mock import patch

import torch
from torch._inductor.lowering import lowerings


class IntegerRangeLoweringTests(unittest.TestCase):
    def test_start_step_is_available_to_delegating_backends(self):
        operator = torch.ops.aten.arange.start_step
        self.assertIn(operator, lowerings)

    def test_start_step_delegates_integer_bounds_to_iota(self):
        operator = torch.ops.aten.arange.start_step
        sentinel = object()
        with patch("torch._inductor.lowering.iota", return_value=sentinel) as target_mock:
            result = lowerings[operator](
                2, 11, 3, dtype=torch.int64, device=torch.device("cpu")
            )
        self.assertIs(result, sentinel)
        target_mock.assert_called_once()

    def test_iota_registration_remains_available(self):
        self.assertIn(torch.ops.prims.iota.default, lowerings)

    def test_public_integer_range_values_remain_stable(self):
        actual = torch.ops.aten.arange.start_step(
            -3, 8, 2, dtype=torch.int64, device=torch.device("cpu")
        )
        torch.testing.assert_close(actual, torch.tensor([-3, -1, 1, 3, 5, 7]))


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_180471 = r'''
import unittest

import sympy
import torch
from torch.fx.experimental.symbolic_shapes import _iterate_exprs, free_symbols


class MappingShapeMetadataTests(unittest.TestCase):
    def test_tensor_mapping_is_accepted(self):
        self.assertEqual(list(_iterate_exprs({"Out": torch.randn(2, 3)})), [])

    def test_mapping_keys_and_values_are_traversed(self):
        left, right = sympy.symbols("left right")
        values = list(_iterate_exprs({left: right + 1}))
        self.assertIn(left, values)
        self.assertIn(right + 1, values)
        self.assertEqual(free_symbols({left + right: 1}), {left, right})

    def test_sequence_metadata_remains_recursive(self):
        left, right = sympy.symbols("left right")
        values = list(_iterate_exprs([left, (right + 2,)]))
        self.assertEqual(set(values), {left, right + 2})

    def test_scalar_and_tensor_leaves_remain_ignored(self):
        self.assertEqual(list(_iterate_exprs(3)), [])
        self.assertEqual(list(_iterate_exprs(torch.ones(1))), [])


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_186481 = r'''
import unittest

import torch
from torch._inductor.fx_passes.joint_graph import scatter_upon_const_tensor


class CaptureMatch:
    def __init__(self):
        self.value = None

    def replace_by_example(self, function, arguments):
        self.value = function(*arguments)


class ScatterRewriteDtypeTests(unittest.TestCase):
    @staticmethod
    def reference(indices, dtype):
        rows, columns = indices.numel(), 64
        result = torch.full((rows, columns), 3.25, dtype=dtype)
        return result.scatter(1, indices.unsqueeze(1), -1.5)

    def assert_rewrite_dtype(self, dtype):
        indices = torch.tensor([1, 7, 13, 31, 47, 63], dtype=torch.int64)
        expected = self.reference(indices, dtype)
        match = CaptureMatch()
        scatter_upon_const_tensor(
            match,
            [indices.numel(), 64],
            3.25,
            dtype,
            1,
            indices.unsqueeze(1),
            -1.5,
        )
        actual = match.value
        self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual, expected)

    def test_float16_scatter_preserves_source_dtype(self):
        self.assert_rewrite_dtype(torch.float16)

    def test_bfloat16_scatter_preserves_source_dtype(self):
        self.assert_rewrite_dtype(torch.bfloat16)

    def test_float32_scatter_remains_stable(self):
        self.assert_rewrite_dtype(torch.float32)

    def test_eager_low_precision_scatter_remains_stable(self):
        indices = torch.tensor([0, 3, 9], dtype=torch.int64)
        result = self.reference(indices, torch.float16)
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(torch.all(result[torch.arange(3), indices] == -1.5))


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_187863 = r'''
import unittest

import torch


class IgnoredLeaf(torch.nn.Module):
    __jit_ignored_attributes__ = ["sub"]

    def __init__(self):
        super().__init__()
        self.sub = torch.nn.Linear(4, 4)

    def forward(self, value):
        if torch.jit.is_scripting():
            return torch.zeros_like(value)
        return self._real(value)

    @torch.jit.ignore
    def _real(self, value):
        return self.sub(value)


class Wrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, value):
        return self.inner.forward(value)


class PairWrapper(torch.nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, value):
        return self.left.forward(value) + self.right.forward(value)


class AlreadyScriptedPrepareTests(unittest.TestCase):
    def test_scripted_child_with_ignored_leaf_can_be_nested(self):
        inner = torch.jit.script(IgnoredLeaf())
        self.assertIn("sub", dict(inner.named_children()))
        outer = torch.jit.script(Wrapper(inner))
        value = torch.randn(2, 4)
        torch.testing.assert_close(outer(value), torch.zeros_like(value))

    def test_two_scripted_children_with_ignored_leaves_can_be_nested(self):
        left = torch.jit.script(IgnoredLeaf())
        right = torch.jit.script(IgnoredLeaf())
        outer = torch.jit.script(PairWrapper(left, right))
        value = torch.randn(3, 4)
        torch.testing.assert_close(outer(value), torch.zeros_like(value))

    def test_plain_scripted_child_can_still_be_nested(self):
        inner = torch.jit.script(torch.nn.ReLU())
        outer = torch.jit.script(Wrapper(inner))
        value = torch.tensor([[-1.0, 2.0]])
        torch.testing.assert_close(outer(value), torch.relu(value))

    def test_unscripted_child_is_prepared_normally(self):
        outer = torch.jit.script(Wrapper(torch.nn.ReLU()))
        value = torch.tensor([[-2.0, 4.0]])
        torch.testing.assert_close(outer(value), torch.relu(value))


if __name__ == "__main__":
    unittest.main()
'''


def _source_fields(pr_number: int) -> dict[str, object]:
    candidate = _candidate(pr_number)
    return {
        "pr_number": pr_number,
        "base_commit": candidate["base_commit"],
        "merge_commit": candidate["merge_commit"],
    }


TASKS = (
    {
        **_source_fields(179028),
        "retired_task_id": "pytorch__176922__aoti_profile_repeated_dynamic_kernel",
        "task_dir": "tasks/pytorch/179028_integer_range_lowering_registry",
        "task_id": "pytorch__179028__integer_range_lowering_registry",
        "public_task_id": "opbench-v07-t0039",
        "screening_index": 22,
        "source_ref": "pytorch-c695a129-integer-range-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/lowering.py"],
        "gold": _gold(179028, ["torch/_inductor/lowering.py"]),
        "hidden": _new_file_patch(
            "test/op_bench/test_integer_range_lowering_registry.py", HIDDEN_179028
        ),
        "test_file": "test/op_bench/test_integer_range_lowering_registry.py",
        "f2p": [
            "IntegerRangeLoweringTests.test_start_step_is_available_to_delegating_backends",
            "IntegerRangeLoweringTests.test_start_step_delegates_integer_bounds_to_iota",
        ],
        "p2p": [
            "IntegerRangeLoweringTests.test_iota_registration_remains_available",
            "IntegerRangeLoweringTests.test_public_integer_range_values_remain_stable",
        ],
        "statement": {
            "title": "A delegated integer range fails before compiler code generation",
            "body": (
                "A backend delegates an integer start/end/step range operation to the standard compiler path. "
                "Direct eager execution produces the expected values, but delegated compilation fails before "
                "code generation because that overload is unavailable to the backend. Repair the compiler path "
                "for positive and negative integer bounds while preserving ordinary range behavior."
            ),
            "labels": ["module: inductor", "module: lowering", "bug"],
        },
        "known_constraints": [
            "The failing overload supplies explicit integer start, end, and step values.",
            "A secondary backend reaches the operation by delegating to the standard compiler implementation.",
            "Direct eager values and the ordinary range overload are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "aten.arange", "component": "TorchInductor",
            "problem_type": "delegated-start-step-range", "tags": ["range", "integer", "compile", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "dtype", "schema", "compatibility"],
            "trigger_tags": ["scalar_or_low_rank"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["wrong_dispatch"], "component_tags": ["inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt names only a public range behavior and delegation boundary; it reveals no source path, registry, internal overload object, helper, decomposition, or repair site.",
            "diagnosis": "Diagnosis must distinguish eager operator support from backend-facing compiler availability and follow delegation through decomposition and lowering registration.",
            "repair_regression": "The repair must preserve start/end/step semantics and dtype while making the delegated path callable; two failures and two independent controls reject aliases or eager-only workarounds.",
        },
        "behavior_tokens": ["integer_range", "delegated_backend", "explicit_step", "compiler_registry", "dtype"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 16,
    },
    {
        **_source_fields(180471),
        "retired_task_id": "pytorch__177860__flex_attention_captured_grad",
        "task_dir": "tasks/pytorch/180471_mapping_shape_metadata",
        "task_id": "pytorch__180471__mapping_shape_metadata",
        "public_task_id": "opbench-v07-t0033",
        "screening_index": 41,
        "source_ref": "pytorch-fadf0e3f-mapping-shape-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260423-torch2.13.0dev-cpu-py311",
        "overlay_paths": ["torch/fx/experimental/symbolic_shapes.py"],
        "gold": _gold(180471, ["torch/fx/experimental/symbolic_shapes.py"]),
        "hidden": _new_file_patch("test/op_bench/test_mapping_shape_metadata.py", HIDDEN_180471),
        "test_file": "test/op_bench/test_mapping_shape_metadata.py",
        "f2p": [
            "MappingShapeMetadataTests.test_tensor_mapping_is_accepted",
            "MappingShapeMetadataTests.test_mapping_keys_and_values_are_traversed",
        ],
        "p2p": [
            "MappingShapeMetadataTests.test_sequence_metadata_remains_recursive",
            "MappingShapeMetadataTests.test_scalar_and_tensor_leaves_remain_ignored",
        ],
        "statement": {
            "title": "Compiler shape analysis rejects mapping-valued operator metadata",
            "body": (
                "A compiled operator returns named tensor outputs represented as mapping-valued metadata. "
                "Shape analysis aborts while checking whether the graph is compatible with optimized execution. "
                "Repair analysis so mapping keys and values participate in symbolic dependency discovery, while "
                "existing sequence traversal and concrete tensor leaves retain their behavior."
            ),
            "labels": ["module: dynamic shapes", "module: fx", "bug"],
        },
        "known_constraints": [
            "Operator metadata may be an immutable mapping from output names to tensor-like values.",
            "Symbolic expressions may occur in either mapping keys or mapping values.",
            "Nested sequences, scalar leaves, and concrete tensor leaves are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "symbolic shape dependency analysis", "component": "FX symbolic shapes",
            "problem_type": "mapping-valued-metadata-traversal", "tags": ["dynamic-shape", "metadata", "mapping", "compile", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["shape", "compatibility"], "trigger_tags": ["dynamic_shape"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_validation"], "component_tags": ["dynamo", "inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt exposes the metadata container category and public analysis phase but no private test, source path, traversal helper, accepted leaf types, assertion, or repair site.",
            "diagnosis": "Diagnosis requires following operator metadata into free-symbol discovery and recognizing that container recursion covers sequences but not mappings, including symbolic keys.",
            "repair_regression": "The repair must traverse both sides of a mapping, accept concrete outputs, preserve nested sequence recursion, and keep scalar/tensor leaves ignored across four focused controls.",
        },
        "behavior_tokens": ["mapping_metadata", "symbolic_dependency", "container_traversal", "optimized_execution", "fx"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 15,
    },
    {
        **_source_fields(186481),
        "retired_task_id": "pytorch__179333__dynamic_large_offset_template",
        "task_dir": "tasks/pytorch/186481_scatter_rewrite_dtype",
        "task_id": "pytorch__186481__scatter_rewrite_dtype",
        "public_task_id": "opbench-v07-t0041",
        "screening_index": 82,
        "source_ref": "pytorch-adf92da6-scatter-dtype-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/fx_passes/joint_graph.py"],
        "gold": _gold(186481, ["torch/_inductor/fx_passes/joint_graph.py"]),
        "hidden": _new_file_patch("test/op_bench/test_scatter_rewrite_dtype.py", HIDDEN_186481),
        "test_file": "test/op_bench/test_scatter_rewrite_dtype.py",
        "f2p": [
            "ScatterRewriteDtypeTests.test_float16_scatter_preserves_source_dtype",
            "ScatterRewriteDtypeTests.test_bfloat16_scatter_preserves_source_dtype",
        ],
        "p2p": [
            "ScatterRewriteDtypeTests.test_float32_scatter_remains_stable",
            "ScatterRewriteDtypeTests.test_eager_low_precision_scatter_remains_stable",
        ],
        "statement": {
            "title": "Compiled scalar scatter widens a low-precision destination",
            "body": (
                "A graph fills a low-precision tensor and scatters a scalar into one column per row. Eager "
                "execution retains the destination dtype, but the optimized compiled graph returns a wider "
                "numeric dtype and changes dependent numerics. Repair compilation for both common 16-bit formats. "
                "Float32 compilation and eager low-precision scatter must remain unchanged."
            ),
            "labels": ["module: inductor", "module: dtype", "bug"],
        },
        "known_constraints": [
            "The destination is created from a scalar fill and updated by scalar scatter values.",
            "Both supported 16-bit compiled outputs must retain the destination dtype.",
            "Float32 compilation and eager low-precision behavior are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "aten.scatter.value", "component": "TorchInductor joint graph",
            "problem_type": "scatter-rewrite-dtype-promotion", "tags": ["scatter", "dtype", "low-precision", "compile", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "tensor_metadata",
            "contract_detail_tags": ["value", "numerical", "dtype", "compatibility"],
            "trigger_tags": ["scalar_or_low_rank", "mixed_dtype_or_precision_mode"],
            "execution_context": {"devices": ["cpu"], "modes": ["eager", "compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "wrong_result", "root_cause_tags": ["incorrect_cast", "incorrect_lowering"], "component_tags": ["inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt describes public fill-and-scatter behavior without exposing the optimizer pass, rewrite pattern, replacement operator, scalar branch representation, source path, or fix.",
            "diagnosis": "Diagnosis must compare eager scatter semantics with the optimized graph, isolate dtype promotion inside a joint-graph rewrite, and distinguish declared output metadata from actual buffer dtype.",
            "repair_regression": "The repair must preserve two low-precision dtypes and values without changing float32 compilation or eager execution; four selectors prevent disabling compilation or the optimization globally.",
        },
        "behavior_tokens": ["scalar_scatter", "const_destination", "low_precision_dtype", "joint_graph_rewrite", "promotion"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 22,
    },
    {
        **_source_fields(187863),
        "retired_task_id": "pytorch__179505__broadcast_batch_autotune",
        "task_dir": "tasks/pytorch/187863_nested_script_prepare",
        "task_id": "pytorch__187863__nested_script_prepare",
        "public_task_id": "opbench-v07-t0056",
        "screening_index": 89,
        "source_ref": "pytorch-71519a98-nested-script-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260707-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/jit/_script.py"],
        "gold": _gold(187863, ["torch/jit/_script.py"]),
        "hidden": _new_file_patch("test/op_bench/test_nested_script_prepare.py", HIDDEN_187863),
        "test_file": "test/op_bench/test_nested_script_prepare.py",
        "f2p": [
            "AlreadyScriptedPrepareTests.test_scripted_child_with_ignored_leaf_can_be_nested",
            "AlreadyScriptedPrepareTests.test_two_scripted_children_with_ignored_leaves_can_be_nested",
        ],
        "p2p": [
            "AlreadyScriptedPrepareTests.test_plain_scripted_child_can_still_be_nested",
            "AlreadyScriptedPrepareTests.test_unscripted_child_is_prepared_normally",
        ],
        "statement": {
            "title": "Nesting a precompiled module rejects an ignored live submodule",
            "body": (
                "A TorchScript module deliberately keeps one eager-only submodule live, then is embedded in a larger "
                "module that is converted again. The second conversion fails by trying to replace state inside "
                "the precompiled submodule. Repair nested conversion for one or multiple such modules while "
                "preserving ordinary precompiled and newly converted nesting behavior."
            ),
            "labels": ["module: jit", "module: scripting", "bug"],
        },
        "known_constraints": [
            "The inner module is already TorchScript-compiled before it is registered on the outer module.",
            "An explicitly ignored attribute remains a live eager descendant of that compiled module.",
            "Plain precompiled submodules and ordinary eager submodules are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "torch.jit.script", "component": "TorchScript",
            "problem_type": "already-scripted-child-prepare", "tags": ["jit", "module", "nested", "state", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "mutation_state",
            "contract_detail_tags": ["state", "serialization", "exception", "compatibility"],
            "trigger_tags": ["mutation_or_alias"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_validation"], "component_tags": [],
        },
        "dimension_evidence": {
            "localization": "The Prompt exposes the public two-level scripting scenario but no source path, preparation function, internal module container, descent branch, type guard, or patch location.",
            "diagnosis": "Diagnosis must distinguish preparation of a new child from re-walking immutable scripted state and explain why an ignored live descendant makes only the second scripting pass fail.",
            "repair_regression": "The repair must support one and two affected children while retaining ordinary already-scripted and unscripted preparation paths, without suppressing legitimate module assignment errors.",
        },
        "behavior_tokens": ["nested_scripting", "already_scripted_child", "ignored_attribute", "module_state", "prepare_pass"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 18,
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
    spec["tracked_file_count"] = len(
        _git_text(destination, "ls-files", "--deduplicate").splitlines()
    )


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
