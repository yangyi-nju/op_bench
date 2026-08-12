#!/usr/bin/env python3
"""Build five exact-commit offline replacements for the final v0.7 slots."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for selected in (SRC, SCRIPTS):
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))

import build_v07_overlay_replacements_batch8 as support  # noqa: E402
import build_v07_replacement_tasks as replacement_builder  # noqa: E402
from op_bench.runtime.canonical import canonical_json  # noqa: E402


HIDDEN_178617 = r'''
import unittest

from torch._inductor.kernel.bmm import bmm_grid
from torch._inductor.kernel.mm_common import load_kernel_template
from torch._inductor.runtime.runtime_utils import get_max_y_grid


class LargeBatchGridTests(unittest.TestCase):
    META = {"BLOCK_M": 16, "BLOCK_N": 32}

    def test_large_batch_is_split_across_launch_axes(self):
        batch = get_max_y_grid() + 4097
        grid_x, grid_y, grid_z = bmm_grid(batch, 64, 96, self.META)
        self.assertLessEqual(grid_y, get_max_y_grid())
        self.assertGreater(grid_z, 1)
        self.assertGreaterEqual(grid_y * grid_z, batch)
        self.assertEqual(grid_x, 12)

    def test_last_batch_index_remains_representable(self):
        batch = 2 * get_max_y_grid() + 19
        _, grid_y, grid_z = bmm_grid(batch, 32, 32, self.META)
        reconstructed = (grid_y - 1) + (grid_z - 1) * grid_y
        self.assertGreaterEqual(reconstructed, batch - 1)
        self.assertLessEqual(grid_y, get_max_y_grid())

    def test_template_reconstructs_and_masks_the_batch_index(self):
        source = load_kernel_template("triton_bmm")
        self.assertIn("tl.program_id(2)", source)
        self.assertIn("tl.num_programs(1)", source)
        self.assertIn("idx_q < BATCH", source)

    def test_small_batch_preserves_single_depth_launch(self):
        self.assertEqual(bmm_grid(37, 64, 96, self.META), (12, 37, 1))

    def test_single_batch_preserves_grid_shape(self):
        self.assertEqual(bmm_grid(1, 16, 32, self.META), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_179278 = r'''
import unittest

import torch
from torch.export.unflatten import _reorder_submodules


class RepeatedChildReorderTests(unittest.TestCase):
    @staticmethod
    def repeated_parent():
        parent = torch.nn.Module()
        repeated = torch.nn.Linear(3, 3)
        parent.add_module("node", repeated)
        parent.add_module("node@1", repeated)
        return parent, repeated

    def test_alias_qualified_child_does_not_raise(self):
        parent, _ = self.repeated_parent()
        _reorder_submodules(parent, {"": 0, "node": 1})
        self.assertEqual(list(parent._modules), ["node", "node@1"])

    def test_shared_identity_survives_reordering(self):
        parent, repeated = self.repeated_parent()
        _reorder_submodules(parent, {"": 0, "node": 1})
        self.assertIs(parent._modules["node"], repeated)
        self.assertIs(parent._modules["node@1"], repeated)

    def test_alias_uses_base_child_order(self):
        parent = torch.nn.Module()
        parent.add_module("late", torch.nn.ReLU())
        repeated = torch.nn.Linear(2, 2)
        parent.add_module("node@1", repeated)
        parent.add_module("node", repeated)
        _reorder_submodules(parent, {"": 0, "node": 1, "late": 2})
        names = list(parent._modules)
        self.assertLess(names.index("node@1"), names.index("late"))
        self.assertLess(names.index("node"), names.index("late"))

    def test_ordinary_children_follow_declared_order(self):
        parent = torch.nn.Module()
        parent.add_module("second", torch.nn.ReLU())
        parent.add_module("first", torch.nn.Sigmoid())
        _reorder_submodules(parent, {"": 0, "first": 1, "second": 2})
        self.assertEqual(list(parent._modules), ["first", "second"])

    def test_nested_ordinary_children_remain_available(self):
        parent = torch.nn.Module()
        child = torch.nn.Module()
        child.add_module("leaf", torch.nn.ReLU())
        parent.add_module("child", child)
        _reorder_submodules(parent, {"": 0, "child": 1, "child.leaf": 2})
        self.assertIsInstance(parent.child.leaf, torch.nn.ReLU)


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_180370 = r'''
import unittest
from collections import OrderedDict
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch._inductor.codegen.cpp_wrapper_cpu_array_ref as arrayref_module
from torch._inductor.codegen.cpp_wrapper_cpu_array_ref import CppWrapperCpuArrayRef
from torch._inductor.utils import IndentedBuffer


class FakeTensorBox:
    pass


class FakeBuffer:
    def __init__(self, name):
        self.name = name

    def codegen_reference(self):
        return self.name

    def get_name(self):
        return self.name


class FakeShapePredicate:
    pass


def wrapper_instance():
    wrapper = object.__new__(CppWrapperCpuArrayRef)
    wrapper.wrapper_call = IndentedBuffer()
    wrapper.writeline = wrapper.wrapper_call.writeline
    wrapper.allow_stack_allocation = False
    wrapper.stack_allocated_buffers = {}
    return wrapper


class MinimalArrayRefControlFlowTests(unittest.TestCase):
    def test_subgraph_tensor_inputs_receive_owned_handles(self):
        wrapper = wrapper_instance()
        subgraph = SimpleNamespace(
            graph=SimpleNamespace(
                graph_inputs=OrderedDict(
                    [("inner_tensor", FakeTensorBox()), ("inner_scalar", object())]
                )
            )
        )
        with patch.object(arrayref_module.ir, "TensorBox", FakeTensorBox):
            wrapper.codegen_subgraph_prefix(subgraph, ["outer_tensor", "outer_scalar"], [])
        code = wrapper.wrapper_call.getvalue()
        self.assertIn("AtenTensorHandle inner_tensor_handle;", code)
        self.assertIn("borrow_arrayref_tensor_as_tensor(outer_tensor)", code)
        self.assertIn("RAIIAtenTensorHandle inner_tensor(inner_tensor_handle);", code)
        self.assertNotIn("inner_scalar_handle", code)

    def test_tensor_predicate_while_loop_emits_scalar_check(self):
        wrapper = wrapper_instance()
        calls = []

        def record_subgraph(self, subgraph, inputs, outputs):
            calls.append((subgraph, list(inputs), list(outputs)))

        def record_item(self, dtype, source, target):
            self.writeline(f"ITEM {dtype} {source} {target};")

        wrapper.codegen_subgraph = MethodType(record_subgraph, wrapper)
        wrapper.codegen_tensor_item = MethodType(record_item, wrapper)
        cond_graph = SimpleNamespace(graph_outputs=[object()])
        body_graph = SimpleNamespace(graph_outputs=[])
        loop = SimpleNamespace(
            get_name=lambda: "loop0",
            carried_inputs=[FakeBuffer("arg0")],
            additional_inputs=[FakeBuffer("arg1")],
            outputs=[FakeBuffer("buf0")],
            cond_subgraph=SimpleNamespace(graph=cond_graph),
            body_subgraph=SimpleNamespace(graph=body_graph),
        )
        with (
            patch.object(arrayref_module.ir, "ShapeAsConstantBuffer", FakeShapePredicate),
            patch.object(arrayref_module, "EnterSubgraphLine", lambda *args: "ENTER"),
            patch.object(arrayref_module, "ExitSubgraphLine", lambda *args: "EXIT"),
        ):
            wrapper.codegen_while_loop(loop)
        code = wrapper.wrapper_call.getvalue()
        self.assertIn("RAIIAtenTensorHandle loop0_cond_result;", code)
        self.assertIn("while (1) {", code)
        self.assertIn("ITEM torch.bool loop0_cond_result loop0_cond_result_scalar;", code)
        self.assertIn("if (!loop0_cond_result_scalar) break;", code)
        self.assertEqual(calls[0][1:], (["buf0", "arg1"], ["loop0_cond_result"]))
        self.assertEqual(calls[1][1:], (["buf0", "arg1"], ["buf0"]))

    def test_scalar_predicate_avoids_tensor_item_conversion(self):
        wrapper = wrapper_instance()
        wrapper.codegen_subgraph = MethodType(lambda *args: None, wrapper)
        wrapper.codegen_tensor_item = MethodType(
            lambda *args: self.fail("scalar predicate must not be converted"), wrapper
        )
        cond_graph = SimpleNamespace(graph_outputs=[FakeShapePredicate()])
        loop = SimpleNamespace(
            get_name=lambda: "loop1",
            carried_inputs=[],
            additional_inputs=[],
            outputs=[],
            cond_subgraph=SimpleNamespace(graph=cond_graph),
            body_subgraph=SimpleNamespace(graph=SimpleNamespace(graph_outputs=[])),
        )
        with (
            patch.object(arrayref_module.ir, "ShapeAsConstantBuffer", FakeShapePredicate),
            patch.object(arrayref_module, "EnterSubgraphLine", lambda *args: "ENTER"),
            patch.object(arrayref_module, "ExitSubgraphLine", lambda *args: "EXIT"),
        ):
            wrapper.codegen_while_loop(loop)
        code = wrapper.wrapper_call.getvalue()
        self.assertIn("bool loop1_cond_result;", code)
        self.assertIn("if (!loop1_cond_result) break;", code)

    def test_stack_output_is_explicitly_rejected(self):
        wrapper = wrapper_instance()
        with self.assertRaises(NotImplementedError):
            wrapper.codegen_while_loop(object(), stack_output=True)

    def test_borrow_safety_contract_remains_unchanged(self):
        wrapper = wrapper_instance()
        self.assertTrue(wrapper.is_safe_to_use_borrow_arrayref_tensor_as_tensor())
        wrapper.allow_stack_allocation = True
        self.assertFalse(wrapper.is_safe_to_use_borrow_arrayref_tensor_as_tensor())


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_188758 = r'''
import unittest
from unittest.mock import patch

import torch
import torch._inductor.lowering as lowering


class FakeConstant:
    pass


class ExpandSchemaCompatibilityTests(unittest.TestCase):
    @staticmethod
    def exercise(sizes, **kwargs):
        value = FakeConstant()
        sentinel = object()
        with (
            patch.object(lowering, "promote_constants", lambda values: [value]),
            patch.object(lowering.ir, "BaseConstant", FakeConstant),
            patch.object(lowering.ExpandView, "create", return_value=sentinel) as create,
        ):
            result = lowering.expand(value, sizes, **kwargs)
        create.assert_called_once_with(value, tuple(sizes))
        return result, sentinel

    def test_explicit_schema_keyword_is_accepted(self):
        result, sentinel = self.exercise([3, 4], implicit=False)
        self.assertIs(result, sentinel)

    def test_schema_keyword_is_inert_for_multiple_shapes(self):
        for sizes, flag in (([5, 3], False), ([2, 7], True)):
            with self.subTest(sizes=sizes, flag=flag):
                result, sentinel = self.exercise(sizes, implicit=flag)
                self.assertIs(result, sentinel)

    def test_ordinary_expand_without_keyword_remains_stable(self):
        value = torch.arange(6.0).view(1, 6)
        actual = value.expand(4, -1)
        self.assertEqual(actual.shape, (4, 6))
        for row in actual:
            torch.testing.assert_close(row, value[0])

    def test_eager_schema_behavior_is_unchanged(self):
        value = torch.arange(3.0).view(1, 3)
        actual = torch.ops.aten.expand.default(value, [2, 3], implicit=False)
        self.assertEqual(actual.shape, (2, 3))
        torch.testing.assert_close(actual[0], value[0])
        torch.testing.assert_close(actual[1], value[0])


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_187209 = r'''
import contextlib
import unittest

import sympy
import torch
from torch._inductor.codegen.simd import CantSplit, SIMDKernel
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V


class ResidualTilingCompatibilityTests(unittest.TestCase):
    @staticmethod
    def graph_context():
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        stack = contextlib.ExitStack()
        stack.enter_context(V.set_graph_handler(graph))
        return stack

    def test_nonunit_residual_extent_uses_compatibility_failure(self):
        with self.graph_context():
            with self.assertRaises(CantSplit):
                SIMDKernel._split_iteration_ranges(
                    [sympy.Integer(2), sympy.Integer(2)],
                    [[sympy.Integer(2)], []],
                )

    def test_submultiple_domain_is_reported_incompatible(self):
        with self.graph_context():
            compatible = SIMDKernel.is_compatible(
                [sympy.Integer(4), sympy.Integer(8)],
                [[sympy.Integer(4)], []],
            )
        self.assertFalse(compatible)

    def test_exactly_consumed_domain_remains_compatible(self):
        with self.graph_context():
            compatible = SIMDKernel.is_compatible(
                [sympy.Integer(4), sympy.Integer(8)],
                [[sympy.Integer(4), sympy.Integer(8)], []],
            )
        self.assertTrue(compatible)

    def test_excess_lengths_remain_incompatible(self):
        with self.graph_context():
            compatible = SIMDKernel.is_compatible(
                [sympy.Integer(1), sympy.Integer(2), sympy.Integer(2)],
                [[], [sympy.Integer(2), sympy.Integer(2), sympy.Integer(2)]],
            )
        self.assertFalse(compatible)


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_188862 = r'''
import itertools
import unittest

import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRangeAnalysis, ValueRanges


class BooleanMinMaxPropagationTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_compiled_boolean_minimum_and_maximum_match_eager(self):
        def function(value):
            left = value > -0.25
            right = value < 0.75
            return torch.minimum(left, right), torch.maximum(left, right)

        value = torch.linspace(-1, 1, 33)
        expected = function(value)
        actual = torch.compile(function, fullgraph=True)(value)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_boolean_ranges_follow_logical_truth_tables(self):
        values = [sympy.false, sympy.true]
        for left, right in itertools.product(values, repeat=2):
            lhs, rhs = ValueRanges(left, left), ValueRanges(right, right)
            minimum = ValueRangeAnalysis.minimum(lhs, rhs)
            maximum = ValueRangeAnalysis.maximum(lhs, rhs)
            self.assertEqual(minimum.lower, sympy.true if bool(left) and bool(right) else sympy.false)
            self.assertEqual(maximum.lower, sympy.true if bool(left) or bool(right) else sympy.false)

    def test_unknown_boolean_ranges_remain_boolean(self):
        unknown = ValueRanges(sympy.false, sympy.true)
        self.assertTrue(ValueRangeAnalysis.minimum(unknown, unknown).is_bool)
        self.assertTrue(ValueRangeAnalysis.maximum(unknown, unknown).is_bool)

    def test_numeric_range_minimum_and_maximum_remain_stable(self):
        left = ValueRanges(sympy.Integer(-3), sympy.Integer(4))
        right = ValueRanges(sympy.Integer(2), sympy.Integer(9))
        minimum = ValueRangeAnalysis.minimum(left, right)
        maximum = ValueRangeAnalysis.maximum(left, right)
        self.assertEqual((minimum.lower, minimum.upper), (-3, 4))
        self.assertEqual((maximum.lower, maximum.upper), (2, 9))

    def test_compiled_numeric_minimum_remains_stable(self):
        def function(left, right):
            return torch.minimum(left, right), torch.maximum(left, right)

        left = torch.tensor([-2.0, 4.0, 1.0])
        right = torch.tensor([3.0, 0.0, 1.0])
        expected = function(left, right)
        actual = torch.compile(function, fullgraph=True)(left, right)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
'''


TASKS = (
    {
        **support._source_fields(178617),
        "retired_task_id": "pytorch__184514__unsigned_scan_accumulator",
        "task_dir": "tasks/pytorch/178617_large_batch_bmm_grid",
        "task_id": "pytorch__178617__large_batch_bmm_grid",
        "public_task_id": "opbench-v07-t0063",
        "screening_index": 18,
        "source_ref": "pytorch-048c2ae8-large-bmm-grid-overlay",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260417-torch2.13.0dev-cu126-devel-py311",
        "overlay_paths": [
            "torch/_inductor/kernel/bmm.py",
            "torch/_inductor/kernel/templates/triton_bmm.py.jinja",
        ],
        "gold": support._gold(
            178617,
            [
                "torch/_inductor/kernel/bmm.py",
                "torch/_inductor/kernel/templates/triton_bmm.py.jinja",
            ],
        ),
        "hidden": support._new_file_patch(
            "test/op_bench/test_large_batch_bmm_grid.py", HIDDEN_178617
        ),
        "test_file": "test/op_bench/test_large_batch_bmm_grid.py",
        "f2p": [
            "LargeBatchGridTests.test_large_batch_is_split_across_launch_axes",
            "LargeBatchGridTests.test_last_batch_index_remains_representable",
            "LargeBatchGridTests.test_template_reconstructs_and_masks_the_batch_index",
        ],
        "p2p": [
            "LargeBatchGridTests.test_small_batch_preserves_single_depth_launch",
            "LargeBatchGridTests.test_single_batch_preserves_grid_shape",
        ],
        "statement": {
            "title": "Dynamic batched matrix multiplication exceeds a launch-axis limit",
            "body": (
                "A dynamically compiled CUDA batched matrix multiplication works for ordinary batches but "
                "cannot launch once the batch dimension exceeds the device limit of a single grid axis. "
                "Support larger batches without dropping or duplicating matrices. Small and single-batch "
                "launch geometry must remain unchanged."
            ),
            "labels": ["module: inductor", "module: triton", "bug"],
        },
        "known_constraints": [
            "The failing batch is larger than the CUDA y-axis launch limit.",
            "Every batch matrix must map to exactly one valid program instance.",
            "Ordinary batches must retain a unit-depth launch.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "torch.bmm", "component": "TorchInductor Triton template",
            "problem_type": "large-batch-launch-grid", "tags": ["bmm", "cuda", "dynamic-shape", "grid", "triton"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "shape", "compatibility"],
            "trigger_tags": ["extreme_value_or_size", "dynamic_shape", "device_specific"],
            "execution_context": {"devices": ["cuda"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": "The Prompt identifies the public operator, dynamic trigger, and device launch boundary but no template path, grid callback, index formula, program-axis mapping, or patch site.",
            "diagnosis": "Diagnosis must connect a tensor batch extent to a hardware grid-axis limit, then reason about reconstructing a linear batch index across two launch axes without invalid pointer arithmetic.",
            "repair_regression": "Three failure selectors cover launch splitting, terminal-index coverage, and template masking; two controls require unchanged ordinary and singleton launch geometry.",
        },
        "behavior_tokens": ["bmm", "large_batch", "grid_axis_limit", "batch_index_coverage", "dynamic_launch"],
        "risk_signals": [], "estimated_runtime_min": 22,
    },
    {
        **support._source_fields(179278),
        "retired_task_id": "pytorch__183472__sparse_transposed_output_shape",
        "task_dir": "tasks/pytorch/179278_shared_module_reorder",
        "task_id": "pytorch__179278__shared_module_reorder",
        "public_task_id": "opbench-v07-t0061",
        "screening_index": 23,
        "source_ref": "pytorch-c76c52c3-shared-reorder-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260423-torch2.13.0dev-cpu-py311",
        "overlay_paths": ["torch/export/unflatten.py"],
        "gold": support._gold(179278, ["torch/export/unflatten.py"]),
        "hidden": support._new_file_patch(
            "test/op_bench/test_shared_module_reorder.py", HIDDEN_179278
        ),
        "test_file": "test/op_bench/test_shared_module_reorder.py",
        "f2p": [
            "RepeatedChildReorderTests.test_alias_qualified_child_does_not_raise",
            "RepeatedChildReorderTests.test_shared_identity_survives_reordering",
            "RepeatedChildReorderTests.test_alias_uses_base_child_order",
        ],
        "p2p": [
            "RepeatedChildReorderTests.test_ordinary_children_follow_declared_order",
            "RepeatedChildReorderTests.test_nested_ordinary_children_remain_available",
        ],
        "statement": {
            "title": "Export reconstruction fails while ordering a reused child module",
            "body": (
                "Reconstructing an exported module that registers the same child in multiple positions fails "
                "during hierarchy ordering, before the module can run. Preserve the single object instance and "
                "place every registration according to the canonical child order. Ordinary and nested unique "
                "modules must keep their current ordering behavior."
            ),
            "labels": ["module: export", "module: unflatten", "bug"],
        },
        "known_constraints": [
            "One child module is registered under more than one position in the hierarchy.",
            "Reconstruction must preserve object identity rather than cloning the shared child.",
            "Non-shared children and nested modules are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "torch.export reconstruction", "component": "module hierarchy reconstruction",
            "problem_type": "shared-submodule-ordering", "tags": ["export", "unflatten", "aliasing", "module", "serialization"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "mutation_state",
            "contract_detail_tags": ["alias", "serialization", "compatibility"],
            "trigger_tags": ["mutation_or_alias"],
            "execution_context": {"devices": ["cpu"], "modes": ["eager"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_validation"], "component_tags": ["dynamo"],
        },
        "dimension_evidence": {
            "localization": "The Prompt describes shared hierarchy reconstruction but exposes no source path, internal name encoding, ordering table, recursive helper, lookup expression, or fix.",
            "diagnosis": "Diagnosis requires tracing how one module acquires multiple internal identities while only its canonical hierarchy identity participates in the reconstruction order.",
            "repair_regression": "Three selectors exercise alias lookup, identity, and relative ordering; two controls protect ordinary reordering and nested traversal.",
        },
        "behavior_tokens": ["export_unflatten", "shared_module", "alias_identity", "hierarchy_order", "recursive_reconstruction"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 18,
    },
    {
        **support._source_fields(180370),
        "retired_task_id": "pytorch__183959__cuda_low_precision_nan_sign",
        "task_dir": "tasks/pytorch/180370_arrayref_while_loop",
        "task_id": "pytorch__180370__arrayref_while_loop",
        "public_task_id": "opbench-v07-t0040",
        "screening_index": 37,
        "source_ref": "pytorch-9b22041c-arrayref-loop-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260423-torch2.13.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py"],
        "gold": support._gold(
            180370, ["torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py"]
        ),
        "hidden": support._new_file_patch(
            "test/op_bench/test_arrayref_while_loop.py", HIDDEN_180370
        ),
        "test_file": "test/op_bench/test_arrayref_while_loop.py",
        "f2p": [
            "MinimalArrayRefControlFlowTests.test_subgraph_tensor_inputs_receive_owned_handles",
            "MinimalArrayRefControlFlowTests.test_tensor_predicate_while_loop_emits_scalar_check",
            "MinimalArrayRefControlFlowTests.test_scalar_predicate_avoids_tensor_item_conversion",
        ],
        "p2p": [
            "MinimalArrayRefControlFlowTests.test_stack_output_is_explicitly_rejected",
            "MinimalArrayRefControlFlowTests.test_borrow_safety_contract_remains_unchanged",
        ],
        "statement": {
            "title": "Minimal array-reference wrapper cannot emit a functional while loop",
            "body": (
                "A CPU AOT graph containing a functional while loop cannot be emitted through the minimal "
                "array-reference interface. Carried tensors must remain valid across the condition and body, "
                "tensor and scalar predicates must both terminate correctly, and unsupported stack-return mode "
                "must fail explicitly. Existing borrowing safety rules must remain intact."
            ),
            "labels": ["module: inductor", "module: cpp wrapper", "bug"],
        },
        "known_constraints": [
            "Carried tensor values cross two inlined subgraphs and require valid ownership for both.",
            "The condition can be represented by either a scalar or a tensor result.",
            "Borrowing safety and unsupported stack-return behavior are regression boundaries.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "torch.while_loop", "component": "AOTInductor minimal array-reference wrapper",
            "problem_type": "control-flow-wrapper-lifetime", "tags": ["aoti", "while-loop", "arrayref", "ownership", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "mutation_state",
            "contract_detail_tags": ["alias", "exception", "compatibility", "liveness"],
            "trigger_tags": ["mutation_or_alias"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt identifies functional control flow, ownership, and the public wrapper mode but no source path, wrapper subclass, emission hook, handle type, buffer names, or implementation sequence.",
            "diagnosis": "Diagnosis must follow carried tensors through condition and body subgraphs, distinguish scalar from tensor predicates, and reason about borrowed versus owned handles.",
            "repair_regression": "Four failure selectors cover input ownership, both predicate forms, and explicit unsupported mode; a borrowing-safety control prevents weakening lifetime checks.",
        },
        "behavior_tokens": ["while_loop", "minimal_arrayref", "carried_tensor", "predicate_conversion", "handle_lifetime"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 24,
    },
    {
        **support._source_fields(188758),
        "retired_task_id": "pytorch__188575__inference_boolean_mask_export",
        "task_dir": "tasks/pytorch/188758_expand_schema_keyword",
        "task_id": "pytorch__188758__expand_schema_keyword",
        "public_task_id": "opbench-v07-t0034",
        "screening_index": 94,
        "source_ref": "pytorch-75f92218-expand-schema-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260710-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/lowering.py"],
        "gold": support._gold(188758, ["torch/_inductor/lowering.py"]),
        "hidden": support._new_file_patch(
            "test/op_bench/test_expand_schema_keyword.py", HIDDEN_188758
        ),
        "test_file": "test/op_bench/test_expand_schema_keyword.py",
        "f2p": [
            "ExpandSchemaCompatibilityTests.test_explicit_schema_keyword_is_accepted",
            "ExpandSchemaCompatibilityTests.test_schema_keyword_is_inert_for_multiple_shapes",
        ],
        "p2p": [
            "ExpandSchemaCompatibilityTests.test_ordinary_expand_without_keyword_remains_stable",
            "ExpandSchemaCompatibilityTests.test_eager_schema_behavior_is_unchanged",
        ],
        "statement": {
            "title": "Tensor broadcasting rejects a schema-valid optional argument",
            "body": (
                "A graph invokes tensor broadcasting with an optional schema argument. Eager execution accepts "
                "the call, but static and dynamic graph execution reject it before code generation. Make this "
                "path honor the public operator schema without changing ordinary broadcast values or shapes."
            ),
            "labels": ["module: inductor", "module: lowering", "bug"],
        },
        "known_constraints": [
            "The optional argument is valid operator-schema metadata and does not alter expansion values.",
            "Both static and dynamic compiled shapes must work.",
            "Calls that omit the optional argument are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "tensor broadcasting view", "component": "TorchInductor lowering",
            "problem_type": "schema-keyword-compatibility", "tags": ["expand", "autograd", "schema", "dynamic-shape", "compile"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["shape", "schema", "exception", "compatibility"],
            "trigger_tags": ["dynamic_shape"],
            "execution_context": {"devices": ["cpu"], "modes": ["eager", "compile"], "phases": ["forward", "backward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["wrong_dispatch"], "component_tags": ["dispatcher", "inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt names the public operator and schema-level symptom but no source path, lowering function, parameter name, registration site, internal call stack, or patch.",
            "diagnosis": "Diagnosis requires comparing the dispatcher schema with the compiler lowering signature and recognizing metadata that is semantically inert but still part of call compatibility.",
            "repair_regression": "Static and dynamic compiled selectors must accept the schema call while ordinary compiled and eager expansion controls preserve shape and value semantics.",
        },
        "behavior_tokens": ["expand", "schema_metadata", "autograd_graph", "dynamic_compile", "lowering_signature"],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file"], "estimated_runtime_min": 16,
    },
    {
        **support._source_fields(187209),
        "retired_task_id": "pytorch__188862__boolean_minmax_propagation",
        "task_dir": "tasks/pytorch/187209_residual_tiling_fallback",
        "task_id": "pytorch__187209__residual_tiling_fallback",
        "public_task_id": "opbench-v07-t0044",
        "screening_index": 87,
        "source_ref": "pytorch-71b54592-residual-tiling-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260710-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/codegen/simd.py"],
        "gold": support._gold(187209, ["torch/_inductor/codegen/simd.py"]),
        "hidden": support._new_file_patch(
            "test/op_bench/test_residual_tiling_fallback.py", HIDDEN_187209
        ),
        "test_file": "test/op_bench/test_residual_tiling_fallback.py",
        "f2p": [
            "ResidualTilingCompatibilityTests.test_nonunit_residual_extent_uses_compatibility_failure",
            "ResidualTilingCompatibilityTests.test_submultiple_domain_is_reported_incompatible",
        ],
        "p2p": [
            "ResidualTilingCompatibilityTests.test_exactly_consumed_domain_remains_compatible",
            "ResidualTilingCompatibilityTests.test_excess_lengths_remain_incompatible",
        ],
        "statement": {
            "title": "Template fusion aborts when a smaller consumer leaves a tiling extent",
            "body": (
                "A compiled template considers fusing a consumer whose domain is a strict fraction of the "
                "template domain. Each supplied length cleanly fits its corresponding group, but a non-unit "
                "extent remains and compilation aborts instead of declining the fusion. Treat this as an "
                "incompatible mapping while preserving exact mappings and rejection of excess dimensions."
            ),
            "labels": ["module: inductor", "module: codegen", "bug"],
        },
        "known_constraints": [
            "Every supplied length divides the group it consumes, but the groups are not fully consumed.",
            "The caller must receive an incompatibility result so it can skip the fusion.",
            "Exact mappings and cases with too many lengths are distinct regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "compiled template fusion", "component": "TorchInductor iteration tiling",
            "problem_type": "residual-extent-compatibility", "tags": ["inductor", "fusion", "tiling", "epilogue", "fallback"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["exception", "compatibility"],
            "trigger_tags": [],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_validation"], "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": "The Prompt describes the fusion-domain relation and fallback contract but no source path, kernel class, range-splitting helper, exception type, final assertion, or repair line.",
            "diagnosis": "Diagnosis must distinguish a leftover group extent after clean consumption from the separate case where input lengths outlive all groups; only the former currently escapes compatibility handling.",
            "repair_regression": "Two failure selectors require a typed incompatibility at the residual-extent boundary; exact consumption and excess-length controls preserve both neighboring branches.",
        },
        "behavior_tokens": ["residual_group_extent", "submultiple_epilogue", "fusion_rejection", "cant_split", "exact_consumption_control"],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file"], "estimated_runtime_min": 18,
    },
)


def main() -> int:
    for spec in TASKS:
        support._materialize(spec)
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
