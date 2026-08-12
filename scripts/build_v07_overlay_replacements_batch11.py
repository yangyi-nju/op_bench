#!/usr/bin/env python3
"""Build four distinct exact-commit replacements for the last v0.7 slots."""

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


HIDDEN_176922 = r'''
import sys
import unittest
from types import SimpleNamespace

import sympy
from torch._inductor import config
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.utils import IndentedBuffer


def wrapper_instance():
    wrapper = object.__new__(CppWrapperCpu)
    wrapper.kernel_numel_expr = set()
    wrapper.wrapper_call = IndentedBuffer()
    wrapper.writeline = wrapper.wrapper_call.writeline
    return wrapper


def symbolic_argument(name="kernel_xnumel", value=128):
    return SimpleNamespace(inner=sympy.Symbol(name), inner_expr=sympy.Integer(value))


class ProfiledKernelScopeTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform in ("linux", "win32"), "profiling scopes are platform-specific")
    def test_repeated_profiled_call_redeclares_numel(self):
        wrapper = wrapper_instance()
        argument = symbolic_argument()
        graph = object()
        with config.patch({"cpp.enable_kernel_profile": True}):
            wrapper._generate_symbolic_call_arg_helper(argument, graph)
            wrapper._generate_symbolic_call_arg_helper(argument, graph)
        lines = [line.strip() for line in wrapper.wrapper_call.getvalue().splitlines()]
        self.assertEqual(sum(line.startswith("int64_t kernel_xnumel =") for line in lines), 2)
        self.assertEqual(sum(line.startswith("kernel_xnumel =") for line in lines), 0)

    @unittest.skipUnless(sys.platform in ("linux", "win32"), "profiling scopes are platform-specific")
    def test_three_profiled_scopes_each_receive_a_declaration(self):
        wrapper = wrapper_instance()
        argument = symbolic_argument(value=257)
        graph = object()
        with config.patch({"cpp.enable_kernel_profile": True}):
            for _ in range(3):
                wrapper._generate_symbolic_call_arg_helper(argument, graph)
        code = wrapper.wrapper_call.getvalue()
        self.assertEqual(code.count("int64_t kernel_xnumel ="), 3)

    def test_unprofiled_repeated_call_reuses_declaration(self):
        wrapper = wrapper_instance()
        argument = symbolic_argument()
        graph = object()
        with config.patch({"cpp.enable_kernel_profile": False}):
            wrapper._generate_symbolic_call_arg_helper(argument, graph)
            wrapper._generate_symbolic_call_arg_helper(argument, graph)
        lines = [line.strip() for line in wrapper.wrapper_call.getvalue().splitlines()]
        self.assertEqual(sum(line.startswith("int64_t kernel_xnumel =") for line in lines), 1)
        self.assertEqual(sum(line.startswith("kernel_xnumel =") for line in lines), 1)

    @unittest.skipUnless(sys.platform in ("linux", "win32"), "profiling scopes are platform-specific")
    def test_distinct_graph_scopes_remain_independent(self):
        wrapper = wrapper_instance()
        argument = symbolic_argument("kernel_rnumel", 64)
        with config.patch({"cpp.enable_kernel_profile": True}):
            wrapper._generate_symbolic_call_arg_helper(argument, object())
            wrapper._generate_symbolic_call_arg_helper(argument, object())
        self.assertEqual(
            wrapper.wrapper_call.getvalue().count("int64_t kernel_rnumel ="),
            2,
        )


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_179333 = r'''
import inspect
import unittest
from types import SimpleNamespace

import torch
from torch._inductor.select_algorithm import TritonTemplateKernel


def kernel_with(feature_dtype, override):
    kernel = object.__new__(TritonTemplateKernel)
    kernel.features = SimpleNamespace(select_index_dtype=lambda: feature_dtype)
    kernel._index_dtype_override = override
    return kernel


class TemplateIndexWidthTests(unittest.TestCase):
    def test_explicit_wide_index_overrides_small_output_heuristic(self):
        kernel = kernel_with(torch.int32, "tl.int64")
        self.assertEqual(kernel.index_dtype, "tl.int64")

    def test_explicit_narrow_index_is_preserved(self):
        kernel = kernel_with(torch.int64, "tl.int32")
        self.assertEqual(kernel.index_dtype, "tl.int32")

    def test_kernel_constructor_accepts_computed_index_width(self):
        parameters = inspect.signature(TritonTemplateKernel.__init__).parameters
        self.assertIn("index_dtype_override", parameters)
        self.assertIsNone(parameters["index_dtype_override"].default)

    def test_default_narrow_feature_selection_remains_stable(self):
        kernel = kernel_with(torch.int32, None)
        self.assertEqual(kernel.index_dtype, "tl.int32")

    def test_default_wide_feature_selection_remains_stable(self):
        kernel = kernel_with(torch.int64, None)
        self.assertEqual(kernel.index_dtype, "tl.int64")


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_181119 = r'''
import unittest
from collections import OrderedDict
from types import MethodType, SimpleNamespace

import torch
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V


class FakeInput:
    def __init__(self, shape, stride, dtype=torch.float32):
        self.shape = shape
        self.stride = stride
        self.dtype = dtype

    def get_size(self):
        return self.shape

    def get_stride(self):
        return self.stride

    def get_device(self):
        return torch.device("cpu")

    def get_dtype(self):
        return self.dtype


class FakeSizeVars:
    backed_var_to_val = {}

    @staticmethod
    def optimization_hints(values, fallback):
        return list(values)


def render_get_args(example_inputs):
    graph_inputs = OrderedDict(
        (f"arg{index}", FakeInput(list(value.shape), list(value.stride()), value.dtype))
        for index, value in enumerate(example_inputs)
    )
    graph = SimpleNamespace(
        constants={},
        torchbind_constants={},
        graph_inputs=graph_inputs,
        example_inputs=example_inputs,
        sizevars=FakeSizeVars(),
    )
    wrapper = object.__new__(PythonWrapperCodegen)
    wrapper.codegen_python_shape_tuple = MethodType(
        lambda self, values: repr(tuple(values)), wrapper
    )
    output = IndentedBuffer()
    with V.set_graph_handler(graph):
        wrapper.benchmark_compiled_module(output)
    return output.getvalue()


def execute_get_args(code):
    namespace = {"torch": torch}
    exec(code, namespace)
    return namespace["get_args"]()


class BenchmarkAliasStorageTests(unittest.TestCase):
    def test_two_views_reuse_one_generated_storage(self):
        base = torch.arange(64, dtype=torch.float32)
        left = torch.as_strided(base, (3, 4), (4, 1), 2)
        right = torch.as_strided(base, (3, 4), (4, 1), 10)
        code = render_get_args([left, right])
        self.assertEqual(code.count("_shared_storage_0 = rand_strided("), 1)
        self.assertEqual(code.count("torch.as_strided(_shared_storage_0"), 2)
        recreated_left, recreated_right = execute_get_args(code)
        self.assertEqual(
            recreated_left.untyped_storage().data_ptr(),
            recreated_right.untyped_storage().data_ptr(),
        )

    def test_three_aliases_share_the_same_generated_storage(self):
        base = torch.arange(96, dtype=torch.float32)
        views = [torch.as_strided(base, (2, 5), (5, 1), offset) for offset in (0, 10, 20)]
        recreated = execute_get_args(render_get_args(views))
        pointers = {value.untyped_storage().data_ptr() for value in recreated}
        self.assertEqual(len(pointers), 1)
        self.assertEqual(len(recreated), 3)

    def test_independent_inputs_remain_independent(self):
        left = torch.randn(2, 3)
        right = torch.randn(2, 3)
        code = render_get_args([left, right])
        self.assertNotIn("_shared_storage_", code)
        recreated_left, recreated_right = execute_get_args(code)
        self.assertNotEqual(
            recreated_left.untyped_storage().data_ptr(),
            recreated_right.untyped_storage().data_ptr(),
        )

    def test_single_input_generation_remains_stable(self):
        code = render_get_args([torch.randn(4, 7)])
        self.assertNotIn("_shared_storage_", code)
        (value,) = execute_get_args(code)
        self.assertEqual(value.shape, (4, 7))
        self.assertEqual(value.stride(), (7, 1))


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_188575 = r'''
import unittest
from types import SimpleNamespace

from torch._subclasses.fake_tensor import SymNumberMemoDescriptor


class MemoHolder:
    cached = SymNumberMemoDescriptor()

    def __init__(self, *, inference):
        self._cached = None
        self._cached_vc = None
        self._cached_epoch = None
        self._inference = inference
        self._version = 0
        self.fake_mode = SimpleNamespace(epoch=1)

    def is_inference(self):
        return self._inference


class InferenceMemoDescriptorTests(unittest.TestCase):
    def test_inference_value_is_stored_and_retrieved(self):
        holder = MemoHolder(inference=True)
        holder.cached = 17
        self.assertEqual(holder.cached, 17)

    def test_inference_value_can_be_updated_in_same_epoch(self):
        holder = MemoHolder(inference=True)
        holder.cached = 5
        holder.cached = 11
        self.assertEqual(holder.cached, 11)
        self.assertIsNone(holder._cached_vc)

    def test_normal_value_tracks_version_counter(self):
        holder = MemoHolder(inference=False)
        holder.cached = 23
        self.assertEqual(holder.cached, 23)
        holder._version += 1
        self.assertIsNone(holder.cached)

    def test_epoch_change_invalidates_value(self):
        holder = MemoHolder(inference=False)
        holder.cached = 31
        holder.fake_mode.epoch += 1
        self.assertIsNone(holder.cached)

    def test_explicit_clear_remains_stable(self):
        holder = MemoHolder(inference=False)
        holder.cached = 7
        holder.cached = None
        self.assertIsNone(holder.cached)
        self.assertIsNone(holder._cached_vc)
        self.assertIsNone(holder._cached_epoch)


if __name__ == "__main__":
    unittest.main()
'''


TASKS = (
    {
        **support._source_fields(176922),
        "retired_task_id": "pytorch__178617__large_batch_bmm_grid",
        "task_dir": "tasks/pytorch/176922_profiled_kernel_scope",
        "task_id": "pytorch__176922__profiled_kernel_scope",
        "public_task_id": "opbench-v07-t0063",
        "screening_index": 4,
        "source_ref": "pytorch-4924764e-profile-scope-overlay-v2",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/codegen/cpp_wrapper_cpu.py"],
        "gold": support._gold(176922, ["torch/_inductor/codegen/cpp_wrapper_cpu.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_profiled_kernel_scope.py", HIDDEN_176922),
        "test_file": "test/op_bench/test_profiled_kernel_scope.py",
        "f2p": [
            "ProfiledKernelScopeTests.test_repeated_profiled_call_redeclares_numel",
            "ProfiledKernelScopeTests.test_three_profiled_scopes_each_receive_a_declaration",
        ],
        "p2p": [
            "ProfiledKernelScopeTests.test_unprofiled_repeated_call_reuses_declaration",
            "ProfiledKernelScopeTests.test_distinct_graph_scopes_remain_independent",
        ],
        "statement": {
            "title": "Kernel profiling loses a dynamic launch size between repeated calls",
            "body": (
                "An ahead-of-time CUDA wrapper invokes equivalent generated work more than once while kernel "
                "profiling is enabled. Each profiled call is emitted in its own native scope, but later calls "
                "refer to launch-size state established only in an earlier scope, so packaging fails. Ordinary "
                "unprofiled behavior and isolation between different graph scopes must remain unchanged."
            ),
            "labels": ["module: aotinductor", "module: profiling", "bug"],
        },
        "known_constraints": [
            "Profiling places repeated generated calls in separate native scopes.",
            "The dynamic launch extent is required by every repeated call.",
            "Unprofiled calls and distinct graph scopes are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "profiled AOT kernel call", "component": "AOTInductor C++ wrapper",
            "problem_type": "profile-scope-declaration-lifetime", "tags": ["aoti", "profiling", "dynamic-shape", "codegen", "cuda"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["shape", "exception", "compatibility", "liveness"],
            "trigger_tags": ["dynamic_shape", "device_specific"],
            "execution_context": {"devices": ["cuda"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor", "cuda_kernel"],
        },
        "dimension_evidence": {
            "localization": "The Prompt exposes scope lifetime and repeated profiled calls but no source path, wrapper class, helper method, cache key, emitted type, or repair condition.",
            "diagnosis": "Diagnosis requires connecting generated native block scope to compiler-side declaration caching and distinguishing profiling scope lifetime from graph lifetime.",
            "repair_regression": "Two repeated-call selectors require per-scope declarations; unprofiled reuse and distinct-graph controls preserve both neighboring cache policies.",
        },
        "behavior_tokens": ["profile_scope", "repeated_kernel_call", "dynamic_launch_extent", "declaration_lifetime", "unprofiled_reuse"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 20,
    },
    {
        **support._source_fields(179333),
        "retired_task_id": "pytorch__180370__arrayref_while_loop",
        "task_dir": "tasks/pytorch/179333_template_index_width",
        "task_id": "pytorch__179333__template_index_width",
        "public_task_id": "opbench-v07-t0040",
        "screening_index": 24,
        "source_ref": "pytorch-6cda861d-template-index-overlay-v2",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/select_algorithm.py"],
        "gold": support._gold(179333, ["torch/_inductor/select_algorithm.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_template_index_width.py", HIDDEN_179333),
        "test_file": "test/op_bench/test_template_index_width.py",
        "f2p": [
            "TemplateIndexWidthTests.test_explicit_wide_index_overrides_small_output_heuristic",
            "TemplateIndexWidthTests.test_explicit_narrow_index_is_preserved",
            "TemplateIndexWidthTests.test_kernel_constructor_accepts_computed_index_width",
        ],
        "p2p": [
            "TemplateIndexWidthTests.test_default_narrow_feature_selection_remains_stable",
            "TemplateIndexWidthTests.test_default_wide_feature_selection_remains_stable",
        ],
        "statement": {
            "title": "A dynamic matrix view overflows template pointer indexing",
            "body": (
                "A CUDA matrix multiplication slices a dynamically sized input at a storage offset beyond the "
                "32-bit range. The output itself is small, so generated template code chooses narrow pointer "
                "indices and overflows when applying the input offset. Preserve the width computed from all "
                "template buffers while retaining default narrow and wide selection for ordinary kernels."
            ),
            "labels": ["module: inductor", "module: triton", "bug"],
        },
        "known_constraints": [
            "The large value is an input storage offset rather than the output element count.",
            "Dynamic slicing makes the offset part of the generated kernel signature.",
            "Default narrow and wide feature selection are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "dynamic CUDA matrix multiplication", "component": "TorchInductor Triton template",
            "problem_type": "large-storage-offset-index-width", "tags": ["matmul", "storage-offset", "int64", "dynamic-shape", "cuda"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "efficiency_safety",
            "contract_detail_tags": ["value", "compatibility", "memory"],
            "trigger_tags": ["extreme_value_or_size", "dynamic_shape", "device_specific"],
            "execution_context": {"devices": ["cuda"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "wrong_result", "root_cause_tags": ["overflow", "incorrect_lowering"], "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": "The Prompt distinguishes input offset from output size but gives no source path, template class, constructor argument, index property, heuristic call, or patch site.",
            "diagnosis": "Diagnosis must trace a dynamic storage offset through template signature generation and discover that a later kernel object recomputes width from a smaller output domain.",
            "repair_regression": "Three selectors bind explicit width propagation and constructor support; two controls require the legacy feature-based width path when no override exists.",
        },
        "behavior_tokens": ["large_storage_offset", "template_signature", "index_width_override", "small_output", "pointer_arithmetic"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 22,
    },
    {
        **support._source_fields(181119),
        "retired_task_id": "pytorch__188758__expand_schema_keyword",
        "task_dir": "tasks/pytorch/181119_benchmark_alias_storage_v2",
        "task_id": "pytorch__181119__benchmark_alias_storage_v2",
        "public_task_id": "opbench-v07-t0034",
        "screening_index": 50,
        "source_ref": "pytorch-4336c967-benchmark-alias-overlay-v2",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260423-torch2.13.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/codegen/wrapper.py"],
        "gold": support._gold(181119, ["torch/_inductor/codegen/wrapper.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_benchmark_alias_storage_v2.py", HIDDEN_181119),
        "test_file": "test/op_bench/test_benchmark_alias_storage_v2.py",
        "f2p": [
            "BenchmarkAliasStorageTests.test_two_views_reuse_one_generated_storage",
            "BenchmarkAliasStorageTests.test_three_aliases_share_the_same_generated_storage",
        ],
        "p2p": [
            "BenchmarkAliasStorageTests.test_independent_inputs_remain_independent",
            "BenchmarkAliasStorageTests.test_single_input_generation_remains_stable",
        ],
        "statement": {
            "title": "Generated performance inputs lose shared storage relationships",
            "body": (
                "A compiled graph accepts multiple tensor views backed by the same non-empty storage. Its "
                "generated timing helper loses that relationship, so the measurement no longer represents "
                "alias-sensitive behavior and can use far more memory. The failure occurs for groups of two "
                "and three views; independent inputs and graphs with one input remain unaffected."
            ),
            "labels": ["module: inductor", "module: benchmarking", "bug"],
        },
        "known_constraints": [
            "The affected inputs are distinct views of one non-empty storage.",
            "Generated shapes and strides must be preserved for every view.",
            "Independent and single inputs must not be grouped.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "generated compiled-module timing harness", "component": "TorchInductor Python wrapper",
            "problem_type": "benchmark-input-alias-loss", "tags": ["benchmark", "alias", "storage", "view", "memory"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "mutation_state",
            "contract_detail_tags": ["shape", "stride", "alias", "compatibility", "memory"],
            "trigger_tags": ["noncontiguous_or_special_layout", "mutation_or_alias"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "wrong_result", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt identifies generated benchmark semantics and storage sharing but no source path, wrapper method, storage key type, emitted helper name, allocation variable, or fix.",
            "diagnosis": "Diagnosis must distinguish tensor identity from storage identity, reconstruct shape/stride views, and preserve grouping without merging independent inputs.",
            "repair_regression": "Two failure selectors cover two- and three-view groups plus executable generated code; independent and singleton controls prevent over-grouping.",
        },
        "behavior_tokens": ["benchmark_harness", "shared_storage", "aliased_views", "generated_inputs", "allocation_group"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 22,
    },
    {
        **support._source_fields(188575),
        "retired_task_id": "pytorch__187209__residual_tiling_fallback",
        "task_dir": "tasks/pytorch/188575_inference_memo_descriptor_v2",
        "task_id": "pytorch__188575__inference_memo_descriptor_v2",
        "public_task_id": "opbench-v07-t0044",
        "screening_index": 92,
        "source_ref": "pytorch-14c2be65-inference-memo-overlay-v2",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260710-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/_subclasses/fake_tensor.py"],
        "gold": support._gold(188575, ["torch/_subclasses/fake_tensor.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_inference_memo_descriptor_v2.py", HIDDEN_188575),
        "test_file": "test/op_bench/test_inference_memo_descriptor_v2.py",
        "f2p": [
            "InferenceMemoDescriptorTests.test_inference_value_is_stored_and_retrieved",
            "InferenceMemoDescriptorTests.test_inference_value_can_be_updated_in_same_epoch",
        ],
        "p2p": [
            "InferenceMemoDescriptorTests.test_normal_value_tracks_version_counter",
            "InferenceMemoDescriptorTests.test_epoch_change_invalidates_value",
            "InferenceMemoDescriptorTests.test_explicit_clear_remains_stable",
        ],
        "statement": {
            "title": "Inference fake tensors lose a reused symbolic result",
            "body": (
                "Two fake tensors are filtered by the same boolean mask while inference mode is active. The "
                "first symbolic result is not retained, so repeating the mask operation allocates an unrelated "
                "symbol and later shape reasoning fails. Preserve reusable inference-mode memo values without "
                "weakening normal version, tracing-epoch, or explicit-clear invalidation."
            ),
            "labels": ["module: fake tensor", "module: symbolic shapes", "bug"],
        },
        "known_constraints": [
            "Inference tensors do not expose ordinary version-counter semantics.",
            "Repeated use within one tracing epoch must retrieve the same memoized value.",
            "Normal mutation, epoch change, and explicit clear remain invalidation controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "inference fake-tensor symbolic memo", "component": "FakeTensor symbolic shape propagation",
            "problem_type": "inference-memo-loss", "tags": ["fake-tensor", "inference", "symbolic-shape", "memoization", "mask"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "mutation_state",
            "contract_detail_tags": ["shape", "mutation", "state", "compatibility", "liveness"],
            "trigger_tags": ["dynamic_shape", "mutation_or_alias"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_validation"], "component_tags": ["dynamo"],
        },
        "dimension_evidence": {
            "localization": "The Prompt states inference-mode memo and invalidation contracts but no source path, descriptor class, backing attribute names, version field, epoch field, or setter/getter repair.",
            "diagnosis": "Diagnosis must account for the absence of inference version counters while retaining a separate tracing-epoch invalidation model and normal mutation semantics.",
            "repair_regression": "Two inference selectors require storage and update; three controls cover normal version invalidation, epoch invalidation, and explicit clear.",
        },
        "behavior_tokens": ["inference_fake_tensor", "symbolic_memo", "version_counter_absence", "epoch_invalidation", "mask_reuse"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 20,
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
