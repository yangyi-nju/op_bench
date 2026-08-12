#!/usr/bin/env python3
"""Build three code-generation focused offline replacements for v0.7."""

from __future__ import annotations

from itertools import count
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


HIDDEN_182825 = r'''
import unittest

from torch._inductor.codegen.cpp_wrapper_gpu import DeferredTritonCallWrapper
from torch._inductor.utils import IndentedBuffer


class DeviceCodegen:
    @staticmethod
    def cpp_device_ptr():
        return "CUdeviceptr"


class Wrapper:
    device_codegen = DeviceCodegen()

    @staticmethod
    def codegen_dtype(dtype):
        return "aoti_torch_dtype_uint8"

    @staticmethod
    def codegen_device(device):
        return "aoti_torch_device_type_cuda, device_idx_"


class LazyScratchGridTests(unittest.TestCase):
    @staticmethod
    def generated():
        deferred = DeferredTritonCallWrapper(
            wrapper_name="call_kernel",
            kernel_name="sample_kernel",
            kernel_name_to_body={},
            arg_types=[],
        )
        prefix = IndentedBuffer()
        call_args = deferred._generate_lazy_scratch(prefix, Wrapper(), "base_arg")
        return prefix.getvalue(), call_args

    def test_global_scratch_scales_with_full_launch_grid(self):
        code, _ = self.generated()
        self.assertIn(
            "sample_kernel_result.global_scratch * static_cast<int64_t>(grid_0) * grid_1 * grid_2",
            code,
        )

    def test_profile_scratch_scales_with_full_launch_grid(self):
        code, _ = self.generated()
        self.assertIn(
            "sample_kernel_result.profile_scratch * static_cast<int64_t>(grid_0) * grid_1 * grid_2",
            code,
        )

    def test_scratch_pointers_remain_kernel_arguments(self):
        _, call_args = self.generated()
        self.assertEqual(
            call_args,
            "base_arg, &global_scratch_ptr, &profile_scratch_ptr",
        )

    def test_zero_scratch_guards_remain_present(self):
        code, _ = self.generated()
        self.assertIn("if (sample_kernel_result.global_scratch > 0)", code)
        self.assertIn("if (sample_kernel_result.profile_scratch > 0)", code)


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_184514 = r'''
import unittest

import torch


class UnsignedScanCompilationTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        torch.manual_seed(0)

    def tearDown(self):
        torch._dynamo.reset()

    @staticmethod
    def compile_and_compare(function, value):
        expected = function(value)
        actual = torch.compile(function, fullgraph=True)(value)
        torch.testing.assert_close(actual, expected)

    def test_uint8_cumulative_sum_compiles(self):
        value = torch.randint(0, 5, (8, 16), dtype=torch.uint8, device="cuda")
        self.compile_and_compare(lambda x: x.cumsum(-1), value)

    def test_uint8_cumulative_product_compiles(self):
        value = torch.randint(0, 3, (4, 12), dtype=torch.uint8, device="cuda")
        self.compile_and_compare(lambda x: x.cumprod(-1), value)

    def test_signed_integer_scan_remains_stable(self):
        value = torch.randint(-3, 4, (5, 9), dtype=torch.int32, device="cuda")
        self.compile_and_compare(lambda x: x.cumsum(0), value)

    def test_floating_scan_remains_stable(self):
        value = torch.randn(6, 11, device="cuda")
        self.compile_and_compare(lambda x: x.cumsum(1), value)


if __name__ == "__main__":
    unittest.main()
'''


HIDDEN_185257 = r'''
import unittest
from itertools import count
from types import SimpleNamespace

from torch._inductor import config
from torch._inductor.codegen.cpp_wrapper_cpu_array_ref import CppWrapperCpuArrayRef
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V


class FakeGraph:
    constants = {}
    graph_outputs = [object(), object()]
    is_const_graph = False
    aot_mode = False


def wrapper_instance():
    wrapper = object.__new__(CppWrapperCpuArrayRef)
    wrapper.cached_output_id = count()
    wrapper.scalar_to_tensor_id = count()
    wrapper.wrapper_call = IndentedBuffer()
    wrapper.v2_raw_wrapper_body = IndentedBuffer()
    wrapper.v2_raw_output_refs = None
    return wrapper


class CachedOutputInitializationTests(unittest.TestCase):
    def test_tensor_interface_constructor_expression_uses_braces(self):
        wrapper = wrapper_instance()
        with config.patch({"aot_inductor.use_minimal_arrayref_interface": False}):
            with V.set_graph_handler(FakeGraph()):
                wrapper.generate_return(["RAIIAtenTensorHandle(tmp_x)"])
        code = wrapper.wrapper_call.getvalue()
        self.assertRegex(
            code,
            r"cached_output_\d+\{RAIIAtenTensorHandle\(tmp_x\)\};",
        )

    def test_raw_array_interface_constructor_expression_uses_braces(self):
        wrapper = wrapper_instance()
        code = IndentedBuffer()
        with V.set_graph_handler(FakeGraph()):
            wrapper._codegen_v2_raw_outputs(
                code, ["RAIIAtenTensorHandle(tmp_x)"]
            )
        self.assertRegex(
            code.getvalue(),
            r"cached_output_\d+\{RAIIAtenTensorHandle\(tmp_x\)\};",
        )

    def test_null_output_remains_ignored(self):
        wrapper = wrapper_instance()
        with config.patch({"aot_inductor.use_minimal_arrayref_interface": False}):
            with V.set_graph_handler(FakeGraph()):
                wrapper.generate_return(["nullptr"])
        self.assertNotIn("ThreadLocalCachedOutput", wrapper.wrapper_call.getvalue())

    def test_handle_release_branch_remains_emitted(self):
        wrapper = wrapper_instance()
        with config.patch({"aot_inductor.use_minimal_arrayref_interface": False}):
            with V.set_graph_handler(FakeGraph()):
                wrapper.generate_return(["plain_output"])
        self.assertIn("plain_output.release()", wrapper.wrapper_call.getvalue())


if __name__ == "__main__":
    unittest.main()
'''


TASKS = (
    {
        **support._source_fields(182825),
        "retired_task_id": "pytorch__183973__unrealized_addmm_bias",
        "task_dir": "tasks/pytorch/182825_lazy_scratch_grid_extent",
        "task_id": "pytorch__182825__lazy_scratch_grid_extent",
        "public_task_id": "opbench-v07-t0050",
        "screening_index": 62,
        "source_ref": "pytorch-b8777a2c-lazy-scratch-overlay",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/codegen/cpp_wrapper_gpu.py"],
        "gold": support._gold(182825, ["torch/_inductor/codegen/cpp_wrapper_gpu.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_lazy_scratch_grid_extent.py", HIDDEN_182825),
        "test_file": "test/op_bench/test_lazy_scratch_grid_extent.py",
        "f2p": [
            "LazyScratchGridTests.test_global_scratch_scales_with_full_launch_grid",
            "LazyScratchGridTests.test_profile_scratch_scales_with_full_launch_grid",
        ],
        "p2p": [
            "LazyScratchGridTests.test_scratch_pointers_remain_kernel_arguments",
            "LazyScratchGridTests.test_zero_scratch_guards_remain_present",
        ],
        "statement": {
            "title": "Lazy CUDA launch under-allocates per-block scratch memory",
            "body": (
                "A lazily compiled CUDA kernel reports scratch memory required by each launch block. Warmup "
                "succeeds, but the cached C++ launch allocates space for only one block, so multiple blocks "
                "can overwrite one another. Repair global and profiling scratch allocation for a three-dimensional "
                "runtime grid while preserving zero-size guards and kernel argument wiring."
            ),
            "labels": ["module: inductor", "module: cpp wrapper", "bug"],
        },
        "known_constraints": [
            "The runtime metadata reports scratch bytes per launch block rather than per kernel launch.",
            "All three runtime grid dimensions contribute to the total allocation.",
            "Zero-size guards and both scratch pointer arguments are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "lazy CUDA kernel launch", "component": "AOTInductor C++ wrapper",
            "problem_type": "per-block-scratch-allocation", "tags": ["cuda", "scratch", "grid", "memory", "cpp-wrapper"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "efficiency_safety",
            "contract_detail_tags": ["compatibility", "memory", "liveness"],
            "trigger_tags": ["extreme_value_or_size", "device_specific"],
            "execution_context": {"devices": ["cuda"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "wrong_result", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor", "cuda_kernel"],
        },
        "dimension_evidence": {
            "localization": "The Prompt exposes the per-block versus per-launch contract but no source path, wrapper class, emitted C++ variables, scratch field names, allocation helper, or fix.",
            "diagnosis": "Diagnosis must connect device-side per-block scratch indexing to host allocation size and reason across three runtime grid dimensions and two independent workspaces.",
            "repair_regression": "The repair must scale global and profiling allocations without changing pointer ABI or zero-size guards; four code-generation assertions reject constant sizing or launch rewiring.",
        },
        "behavior_tokens": ["lazy_launch", "per_cta_scratch", "runtime_grid", "cached_cpp_launch", "memory_safety"],
        "risk_signals": ["gold_patch_single_file"], "estimated_runtime_min": 20,
    },
    {
        **support._source_fields(184514),
        "retired_task_id": "pytorch__181941__aoti_concurrent_const_fold",
        "task_dir": "tasks/pytorch/184514_unsigned_scan_accumulator",
        "task_id": "pytorch__184514__unsigned_scan_accumulator",
        "public_task_id": "opbench-v07-t0063",
        "screening_index": 71,
        "source_ref": "pytorch-6574ed24-unsigned-scan-overlay",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/codegen/triton.py"],
        "gold": support._gold(184514, ["torch/_inductor/codegen/triton.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_unsigned_scan_accumulator.py", HIDDEN_184514),
        "test_file": "test/op_bench/test_unsigned_scan_accumulator.py",
        "f2p": [
            "UnsignedScanCompilationTests.test_uint8_cumulative_sum_compiles",
            "UnsignedScanCompilationTests.test_uint8_cumulative_product_compiles",
        ],
        "p2p": [
            "UnsignedScanCompilationTests.test_signed_integer_scan_remains_stable",
            "UnsignedScanCompilationTests.test_floating_scan_remains_stable",
        ],
        "statement": {
            "title": "CUDA cumulative scans fail for unsigned inputs",
            "body": (
                "Compiling cumulative sum and product over unsigned CUDA tensors fails while constructing the "
                "generated scan kernel, although eager execution succeeds. Repair non-persistent scan code "
                "generation so unsigned inputs compile and match eager values. Signed integer and floating-point "
                "scans must retain their current behavior."
            ),
            "labels": ["module: inductor", "module: triton", "bug"],
        },
        "known_constraints": [
            "The failure affects non-persistent cumulative scans over unsigned element types.",
            "Both cumulative sum and cumulative product must compile and match eager execution.",
            "Signed integer and floating-point scans are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "CUDA cumulative scan", "component": "TorchInductor Triton codegen",
            "problem_type": "unsigned-scan-initialization", "tags": ["scan", "cumsum", "cumprod", "unsigned", "cuda"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["numerical", "dtype", "exception", "compatibility"],
            "trigger_tags": ["mixed_dtype_or_precision_mode", "device_specific"],
            "execution_context": {"devices": ["cuda"], "modes": ["eager", "compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_cast", "incorrect_lowering"], "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": "The Prompt identifies scan behavior and unsigned dtype but no source path, kernel class, accumulator allocation, default literal, Triton builder API, or fix.",
            "diagnosis": "Diagnosis requires tracing non-persistent scan initialization into generated typed constants and explaining why a dead first-iteration value is invalid only for unsigned builders.",
            "repair_regression": "Although Gold is one line, two unsigned scan families plus signed and floating controls prove the value is safe and reject disabling scan fusion or changing result dtype.",
        },
        "behavior_tokens": ["unsigned_scan", "nonpersistent_accumulator", "typed_constant", "cumulative_sum", "cumulative_product"],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file"], "estimated_runtime_min": 22,
    },
    {
        **support._source_fields(185257),
        "retired_task_id": "pytorch__179278__shared_module_unflatten_order",
        "task_dir": "tasks/pytorch/185257_cached_output_initialization",
        "task_id": "pytorch__185257__cached_output_initialization",
        "public_task_id": "opbench-v07-t0064",
        "screening_index": 76,
        "source_ref": "pytorch-b6f9631d-cached-output-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260612-torch2.14.0dev-cpu-py311",
        "overlay_paths": ["torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py"],
        "gold": support._gold(185257, ["torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py"]),
        "hidden": support._new_file_patch("test/op_bench/test_cached_output_initialization.py", HIDDEN_185257),
        "test_file": "test/op_bench/test_cached_output_initialization.py",
        "f2p": [
            "CachedOutputInitializationTests.test_tensor_interface_constructor_expression_uses_braces",
            "CachedOutputInitializationTests.test_raw_array_interface_constructor_expression_uses_braces",
        ],
        "p2p": [
            "CachedOutputInitializationTests.test_null_output_remains_ignored",
            "CachedOutputInitializationTests.test_handle_release_branch_remains_emitted",
        ],
        "statement": {
            "title": "Generated C++ misparses a cached tensor-view output",
            "body": (
                "A CPU AOT graph returns a dtype-reinterpreted view represented by a constructor expression. "
                "The generated wrapper intends to create thread-local cached output state, but C++ parses the "
                "line as a function declaration and rejects its storage qualifier. Repair both tensor and raw-array "
                "wrapper paths while preserving null-output handling and handle-release behavior."
            ),
            "labels": ["module: inductor", "module: cpp wrapper", "bug"],
        },
        "known_constraints": [
            "The output expression is a constructor call rather than a simple variable name.",
            "Both the tensor-return and raw array-reference wrapper paths must emit an unambiguous definition.",
            "Null outputs and the ordinary handle release branch are regression controls.",
        ],
        "operator": {
            "framework": "pytorch", "operator_name": "AOTInductor cached output", "component": "CPU C++ wrapper codegen",
            "problem_type": "constructor-expression-declaration-ambiguity", "tags": ["aoti", "cpp", "arrayref", "view", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2", "contract_family": "api_behavior",
            "contract_detail_tags": ["layout", "exception", "compatibility"],
            "trigger_tags": ["noncontiguous_or_special_layout"],
            "execution_context": {"devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False},
            "failure_type": "unexpected_error", "root_cause_tags": ["incorrect_lowering"], "component_tags": ["inductor"],
        },
        "dimension_evidence": {
            "localization": "The Prompt explains the generated C++ ambiguity but gives no source path, codegen class, cache helper type, internal buffer, initialization token, or patch location.",
            "diagnosis": "Diagnosis must connect a view constructor expression to C++ declaration grammar and find two distinct wrapper emission sites that share the same ambiguity.",
            "repair_regression": "The four selectors cover both affected output interfaces and two unrelated branches; a narrow syntax repair must retain ownership and null-output semantics.",
        },
        "behavior_tokens": ["cached_output", "constructor_expression", "cpp_declaration", "arrayref_interface", "thread_local"],
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
