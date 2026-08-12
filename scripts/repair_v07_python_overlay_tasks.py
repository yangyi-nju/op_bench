#!/usr/bin/env python3
"""Apply the source-consistent Python overlay repair to failed v0.7 Tasks."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


TASK_DIRECTORIES = (
    "176922_aoti_profile_repeated_dynamic_kernel",
    "177860_flex_attention_captured_grad",
    "178667_semi_structured_cpu_conversion",
    "179333_dynamic_large_offset_template",
    "179505_broadcast_batch_autotune",
    "179686_export_forward_mode_roundtrip",
    "180297_aoti_triton_nonfinite_scalar",
    "181119_benchmark_alias_storage",
    "182971_aoti_user_stream_events",
    "183472_sparse_transposed_output_shape",
    "184624_aoti_shared_dynamic_slice",
    "185314_functional_export_argument_collision",
    "186008_opaque_partition_cost_policy",
    "188485_undefined_tensor_boxing",
    "188575_inference_boolean_mask_export",
    "188907_autotune_failure_resource_lifetime",
)
TRITON_TASK = "188907_autotune_failure_resource_lifetime"
SPARSE_TRANSPOSE_TASK = "183472_sparse_transposed_output_shape"
TREE_OVERLAY_TASK = "178667_semi_structured_cpu_conversion"
DEPENDENCY_PATHS = {
    "181119_benchmark_alias_storage": (
        "torch/_inductor/scheduler.py",
    ),
}
MATCHED_OVERLAY_OVERRIDES = {
    "179686_export_forward_mode_roundtrip": (
        "torch/_dynamo/trace_rules.py",
        "torch/_dynamo/variables/ctx_manager.py",
        "torch/_export/utils.py",
        "torch/_export/verifier.py",
        "torch/_functorch/eager_transforms.py",
        "torch/_functorch/predispatch.py",
        "torch/autograd/forward_ad.py",
        "torch/fx/experimental/proxy_tensor.py",
        "torch/fx/experimental/symbolic_shapes.py",
    ),
    "180297_aoti_triton_nonfinite_scalar": (
        "torch/_inductor/codegen/cpp_wrapper_cpu.py",
        "torch/_inductor/codegen/cpp_wrapper_gpu.py",
        "torch/_inductor/codegen/wrapper.py",
    ),
    "182971_aoti_user_stream_events": (
        "torch/_functorch/_aot_autograd/streams.py",
        "torch/_inductor/codecache.py",
        "torch/_inductor/codegen/aoti_runtime/streams.h",
        "torch/_inductor/codegen/cpp_wrapper_gpu.py",
        "torch/_inductor/codegen/wrapper.py",
        "torch/_inductor/lowering.py",
        "torch/_inductor/scheduler.py",
        "torch/_inductor/stream_utils.py",
    ),
    "184624_aoti_shared_dynamic_slice": (
        "torch/_inductor/codegen/cpp_wrapper_cpu.py",
        "torch/_inductor/codegen/wrapper.py",
        "torch/_inductor/graph.py",
        "torch/_subclasses/meta_utils.py",
    ),
    "186008_opaque_partition_cost_policy": (
        "torch/_functorch/config.py",
        "torch/_functorch/partitioners.py",
    ),
    "188485_undefined_tensor_boxing": (
        "torch/_inductor/codegen/cpp_wrapper_cpu.py",
    ),
    "188575_inference_boolean_mask_export": (
        "torch/_subclasses/fake_tensor.py",
    ),
    "188907_autotune_failure_resource_lifetime": (
        "torch/_inductor/runtime/triton_heuristics.py",
    ),
}
ENVIRONMENT_REF_OVERRIDES = {
    "179686_export_forward_mode_roundtrip": (
        "pytorch-nightly-20260423-torch2.13.0dev-cpu-py311"
    ),
    "180297_aoti_triton_nonfinite_scalar": (
        "pytorch-nightly-20260417-torch2.13.0dev-cu126-devel-py311"
    ),
    "182971_aoti_user_stream_events": (
        "pytorch-nightly-20260707-torch2.14.0dev-cu126-devel-py311"
    ),
    "183472_sparse_transposed_output_shape": (
        "pytorch-nightly-20260710-torch2.14.0dev-cu126-py311"
    ),
    "184624_aoti_shared_dynamic_slice": (
        "pytorch-nightly-20260612-torch2.14.0dev-cu126-devel-py311"
    ),
    "186008_opaque_partition_cost_policy": (
        "pytorch-nightly-20260612-torch2.14.0dev-cpu-py311"
    ),
    "188485_undefined_tensor_boxing": (
        "pytorch-nightly-20260707-torch2.14.0dev-cpu-py311"
    ),
    "188575_inference_boolean_mask_export": (
        "pytorch-nightly-20260707-torch2.14.0dev-cpu-py311"
    ),
}


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{label}: expected object")
    return value


def repair() -> tuple[str, ...]:
    repaired: list[str] = []
    for directory in TASK_DIRECTORIES:
        path = ROOT / "tasks/pytorch" / directory / "task.json"
        manifest = _object(json.loads(path.read_bytes()), str(path))
        environment = _object(manifest.get("environment"), "environment")
        source_loading = _object(
            environment.get("source_loading"), "environment.source_loading"
        )
        if source_loading.get("mode") != "python_overlay":
            raise ContractError(f"{directory}: expected python_overlay")
        if directory == TREE_OVERLAY_TASK:
            source_loading["overlay_tree"] = "torch"
        else:
            source_loading.pop("overlay_tree", None)
        overlay_paths = source_loading.get("overlay_paths")
        if not isinstance(overlay_paths, list):
            raise ContractError("environment.source_loading.overlay_paths")
        override = MATCHED_OVERLAY_OVERRIDES.get(directory)
        if override is not None:
            overlay_paths[:] = override
        for dependency in DEPENDENCY_PATHS.get(directory, ()):
            if dependency not in overlay_paths:
                overlay_paths.append(dependency)
        environment_ref = ENVIRONMENT_REF_OVERRIDES.get(directory)
        if environment_ref is not None:
            manifest["environment_ref"] = environment_ref

        if directory == SPARSE_TRANSPOSE_TASK:
            evaluation = _object(manifest.get("evaluation"), "evaluation")
            evaluation["pass_to_pass"] = [
                "SparseTransposedOutputShapeTests.test_cutlass_regular_orientation_remains_stable",
                "SparseTransposedOutputShapeTests.test_cusparselt_input_transpose_without_fused_output_is_stable",
            ]
            complexity_path = (
                ROOT / "tasks/pytorch" / directory / "quality/complexity.json"
            )
            complexity = _object(
                json.loads(complexity_path.read_bytes()),
                "complexity",
            )
            dimensions = _object(
                complexity.get("dimension_evidence"),
                "complexity.dimension_evidence",
            )
            dimensions["repair_regression"] = (
                "The repair coordinates call-site transpose ownership, padded "
                "narrowing, contiguous materialization, and fake output-shape "
                "construction across two backend paths. Two transposed "
                "fail-to-pass cases and two complementary orientation controls "
                "reject an unconditional final transpose or a shape-only "
                "metadata patch."
            )
            complexity["reviewer"] = (
                "codex-primary-semantic-complexity-runtime-reviewer-v07"
            )
            complexity["reviewed_at"] = "2026-08-10T10:22:00Z"
            complexity["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in complexity.items()
                    if key != "content_hash"
                }
            )
            complexity_path.write_text(
                canonical_json(complexity), encoding="utf-8"
            )

        if directory == TRITON_TASK:
            manifest["environment_ref"] = (
                "pytorch-nightly-20260710-torch2.14.0dev-cu126-py311"
            )
            manifest["runtime_tier"] = "cuda_python_overlay"
            environment["tier"] = "cuda_python_overlay"
            environment["preflight_commands"] = [
                "{python} --version",
                (
                    "{python} -c \"import expecttest, torch, triton; "
                    "print(torch.__version__); print(torch.version.git_version); "
                    "print(torch.version.cuda); print(torch.cuda.is_available()); "
                    "print(torch.__file__); print(triton.__version__)\""
                ),
            ]
            taxonomy = _object(manifest.get("taxonomy"), "taxonomy")
            execution = _object(
                taxonomy.get("execution_context"),
                "taxonomy.execution_context",
            )
            execution["devices"] = ["cuda"]
            operator = _object(manifest.get("operator"), "operator")
            tags = operator.get("tags")
            if not isinstance(tags, list):
                raise ContractError("operator.tags: expected list")
            operator["tags"] = [
                "cuda" if tag == "cpu" else tag for tag in tags
            ]
            evaluation = _object(manifest.get("evaluation"), "evaluation")
            evaluation["fail_to_pass"] = [
                "AutotuneFailureResourceTests.test_mixed_candidates_release_caller_resource"
            ]
            evaluation["pass_to_pass"] = [
                "AutotuneFailureResourceTests.test_all_failed_candidates_release_caller_resource",
                "AutotuneFailureResourceTests.test_all_successful_candidates_preserve_launchers",
                "AutotuneFailureResourceTests.test_low_stage_retry_behavior_is_preserved",
            ]
            statement = _object(manifest.get("statement"), "statement")
            statement["body"] = (
                "Compiler autotuning can prepare several candidate implementations "
                "before choosing executable launchers. When an earlier candidate "
                "succeeds and a later candidate fails, selection returns the valid "
                "launcher, but with cyclic collection disabled a caller-owned "
                "benchmark resource remains reachable afterward and the retained "
                "memory accumulates across compiled kernels. Repair resource lifetime "
                "so caller state is released immediately while preserving selected "
                "launchers, all-invalid failure reporting and resource release, and "
                "the existing lower-stage retry behavior."
            )
            complexity_path = (
                ROOT
                / "tasks/pytorch"
                / directory
                / "quality/complexity.json"
            )
            complexity = _object(
                json.loads(complexity_path.read_bytes()),
                "complexity",
            )
            dimensions = _object(
                complexity.get("dimension_evidence"),
                "complexity.dimension_evidence",
            )
            dimensions["diagnosis"] = (
                "Correct diagnosis must explain why a handled build failure can keep "
                "its traceback and caller frame chain alive after a valid launcher is "
                "returned, why refcount release differs when cyclic collection is "
                "disabled, and why the all-invalid error path is an important "
                "unchanged contrast rather than a second failing path."
            )
            dimensions["repair_regression"] = (
                "The repair must release failure-only state on mixed-candidate normal "
                "completion while preserving launcher order, all-invalid error "
                "reporting and resource release, all-success behavior, and the "
                "lower-stage retry control."
            )
            complexity["reviewed_at"] = "2026-08-10T09:45:00Z"
            complexity["reviewer"] = (
                "codex-primary-semantic-complexity-reviewer-v07"
            )
            complexity["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in complexity.items()
                    if key != "content_hash"
                }
            )
            complexity_path.write_bytes(
                canonical_json(complexity).encode("utf-8")
            )

        path.write_bytes(canonical_json(manifest).encode("utf-8"))
        repaired.append(str(manifest.get("task_id")))
    return tuple(repaired)


def main() -> int:
    repaired = repair()
    print(
        canonical_json(
            {"repaired_task_count": len(repaired), "tasks": list(repaired)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
