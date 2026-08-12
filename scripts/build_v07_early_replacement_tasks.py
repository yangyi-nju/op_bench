#!/usr/bin/env python3
"""Build two v0.7 replacements from the local pre-April PyTorch history."""

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


LOCAL_HISTORY = (
    ROOT
    / ".op_bench_cache/sources/pytorch/pytorch"
    / "f1a9f64dfd40dd4c1a7a6f2a51d670d53545607d"
    / "source"
)


GOLD_177419 = """diff --git a/torch/_inductor/scheduler.py b/torch/_inductor/scheduler.py
--- a/torch/_inductor/scheduler.py
+++ b/torch/_inductor/scheduler.py
@@ -361,6 +361,9 @@ class MixOrderReduction:
         if not V.graph.sizevars.statically_known_leq(ncol, 1024 * 16):
             return False
 
+        if MixOrderReduction.is_split_reduction(contiguous_node):
+            return False
+
         # Other reduction types like max/min is not supported yet.
         # There are no real use case as well.
         out = all(
"""


HIDDEN_177419 = """diff --git a/test/op_bench/test_split_reduction_backward.py b/test/op_bench/test_split_reduction_backward.py
new file mode 100644
--- /dev/null
+++ b/test/op_bench/test_split_reduction_backward.py
@@ -0,0 +1,72 @@
+import unittest
+
+import torch
+from torch._inductor import config, metrics
+
+
+@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
+class SplitReductionBackwardTests(unittest.TestCase):
+    def setUp(self):
+        torch._dynamo.reset()
+        metrics.reset()
+        torch.manual_seed(0)
+
+    def tearDown(self):
+        torch._dynamo.reset()
+
+    @staticmethod
+    def reference_workload(value, weight, eps):
+        original_dtype = value.dtype
+        promoted = value.float()
+        inverse_rms = torch.rsqrt(
+            promoted.square().sum(dim=-1) / promoted.shape[-1] + eps
+        )
+        return (promoted * inverse_rms[:, None] * weight).to(original_dtype)
+
+    def collect_derivatives(self, callable_obj, value, weight, upstream):
+        value.grad = None
+        weight.grad = None
+        output = callable_obj(value, weight, 1e-5)
+        output.backward(upstream)
+        return value.grad.detach().clone(), weight.grad.detach().clone()
+
+    def run_case(self, rows, columns):
+        value = torch.randn(
+            rows, columns, device="cuda", dtype=torch.float32, requires_grad=True
+        )
+        weight = torch.randn(
+            columns, device="cuda", dtype=torch.float32, requires_grad=True
+        )
+        upstream = torch.randn_like(value)
+        expected = self.collect_derivatives(
+            self.reference_workload, value, weight, upstream
+        )
+        compiled = torch.compile(self.reference_workload, fullgraph=True)
+        actual = self.collect_derivatives(compiled, value, weight, upstream)
+        for observed, reference in zip(actual, expected):
+            torch.testing.assert_close(observed, reference, atol=1e-3, rtol=1e-3)
+        return actual
+
+    def test_large_feature_split_backward_matches_eager(self):
+        patches = {
+            "triton.mix_order_reduction": True,
+            "triton.mix_order_reduction_non_strict_mode": True,
+        }
+        with config.patch(patches):
+            derivatives = self.run_case(13, 8472)
+        self.assertEqual(derivatives[0].shape, (13, 8472))
+        self.assertEqual(derivatives[1].shape, (8472,))
+
+    def test_unsplit_backward_remains_stable(self):
+        patches = {
+            "triton.mix_order_reduction": True,
+            "triton.mix_order_reduction_non_strict_mode": True,
+        }
+        with config.patch(patches):
+            derivatives = self.run_case(8, 1024)
+        self.assertEqual(derivatives[0].shape, (8, 1024))
+        self.assertEqual(derivatives[1].shape, (1024,))
+
+
+if __name__ == "__main__":
+    unittest.main()
"""


GOLD_178076 = """diff --git a/torch/_decomp/decompositions.py b/torch/_decomp/decompositions.py
--- a/torch/_decomp/decompositions.py
+++ b/torch/_decomp/decompositions.py
@@ -5399,7 +5399,9 @@ def isin(elements, test_elements, *, assume_unique=False, invert=False):
         else:
             return torch.eq(elements, test_elements)
 
-    if test_elements.numel() < 10.0 * pow(elements.numel(), 0.145):
+    from torch.fx.experimental.symbolic_shapes import guard_or_false
+
+    if guard_or_false(test_elements.numel() < 10.0 * pow(elements.numel(), 0.145)):
         return isin_default(elements, test_elements, invert=invert)
     else:
         return isin_sorting(
"""


HIDDEN_178076 = """diff --git a/test/op_bench/test_dynamic_isin_decomposition.py b/test/op_bench/test_dynamic_isin_decomposition.py
new file mode 100644
--- /dev/null
+++ b/test/op_bench/test_dynamic_isin_decomposition.py
@@ -0,0 +1,63 @@
+import unittest
+
+import torch
+from torch.export import Dim, export
+
+
+class IsinWorkflow(torch.nn.Module):
+    def __init__(self, *, invert=False):
+        super().__init__()
+        self.invert = invert
+
+    def forward(self, elements, test_elements):
+        return torch.isin(elements, test_elements, invert=self.invert)
+
+
+class DynamicIsinDecompositionTests(unittest.TestCase):
+    def export_dynamic(self, module, elements, test_elements):
+        return export(
+            module,
+            (elements, test_elements),
+            dynamic_shapes={
+                "elements": {0: Dim("element_count", min=1, max=32)},
+                "test_elements": {0: Dim("test_count", min=1, max=16)},
+            },
+        )
+
+    def assert_case(self, decomposed, module, elements, test_elements):
+        expected = module(elements, test_elements)
+        actual = decomposed.module()(elements, test_elements)
+        torch.testing.assert_close(actual, expected)
+        self.assertEqual(actual.dtype, torch.bool)
+        self.assertEqual(actual.shape, elements.shape)
+
+    def test_dynamic_membership_decomposition_round_trip(self):
+        for invert in (False, True):
+            with self.subTest(invert=invert):
+                module = IsinWorkflow(invert=invert)
+                initial_elements = torch.tensor([1, 2, 3, 4, 5])
+                initial_tests = torch.tensor([2, 4])
+                captured = self.export_dynamic(
+                    module, initial_elements, initial_tests
+                )
+                decomposed = captured.run_decompositions()
+                for elements, tests in (
+                    (initial_elements, initial_tests),
+                    (torch.tensor([7, 8, 9]), torch.tensor([8])),
+                    (
+                        torch.arange(12, dtype=torch.int64),
+                        torch.tensor([0, 3, 6, 9]),
+                    ),
+                ):
+                    self.assert_case(decomposed, module, elements, tests)
+
+    def test_fixed_membership_decomposition_remains_stable(self):
+        module = IsinWorkflow()
+        elements = torch.tensor([4, 1, 4, 2])
+        tests = torch.tensor([1, 3, 4])
+        decomposed = export(module, (elements, tests)).run_decompositions()
+        self.assert_case(decomposed, module, elements, tests)
+
+
+if __name__ == "__main__":
+    unittest.main()
"""


TASKS = (
    {
        "retired_task_id": "pytorch__181845__symbolic_concat_layout_guard",
        "task_dir": "tasks/pytorch/177419_split_reduction_backward_group",
        "task_id": "pytorch__177419__split_reduction_backward_group",
        "public_task_id": "opbench-v07-t0045",
        "pr_number": 177419,
        "screening_index": 8,
        "base_commit": "030360f5f8dc3bdbc1028f68878e2e9fc1decbec",
        "merge_commit": "910e89b1c31c8e888545fe1364f7e012b3f09218",
        "source_ref": "pytorch-030360f5-split-reduction-overlay",
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/scheduler.py"],
        "gold": GOLD_177419,
        "hidden": HIDDEN_177419,
        "test_file": "test/op_bench/test_split_reduction_backward.py",
        "f2p": ["SplitReductionBackwardTests.test_large_feature_split_backward_matches_eager"],
        "p2p": ["SplitReductionBackwardTests.test_unsplit_backward_remains_stable"],
        "statement": {
            "title": "Feature-axis splitting destabilizes a fused normalization backward pass",
            "body": (
                "A CUDA normalization-like function reduces a large feature axis, applies a learned scale, and "
                "runs backward. Eager execution produces both gradients, but torch.compile cannot produce a valid "
                "backward result when the feature reduction is internally split and combined with another reduction "
                "order. Repair the large-feature case so both gradients match eager execution. A smaller feature "
                "dimension that does not require splitting must retain its current behavior."
            ),
            "labels": ["module: inductor", "module: reductions", "bug"],
        },
        "known_constraints": [
            "The failure appears in backward after a large contiguous feature reduction is split.",
            "Both input and learned-scale gradients must match eager execution.",
            "A smaller unsplit feature reduction is a required regression control.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "normalization backward reduction",
            "component": "torch.compile",
            "problem_type": "split-reduction-group-mismatch",
            "tags": ["normalization", "backward", "reduction", "fusion", "cuda"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "gradient",
            "contract_detail_tags": ["value", "numerical", "gradient", "exception", "compatibility"],
            "trigger_tags": ["extreme_value_or_size", "device_specific"],
            "execution_context": {
                "devices": ["cuda"], "modes": ["compile"], "phases": ["backward"], "distributed": False,
            },
            "failure_type": "unexpected_error",
            "root_cause_tags": ["incorrect_lowering"],
            "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": (
                "The Prompt gives a public normalization-shaped workload and the size-dependent backward trigger but "
                "no source path, scheduler class, fusion predicate, grouping representation, or repair location. A "
                "solver must inspect forward and backward graphs and the generated reduction schedule."
            ),
            "diagnosis": (
                "Correct diagnosis must connect the feature-axis split to a change in dimension factorization between "
                "fusion selection and code generation, distinguish the two reduction orders, and determine why an "
                "otherwise legal fusion becomes invalid only for the split case."
            ),
            "repair_regression": (
                "The repair must prevent the incompatible fusion without disabling the optimization globally, recover "
                "both gradient tensors for the large feature size, and preserve the unsplit control. The tiny Gold "
                "delta is justified only after nontrivial scheduler and generated-code analysis."
            ),
        },
        "behavior_tokens": [
            "normalization_backward", "split_reduction", "dimension_factorization", "fusion", "gradient"
        ],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file", "single_hidden_f2p"],
        "estimated_runtime_min": 24,
    },
    {
        "retired_task_id": "pytorch__188771__nested_reduction_pointer_rewind",
        "task_dir": "tasks/pytorch/178076_dynamic_isin_decomposition",
        "task_id": "pytorch__178076__dynamic_isin_decomposition",
        "public_task_id": "opbench-v07-t0043",
        "pr_number": 178076,
        "screening_index": 13,
        "base_commit": "8b44e3d44dc8d492ff4adaffd55276fb684e7ca1",
        "merge_commit": "86b29cc5ecdc3b15d47ee31796ad3b9e5060d08c",
        "source_ref": "pytorch-8b44e3d4-dynamic-isin-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cpu-py311",
        "overlay_paths": ["torch/_decomp/decompositions.py"],
        "gold": GOLD_178076,
        "hidden": HIDDEN_178076,
        "test_file": "test/op_bench/test_dynamic_isin_decomposition.py",
        "f2p": ["DynamicIsinDecompositionTests.test_dynamic_membership_decomposition_round_trip"],
        "p2p": ["DynamicIsinDecompositionTests.test_fixed_membership_decomposition_remains_stable"],
        "statement": {
            "title": "Dynamic membership export fails while applying decompositions",
            "body": (
                "A CPU module performs tensor membership testing and exports successfully with dynamic lengths for "
                "both inputs. Applying the standard decomposition table to the exported program then raises a "
                "data-dependent shape error before the decomposed module can run. Repair the dynamic export workflow "
                "so normal and inverted membership results match eager execution across several valid lengths. "
                "Fixed-shape decomposition behavior must remain unchanged."
            ),
            "labels": ["module: export", "module: decompositions", "bug"],
        },
        "known_constraints": [
            "Both the values tensor and membership-set tensor have independently dynamic one-dimensional lengths.",
            "Normal and inverted membership modes must survive export and decomposition.",
            "A fixed-shape exported program is a required regression control.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "torch.isin",
            "component": "torch.export",
            "problem_type": "symbolic-decomposition-branch",
            "tags": ["membership", "export", "decomposition", "symbolic-shape", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "shape", "serialization", "exception", "compatibility"],
            "trigger_tags": ["dynamic_shape"],
            "execution_context": {
                "devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False,
            },
            "failure_type": "unexpected_error",
            "root_cause_tags": ["incorrect_validation"],
            "component_tags": ["dynamo"],
        },
        "dimension_evidence": {
            "localization": (
                "The Prompt identifies membership export and the decomposition phase but exposes no source path, "
                "decomposition function, symbolic predicate, threshold expression, guard helper, or repair location. "
                "The solver must reproduce a failure that occurs after export rather than during eager execution."
            ),
            "diagnosis": (
                "Correct diagnosis must distinguish a data-dependent tensor value from an unbacked symbolic size, "
                "identify a decomposition strategy threshold that Python tries to decide during tracing, and choose "
                "a conservative symbolic outcome without changing concrete branch selection."
            ),
            "repair_regression": (
                "The repair must support two independently dynamic lengths, normal and inverted membership semantics, "
                "multiple post-export input sizes, and a fixed-shape control. It must preserve both decomposition "
                "strategies for concrete shapes while preventing symbolic evaluation from escaping into Python."
            ),
        },
        "behavior_tokens": [
            "membership", "dynamic_length", "export_decomposition", "symbolic_branch", "invert"
        ],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file", "single_hidden_f2p"],
        "estimated_runtime_min": 16,
    },
)


def _git_text(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _materialize(spec: dict[str, object]) -> None:
    destination = (
        ROOT
        / ".op_bench_cache/sources/pytorch/pytorch"
        / str(spec["base_commit"])
        / "source"
    )
    expected_tree = _git_text(LOCAL_HISTORY, "rev-parse", f"{spec['base_commit']}^{{tree}}")
    if destination.exists():
        observed_tree = _git_text(destination, "rev-parse", "HEAD^{tree}")
        if observed_tree != expected_tree:
            raise RuntimeError(f"{destination}: existing snapshot has the wrong tree")
    else:
        materialize_frozen_git_revision(
            LOCAL_HISTORY,
            str(spec["base_commit"]),
            destination,
        )
    spec["source_snapshot_commit"] = _git_text(destination, "rev-parse", "HEAD")
    spec["tracked_file_count"] = len(
        _git_text(destination, "ls-files", "--deduplicate").splitlines()
    )


def main() -> int:
    for spec in TASKS:
        _materialize(spec)
    replacement_builder.TASKS = TASKS
    replacement_builder.BASE_SOURCE_REGISTRY = (
        ROOT / "runs/v0.7_quality_admission_staging/source_registry.json"
    )
    for spec in TASKS:
        replacement_builder._build_task(spec)
    replacement_builder._register_sources()
    replacement_builder._update_review_packets()
    print(
        canonical_json(
            {
                "built": [spec["task_id"] for spec in TASKS],
                "source_registry": replacement_builder.STAGING_SOURCE_REGISTRY.relative_to(ROOT).as_posix(),
                "source_count": len(
                    json.loads(
                        replacement_builder.STAGING_SOURCE_REGISTRY.read_bytes()
                    )["sources"]
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
