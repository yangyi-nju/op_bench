#!/usr/bin/env python3
"""Build the source-bound v0.7 replacement tasks that are ready offline.

The command is intentionally deterministic.  It writes task-private artifacts,
formal quality evidence, source-registry entries, and swaps the corresponding
records in the staging review packets.  Runtime Admission is a separate step.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.complexity import (  # noqa: E402
    build_complexity_evidence,
    semantic_duplicate_fingerprint,
)
from op_bench.factory.prompt_quality import build_prompt_quality_evidence  # noqa: E402
from op_bench.factory.quality_release import quality_prompt_source_inputs  # noqa: E402
from op_bench.factory.taxonomy import parse_taxonomy_v2  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


PROMPT_PACKET = ROOT / "runs/v0.7_quality_admission_staging/prompt_review_packet.json"
REASSESSMENT_PACKET = (
    ROOT / "runs/v0.7_quality_admission_staging/reassessment_review_packet.json"
)
SCREENING_ROOT = ROOT / "factory/v0.7/p8/screening"
SOURCE_REGISTRY = ROOT / "sources/registry.json"
BASE_SOURCE_REGISTRY = SOURCE_REGISTRY
STAGING_SOURCE_REGISTRY = ROOT / "sources/staging_v07_replacements.json"
SCANNER_VERSION = "prompt-overlap-v2"


GOLD_181845 = """diff --git a/torch/_inductor/config.py b/torch/_inductor/config.py
--- a/torch/_inductor/config.py
+++ b/torch/_inductor/config.py
@@ -951,7 +951,14 @@ def use_autoheuristic(name: str) -> bool:
 # The check is disabled if set to None.
 max_fusion_unique_io_buffers: int | None = None
 
-# max number of inputs to generate cat as a pointwise op with masked loads
+# max number of inputs to always fuse cat as a pointwise op regardless of
+# per-input op complexity. Beyond this limit (up to max_pointwise_cat_inputs),
+# fusion is only applied when every input has a low op count.
+max_complex_pointwise_cat_inputs = 8
+
+# max number of inputs to generate cat as a pointwise op with masked loads.
+# Inputs beyond max_complex_pointwise_cat_inputs but within this limit are
+# only fused when every input has a simple computation (op count <= 2).
 max_pointwise_cat_inputs = 8
 
 # force concat to be generated as a pointwise op with masked loads
diff --git a/torch/_inductor/ir.py b/torch/_inductor/ir.py
--- a/torch/_inductor/ir.py
+++ b/torch/_inductor/ir.py
@@ -6094,11 +6094,15 @@ def create(cls, inputs: Sequence[IRNode], dim: int) -> StorageBox:
                 # pyrefly: ignore [missing-attribute]
                 "val" in arg.meta
                 and (
-                    # pyrefly: ignore [missing-attribute]
-                    arg.meta["val"].is_contiguous(memory_format=torch.channels_last)
-                    # pyrefly: ignore [missing-attribute]
-                    or arg.meta["val"].is_contiguous(
-                        memory_format=torch.channels_last_3d
+                    is_contiguous_for_memory_format_or_false(
+                        # pyrefly: ignore [missing-attribute]
+                        arg.meta["val"],
+                        memory_format=torch.channels_last,
+                    )
+                    or is_contiguous_for_memory_format_or_false(
+                        # pyrefly: ignore [missing-attribute]
+                        arg.meta["val"],
+                        memory_format=torch.channels_last_3d,
                     )
                 )
                 for arg in fx_node_args
diff --git a/torch/_inductor/lowering.py b/torch/_inductor/lowering.py
--- a/torch/_inductor/lowering.py
+++ b/torch/_inductor/lowering.py
@@ -2157,17 +2157,14 @@ def op_count(x):
 
         return count
 
-    # as of inputs increase, possibility for register spilling also increases
-    # past a certain threshold of inputs we only fuse if the input kernels
-    # are simple
-    # not sure if we want to expose to users via config since logic may change in future
-    MAX_COMPLEX_POINTWISE_CAT = 8
+    # as inputs increase, possibility for register spilling also increases.
+    # Past a certain threshold we only fuse if the input kernels are simple.
     MAX_SIMPLE_OP_COUNT = 2
 
     def additional_pointwise_ops(op: torch._ops.OpOverload):
         return op in (aten.cat.default, aten.constant_pad_nd.default)
 
-    if len(inputs) <= MAX_COMPLEX_POINTWISE_CAT or (
+    if len(inputs) <= config.max_complex_pointwise_cat_inputs or (
         (len(inputs) <= config.max_pointwise_cat_inputs)
         and all(op_count(t) <= MAX_SIMPLE_OP_COUNT for t in inputs)
     ):
"""


HIDDEN_181845 = """diff --git a/test/op_bench/test_symbolic_concat_layout.py b/test/op_bench/test_symbolic_concat_layout.py
new file mode 100644
--- /dev/null
+++ b/test/op_bench/test_symbolic_concat_layout.py
@@ -0,0 +1,71 @@
+import unittest
+
+import torch
+from torch._dynamo import config as dynamo_config
+from torch._inductor import config as inductor_config
+
+
+class SymbolicConcatLayoutTests(unittest.TestCase):
+    def setUp(self):
+        torch._dynamo.reset()
+        torch.manual_seed(0)
+
+    def tearDown(self):
+        torch._dynamo.reset()
+
+    @staticmethod
+    def dynamic_concat(mask, template):
+        selected = torch.nonzero(mask).size(0)
+        inputs = [
+            template.new_zeros(
+                template.size(0), selected, template.size(2), template.size(3)
+            ),
+            template.new_zeros(
+                template.size(0), selected + 1, template.size(2), template.size(3)
+            ),
+        ]
+        return torch.cat(inputs, dim=1)
+
+    def test_data_dependent_concat_matches_eager(self):
+        patches = {
+            "max_complex_pointwise_cat_inputs": 1,
+            "max_pointwise_cat_inputs": 1,
+        }
+        with dynamo_config.patch({"capture_dynamic_output_shape_ops": True}):
+            with inductor_config.patch(patches):
+                compiled = torch.compile(self.dynamic_concat, fullgraph=True)
+                template = torch.randn(2, 3, 4, 10)
+                for mask in (
+                    torch.tensor([1.0, 0.0, 1.0, 1.0]),
+                    torch.tensor([0.0, 1.0, 0.0, 0.0]),
+                ):
+                    with self.subTest(selected=int(mask.count_nonzero())):
+                        expected = self.dynamic_concat(mask, template)
+                        actual = compiled(mask, template)
+                        torch.testing.assert_close(actual, expected)
+                        self.assertEqual(actual.shape, expected.shape)
+
+    def test_static_channels_last_concat_remains_stable(self):
+        def model(left, right):
+            return torch.cat([left.sin(), right.cos()], dim=1)
+
+        patches = {
+            "max_complex_pointwise_cat_inputs": 1,
+            "max_pointwise_cat_inputs": 1,
+        }
+        left = torch.randn(2, 3, 5, 7).contiguous(
+            memory_format=torch.channels_last
+        )
+        right = torch.randn(2, 2, 5, 7).contiguous(
+            memory_format=torch.channels_last
+        )
+        with inductor_config.patch(patches):
+            actual = torch.compile(model, fullgraph=True)(left, right)
+        expected = model(left, right)
+        torch.testing.assert_close(actual, expected)
+        self.assertEqual(actual.shape, expected.shape)
+        self.assertEqual(actual.dtype, expected.dtype)
+
+
+if __name__ == "__main__":
+    unittest.main()
"""


GOLD_188771 = """diff --git a/torch/_inductor/codegen/triton.py b/torch/_inductor/codegen/triton.py
--- a/torch/_inductor/codegen/triton.py
+++ b/torch/_inductor/codegen/triton.py
@@ -4162,18 +4162,19 @@ def codegen_block_ptr(
                     self.block_ptr_to_buffer[block_descriptor] = name
 
                     # Generate block pointer advancements, for later use.
+                    # We record the entry for every level, even when the
+                    # per-level offset is zero. The outer-loop suffix computes
+                    # a rewind as `outer_step - inner_step * inner_num_iter`;
+                    # if a pointer's outer entry is absent, no rewind is
+                    # emitted and its SSA value (in scf.for-based backends
+                    # such as Triton-MTIA) retains the accumulated inner
+                    # advances across outer iterations, silently loading
+                    # out-of-bounds. The emit site below drops pure no-op
+                    # advances so this does not add codegen noise for
+                    # pointers that are truly constant across all levels.
                     for symt in TritonSymbols.reduction_types:
                         advance_offsets = indexing.advance_roffset(symt)
 
-                        # Ignore identity advancements.
-                        if all(
-                            V.graph.sizevars.statically_known_equals(
-                                offset, sympy.Integer(0)
-                            )
-                            for offset in advance_offsets
-                        ):
-                            continue
-
                         advancements = self.pointer_advancements[symt]
                         if block_descriptor in advancements:
                             raise AssertionError(
@@ -6260,8 +6261,6 @@ def codegen_body(self):
                             prev_advancements = self.pointer_advancements[
                                 prev_tree.symt
                             ]
-                            # block_ptr may not exist in the inner loop's advancements
-                            # if its advancement was identity (zero) and was skipped
                             if block_ptr in prev_advancements:
                                 prev_advancement = prev_advancements[block_ptr]
                                 prev_block = TritonSymbols.get_block_size(prev_tree)
@@ -6271,6 +6270,17 @@ def codegen_body(self):
                                     for cur, prev in zip(advancement, prev_advancement)
                                 ]
 
+                        # Drop pure no-op advances to avoid emitting
+                        # `tl.advance(ptr, [0, 0, ...])` for pointers that
+                        # are constant across every level.
+                        if all(
+                            V.graph.sizevars.statically_known_equals(
+                                offset, sympy.Integer(0)
+                            )
+                            for offset in advancement
+                        ):
+                            continue
+
                         self.body.writeline(
                             DeferredLine(
                                 self.block_ptr_to_buffer[block_ptr],
"""


HIDDEN_188771 = """diff --git a/test/op_bench/test_nested_reduction_pointer_state.py b/test/op_bench/test_nested_reduction_pointer_state.py
new file mode 100644
--- /dev/null
+++ b/test/op_bench/test_nested_reduction_pointer_state.py
@@ -0,0 +1,57 @@
+import re
+import unittest
+
+import torch
+from torch._inductor.utils import run_and_get_code
+
+
+@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
+class NestedReductionPointerStateTests(unittest.TestCase):
+    def setUp(self):
+        torch._dynamo.reset()
+        torch.cuda.empty_cache()
+        torch.manual_seed(0)
+
+    def tearDown(self):
+        torch._dynamo.reset()
+
+    @staticmethod
+    def nested_reduction(left, right):
+        product = left * right
+        return product.sum(), product.mean()
+
+    def test_broadcast_reduction_consumes_every_outer_tile(self):
+        size = 1024
+        left = torch.randn(size, device="cuda") + 3.0
+        right = (torch.randn(size, device="cuda") + 3.0).view(-1, 1)
+        compiled = torch.compile(self.nested_reduction, fullgraph=True)
+        actual, sources = run_and_get_code(compiled, left, right)
+        expected = self.nested_reduction(left, right)
+
+        torch.testing.assert_close(actual[0], expected[0], rtol=2e-4, atol=1e-5)
+        torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=1e-5)
+
+        generated = "\\n".join(sources)
+        negative_advance = re.compile(
+            r"tl\\.advance\\([^\\n]*,\\s*\\[[^\\]]*-[1-9][0-9]*"
+        )
+        self.assertRegex(
+            generated,
+            negative_advance,
+            "nested reduction must restore carried pointer state between outer tiles",
+        )
+
+    def test_single_axis_reduction_remains_stable(self):
+        def model(value):
+            return value.square().sum(dim=1)
+
+        value = torch.randn(128, 257, device="cuda")
+        expected = model(value)
+        actual = torch.compile(model, fullgraph=True)(value)
+        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
+        self.assertEqual(actual.shape, expected.shape)
+        self.assertEqual(actual.dtype, expected.dtype)
+
+
+if __name__ == "__main__":
+    unittest.main()
"""


TASKS = (
    {
        "retired_task_id": "pytorch__180583__symbolic_weighted_bce",
        "task_dir": "tasks/pytorch/181845_symbolic_concat_layout_guard",
        "task_id": "pytorch__181845__symbolic_concat_layout_guard",
        "public_task_id": "opbench-v07-t0045",
        "pr_number": 181845,
        "screening_index": 55,
        "base_commit": "555d2474759010ac365a3c99cdac9c73ab106f1c",
        "merge_commit": "3367e1e0cc262d5a5d6e59f4dd9f7c75152a4c56",
        "source_ref": "pytorch-555d2474-symbolic-concat-overlay",
        "source_snapshot_commit": "555d2474759010ac365a3c99cdac9c73ab106f1c",
        "tracked_file_count": 21866,
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cpu-py311",
        "overlay_paths": [
            "torch/_inductor/config.py",
            "torch/_inductor/ir.py",
            "torch/_inductor/lowering.py",
        ],
        "gold": GOLD_181845,
        "hidden": HIDDEN_181845,
        "test_file": "test/op_bench/test_symbolic_concat_layout.py",
        "f2p": ["SymbolicConcatLayoutTests.test_data_dependent_concat_matches_eager"],
        "p2p": ["SymbolicConcatLayoutTests.test_static_channels_last_concat_remains_stable"],
        "statement": {
            "title": "Full-graph concatenation rejects a data-dependent channel extent",
            "body": (
                "A CPU graph builds two same-rank tensors whose channel extents depend on "
                "the number of selected input elements, then concatenates them. Eager "
                "execution succeeds, but full-graph compilation raises a data-dependent "
                "shape guard before the result is produced. Repair torch.compile execution so "
                "multiple selection counts match eager shapes and values. Static "
                "channels-last concatenation must retain its current behavior."
            ),
            "labels": ["module: inductor", "module: dynamic shapes", "bug"],
        },
        "known_constraints": [
            "The failing CPU graph derives a concatenation extent from runtime selection cardinality.",
            "The same full-graph function must work for more than one valid cardinality.",
            "Static channels-last inputs must remain numerically and structurally compatible.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "torch.cat",
            "component": "torch.compile",
            "problem_type": "data-dependent-layout-guard",
            "tags": ["concat", "symbolic-shape", "layout", "fullgraph", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "shape", "layout", "exception", "compatibility"],
            "trigger_tags": ["noncontiguous_or_special_layout", "dynamic_shape"],
            "execution_context": {
                "devices": ["cpu"], "modes": ["compile"], "phases": ["forward"], "distributed": False,
            },
            "failure_type": "unexpected_error",
            "root_cause_tags": ["incorrect_validation"],
            "component_tags": ["dynamo", "inductor"],
        },
        "dimension_evidence": {
            "localization": (
                "The sanitized Prompt exposes the public concatenation behavior and dynamic trigger but no "
                "source path, internal IR class, configuration key, layout helper, or repair location. The "
                "solver must reproduce a compiler-only guard and trace it across graph lowering and layout analysis."
            ),
            "diagnosis": (
                "Correct diagnosis must distinguish a real invalid runtime guard from ordinary dynamic-output "
                "capture, and identify that symbolic layout metadata is being forced through a boolean contiguity "
                "query while concat path selection independently controls whether that query is reached."
            ),
            "repair_regression": (
                "The repair spans layout analysis and the pointwise-concat selection boundary. It must treat an "
                "unprovable symbolic layout predicate conservatively, preserve complex-input thresholds, execute "
                "multiple data-dependent extents, and retain static channels-last behavior."
            ),
        },
        "behavior_tokens": [
            "concat", "data_dependent_extent", "layout_predicate", "dynamic_output", "path_selection"
        ],
        "risk_signals": ["single_hidden_f2p"],
        "estimated_runtime_min": 18,
    },
    {
        "retired_task_id": "pytorch__180884__aoti_custom_op_error_context",
        "task_dir": "tasks/pytorch/188771_nested_reduction_pointer_rewind",
        "task_id": "pytorch__188771__nested_reduction_pointer_rewind",
        "public_task_id": "opbench-v07-t0043",
        "pr_number": 188771,
        "screening_index": 95,
        "base_commit": "a8cccea440638f1e3061079c474b9a3ec78a242a",
        "merge_commit": "394d981e828dfc97efc6dcc7a1aeee9a933281c3",
        "source_ref": "pytorch-a8cccea4-nested-reduction-overlay",
        "source_snapshot_commit": "111dac85187d50c8e1f17ad8b1bb84dd1892ab39",
        "tracked_file_count": 21574,
        "runtime_tier": "cuda_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cu126-py311",
        "overlay_paths": ["torch/_inductor/codegen/triton.py"],
        "gold": GOLD_188771,
        "hidden": HIDDEN_188771,
        "test_file": "test/op_bench/test_nested_reduction_pointer_state.py",
        "f2p": ["NestedReductionPointerStateTests.test_broadcast_reduction_consumes_every_outer_tile"],
        "p2p": ["NestedReductionPointerStateTests.test_single_axis_reduction_remains_stable"],
        "statement": {
            "title": "Full-graph nested reduction drops outer broadcast tiles",
            "body": (
                "A CUDA graph broadcasts a vector with a column vector and reduces the resulting matrix with "
                "sum and mean. Eager execution consumes the full matrix, while torch.compile execution can retain only "
                "the earliest outer tiles, producing a large numerical error on nonzero-mean inputs. Repair the "
                "nested reduction so every outer tile contributes and both reductions match eager execution. "
                "Ordinary single-axis reductions must remain unchanged."
            ),
            "labels": ["module: inductor", "module: triton", "bug"],
        },
        "known_constraints": [
            "The CUDA failure requires a fused broadcast followed by a reduction over both materialized axes.",
            "Nonzero-mean inputs make missing outer tiles observable in both sum and mean.",
            "Single-axis torch.compile reductions must preserve their existing results and metadata.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "broadcast reduction",
            "component": "TorchInductor Triton codegen",
            "problem_type": "nested-reduction-state-carry",
            "tags": ["broadcast", "reduction", "nested-loop", "cuda", "wrong-result"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "result",
            "contract_detail_tags": ["value", "numerical", "compatibility"],
            "trigger_tags": ["extreme_value_or_size", "device_specific"],
            "execution_context": {
                "devices": ["cuda"], "modes": ["compile"], "phases": ["forward"], "distributed": False,
            },
            "failure_type": "wrong_result",
            "root_cause_tags": ["incorrect_lowering"],
            "component_tags": ["inductor", "triton"],
        },
        "dimension_evidence": {
            "localization": (
                "The Prompt identifies only a CUDA broadcast-reduction wrong result. It exposes no source path, "
                "code generator, loop-tree object, pointer representation, emitted primitive, or repair location. "
                "The solver must inspect generated kernels and relate a numerical truncation to nested loop state."
            ),
            "diagnosis": (
                "Correct diagnosis must explain why a value advanced through the inner reduction is carried into "
                "the next outer iteration, why a zero outer-axis step still matters, and why shifted inputs reveal "
                "the loss while near-zero means can hide it as accumulation noise."
            ),
            "repair_regression": (
                "The repair must preserve per-level advancement information, derive the compensating outer-loop "
                "state restoration, suppress true no-op emissions, recover sum and mean numerics, and avoid changing "
                "ordinary single-axis reductions. This requires coordinated bookkeeping and code-emission reasoning."
            ),
        },
        "behavior_tokens": [
            "broadcast_reduction", "nested_loop_state", "outer_tile", "numerical_truncation", "cuda"
        ],
        "risk_signals": ["gold_patch_single_file", "single_hidden_f2p"],
        "estimated_runtime_min": 28,
    },
)


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected object")
    return value


def _task_manifest(spec: dict[str, object]) -> dict[str, object]:
    test_file = str(spec["test_file"])
    test_command = f"{{python}} {test_file} {{test}}"
    return {
        "version": "v1",
        "task_id": spec["task_id"],
        "statement": spec["statement"],
        "operator": spec["operator"],
        "source_ref": spec["source_ref"],
        "source": {
            "repo": "pytorch/pytorch",
            "repo_url": "https://github.com/pytorch/pytorch.git",
            "checkout_mode": "git",
            "base_commit": spec["base_commit"],
            "merge_commit": spec["merge_commit"],
            "pr_number": spec["pr_number"],
            "pr_url": f"https://github.com/pytorch/pytorch/pull/{spec['pr_number']}",
            "issue_number": None,
            "issue_url": None,
            "snapshot_path": (
                "../../../.op_bench_cache/sources/pytorch/pytorch/"
                f"{spec['base_commit']}/source"
            ),
        },
        "environment_ref": spec["environment_ref"],
        "runtime_tier": spec["runtime_tier"],
        "environment": {
            "backend": "remote_docker",
            "tier": spec["runtime_tier"],
            "preflight_commands": [
                "{python} --version",
                (
                    "{python} -c \"import torch; print(torch.__version__); "
                    "print(torch.version.git_version); print(torch.__file__)\""
                ),
            ],
            "source_loading": {
                "mode": "python_overlay",
                "installed_package": "torch",
                "runtime_site_packages": "/tmp/op_bench_runtime/site-packages",
                "overlay_paths": spec["overlay_paths"],
                "sync_before_tests": True,
            },
        },
        "agent_visible": {
            "public_task_id": spec["public_task_id"],
            "known_constraints": spec["known_constraints"],
            "allowed_test_commands": [test_command],
            "repo_setup_commands": [],
        },
        "artifacts": {
            "gold_patch": "artifacts/gold.patch",
            "hidden_test_patch": "artifacts/hidden_test.patch",
        },
        "evaluation": {
            "setup_commands": [],
            "test_command": test_command,
            "fail_to_pass": spec["f2p"],
            "pass_to_pass": spec["p2p"],
            "timeout_sec": 900,
        },
        "patch_scope": {"mode": "enforced", "allowed_paths": spec["overlay_paths"]},
        "metadata": {
            "difficulty": "hard",
            "estimated_runtime_min": spec["estimated_runtime_min"],
            "deterministic": True,
            "layer": "A",
            "curation_status": "draft",
            "admission_status": "candidate",
            "source_loading_verified": False,
        },
        "admission": {
            "status": "candidate",
            "verified_at": "2026-08-10T04:15:00Z",
            "evidence": "admission/evidence.json",
        },
        "quality": {
            "origin": "replacement",
            "prompt_evidence": "quality/prompt.json",
            "complexity_evidence": "quality/complexity.json",
            "readmission_evidence": "quality/readmission.json",
        },
        "taxonomy": spec["taxonomy"],
    }


def _build_task(spec: dict[str, object]) -> None:
    task_dir = ROOT / str(spec["task_dir"])
    (task_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (task_dir / "quality").mkdir(parents=True, exist_ok=True)
    (task_dir / "artifacts/gold.patch").write_text(str(spec["gold"]), encoding="utf-8")
    (task_dir / "artifacts/hidden_test.patch").write_text(
        str(spec["hidden"]), encoding="utf-8"
    )
    _write_canonical(task_dir / "task.json", _task_manifest(spec))

    taxonomy = parse_taxonomy_v2(spec["taxonomy"])
    complexity = build_complexity_evidence(
        task_id=str(spec["task_id"]),
        localization=2,
        diagnosis=2,
        repair_regression=2,
        dimension_evidence=spec["dimension_evidence"],
        hard_rejections=(),
        risk_signals=tuple(spec["risk_signals"]),
        duplicate_fingerprint=semantic_duplicate_fingerprint(
            taxonomy, spec["behavior_tokens"]
        ),
        duplicate_decision="distinct",
        blind_pilot=None,
        second_review=False,
        reviewer=f"pr{spec['pr_number']}-source-bound-complexity-reviewer",
        reviewed_at="2026-08-10T04:16:00Z",
    )
    _write_canonical(task_dir / "quality/complexity.json", complexity.to_dict())

    task = TaskManifest.load(task_dir / "task.json")
    view, private_index = quality_prompt_source_inputs(
        task, scanner_version=SCANNER_VERSION
    )
    prompt = build_prompt_quality_evidence(
        task_id=task.task_id,
        public_task_id=str(task.public_task_id),
        rendered_prompt=render_mcp_prompt(view),
        agent_task_view=view,
        private_index=private_index,
        scanner_version=SCANNER_VERSION,
        blind_review={
            "decision": "accepted",
            "reviewer": "codex-primary-public-blind-replacement-pass-v07",
            "reviewed_at": "2026-08-10T04:17:00Z",
        },
        semantic_review={
            "decision": "equivalent",
            "reviewer": "codex-primary-private-semantic-replacement-pass-v07",
            "reviewed_at": "2026-08-10T04:17:01Z",
        },
        decision="accepted",
        created_at="2026-08-10T04:17:02Z",
    )
    if prompt.findings:
        raise RuntimeError(f"{task.task_id}: Prompt overlap findings are not allowed")
    _write_canonical(task_dir / "quality/prompt.json", prompt.to_dict())

    source = (
        ROOT
        / ".op_bench_cache/sources/pytorch/pytorch"
        / str(spec["base_commit"])
        / "source"
    )
    for artifact in ("gold.patch", "hidden_test.patch"):
        completed = subprocess.run(
            ["git", "-C", str(source), "apply", "--check", str(task_dir / "artifacts" / artifact)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{spec['task_id']}: {artifact} does not apply: {completed.stderr.strip()}"
            )


def _register_sources() -> None:
    registry = _load_object(BASE_SOURCE_REGISTRY)
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise TypeError("sources/registry.json: sources must be a list")
    by_id = {
        item.get("id"): item for item in sources if isinstance(item, dict)
    }
    for spec in TASKS:
        entry = {
            "id": spec["source_ref"],
            "repo_url": "https://github.com/pytorch/pytorch.git",
            "commit": spec["base_commit"],
            "local_path": (
                "../.op_bench_cache/sources/pytorch/pytorch/"
                f"{spec['base_commit']}/source"
            ),
            "checksum": None,
            "snapshot_git_commit": spec["source_snapshot_commit"],
            "tracked_file_count": spec["tracked_file_count"],
            "submodules": {"policy": "none_required", "status": "not_initialized"},
            "source_loading_modes": ["python_overlay"],
            "related_tasks": [spec["task_id"]],
            "snapshot_mode": "overlay",
            "notes": (
                "v0.7 source-bound replacement. Only the declared Python production "
                "files are overlaid into a registered nightly wheel."
            ),
        }
        existing = by_id.get(spec["source_ref"])
        if existing is None:
            sources.append(entry)
        elif existing != entry:
            raise RuntimeError(f"conflicting source registry entry: {spec['source_ref']}")
    STAGING_SOURCE_REGISTRY.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _screening_refs(position: int) -> tuple[dict[str, object], dict[str, object]]:
    index = _load_object(SCREENING_ROOT / "screening_index.json")
    records = index.get("records")
    if not isinstance(records, list) or not isinstance(records[position], dict):
        raise TypeError(f"screening record {position} is unavailable")
    record = records[position]
    candidate = record.get("candidate")
    decision = record.get("decision")
    if not isinstance(candidate, dict) or not isinstance(decision, dict):
        raise TypeError(f"screening record {position} has invalid references")
    return copy.deepcopy(candidate), copy.deepcopy(decision)


def _complexity_record(spec: dict[str, object]) -> dict[str, object]:
    task_dir = ROOT / str(spec["task_dir"])
    complexity = _load_object(task_dir / "quality/complexity.json")
    return {
        "artifact_path": f"{spec['task_dir']}/quality/complexity.json",
        "content_hash": complexity["content_hash"],
        "difficulty": complexity["difficulty"],
        "dimension_evidence": complexity["dimension_evidence"],
        "duplicate_decision": complexity["duplicate_decision"],
        "duplicate_fingerprint": complexity["duplicate_fingerprint"],
        "risk_signals": complexity["risk_signals"],
        "total": complexity["total"],
    }


def _review_packet_record(spec: dict[str, object]) -> dict[str, object]:
    task_dir = ROOT / str(spec["task_dir"])
    manifest = _load_object(task_dir / "task.json")
    gold = (task_dir / "artifacts/gold.patch").read_bytes()
    hidden = (task_dir / "artifacts/hidden_test.patch").read_bytes()
    candidate, decision = _screening_refs(int(spec["screening_index"]))
    deferred = _load_object(
        SCREENING_ROOT / str(decision["relative_path"])
    ).get("preliminary_review_reasons")
    if not isinstance(deferred, list) or not deferred:
        raise RuntimeError(f"{spec['task_id']}: deferred reasons are unavailable")
    questions = {
        "review.title_needs_behavior_confirmation": (
            "Does the exact Base/Gold behavior and hidden oracle confirm that the sanitized public "
            "title describes a real operator-domain failure rather than a title-only or mechanical change?"
        ),
        "review.ambiguous_change_context": (
            "Do the exact Base/Gold behavior, implementation scope, and hidden oracle resolve the "
            "candidate's ambiguous public change context?"
        ),
        "review.small_source_delta": (
            "Does the exact Base/Gold behavior and hidden oracle demonstrate that the small source "
            "delta still requires nontrivial operator-domain localization, diagnosis, and regression reasoning?"
        ),
    }
    unsupported = [reason for reason in deferred if reason not in questions]
    if unsupported:
        raise RuntimeError(f"{spec['task_id']}: unsupported deferred reasons {unsupported}")
    return {
        "candidate": candidate,
        "decision": decision,
        "deferred_reasons": deferred,
        "reason_reviews": [
            {
                "reason": reason,
                "review_question": questions[reason],
                "allowed_resolution": ["resolved", "confirmed_blocker"],
            }
            for reason in deferred
        ],
        "required_output": {
            "decision": ["accepted_for_build", "rejected"],
            "rationale": "source-bound rationale",
            "reason_resolutions": "one exact resolution per deferred reason",
            "reviewed_at": "UTC RFC3339 seconds",
            "reviewer": "non-empty independent reviewer identity",
        },
        "review_state": "pending",
        "reassessment": None,
        "screening_index": spec["screening_index"],
        "pr_number": spec["pr_number"],
        "task_id": spec["task_id"],
        "public_task_id": spec["public_task_id"],
        "task_path": spec["task_dir"],
        "public_statement": spec["statement"],
        "source": {
            "repository": "pytorch/pytorch",
            "base_commit": spec["base_commit"],
            "merge_commit": spec["merge_commit"],
        },
        "patch_scope_paths": spec["overlay_paths"],
        "fail_to_pass_count": len(spec["f2p"]),
        "pass_to_pass_count": len(spec["p2p"]),
        "complexity": _complexity_record(spec),
        "private_review_inputs": {
            "task_manifest_hash": canonical_sha256(manifest),
            "gold_patch_path": f"{spec['task_dir']}/artifacts/gold.patch",
            "gold_patch_hash": "sha256:" + __import__("hashlib").sha256(gold).hexdigest(),
            "hidden_test_patch_path": f"{spec['task_dir']}/artifacts/hidden_test.patch",
            "hidden_test_patch_hash": "sha256:" + __import__("hashlib").sha256(hidden).hexdigest(),
        },
    }


def _update_review_packets() -> None:
    prompt = _load_object(PROMPT_PACKET)
    prompt_records = prompt.get("records")
    if not isinstance(prompt_records, list):
        raise TypeError("Prompt packet records must be a list")
    offline_hash = None
    for value in prompt_records:
        if isinstance(value, dict):
            semantic = value.get("semantic_review_inputs")
            if isinstance(semantic, dict) and isinstance(
                semantic.get("offline_readiness_content_hash"), str
            ):
                offline_hash = semantic["offline_readiness_content_hash"]
                break
    if offline_hash is None:
        raise RuntimeError("offline readiness binding is unavailable")
    for spec in TASKS:
        replacement = {
            "task_id": spec["task_id"],
            "public_task_id": spec["public_task_id"],
            "task_path": spec["task_dir"],
            "pr_number": spec["pr_number"],
            "screening_index": spec["screening_index"],
            "scanner_version": SCANNER_VERSION,
            "semantic_review_inputs": {
                "offline_readiness_content_hash": offline_hash
            },
        }
        current_matches = [
            index
            for index, value in enumerate(prompt_records)
            if isinstance(value, dict) and value.get("task_id") == spec["task_id"]
        ]
        retired_matches = [
            index
            for index, value in enumerate(prompt_records)
            if isinstance(value, dict)
            and value.get("task_id") == spec["retired_task_id"]
        ]
        matches = current_matches or retired_matches
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one current or retired Prompt record for {spec['task_id']}"
            )
        prompt_records[matches[0]] = replacement
    prompt["records"] = prompt_records
    prompt["content_hash"] = canonical_sha256(
        {key: value for key, value in prompt.items() if key != "content_hash"}
    )
    _write_canonical(PROMPT_PACKET, prompt)

    reassessment = _load_object(REASSESSMENT_PACKET)
    reassessment_records = reassessment.get("records")
    if not isinstance(reassessment_records, list):
        raise TypeError("Reassessment packet records must be a list")
    replaced = {
        str(identifier)
        for spec in TASKS
        for identifier in (spec["retired_task_id"], spec["task_id"])
    }
    reassessment_records = [
        value
        for value in reassessment_records
        if not isinstance(value, dict) or value.get("task_id") not in replaced
    ]
    for spec in TASKS:
        _, decision_ref = _screening_refs(int(spec["screening_index"]))
        decision = _load_object(SCREENING_ROOT / str(decision_ref["relative_path"]))
        deferred = decision.get("preliminary_review_reasons")
        if not isinstance(deferred, list):
            raise TypeError(
                f"{spec['task_id']}: preliminary_review_reasons must be a list"
            )
        if deferred:
            reassessment_records.append(_review_packet_record(spec))
    reassessment_records.sort(
        key=lambda value: value.get("screening_index", -1)
        if isinstance(value, dict)
        else -1
    )
    reassessment["records"] = reassessment_records
    reassessment["task_count"] = len(reassessment_records)
    reassessment["pending_count"] = sum(
        isinstance(value, dict) and value.get("review_state") == "pending"
        for value in reassessment_records
    )
    reassessment["accepted_count"] = sum(
        isinstance(value, dict) and value.get("review_state") == "accepted_for_build"
        for value in reassessment_records
    )
    reassessment["rejected_count"] = sum(
        isinstance(value, dict) and value.get("review_state") == "rejected"
        for value in reassessment_records
    )
    reassessment["content_hash"] = canonical_sha256(
        {key: value for key, value in reassessment.items() if key != "content_hash"}
    )
    _write_canonical(REASSESSMENT_PACKET, reassessment)


def main() -> int:
    for spec in TASKS:
        _build_task(spec)
    _register_sources()
    _update_review_packets()
    print(
        canonical_json(
            {
                "built": [spec["task_id"] for spec in TASKS],
                "prompt_record_count": _load_object(PROMPT_PACKET)["task_count"],
                "reassessment_record_count": _load_object(REASSESSMENT_PACKET)[
                    "task_count"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
