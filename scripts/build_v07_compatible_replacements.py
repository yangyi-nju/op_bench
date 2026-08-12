#!/usr/bin/env python3
"""Build v0.7 replacements whose Base revisions match the frozen nightly."""

from __future__ import annotations

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


def _git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _gold(base: str, merge: str, paths: list[str]) -> str:
    return _git_text(
        LOCAL_HISTORY,
        "diff",
        "--no-ext-diff",
        "--no-renames",
        base,
        merge,
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


HIDDEN_179278_BODY = r'''
import unittest

import torch
from torch.export import export, unflatten


class SharedBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = torch.nn.Linear(width, width)

    def forward(self, value):
        return torch.tanh(self.projection(value))


class SharedBlockModel(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        shared = SharedBlock(width)
        self.blocks = torch.nn.Sequential(
            shared,
            torch.nn.ReLU(),
            shared,
            torch.nn.Sigmoid(),
        )

    def forward(self, value):
        return self.blocks(value)


class SharedNormModel(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        shared = torch.nn.LayerNorm(width)
        self.path = torch.nn.Sequential(shared, torch.nn.GELU(), shared)

    def forward(self, value):
        return self.path(value)


class IndependentModel(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.path = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
        )

    def forward(self, value):
        return self.path(value)


class SharedModuleUnflattenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def assert_round_trip(self, model, sample):
        captured = export(model, (sample,), strict=True)
        restored = unflatten(captured)
        torch.testing.assert_close(restored(sample), model(sample))
        return restored

    def test_repeated_parameterized_block_round_trips_with_identity(self):
        model = SharedBlockModel(8)
        restored = self.assert_round_trip(model, torch.randn(3, 8))
        self.assertIs(
            getattr(restored.blocks, "0"),
            getattr(restored.blocks, "2"),
        )

    def test_repeated_normalization_round_trips_with_identity(self):
        model = SharedNormModel(8)
        restored = self.assert_round_trip(model, torch.randn(4, 8))
        self.assertIs(
            getattr(restored.path, "0"),
            getattr(restored.path, "2"),
        )

    def test_independent_submodules_keep_order_and_values(self):
        model = IndependentModel(8)
        restored = self.assert_round_trip(model, torch.randn(2, 8))
        self.assertEqual(list(restored.path._modules), ["0", "1", "2"])
        self.assertIsNot(
            getattr(restored.path, "0"),
            getattr(restored.path, "2"),
        )

    def test_parameter_free_shared_activation_remains_stable(self):
        shared = torch.nn.ReLU()
        model = torch.nn.Sequential(shared, torch.nn.Sigmoid(), shared)
        restored = self.assert_round_trip(model, torch.randn(5, 8))
        self.assertIs(getattr(restored, "0"), getattr(restored, "2"))


if __name__ == "__main__":
    unittest.main()
'''


TASKS = (
    {
        "retired_task_id": "pytorch__182971__aoti_user_stream_events",
        "task_dir": "tasks/pytorch/179278_shared_module_unflatten_order",
        "task_id": "pytorch__179278__shared_module_unflatten_order",
        "public_task_id": "opbench-v07-t0064",
        "pr_number": 179278,
        "screening_index": 23,
        "base_commit": "f1a9f64dfd40dd4c1a7a6f2a51d670d53545607d",
        "merge_commit": "fdcd168b127e9a13a4c11b9649f94e004390a6de",
        "source_ref": "pytorch-f1a9f64d-shared-unflatten-overlay",
        "runtime_tier": "cpu_python_overlay",
        "environment_ref": "pytorch-nightly-20260407-torch2.12.0dev-cpu-py311",
        "overlay_paths": ["torch/export/unflatten.py"],
        "gold": _gold(
            "f1a9f64dfd40dd4c1a7a6f2a51d670d53545607d",
            "fdcd168b127e9a13a4c11b9649f94e004390a6de",
            ["torch/export/unflatten.py"],
        ),
        "hidden": _new_file_patch(
            "test/op_bench/test_shared_module_unflatten_order.py",
            HIDDEN_179278_BODY,
        ),
        "test_file": "test/op_bench/test_shared_module_unflatten_order.py",
        "f2p": [
            "SharedModuleUnflattenTests.test_repeated_parameterized_block_round_trips_with_identity",
            "SharedModuleUnflattenTests.test_repeated_normalization_round_trips_with_identity",
        ],
        "p2p": [
            "SharedModuleUnflattenTests.test_independent_submodules_keep_order_and_values",
            "SharedModuleUnflattenTests.test_parameter_free_shared_activation_remains_stable",
        ],
        "statement": {
            "title": "Export reconstruction fails for reused stateful submodules",
            "body": (
                "A CPU module reuses the same stateful submodule at multiple positions in a sequential hierarchy. "
                "Export succeeds, but reconstructing a runnable module fails before inference. Repair the "
                "round trip so values match eager execution and the repeated positions still reference one "
                "shared instance. Independent components must retain their order, and a reused parameter-free "
                "activation must remain stable."
            ),
            "labels": ["module: export", "module: serialization", "bug"],
        },
        "known_constraints": [
            "The same parameterized submodule is registered at more than one position in a nested module hierarchy.",
            "Reconstruction must preserve both numerical output and object identity for reused submodules.",
            "Independent components and a reused parameter-free activation are regression controls.",
        ],
        "operator": {
            "framework": "pytorch",
            "operator_name": "torch.export module reconstruction",
            "component": "torch.export",
            "problem_type": "shared-submodule-reconstruction-order",
            "tags": ["export", "unflatten", "alias", "module-hierarchy", "cpu"],
        },
        "taxonomy": {
            "taxonomy_version": "v2",
            "contract_family": "api_behavior",
            "contract_detail_tags": ["value", "alias", "state", "serialization", "compatibility"],
            "trigger_tags": ["mutation_or_alias"],
            "execution_context": {
                "devices": ["cpu"],
                "modes": ["compile"],
                "phases": ["forward"],
                "distributed": False,
            },
            "failure_type": "unexpected_error",
            "root_cause_tags": ["incorrect_validation"],
            "component_tags": ["dynamo"],
        },
        "dimension_evidence": {
            "localization": (
                "The Prompt exposes only a public export/reconstruction failure with shared module identity. "
                "It names no source path, hierarchy ordering helper, internal qualified-name representation, "
                "filtered ordering table, or repair location."
            ),
            "diagnosis": (
                "Correct diagnosis must distinguish parameter aliasing from repeated registration, follow how "
                "export represents duplicate hierarchy positions, and explain why reconstruction orders a child "
                "that is absent from the canonical hierarchy ordering map."
            ),
            "repair_regression": (
                "The repair must choose a deterministic ordering for duplicate positions without breaking the "
                "canonical order, preserve stateful and parameter-free identity, and retain independent module "
                "ordering. Two aliasing failures and two controls reject cloning or broad exception suppression."
            ),
        },
        "behavior_tokens": [
            "export_roundtrip",
            "shared_submodule",
            "hierarchy_order",
            "object_identity",
            "alias_registration",
        ],
        "risk_signals": ["gold_patch_lte_4_lines", "gold_patch_single_file"],
        "estimated_runtime_min": 12,
    },
)


def _materialize(spec: dict[str, object]) -> None:
    commit = str(spec["base_commit"])
    destination = ROOT / ".op_bench_cache/sources/pytorch/pytorch" / commit / "source"
    if not destination.exists():
        materialize_frozen_git_revision(
            source_repository=LOCAL_HISTORY,
            revision=commit,
            destination=destination,
        )
    spec["source_snapshot_commit"] = _git_text(
        destination, "rev-parse", "HEAD"
    ).strip()
    spec["tracked_file_count"] = len(
        _git_text(destination, "ls-files", "--deduplicate").splitlines()
    )


def main() -> int:
    for spec in TASKS:
        _materialize(spec)
    replacement_builder.TASKS = TASKS
    replacement_builder.BASE_SOURCE_REGISTRY = (
        ROOT / "sources/staging_v07_replacements.json"
    )
    for spec in TASKS:
        replacement_builder._build_task(spec)
    replacement_builder._register_sources()
    replacement_builder._update_review_packets()
    print(
        canonical_json(
            {
                "built": [spec["task_id"] for spec in TASKS],
                "source_count": len(
                    replacement_builder._load_object(
                        replacement_builder.STAGING_SOURCE_REGISTRY
                    )["sources"]
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
