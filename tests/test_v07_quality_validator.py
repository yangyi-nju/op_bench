from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from op_bench.factory.artifacts import FactoryArtifactStore
from op_bench.factory.complexity import build_complexity_evidence
from op_bench.factory.prompt_quality import (
    build_private_answer_index,
    build_prompt_quality_evidence,
)
from op_bench.factory.quality_release import (
    validate_historical_index,
    validate_quality_task,
)
from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.registry import (
    EnvironmentRegistry,
    SourceRegistry,
    resolve_task_assets,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt
from op_bench.runtime.legacy import LegacyV05Defaults, full_task_spec_from_v05
from op_bench.runtime.task_view import project_agent_task_view
from op_bench.task import TaskManifest
from scripts import validate_v07_quality as validator_cli


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "pytorch_v0.7" / "dataset.json"
HISTORICAL_INDEX = ROOT / "factory/v0.7/p7/historical_readmission.json"
VALIDATOR = ROOT / "scripts/validate_v07_quality.py"


def historical_task() -> TaskManifest:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    entry = dataset["tasks"][0]
    original = TaskManifest.load(ROOT / entry["task_path"] / "task.json")
    data = copy.deepcopy(original.data)
    data.pop("taxonomy", None)
    data.pop("quality", None)
    data["agent_visible"].pop("public_task_id", None)
    return TaskManifest(task_dir=original.task_dir, data=data)


def _quality_view(task: TaskManifest) -> dict[str, object]:
    spec = full_task_spec_from_v05(task)
    defaults = LegacyV05Defaults.standard()
    capability = replace(
        defaults.capability_policy,
        policy_id="opbench-v0.7-repository-root-v1",
        writable_paths=(".",),
        registered_tests=tuple(
            sorted(selector.selector_id for selector in spec.public_tests)
        ),
    )
    return project_agent_task_view(
        spec,
        capability,
        defaults.budget_policy,
    ).to_dict()


def _resolved_task(task: TaskManifest) -> TaskManifest:
    return resolve_task_assets(
        task,
        environment_registry=EnvironmentRegistry.load(
            ROOT / "environments/registry.json"
        ),
        source_registry=SourceRegistry.load(
            ROOT / "sources/registry.json"
        ),
    )


def complete_quality_task(
    task_root: Path,
    *,
    difficulty: str = "hard",
) -> TaskManifest:
    original = historical_task()
    task_dir = task_root / "task"
    task_dir.mkdir(parents=True)
    data = copy.deepcopy(original.data)
    data["agent_visible"]["public_task_id"] = "opbench-v07-t0001"
    data["agent_visible"]["known_constraints"] = [
        "CPU-only behavior through the public API."
    ]
    data["statement"] = {
        "title": "Public API behavior differs for a supported input",
        "body": (
            "A supported input to the public API raises an unexpected error. "
            "Preserve the behavior of ordinary inputs."
        ),
        "labels": ["public-api"],
    }
    data["operator"]["operator_name"] = "torch.nn.LazyLinear"
    data["taxonomy"] = {
        "taxonomy_version": "v2",
        "contract_family": "api_behavior",
        "contract_detail_tags": ["schema"],
        "trigger_tags": [],
        "execution_context": {
            "devices": ["cpu"],
            "modes": ["eager"],
            "phases": ["forward"],
            "distributed": False,
        },
        "failure_type": "unexpected_error",
        "root_cause_tags": [],
        "component_tags": [],
    }
    data["metadata"]["difficulty"] = difficulty
    data["artifacts"] = {
        "gold_patch": "artifacts/gold.patch",
        "hidden_test_patch": "artifacts/hidden.patch",
    }
    data["admission"] = {
        "status": "verified",
        "evidence": "admission/evidence.json",
        "verified_at": "2026-07-29T00:00:00Z",
    }
    (task_dir / "artifacts").mkdir()
    gold_patch = (
        "diff --git a/torch/public.py b/torch/public.py\n"
        "--- a/torch/public.py\n"
        "+++ b/torch/public.py\n"
        "@@ -1 +1 @@\n"
        "-old = 1\n"
        "+new = 2\n"
    )
    hidden_patch = (
        "diff --git a/test/public.py b/test/public.py\n"
        "--- a/test/public.py\n"
        "+++ b/test/public.py\n"
        "@@ -1 +1 @@\n"
        "-assert old\n"
        "+assert new\n"
    )
    (task_dir / "artifacts/gold.patch").write_text(gold_patch, encoding="utf-8")
    (task_dir / "artifacts/hidden.patch").write_text(
        hidden_patch,
        encoding="utf-8",
    )
    (task_dir / "admission").mkdir()
    (task_dir / "task.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task = _resolved_task(TaskManifest(task_dir=task_dir, data=data))
    replay_hash = replay_spec_hash(task)
    created_at = "2026-07-29T00:00:00Z"
    admission = {
        "schema_version": "v1",
        "evidence_id": (
            f"{task.task_id}:{replay_hash.removeprefix('sha256:')[:12]}:"
            f"{created_at}"
        ),
        "task_id": task.task_id,
        "task_manifest_hash": replay_hash,
        "task_manifest_hash_kind": REPLAY_SPEC_HASH_KIND,
        "created_at": created_at,
        "source": {
            "id": task.source_ref,
            "repo_url": task.repo_url,
            "base_commit": task.base_commit,
            "snapshot_hash": task.source_snapshot_hash,
            "snapshot_method": task.source_snapshot_method,
        },
        "environment": {
            "id": task.environment_ref,
            "runtime_tier": task.runtime_tier,
            "backend": task.environment_backend,
            "image": task.environment_image,
            "image_digest": task.environment_image_digest,
            "digest_kind": task.environment_digest_kind,
            "platform": task.environment_platform,
        },
        "baseline": {
            "task_id": task.task_id,
            "mode": "baseline",
            "status": "baseline_reproduced",
            "fail_to_pass_total": len(task.fail_to_pass_tests),
            "fail_to_pass_passed": 0,
            "pass_to_pass_total": len(task.pass_to_pass_tests),
            "pass_to_pass_passed": len(task.pass_to_pass_tests),
            "duration_sec": 1,
        },
        "gold": {
            "task_id": task.task_id,
            "mode": "gold",
            "status": "resolved",
            "fail_to_pass_total": len(task.fail_to_pass_tests),
            "fail_to_pass_passed": len(task.fail_to_pass_tests),
            "pass_to_pass_total": len(task.pass_to_pass_tests),
            "pass_to_pass_passed": len(task.pass_to_pass_tests),
            "duration_sec": 1,
        },
        "admission": {
            "decision": "verified",
            "failure_classification": None,
            "verified": True,
        },
    }
    (task_dir / "admission/evidence.json").write_text(
        json.dumps(admission),
        encoding="utf-8",
    )

    private_index = build_private_answer_index(
        gold_patch=gold_patch,
        hidden_test_patch=hidden_patch,
        patch_scope=tuple(task.patch_scope_paths),
        hidden_selectors=tuple(
            [*task.fail_to_pass_tests, *task.pass_to_pass_tests]
        ),
    )
    view = _quality_view(task)
    prompt = build_prompt_quality_evidence(
        task_id=task.task_id,
        public_task_id="opbench-v07-t0001",
        rendered_prompt=render_mcp_prompt(view),
        agent_task_view=view,
        private_index=private_index,
        scanner_version="prompt-overlap-v1",
        blind_review={
            "decision": "accepted",
            "reviewer": "blind-reviewer",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        semantic_review={
            "decision": "equivalent",
            "reviewer": "semantic-reviewer",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        decision="accepted",
        created_at="2026-07-29T00:00:00Z",
    )
    complexity = build_complexity_evidence(
        task_id=task.task_id,
        localization=2,
        diagnosis=2,
        repair_regression=2,
        dimension_evidence={
            "localization": "Requires tracing the public call across components.",
            "diagnosis": "Requires separating schema behavior from implementation.",
            "repair_regression": "Requires preserving neighboring API behavior.",
        },
        hard_rejections=(),
        risk_signals=(),
        duplicate_fingerprint="sha256:" + "a" * 64,
        duplicate_decision="distinct",
        blind_pilot=None,
        second_review=False,
        reviewer="complexity-reviewer",
        reviewed_at="2026-07-29T00:00:00Z",
    )
    with FactoryArtifactStore(task_dir) as store:
        prompt_ref = store.write_contract("quality/prompt.json", prompt)
        complexity_ref = store.write_contract("quality/complexity.json", complexity)
        readmission_payload = {
            "contract_type": "quality_readmission",
            "schema_version": "v1",
            "task_id": task.task_id,
            "public_task_id": "opbench-v07-t0001",
            "origin": "retained_historical",
            "disposition": "retained",
            "taxonomy_hash": canonical_sha256(data["taxonomy"]),
            "prompt_evidence": prompt_ref.to_dict(),
            "complexity_evidence": complexity_ref.to_dict(),
            "admission_evidence_hash": (
                "sha256:"
                + hashlib.sha256(
                    (task_dir / "admission/evidence.json").read_bytes()
                ).hexdigest()
            ),
            "created_at": "2026-07-29T00:00:00Z",
        }
        readmission_payload["content_hash"] = canonical_sha256(
            readmission_payload
        )
        readmission_ref = store.write_json(
            "quality/readmission.json",
            readmission_payload,
            artifact_type="quality_readmission",
            artifact_id=task.task_id,
        )
    data["quality"] = {
        "prompt_evidence": prompt_ref.relative_path,
        "complexity_evidence": complexity_ref.relative_path,
        "readmission_evidence": readmission_ref.relative_path,
        "origin": "retained_historical",
    }
    return _resolved_task(TaskManifest(task_dir=task_dir, data=data))


def _rewrite_admission(
    task: TaskManifest,
    mutation,
) -> tuple[str, ...]:
    admission_path = task.task_dir / "admission/evidence.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    mutation(admission)
    admission_path.write_text(
        json.dumps(admission, sort_keys=True),
        encoding="utf-8",
    )
    readmission_path = task.task_dir / "quality/readmission.json"
    readmission = json.loads(readmission_path.read_text(encoding="utf-8"))
    readmission["admission_evidence_hash"] = (
        "sha256:" + hashlib.sha256(admission_path.read_bytes()).hexdigest()
    )
    readmission["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in readmission.items()
            if key != "content_hash"
        }
    )
    readmission_path.write_text(
        canonical_json(readmission),
        encoding="utf-8",
    )
    return validate_quality_task(ROOT, task, require_verified=True)


class V07QualityValidatorTests(unittest.TestCase):
    def test_release_cli_can_validate_final_agent_execution_index(self) -> None:
        output = io.StringIO()
        with (
            patch.object(validator_cli, "build_release_outputs", return_value={}),
            patch.object(
                validator_cli,
                "validate_quality_replay_index",
                return_value=[],
            ),
            patch.object(
                validator_cli,
                "validate_quality_validation_index",
                return_value=[],
            ) as validate_agent,
            redirect_stdout(output),
        ):
            status = validator_cli.main(
                [
                    "--release",
                    "factory/v0.7/p9/release_manifest.json",
                    "--replay-index",
                    "runs/v0.7_quality_replay/index.json",
                    "--validation-contract",
                    "factory/v0.7/p9/validation_contract.json",
                    "--run-root",
                    "runs/v0.7_quality_validation",
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn("122/122 Agent Attempts passed validation", output.getvalue())
        validate_agent.assert_called_once()

    def test_historical_index_revalidates_exact_dispositions_and_retained_gates(
        self,
    ) -> None:
        self.assertEqual(
            validate_historical_index(ROOT, HISTORICAL_INDEX),
            (),
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            (
                str(ROOT / ".venv/bin/python"),
                str(VALIDATOR),
                "--historical-index",
                str(HISTORICAL_INDEX),
            ),
            cwd=ROOT,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("retained=14, deferred=1, retired=10", completed.stdout)

    def test_historical_index_rejects_disposition_and_reference_tampering(
        self,
    ) -> None:
        original = json.loads(HISTORICAL_INDEX.read_text(encoding="utf-8"))
        mutations = {
            "disposition": lambda payload: payload["records"][0].__setitem__(
                "disposition",
                "retained",
            ),
            "reference": lambda payload: payload["records"][2][
                "prompt_evidence"
            ].__setitem__("content_hash", "sha256:" + "0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload = copy.deepcopy(original)
                mutate(payload)
                payload["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "content_hash"
                    }
                )
                index = Path(directory) / "historical_readmission.json"
                index.write_text(canonical_json(payload), encoding="utf-8")
                errors = validate_historical_index(ROOT, index)
                self.assertTrue(errors)
                self.assertTrue(
                    any(
                        "disposition" in error or "content_hash" in error
                        for error in errors
                    ),
                    errors,
                )

    def test_historical_validator_rebuilds_reviews_and_artifact_bytes(
        self,
    ) -> None:
        from op_bench.factory import quality_release

        original_loader = quality_release.load_regular_file_bytes
        review_path = (
            ROOT
            / "factory/v0.7/p7/reviews"
            / "pytorch__124385__load_state_dict_prefix.json"
        ).resolve()
        artifact_path = (
            ROOT
            / "tasks/pytorch/124385_load_state_dict_prefix"
            / "quality/complexity.json"
        ).resolve()

        def changed_review(path: Path) -> bytes:
            encoded = original_loader(path)
            if path.resolve() != review_path:
                return encoded
            payload = json.loads(encoded.decode("utf-8"))
            payload["complexity"]["blind_pilot"] = {
                "decision": "accepted",
                "counts_toward_final": False,
            }
            payload["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "content_hash"
                }
            )
            return (canonical_json(payload) + "\n").encode("utf-8")

        with patch.object(
            quality_release,
            "load_regular_file_bytes",
            side_effect=changed_review,
        ):
            review_errors = validate_historical_index(ROOT, HISTORICAL_INDEX)
        self.assertTrue(
            any("review_rebuild" in error for error in review_errors),
            review_errors,
        )

        def changed_artifact(path: Path) -> bytes:
            encoded = original_loader(path)
            return encoded + b" " if path.resolve() == artifact_path else encoded

        with patch.object(
            quality_release,
            "load_regular_file_bytes",
            side_effect=changed_artifact,
        ):
            artifact_errors = validate_historical_index(ROOT, HISTORICAL_INDEX)
        self.assertIn(
            (
                "tasks/pytorch/124385_load_state_dict_prefix/"
                "quality/complexity.json: bytes differ from exact review rebuild"
            ),
            artifact_errors,
        )

    def test_verified_quality_task_requires_all_quality_evidence(self) -> None:
        errors = validate_quality_task(ROOT, historical_task(), require_verified=True)
        self.assertIn("taxonomy: required for formal v0.7", errors)
        self.assertIn("agent_visible.public_task_id: required", errors)
        self.assertIn("quality.prompt_evidence: required", errors)
        self.assertIn("quality.complexity_evidence: required", errors)

    def test_easy_task_cannot_be_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = complete_quality_task(Path(directory), difficulty="easy")
            errors = validate_quality_task(ROOT, task, require_verified=True)
        self.assertIn("metadata.difficulty: easy is forbidden", errors)

    def test_complete_task_revalidates_typed_source_bound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = complete_quality_task(Path(directory))
            self.assertEqual(
                validate_quality_task(ROOT, task, require_verified=True),
                (),
            )

            mutated = copy.deepcopy(task.data)
            mutated["statement"]["body"] += " A changed public fact."
            errors = validate_quality_task(
                ROOT,
                TaskManifest(task_dir=task.task_dir, data=mutated),
                require_verified=True,
            )
        self.assertTrue(
            any(error.startswith("quality.prompt_evidence:") for error in errors)
        )

    def test_verified_validation_rejects_non_retained_disposition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = complete_quality_task(Path(directory))
            readmission_path = task.task_dir / "quality/readmission.json"
            payload = json.loads(readmission_path.read_text(encoding="utf-8"))
            payload["disposition"] = "deferred"
            payload["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "content_hash"
                }
            )
            readmission_path.write_text(
                canonical_json(payload),
                encoding="utf-8",
            )
            errors = validate_quality_task(
                ROOT,
                task,
                require_verified=True,
            )
        self.assertIn(
            "quality.readmission_evidence: disposition: retained required",
            errors,
        )

    def test_status_only_admission_forgery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = complete_quality_task(Path(directory))
            errors = _rewrite_admission(
                task,
                lambda payload: (
                    payload.clear(),
                    payload.update(
                        {
                            "task_id": task.task_id,
                            "baseline": {"status": "baseline_reproduced"},
                            "gold": {"status": "resolved"},
                            "admission": {
                                "decision": "verified",
                                "verified": True,
                            },
                        }
                    ),
                ),
            )
        self.assertTrue(
            any(error.startswith("admission:") for error in errors),
            errors,
        )

    def test_admission_rebinding_axes_are_rejected_after_byte_hash_update(
        self,
    ) -> None:
        mutations = {
            "task_id": lambda payload: payload.__setitem__(
                "task_id", "pytorch__other"
            ),
            "replay_hash": lambda payload: payload.__setitem__(
                "task_manifest_hash", "sha256:" + "f" * 64
            ),
            "source": lambda payload: payload["source"].__setitem__(
                "id", "other-source"
            ),
            "environment": lambda payload: payload["environment"].__setitem__(
                "id", "other-environment"
            ),
            "selector_counts": lambda payload: payload["gold"].__setitem__(
                "fail_to_pass_total",
                payload["gold"]["fail_to_pass_total"] + 1,
            ),
            "execution_identity": lambda payload: payload[
                "baseline"
            ].__setitem__("task_id", "pytorch__other"),
        }
        for name, mutation in mutations.items():
            with self.subTest(axis=name), tempfile.TemporaryDirectory() as directory:
                task = complete_quality_task(Path(directory))
                errors = _rewrite_admission(task, mutation)
                self.assertTrue(
                    any(error.startswith("admission:") for error in errors),
                    errors,
                )


if __name__ == "__main__":
    unittest.main()
