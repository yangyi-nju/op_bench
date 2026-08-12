from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from op_bench.factory.artifacts import (
    FactoryArtifactStore,
    load_factory_contract,
)
from op_bench.factory.contracts import (
    CandidateRecord,
    DecisionRecord,
    FactoryAdmissionRecord,
)
from op_bench.factory.lifecycle import validate_admission_chain
from op_bench.factory.promotion import build_verified_admission_chain
from op_bench.factory.screening import screen_candidate
from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.registry import load_resolved_task
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest
from tests.test_factory_contracts import candidate


BASE_COMMIT = "1" * 40
ROOT = Path(__file__).resolve().parents[1]
PROMOTE_CLI = ROOT / "scripts" / "promote_factory_admission.py"
VALID_STATES = (
    "discovered",
    "screened",
    "bundled",
    "preflight_passed",
    "baseline_reproduced",
    "gold_resolved",
    "reviewed",
    "verified",
)


def task_manifest(*, subclass: str = "B1") -> TaskManifest:
    return TaskManifest(
        task_dir=Path("/fixture/pytorch__170001__empty_reduction"),
        data={
            "task_id": "pytorch__170001__empty_reduction",
            "version": "v1",
            "environment_ref": "pytorch-cpu-fixture",
            "runtime_tier": "cpu_python_overlay",
            "source_ref": "pytorch-source-fixture",
            "admission": {
                "status": "verified",
                "evidence": "admission/evidence.json",
            },
            "source": {
                "repo": "pytorch/pytorch",
                "repo_url": "https://github.com/pytorch/pytorch.git",
                "base_commit": BASE_COMMIT,
                "checkout_mode": "git",
            },
            "environment": {
                "backend": "docker",
                "image": "op-bench/pytorch-cpu:fixture",
            },
            "evaluation": {
                "fail_to_pass": ["TestBoundary.test_empty"],
                "pass_to_pass": ["TestBoundary.test_control"],
                "test_command": "{python} test_boundary.py {test}",
                "timeout_sec": 30,
            },
            "artifacts": {},
            "operator": {
                "framework": "pytorch",
                "problem_dimension": "boundary",
                "problem_subclass": subclass,
                "failure_contract": "wrong-result",
            },
        },
    )


def admission_evidence(task: TaskManifest) -> dict[str, object]:
    return {
        "schema_version": "v1",
        "evidence_id": "admission-fixture",
        "task_id": task.task_id,
        "task_manifest_hash": replay_spec_hash(task),
        "task_manifest_hash_kind": REPLAY_SPEC_HASH_KIND,
        "created_at": "2026-07-27T01:00:00Z",
        "source": {
            "id": task.source_ref,
            "repo_url": task.repo_url,
            "base_commit": task.base_commit,
            "snapshot_hash": None,
            "snapshot_method": None,
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
            "duration_sec": 1.2345,
            "fail_to_pass_total": 1,
            "fail_to_pass_passed": 0,
            "pass_to_pass_total": 1,
            "pass_to_pass_passed": 1,
        },
        "gold": {
            "task_id": task.task_id,
            "mode": "gold",
            "status": "resolved",
            "duration_sec": 2.5,
            "fail_to_pass_total": 1,
            "fail_to_pass_passed": 1,
            "pass_to_pass_total": 1,
            "pass_to_pass_passed": 1,
        },
        "admission": {
            "decision": "verified",
            "verified": True,
            "failure_classification": None,
        },
    }


def review(task: TaskManifest) -> dict[str, object]:
    return {
        "schema_version": "v1",
        "task_id": task.task_id,
        "decision": "approved",
        "root_cause_confirmed": True,
        "scope_confirmed": True,
        "tests_confirmed": True,
        "surrogate_confirmed": None,
        "reviewer": "curator",
        "reviewed_at": "2026-07-27T02:00:00Z",
    }


def changed_task(
    task: TaskManifest,
    update,
) -> TaskManifest:
    data = copy.deepcopy(task.data)
    update(data)
    return TaskManifest(task_dir=task.task_dir, data=data)


def b5_candidate() -> CandidateRecord:
    selected = candidate()
    return replace(
        selected,
        title="Avoid CUDA launch failure at the grid limit",
        description="A tail block at the grid limit must use a bounded launch.",
        keyword_pack_id="boundary-b5-v1",
        matched_keyword_ids=("grid-limit", "tail-block"),
        proposed_subclass="B5",
    )


class FactoryPromotionTests(unittest.TestCase):
    def _assert_rejected(
        self,
        *,
        selected: CandidateRecord,
        decision: DecisionRecord,
        task: TaskManifest,
        admission: dict[str, object],
        review_data: dict[str, object],
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "output"
            with self.assertRaises(ContractError):
                build_verified_admission_chain(
                    candidate=selected,
                    decision=decision,
                    task=task,
                    admission=admission,
                    review=review_data,
                    created_at="2026-07-27T03:00:00Z",
                )
            self.assertFalse(output.exists())

    def _write_cli_inputs(
        self,
        root: Path,
        *,
        approved: bool = True,
    ) -> tuple[TaskManifest, dict[str, Path]]:
        task_dir = root / "task"
        task_dir.mkdir()
        task_data = copy.deepcopy(task_manifest().data)
        task_path = task_dir / "task.json"
        task_path.write_text(
            json.dumps(task_data, indent=2) + "\n",
            encoding="utf-8",
        )
        environment_registry = root / "environments.json"
        environment_registry.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "environments": [
                        {
                            "id": "pytorch-cpu-fixture",
                            "framework": "pytorch",
                            "runtime_tier": "cpu_python_overlay",
                            "backend": "docker",
                            "docker": {
                                "image": "op-bench/pytorch-cpu:fixture",
                                "digest": "sha256:" + ("a" * 64),
                                "digest_kind": "local_image_id",
                                "platform": "linux/amd64",
                            },
                            "preflight": {
                                "workdir": "/tmp",
                                "commands": ["python --version"],
                            },
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        source_registry = root / "sources.json"
        source_registry.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "sources": [
                        {
                            "id": "pytorch-source-fixture",
                            "repo_url": "https://github.com/pytorch/pytorch.git",
                            "commit": BASE_COMMIT,
                            "submodules": {
                                "policy": "none_required",
                                "status": "not_initialized",
                            },
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        task = load_resolved_task(
            task_path,
            environment_registry_path=environment_registry,
            source_registry_path=source_registry,
        )
        selected = candidate()
        decision = screen_candidate(selected)
        contracts = root / "contracts"
        with FactoryArtifactStore(contracts) as store:
            store.write_contract("candidate.json", selected)
            store.write_contract("decision.json", decision)
        admission_path = root / "admission.json"
        admission_path.write_text(
            json.dumps(admission_evidence(task), indent=2) + "\n",
            encoding="utf-8",
        )
        review_data = review(task)
        review_data["scope_confirmed"] = approved
        review_path = root / "review.json"
        review_path.write_text(
            json.dumps(review_data, indent=2) + "\n",
            encoding="utf-8",
        )
        return task, {
            "candidate": contracts / "candidate.json",
            "decision": contracts / "decision.json",
            "task": task_path,
            "admission": admission_path,
            "review": review_path,
            "environment_registry": environment_registry,
            "source_registry": source_registry,
        }

    def _run_cli(
        self,
        paths: dict[str, Path],
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [
                sys.executable,
                str(PROMOTE_CLI),
                "--candidate",
                str(paths["candidate"]),
                "--decision",
                str(paths["decision"]),
                "--task",
                str(paths["task"]),
                "--admission-evidence",
                str(paths["admission"]),
                "--review",
                str(paths["review"]),
                "--environment-registry",
                str(paths["environment_registry"]),
                "--source-registry",
                str(paths["source_registry"]),
                "--output-dir",
                str(output),
                "--created-at",
                "2026-07-27T03:00:00Z",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_inputs_build_verified_chain(self) -> None:
        task = task_manifest()
        selected = candidate()
        decision = screen_candidate(selected)

        records = build_verified_admission_chain(
            candidate=selected,
            decision=decision,
            task=task,
            admission=admission_evidence(task),
            review=review(task),
            created_at="2026-07-27T03:00:00Z",
        )

        self.assertEqual(len(records), 8)
        self.assertEqual(tuple(record.state for record in records), VALID_STATES)
        validate_admission_chain(records)
        self.assertEqual(records[-1].state, "verified")
        normalized_admission = admission_evidence(task)
        for phase, duration_ms in (("baseline", 1235), ("gold", 2500)):
            execution = normalized_admission[phase]
            assert isinstance(execution, dict)
            execution.pop("duration_sec")
            execution["duration_ms"] = duration_ms
        self.assertEqual(
            next(
                evidence.reference.content_hash
                for evidence in records[5].evidence
                if evidence.evidence_type == "gold"
            ),
            canonical_sha256(normalized_admission),
        )

    def test_rejects_identity_status_and_taxonomy_mismatches(self) -> None:
        selected = candidate()
        accepted = screen_candidate(selected)
        valid_task = task_manifest()
        rejected_candidate = replace(
            selected,
            title="Revert empty reduction boundary fix",
        )
        other_candidate = replace(
            selected,
            candidate_id=CandidateRecord.candidate_id_for(
                repository=selected.repository,
                pr_number=170002,
                base_commit="3" * 40,
                merge_commit="4" * 40,
            ),
            pr_number=170002,
            pr_url="https://github.com/pytorch/pytorch/pull/170002",
            base_commit="3" * 40,
            merge_commit="4" * 40,
            raw_metadata=replace(
                selected.raw_metadata,
                artifact_id="pr:pytorch/pytorch#170002",
                relative_path="raw/pr-170002.json",
            ),
        )
        draft_task = changed_task(
            valid_task,
            lambda data: data["admission"].update(status="draft"),
        )
        b2_task = task_manifest(subclass="B2")
        cases = (
            (
                rejected_candidate,
                screen_candidate(rejected_candidate),
                valid_task,
                admission_evidence(valid_task),
            ),
            (
                selected,
                screen_candidate(other_candidate),
                valid_task,
                admission_evidence(valid_task),
            ),
            (
                selected,
                accepted,
                draft_task,
                admission_evidence(draft_task),
            ),
            (
                selected,
                accepted,
                b2_task,
                admission_evidence(b2_task),
            ),
        )

        for index, (case_candidate, decision, task, admission) in enumerate(cases):
            with self.subTest(case=index):
                self._assert_rejected(
                    selected=case_candidate,
                    decision=decision,
                    task=task,
                    admission=admission,
                    review_data=review(task),
                )

    def test_rejects_admission_identity_and_replay_mismatches(self) -> None:
        task = task_manifest()
        selected = candidate()
        decision = screen_candidate(selected)

        def mutated(path: str, value: object) -> dict[str, object]:
            evidence = copy.deepcopy(admission_evidence(task))
            target: dict[str, object] = evidence
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]  # type: ignore[assignment]
            target[parts[-1]] = value
            return evidence

        cases = (
            mutated("task_id", "pytorch__other"),
            mutated("source.id", "pytorch-source-other"),
            mutated("environment.id", "pytorch-environment-other"),
            mutated("environment.runtime_tier", "cuda_python_overlay"),
            mutated("task_manifest_hash", "sha256:" + ("f" * 64)),
            mutated("admission.decision", "gold_failed"),
        )
        for index, admission in enumerate(cases):
            with self.subTest(case=index):
                self._assert_rejected(
                    selected=selected,
                    decision=decision,
                    task=task,
                    admission=admission,
                    review_data=review(task),
                )

    def test_rejects_invalid_test_execution(self) -> None:
        task = task_manifest()
        selected = candidate()
        decision = screen_candidate(selected)

        def mutated(
            phase: str,
            field: str,
            value: object,
        ) -> dict[str, object]:
            evidence = copy.deepcopy(admission_evidence(task))
            execution = evidence[phase]
            assert isinstance(execution, dict)
            execution[field] = value
            return evidence

        cases = (
            mutated("baseline", "status", "resolved"),
            mutated("baseline", "fail_to_pass_passed", 1),
            mutated("baseline", "pass_to_pass_passed", 0),
            mutated("gold", "status", "f2p_failed"),
            mutated("gold", "fail_to_pass_passed", 0),
            mutated("gold", "pass_to_pass_passed", 0),
        )
        for index, admission in enumerate(cases):
            with self.subTest(case=index):
                self._assert_rejected(
                    selected=selected,
                    decision=decision,
                    task=task,
                    admission=admission,
                    review_data=review(task),
                )

    def test_rejects_unapproved_or_incomplete_review(self) -> None:
        task = task_manifest()
        selected = candidate()
        decision = screen_candidate(selected)
        cases: list[dict[str, object]] = []
        for field, value in (
            ("decision", "rejected"),
            ("root_cause_confirmed", False),
            ("scope_confirmed", False),
            ("tests_confirmed", False),
        ):
            selected_review = review(task)
            selected_review[field] = value
            cases.append(selected_review)

        for index, review_data in enumerate(cases):
            with self.subTest(case=index):
                self._assert_rejected(
                    selected=selected,
                    decision=decision,
                    task=task,
                    admission=admission_evidence(task),
                    review_data=review_data,
                )

    def test_b5_requires_surrogate_confirmation(self) -> None:
        task = task_manifest(subclass="B5")
        selected = b5_candidate()

        self._assert_rejected(
            selected=selected,
            decision=screen_candidate(selected),
            task=task,
            admission=admission_evidence(task),
            review_data=review(task),
        )

    def test_cli_writes_atomic_verified_chain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, paths = self._write_cli_inputs(root)
            output = root / "output"

            result = self._run_cli(paths, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            chain_paths = sorted((output / "chain").glob("*.json"))
            self.assertEqual(len(chain_paths), 8)
            records = tuple(load_factory_contract(path) for path in chain_paths)
            self.assertTrue(
                all(isinstance(record, FactoryAdmissionRecord) for record in records)
            )
            self.assertEqual(
                tuple(record.state for record in records),
                VALID_STATES,
            )
            validate_admission_chain(records)
            final = load_factory_contract(output / "admission.json")
            self.assertEqual(final, records[-1])

    def test_cli_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, paths = self._write_cli_inputs(root, approved=False)
            output = root / "output"

            result = self._run_cli(paths, output)

            self.assertEqual(result.returncode, 2)
            self.assertIn("[contract_invalid]", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_rejects_preexisting_output_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, paths = self._write_cli_inputs(root)
            output = root / "output"
            output.mkdir()
            marker = output / "owned-by-user"
            marker.write_text("keep", encoding="utf-8")

            result = self._run_cli(paths, output)

            self.assertEqual(result.returncode, 2)
            self.assertIn("[contract_invalid]", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
