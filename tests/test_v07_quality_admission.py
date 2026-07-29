from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from op_bench.admission import AdmissionRunner
from op_bench.evaluator import EvaluationResult
from op_bench.factory.quality_admission import (
    QualityAdmissionResultIndex,
    QualityAdmissionResultRecord,
    load_quality_admission_result_index,
    quality_admission_bundle_hash,
    run_quality_admission,
    validate_quality_admission_prompt,
)
from op_bench.factory.quality_release import quality_prompt_source_hash
from op_bench.integrity import replay_spec_hash
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest
from tests.test_v07_expansion_artifacts import (
    SCREENING,
    _accepted_fixture,
    _rehash,
    _write_canonical,
)
from tests.test_v07_quality_validator import complete_quality_task

from scripts import run_v07_quality_admission as admission_cli


class V07QualityAdmissionPromptTests(unittest.TestCase):
    def test_exact_prompt_sources_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = complete_quality_task(Path(directory))
            prompt_source_hash = quality_prompt_source_hash(task)

            evidence = validate_quality_admission_prompt(
                task,
                expected_source_hash=prompt_source_hash,
            )

            self.assertEqual(evidence.task_id, task.task_id)
            self.assertEqual(evidence.public_task_id, task.public_task_id)
            self.assertEqual(evidence.decision, "accepted")

    def test_prompt_revalidation_rejects_private_raw_byte_drift(self) -> None:
        mutations = {
            "gold patch bytes": lambda task: task.gold_patch_path.write_bytes(
                task.gold_patch_path.read_bytes() + b"\n"
            ),
            "hidden patch bytes": lambda task: (
                task.hidden_test_patch_path.write_bytes(
                    task.hidden_test_patch_path.read_bytes() + b"\n"
                )
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    original = complete_quality_task(Path(directory))
                    prompt_source_hash = quality_prompt_source_hash(original)
                    task = TaskManifest(
                        task_dir=original.task_dir,
                        data=copy.deepcopy(original.data),
                    )
                    mutate(task)

                    with self.assertRaisesRegex(
                        ContractError, "source hash"
                    ):
                        validate_quality_admission_prompt(
                            task,
                            expected_source_hash=prompt_source_hash,
                        )

    def test_prompt_revalidation_rejects_exact_selector_drift(self) -> None:
        mutations = {
            "patch scope mapping": lambda task: task.data[
                "patch_scope"
            ].__setitem__("allowed_paths", [*task.patch_scope_paths, "torch/extra.py"]),
            "fail-to-pass selector": lambda task: task.data[
                "evaluation"
            ]["fail_to_pass"].append("tests/test_extra.py::test_regression"),
            "pass-to-pass selector": lambda task: task.data[
                "evaluation"
            ]["pass_to_pass"].append("tests/test_extra.py::test_ordinary"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    original = complete_quality_task(Path(directory))
                    prompt_source_hash = quality_prompt_source_hash(original)
                    task = TaskManifest(
                        task_dir=original.task_dir,
                        data=copy.deepcopy(original.data),
                    )
                    mutate(task)

                    with self.assertRaisesRegex(
                        ContractError, "source hash"
                    ):
                        validate_quality_admission_prompt(
                            task,
                            expected_source_hash=prompt_source_hash,
                        )


class _FakeEvaluator:
    def __init__(
        self,
        baseline_status: str = "baseline_reproduced",
        gold_status: str = "resolved",
    ) -> None:
        self.baseline_status = baseline_status
        self.gold_status = gold_status
        self.baseline_calls: list[str] = []
        self.gold_calls: list[str] = []

    def evaluate_baseline(self, task: TaskManifest) -> EvaluationResult:
        self.baseline_calls.append(task.task_id)
        return self._result(task, "baseline", self.baseline_status)

    def evaluate_gold(self, task: TaskManifest) -> EvaluationResult:
        self.gold_calls.append(task.task_id)
        return self._result(task, "gold", self.gold_status)

    @staticmethod
    def _result(
        task: TaskManifest,
        mode: str,
        status: str,
    ) -> EvaluationResult:
        fail_passed = (
            len(task.fail_to_pass_tests)
            if mode == "gold" and status == "resolved"
            else 0
        )
        commands = []
        for selector in task.fail_to_pass_tests:
            commands.append(
                {
                    "command": task.command_for_test(
                        selector,
                        python_executable=task.environment_python_executable,
                    ),
                    "cwd": "/synthetic/workspace",
                    "exit_code": 0 if fail_passed else 1,
                    "stdout": "",
                    "stderr": "",
                    "duration_sec": 0.1,
                    "timed_out": False,
                }
            )
        for selector in task.pass_to_pass_tests:
            commands.append(
                {
                    "command": task.command_for_test(
                        selector,
                        python_executable=task.environment_python_executable,
                    ),
                    "cwd": "/synthetic/workspace",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "duration_sec": 0.1,
                    "timed_out": False,
                }
            )
        return EvaluationResult(
            task_id=task.task_id,
            mode=mode,
            status=status,
            fail_to_pass_total=len(task.fail_to_pass_tests),
            fail_to_pass_passed=fail_passed,
            pass_to_pass_total=len(task.pass_to_pass_tests),
            pass_to_pass_passed=len(task.pass_to_pass_tests),
            duration_sec=1.0,
            environment={"runtime": "synthetic"},
            commands=commands,
        )


def _now() -> datetime:
    return datetime(2026, 7, 30, 3, 4, 5, tzinfo=timezone.utc)


def _runner_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    accepted_path, accepted, record = _accepted_fixture(root)
    complete = complete_quality_task(root / "complete")
    target = root / record["task_path"]
    shutil.rmtree(target)
    shutil.copytree(complete.task_dir, target)

    screening = json.loads(SCREENING.read_text(encoding="utf-8"))
    screening_record = screening["records"][
        record["screening_record_index"]
    ]
    candidate = json.loads(
        (
            root
            / "factory/v0.7/p8/screening"
            / screening_record["candidate"]["relative_path"]
        ).read_text(encoding="utf-8")
    )
    manifest = copy.deepcopy(complete.data)
    manifest["source"].update(
        {
            "repo": "pytorch/pytorch",
            "pr_number": candidate["pr_number"],
            "pr_url": candidate["pr_url"],
            "base_commit": candidate["base_commit"],
            "merge_commit": candidate["merge_commit"],
        }
    )
    manifest["quality"]["origin"] = "new"
    gold_path = target / manifest["artifacts"]["gold_patch"]
    allowed_path = manifest["patch_scope"]["allowed_paths"][0]
    gold_path.write_text(
        gold_path.read_text(encoding="utf-8").replace(
            "torch/public.py", allowed_path
        ),
        encoding="utf-8",
    )
    readmission_path = target / manifest["quality"][
        "readmission_evidence"
    ]
    readmission = json.loads(readmission_path.read_text(encoding="utf-8"))
    readmission["origin"] = "new"
    _rehash(readmission)
    _write_canonical(readmission_path, readmission)
    manifest_path = target / "task.json"
    _write_canonical(manifest_path, manifest)
    task = TaskManifest.load(manifest_path)

    record["task_id"] = task.task_id
    record["public_task_id"] = task.public_task_id
    record["origin"] = "new"
    record["task_manifest_hash"] = canonical_sha256(manifest)
    record["replay_spec_hash"] = replay_spec_hash(task)
    record["prompt_source_hash"] = quality_prompt_source_hash(task)
    _rehash(accepted)
    _write_canonical(accepted_path, accepted)

    environment_registry = root / "environments/registry.json"
    source_registry = root / "sources/registry.json"
    repository_root = Path(__file__).resolve().parents[1]
    environment_payload = json.loads(
        (
            repository_root / "environments/registry.json"
        ).read_text(encoding="utf-8")
    )
    source_payload = json.loads(
        (repository_root / "sources/registry.json").read_text(
            encoding="utf-8"
        )
    )
    source_asset = next(
        item
        for item in source_payload["sources"]
        if item["id"] == manifest["source_ref"]
    )
    source_asset["commit"] = candidate["base_commit"]
    _write_canonical(
        environment_registry,
        environment_payload,
    )
    _write_canonical(
        source_registry,
        source_payload,
    )
    output = root / "factory/v0.7/p8/admission_results.json"
    return (
        accepted_path,
        output,
        environment_registry,
        source_registry,
        accepted,
    )


def _refresh_accepted_task_hashes(
    root: Path,
    accepted_path: Path,
    accepted: dict[str, object],
) -> None:
    record = accepted["tasks"][0]
    task = TaskManifest.load(root / record["task_path"] / "task.json")
    record["task_manifest_hash"] = canonical_sha256(task.data)
    record["replay_spec_hash"] = replay_spec_hash(task)
    record["prompt_source_hash"] = quality_prompt_source_hash(task)
    _rehash(accepted)
    _write_canonical(accepted_path, accepted)


def _mutate_result_order(payload: dict[str, object]) -> None:
    first = copy.deepcopy(payload["results"][0])
    second = copy.deepcopy(payload["results"][0])
    first["screening_record_index"] = 10
    second["screening_record_index"] = 9
    second["task_id"] = "synthetic__second"
    second["public_task_id"] = "opbench-v07-t9999"
    second["task_path"] = "tasks/pytorch/synthetic_second"
    payload["results"] = [first, second]
    payload["task_count"] = 2
    payload["verified_count"] = 2


def _mutate_result_duplicate(payload: dict[str, object]) -> None:
    first = copy.deepcopy(payload["results"][0])
    second = copy.deepcopy(payload["results"][0])
    first["screening_record_index"] = 9
    second["screening_record_index"] = 10
    payload["results"] = [first, second]
    payload["task_count"] = 2
    payload["verified_count"] = 2


def _rewrite_result_bundle_hash(
    root: Path,
    output: Path,
) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    record = payload["results"][0]
    record["admission_bundle_hash"] = quality_admission_bundle_hash(
        root / record["admission_bundle_path"]
    )
    _rehash(payload)
    _write_canonical(output, payload)


class V07QualityAdmissionRunnerTests(unittest.TestCase):
    def test_runner_uses_only_index_path_and_verifies_all_four_gates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                accepted,
            ) = _runner_fixture(root)
            preflight_paths: list[Path] = []

            def preflight(path: Path) -> tuple[bool, list[str]]:
                preflight_paths.append(path)
                return True, ["synthetic preflight passed"]

            evaluator = _FakeEvaluator()
            result = run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=preflight,
                admission_runner=AdmissionRunner(
                    evaluator=evaluator,
                    now=_now,
                ),
            )

            self.assertEqual(
                preflight_paths,
                [root / accepted["tasks"][0]["task_path"]],
            )
            self.assertEqual(
                evaluator.baseline_calls,
                [accepted["tasks"][0]["task_id"]],
            )
            self.assertEqual(result.task_count, 1)
            self.assertEqual(result.verified_count, 1)
            self.assertTrue(result.results[0].verified)
            self.assertEqual(result.results[0].prompt_errors, ())
            self.assertEqual(result.results[0].preflight_status, "passed")
            self.assertEqual(
                result.results[0].admission_decision, "verified"
            )
            self.assertEqual(result.results[0].final_quality_errors, ())
            self.assertTrue(output.is_file())
            stable = (
                root
                / accepted["tasks"][0]["task_path"]
                / "admission/evidence.json"
            )
            self.assertEqual(
                result.results[0].admission_evidence_hash,
                "sha256:" + hashlib.sha256(stable.read_bytes()).hexdigest(),
            )
            bundle = root / result.results[0].admission_bundle_path
            self.assertTrue(bundle.is_dir())
            self.assertEqual(
                set(path.name for path in bundle.iterdir()),
                {
                    "baseline.log",
                    "environment.json",
                    "evidence.json",
                    "gold.log",
                    "source.json",
                },
            )
            self.assertEqual(
                result.results[0].admission_bundle_hash,
                quality_admission_bundle_hash(bundle),
            )

    def test_prompt_failure_stops_preflight_and_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                accepted,
            ) = _runner_fixture(root)
            record = accepted["tasks"][0]
            manifest_path = root / record["task_path"] / "task.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statement"]["body"] += " Prompt drift."
            _write_canonical(manifest_path, manifest)
            _refresh_accepted_task_hashes(
                root, accepted_path, accepted
            )
            preflight_calls: list[Path] = []
            evaluator = _FakeEvaluator()

            result = run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=lambda path: (
                    preflight_calls.append(path) or True,
                    [],
                ),
                admission_runner=AdmissionRunner(
                    evaluator=evaluator,
                    now=_now,
                ),
            )

            self.assertTrue(result.results[0].prompt_errors)
            self.assertEqual(result.results[0].preflight_status, "not_run")
            self.assertEqual(preflight_calls, [])
            self.assertEqual(evaluator.baseline_calls, [])
            self.assertFalse(result.results[0].verified)

    def test_accepted_prompt_source_digest_blocks_rehashed_mutations(
        self,
    ) -> None:
        mutations = {
            "gold bytes": lambda task, manifest: (
                task.gold_patch_path.write_bytes(
                    task.gold_patch_path.read_bytes() + b"\n"
                )
            ),
            "hidden bytes": lambda task, manifest: (
                task.hidden_test_patch_path.write_bytes(
                    task.hidden_test_patch_path.read_bytes() + b"\n"
                )
            ),
            "fail-to-pass selector": lambda task, manifest: manifest[
                "evaluation"
            ]["fail_to_pass"].append(
                "tests/test_extra.py::test_regression"
            ),
            "pass-to-pass selector": lambda task, manifest: manifest[
                "evaluation"
            ]["pass_to_pass"].append(
                "tests/test_extra.py::test_ordinary"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (
                        accepted_path,
                        output,
                        environment_registry,
                        source_registry,
                        accepted,
                    ) = _runner_fixture(root)
                    record = accepted["tasks"][0]
                    manifest_path = (
                        root / record["task_path"] / "task.json"
                    )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    task = TaskManifest.load(manifest_path)
                    mutate(task, manifest)
                    _write_canonical(manifest_path, manifest)
                    task = TaskManifest.load(manifest_path)
                    record["task_manifest_hash"] = canonical_sha256(
                        manifest
                    )
                    record["replay_spec_hash"] = replay_spec_hash(task)
                    _rehash(accepted)
                    _write_canonical(accepted_path, accepted)
                    evaluator = _FakeEvaluator()

                    with self.assertRaisesRegex(
                        ContractError, "prompt_source_hash"
                    ):
                        run_quality_admission(
                            root=root,
                            accepted_index_path=accepted_path,
                            output_path=output,
                            environment_registry_path=environment_registry,
                            source_registry_path=source_registry,
                            created_at="2026-07-30T03:04:05Z",
                            preflight=lambda path: (True, ["passed"]),
                            admission_runner=AdmissionRunner(
                                evaluator=evaluator,
                                now=_now,
                            ),
                        )

                    self.assertEqual(evaluator.baseline_calls, [])

    def test_preflight_failure_stops_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            evaluator = _FakeEvaluator()

            result = run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=lambda path: (False, ["preflight failed"]),
                admission_runner=AdmissionRunner(
                    evaluator=evaluator,
                    now=_now,
                ),
            )

            self.assertEqual(result.results[0].preflight_status, "failed")
            self.assertEqual(evaluator.baseline_calls, [])
            self.assertIsNone(result.results[0].admission_decision)
            self.assertFalse(result.results[0].verified)

    def test_admission_failure_records_baseline_and_gold_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)

            result = run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=lambda path: (True, ["passed"]),
                admission_runner=AdmissionRunner(
                    evaluator=_FakeEvaluator(
                        "baseline_reproduced",
                        "fail_to_pass_failed",
                    ),
                    now=_now,
                ),
            )

            record = result.results[0]
            self.assertEqual(record.baseline_status, "baseline_reproduced")
            self.assertEqual(record.gold_status, "fail_to_pass_failed")
            self.assertEqual(record.admission_decision, "gold_failed")
            self.assertFalse(record.admission_verified)
            self.assertFalse(record.verified)

    def test_registry_resolution_failure_is_a_canonical_task_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            source_payload = json.loads(
                source_registry.read_text(encoding="utf-8")
            )
            task = TaskManifest.load(
                root
                / json.loads(
                    accepted_path.read_text(encoding="utf-8")
                )["tasks"][0]["task_path"]
                / "task.json"
            )
            selected = next(
                item
                for item in source_payload["sources"]
                if item["id"] == task.source_ref
            )
            selected["commit"] = "f" * 40
            _write_canonical(source_registry, source_payload)

            result = run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=lambda path: (True, ["must not run"]),
                admission_runner=AdmissionRunner(
                    evaluator=_FakeEvaluator(),
                    now=_now,
                ),
            )

            self.assertTrue(result.results[0].prompt_errors)
            self.assertEqual(result.results[0].preflight_status, "not_run")
            self.assertFalse(result.results[0].verified)
            self.assertTrue(output.is_file())

    def test_invalid_output_is_rejected_before_any_admission_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                _,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            evaluator = _FakeEvaluator()

            with self.assertRaisesRegex(ContractError, "repository root"):
                run_quality_admission(
                    root=root,
                    accepted_index_path=accepted_path,
                    output_path=root.parent / "outside-results.json",
                    environment_registry_path=environment_registry,
                    source_registry_path=source_registry,
                    created_at="2026-07-30T03:04:05Z",
                    preflight=lambda path: (True, ["passed"]),
                    admission_runner=AdmissionRunner(
                        evaluator=evaluator,
                        now=_now,
                    ),
                )

            self.assertEqual(evaluator.baseline_calls, [])

    def test_output_collisions_are_rejected_before_any_admission_work(
        self,
    ) -> None:
        collision_paths = {
            "accepted index": lambda root, accepted, environment, source: accepted,
            "environment registry": (
                lambda root, accepted, environment, source: environment
            ),
            "source registry": (
                lambda root, accepted, environment, source: source
            ),
            "task manifest": (
                lambda root, accepted, environment, source: root
                / json.loads(accepted.read_text(encoding="utf-8"))[
                    "tasks"
                ][0]["task_path"]
                / "task.json"
            ),
            "task evidence": (
                lambda root, accepted, environment, source: root
                / json.loads(accepted.read_text(encoding="utf-8"))[
                    "tasks"
                ][0]["task_path"]
                / "admission/evidence.json"
            ),
        }
        for name, select_output in collision_paths.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (
                        accepted_path,
                        _,
                        environment_registry,
                        source_registry,
                        _,
                    ) = _runner_fixture(root)
                    evaluator = _FakeEvaluator()
                    preflight_calls: list[Path] = []

                    with self.assertRaisesRegex(
                        ContractError, "output.*collision"
                    ):
                        run_quality_admission(
                            root=root,
                            accepted_index_path=accepted_path,
                            output_path=select_output(
                                root,
                                accepted_path,
                                environment_registry,
                                source_registry,
                            ),
                            environment_registry_path=environment_registry,
                            source_registry_path=source_registry,
                            created_at="2026-07-30T03:04:05Z",
                            preflight=lambda path: (
                                preflight_calls.append(path) or True,
                                ["passed"],
                            ),
                            admission_runner=AdmissionRunner(
                                evaluator=evaluator,
                                now=_now,
                            ),
                        )

                    self.assertEqual(preflight_calls, [])
                    self.assertEqual(evaluator.baseline_calls, [])

    def test_input_ancestor_symlinks_are_rejected_before_loading(
        self,
    ) -> None:
        for input_name in ("accepted", "environment", "source"):
            with self.subTest(input=input_name):
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "repository"
                    root.mkdir()
                    (
                        accepted_path,
                        output,
                        environment_registry,
                        source_registry,
                        _,
                    ) = _runner_fixture(root)
                    outside = base / "outside"
                    outside.mkdir()
                    selected = {
                        "accepted": accepted_path,
                        "environment": environment_registry,
                        "source": source_registry,
                    }[input_name]
                    shutil.copy2(selected, outside / selected.name)
                    link = root / f"{input_name}-link"
                    link.symlink_to(outside, target_is_directory=True)
                    linked = link / selected.name
                    evaluator = _FakeEvaluator()

                    with self.assertRaisesRegex(
                        ContractError, "symlink"
                    ):
                        run_quality_admission(
                            root=root,
                            accepted_index_path=(
                                linked
                                if input_name == "accepted"
                                else accepted_path
                            ),
                            output_path=output,
                            environment_registry_path=(
                                linked
                                if input_name == "environment"
                                else environment_registry
                            ),
                            source_registry_path=(
                                linked
                                if input_name == "source"
                                else source_registry
                            ),
                            created_at="2026-07-30T03:04:05Z",
                            preflight=lambda path: (True, ["passed"]),
                            admission_runner=AdmissionRunner(
                                evaluator=evaluator,
                                now=_now,
                            ),
                        )

                    self.assertEqual(evaluator.baseline_calls, [])

    def test_result_loader_rejects_result_ancestor_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repository"
            root.mkdir()
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=lambda path: (True, ["passed"]),
                admission_runner=AdmissionRunner(
                    evaluator=_FakeEvaluator(),
                    now=_now,
                ),
            )
            outside = base / "outside"
            outside.mkdir()
            shutil.copy2(output, outside / output.name)
            link = root / "result-link"
            link.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ContractError, "symlink"):
                load_quality_admission_result_index(
                    root,
                    link / output.name,
                    accepted_path,
                )

    def test_only_own_prior_result_may_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            kwargs = {
                "root": root,
                "accepted_index_path": accepted_path,
                "output_path": output,
                "environment_registry_path": environment_registry,
                "source_registry_path": source_registry,
                "created_at": "2026-07-30T03:04:05Z",
                "preflight": lambda path: (True, ["passed"]),
            }
            first = run_quality_admission(
                **kwargs,
                admission_runner=AdmissionRunner(
                    evaluator=_FakeEvaluator(),
                    now=_now,
                ),
            )
            second = run_quality_admission(
                **kwargs,
                admission_runner=AdmissionRunner(
                    evaluator=_FakeEvaluator(),
                    now=_now,
                ),
            )
            self.assertEqual(second.to_dict(), first.to_dict())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            _write_canonical(output, {"not": "an admission result"})
            evaluator = _FakeEvaluator()

            with self.assertRaisesRegex(
                ContractError, "output collision"
            ):
                run_quality_admission(
                    root=root,
                    accepted_index_path=accepted_path,
                    output_path=output,
                    environment_registry_path=environment_registry,
                    source_registry_path=source_registry,
                    created_at="2026-07-30T03:04:05Z",
                    preflight=lambda path: (True, ["passed"]),
                    admission_runner=AdmissionRunner(
                        evaluator=evaluator,
                        now=_now,
                    ),
                )

            self.assertEqual(evaluator.baseline_calls, [])

    def test_result_loader_binds_index_registries_order_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                _,
            ) = _runner_fixture(root)
            result = run_quality_admission(
                root=root,
                accepted_index_path=accepted_path,
                output_path=output,
                environment_registry_path=environment_registry,
                source_registry_path=source_registry,
                created_at="2026-07-30T03:04:05Z",
                preflight=lambda path: (True, ["passed"]),
                admission_runner=AdmissionRunner(
                    evaluator=_FakeEvaluator(),
                    now=_now,
                ),
            )

            loaded = load_quality_admission_result_index(
                root, output, accepted_path
            )

            self.assertEqual(loaded.to_dict(), result.to_dict())
            mutations = {
                "identity": lambda payload: payload["results"][0].__setitem__(
                    "task_id", "pytorch__999999__substitution"
                ),
                "count": lambda payload: payload.__setitem__(
                    "task_count", 2
                ),
                "registry": lambda payload: payload.__setitem__(
                    "source_registry_hash", "sha256:" + "f" * 64
                ),
                "admission truth": lambda payload: (
                    payload["results"][0].__setitem__(
                        "admission_decision", "gold_failed"
                    ),
                    payload["results"][0].__setitem__(
                        "admission_verified", False
                    ),
                    payload["results"][0].__setitem__(
                        "gold_status", "fail_to_pass_failed"
                    ),
                    payload["results"][0].__setitem__("verified", False),
                    payload.__setitem__("verified_count", 0),
                ),
                "reordering": _mutate_result_order,
                "duplicate identity": _mutate_result_duplicate,
            }
            original = result.to_dict()
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    payload = copy.deepcopy(original)
                    mutate(payload)
                    _rehash(payload)
                    _write_canonical(output, payload)
                    with self.assertRaises(ContractError):
                        load_quality_admission_result_index(
                            root, output, accepted_path
                        )

    def test_result_loader_replays_each_selector_exit_code(self) -> None:
        mutations = (
            ("baseline", "fail_to_pass", 0),
            ("baseline", "pass_to_pass", 1),
            ("gold", "fail_to_pass", 1),
            ("gold", "pass_to_pass", 1),
        )
        for phase, axis, exit_code in mutations:
            with self.subTest(phase=phase, axis=axis):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (
                        accepted_path,
                        output,
                        environment_registry,
                        source_registry,
                        accepted,
                    ) = _runner_fixture(root)
                    run_quality_admission(
                        root=root,
                        accepted_index_path=accepted_path,
                        output_path=output,
                        environment_registry_path=environment_registry,
                        source_registry_path=source_registry,
                        created_at="2026-07-30T03:04:05Z",
                        preflight=lambda path: (True, ["passed"]),
                        admission_runner=AdmissionRunner(
                            evaluator=_FakeEvaluator(),
                            now=_now,
                        ),
                    )
                    task = TaskManifest.load(
                        root
                        / accepted["tasks"][0]["task_path"]
                        / "task.json"
                    )
                    selector = (
                        task.fail_to_pass_tests[0]
                        if axis == "fail_to_pass"
                        else task.pass_to_pass_tests[0]
                    )
                    result_payload = json.loads(
                        output.read_text(encoding="utf-8")
                    )
                    bundle = (
                        root
                        / result_payload["results"][0][
                            "admission_bundle_path"
                        ]
                    )
                    evidence = json.loads(
                        (bundle / "evidence.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    command = next(
                        item
                        for item in evidence[phase]["commands"]
                        if selector in " ".join(item["command"])
                    )
                    command["exit_code"] = exit_code
                    _write_canonical(bundle / "evidence.json", evidence)
                    _write_canonical(bundle / f"{phase}.log", evidence[phase])
                    _rewrite_result_bundle_hash(root, output)

                    with self.assertRaisesRegex(
                        ContractError, "selector"
                    ):
                        load_quality_admission_result_index(
                            root, output, accepted_path
                        )

    def test_result_loader_replays_selector_totals_and_counts(self) -> None:
        fields = (
            "fail_to_pass_total",
            "fail_to_pass_passed",
            "pass_to_pass_total",
            "pass_to_pass_passed",
        )
        for phase in ("baseline", "gold"):
            for field in fields:
                with self.subTest(phase=phase, field=field):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        (
                            accepted_path,
                            output,
                            environment_registry,
                            source_registry,
                            _,
                        ) = _runner_fixture(root)
                        run_quality_admission(
                            root=root,
                            accepted_index_path=accepted_path,
                            output_path=output,
                            environment_registry_path=environment_registry,
                            source_registry_path=source_registry,
                            created_at="2026-07-30T03:04:05Z",
                            preflight=lambda path: (True, ["passed"]),
                            admission_runner=AdmissionRunner(
                                evaluator=_FakeEvaluator(),
                                now=_now,
                            ),
                        )
                        result_payload = json.loads(
                            output.read_text(encoding="utf-8")
                        )
                        bundle = (
                            root
                            / result_payload["results"][0][
                                "admission_bundle_path"
                            ]
                        )
                        evidence = json.loads(
                            (bundle / "evidence.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        evidence[phase][field] += 1
                        _write_canonical(bundle / "evidence.json", evidence)
                        _write_canonical(
                            bundle / f"{phase}.log",
                            evidence[phase],
                        )
                        _rewrite_result_bundle_hash(root, output)

                        with self.assertRaisesRegex(
                            ContractError, field
                        ):
                            load_quality_admission_result_index(
                                root, output, accepted_path
                            )

    def test_result_loader_binds_full_bundle_stable_and_readmission(
        self,
    ) -> None:
        mutations = ("bundle component", "stable summary", "readmission")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (
                        accepted_path,
                        output,
                        environment_registry,
                        source_registry,
                        accepted,
                    ) = _runner_fixture(root)
                    run_quality_admission(
                        root=root,
                        accepted_index_path=accepted_path,
                        output_path=output,
                        environment_registry_path=environment_registry,
                        source_registry_path=source_registry,
                        created_at="2026-07-30T03:04:05Z",
                        preflight=lambda path: (True, ["passed"]),
                        admission_runner=AdmissionRunner(
                            evaluator=_FakeEvaluator(),
                            now=_now,
                        ),
                    )
                    result_payload = json.loads(
                        output.read_text(encoding="utf-8")
                    )
                    outcome = result_payload["results"][0]
                    task_dir = root / accepted["tasks"][0]["task_path"]
                    bundle = root / outcome["admission_bundle_path"]
                    if mutation == "bundle component":
                        component = json.loads(
                            (bundle / "environment.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        component["runtime_tier"] = "mutated"
                        _write_canonical(
                            bundle / "environment.json",
                            component,
                        )
                        _rewrite_result_bundle_hash(root, output)
                    elif mutation == "stable summary":
                        stable_path = task_dir / "admission/evidence.json"
                        stable = json.loads(
                            stable_path.read_text(encoding="utf-8")
                        )
                        stable["baseline"]["fail_to_pass_total"] += 1
                        _write_canonical(stable_path, stable)
                        result_payload["results"][0][
                            "admission_evidence_hash"
                        ] = "sha256:" + hashlib.sha256(
                            stable_path.read_bytes()
                        ).hexdigest()
                        _rehash(result_payload)
                        _write_canonical(output, result_payload)
                    else:
                        task = TaskManifest.load(task_dir / "task.json")
                        readmission_path = (
                            task_dir
                            / task.data["quality"]["readmission_evidence"]
                        )
                        readmission = json.loads(
                            readmission_path.read_text(encoding="utf-8")
                        )
                        readmission["admission_evidence_hash"] = (
                            "sha256:" + "f" * 64
                        )
                        _rehash(readmission)
                        _write_canonical(readmission_path, readmission)

                    with self.assertRaises(ContractError):
                        load_quality_admission_result_index(
                            root, output, accepted_path
                        )

    def test_result_schema_fields_are_exact(self) -> None:
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas/v07_quality_release.schema.json"
            ).read_text(encoding="utf-8")
        )
        record = schema["$defs"]["quality_admission_result_record"]
        index = schema["$defs"]["quality_admission_result_index"]
        self.assertEqual(
            set(record["required"]),
            set(QualityAdmissionResultRecord.wire_fields()),
        )
        self.assertEqual(
            set(record["properties"]),
            set(QualityAdmissionResultRecord.wire_fields()),
        )
        self.assertEqual(
            set(index["required"]),
            set(QualityAdmissionResultIndex.wire_fields()),
        )
        self.assertEqual(
            set(index["properties"]),
            set(QualityAdmissionResultIndex.wire_fields()),
        )


class V07QualityAdmissionCliTests(unittest.TestCase):
    def test_cli_exposes_only_indexed_inputs_and_no_bypass(self) -> None:
        parser = admission_cli.build_parser()
        destinations = {
            action.dest for action in parser._actions  # noqa: SLF001
        }
        self.assertEqual(
            destinations,
            {
                "help",
                "accepted_index",
                "output",
                "environment_registry",
                "source_registry",
                "created_at",
                "quiet",
            },
        )
        self.assertNotIn("task", destinations)
        self.assertNotIn("allow_incomplete", destinations)

        for unknown in ("--task", "--allow-incomplete"):
            with self.subTest(flag=unknown):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        admission_cli.main([unknown, "substitution"])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_runs_the_exact_building_subset_and_writes_valid_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                accepted_path,
                output,
                environment_registry,
                source_registry,
                accepted,
            ) = _runner_fixture(root)
            executed: list[Path] = []

            def run_with_fakes(**kwargs):
                return run_quality_admission(
                    **kwargs,
                    preflight=lambda path: (
                        executed.append(path) or True,
                        ["passed"],
                    ),
                    admission_runner=AdmissionRunner(
                        evaluator=_FakeEvaluator(),
                        now=_now,
                    ),
                )

            stdout = io.StringIO()
            with (
                patch.object(admission_cli, "ROOT", root),
                patch.object(
                    admission_cli,
                    "run_quality_admission",
                    side_effect=run_with_fakes,
                ),
                redirect_stdout(stdout),
            ):
                status = admission_cli.main(
                    [
                        "--accepted-index",
                        str(accepted_path),
                        "--output",
                        str(output),
                        "--environment-registry",
                        str(environment_registry),
                        "--source-registry",
                        str(source_registry),
                        "--created-at",
                        "2026-07-30T03:04:05Z",
                        "--quiet",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(
                executed,
                [root / accepted["tasks"][0]["task_path"]],
            )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["task_count"], 1)
            self.assertEqual(summary["verified_count"], 1)
            load_quality_admission_result_index(
                root, output, accepted_path
            )

    def test_phase_one_does_not_create_official_placeholder_indexes(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        self.assertFalse(
            (
                repository_root
                / "factory/v0.7/p8/accepted_tasks.json"
            ).exists()
        )
        self.assertFalse(
            (
                repository_root
                / "factory/v0.7/p8/admission_results.json"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
