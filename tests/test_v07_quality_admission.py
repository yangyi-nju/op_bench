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
    run_quality_admission,
    validate_quality_admission_prompt,
)
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

            evidence = validate_quality_admission_prompt(task)

            self.assertEqual(evidence.task_id, task.task_id)
            self.assertEqual(evidence.public_task_id, task.public_task_id)
            self.assertEqual(evidence.decision, "accepted")

    def test_prompt_revalidation_rejects_each_exact_source_drift(self) -> None:
        mutations = {
            "statement": lambda task: task.data["statement"].__setitem__(
                "body", task.data["statement"]["body"] + " Changed."
            ),
            "gold patch": lambda task: task.gold_patch_path.write_text(
                task.gold_patch_path.read_text(encoding="utf-8")
                + "\n+private_gold_change = True\n",
                encoding="utf-8",
            ),
            "hidden patch": lambda task: task.hidden_test_patch_path.write_text(
                task.hidden_test_patch_path.read_text(encoding="utf-8")
                + "\n+assert private_hidden_change\n",
                encoding="utf-8",
            ),
            "patch scope": lambda task: task.data[
                "patch_scope"
            ]["allowed_paths"].append("torch.nn.LazyLinear"),
            "hidden selector": lambda task: task.data[
                "evaluation"
            ]["fail_to_pass"].append("torch.nn.LazyLinear"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    original = complete_quality_task(Path(directory))
                    task = TaskManifest(
                        task_dir=original.task_dir,
                        data=copy.deepcopy(original.data),
                    )
                    mutate(task)

                    with self.assertRaises(ContractError):
                        validate_quality_admission_prompt(task)


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
        return EvaluationResult(
            task_id=task.task_id,
            mode=mode,
            status=status,
            fail_to_pass_total=1,
            fail_to_pass_passed=(
                0 if status == "baseline_reproduced" else 1
            ),
            pass_to_pass_total=1,
            pass_to_pass_passed=1,
            duration_sec=1.0,
            environment={"runtime": "synthetic"},
            commands=[],
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
