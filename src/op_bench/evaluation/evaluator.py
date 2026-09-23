"""Independent evaluation of candidate patches against required task cases."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from op_bench.runtime.workspace import age_cached_workspace, relocate_environment, workspace_environment
from op_bench.runtime.execution import ExecutionError, WitnessCollectionError, create_session, materialize_source, run_process
from op_bench.data.task import TaskSpec, TaskInputError
from op_bench.evaluation import numeric
from op_bench.evaluation.judgment import unittest_witness_error
from op_bench.io import write_json
from op_bench.provenance import software_identity


class PatchEvaluator:
    """Evaluate actual process outcomes in a fresh workspace for every patch.

    This class never consults a reference patch, a gold path list, or an agent
    transcript. Publication/admission review is separate from candidate scoring.
    """

    def evaluate(self, task: TaskSpec, patch: str, output_dir: Path,
                 *, baseline_artifact: Path | None = None,
                 baseline_source: dict | None = None) -> dict[str, Any]:
        output_dir = task.validate_output_path(output_dir)
        if task.variants:
            from op_bench.evaluation.variants import evaluate_variants
            return evaluate_variants(self, task, patch, output_dir,
                                     baseline_artifact=baseline_artifact, baseline_source=baseline_source)
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "result.json").exists():
            raise TaskInputError(f"Refusing to overwrite an existing evaluation: {output_dir}")
        started = time.monotonic()
        phase = "input"
        result: dict[str, Any] = {
            "schema_version": 1, "task_id": task.task_id,
            "task_identity": task.identity_dict(), "software_identity": software_identity(),
            "status": "evaluation_error", "resolved": False,
            "cases": [{"id": case.id, "group": case.group, "status": "not_run",
                       "kind": case.kind, "expected_tests": case.expected_tests,
                       "exit_code": None, "timed_out": False, "output_limited": False, "duration_sec": 0.0}
                      for case in task.tests],
            "stages": [], "duration_sec": 0.0,
            "environment": {"backend": task.environment.backend,
                            "image": task.environment.image,
                            "development_only": task.environment.backend == "local"},
            "source": None, "error": None, "cleanup_errors": [],
            "baseline_artifact": None,
            "sessions": {"build": None, "grading": None},
            "judgment_contract": {
                "numeric": "Candidate receives only current input in a fresh session without grader access; stopped execution yields bounded observations compared by the controller. Shape and finite real values are checked, not internal device, dtype or algorithm.",
                "integration": "command/unittest can import candidate code in the grader process. Unittest witnesses detect missing execution but are not an independent oracle boundary.",
                "local": "Local sessions provide process separation for development, not private-file isolation against arbitrary programs.",
            },
            "artifact_contract": {
                "mode": "preprovisioned_dependencies_and_workspace",
                "preserved": ["workspace"],
                "build_capture_ready": False,
                "not_preserved": ["temporary_files_outside_workspace", "container_installations", "process_environment_changes"],
                "limitation": "prepare/build must produce workspace artifacts; dependencies outside it must be preinstalled in the declared image. Arbitrary external writes are not exhaustively detected. Local execution remains development-only.",
            },
        }
        interrupted: BaseException | None = None
        sessions = {}
        session_role = "build"
        try:
            if not isinstance(patch, str):
                raise TaskInputError("patch must be a string")
            if not task.tests or len({case.id for case in task.tests}) != len(task.tests) or (
                len({(case.argv, case.oracle if case.kind == "numeric" else None) for case in task.tests}) != len(task.tests)
            ):
                raise TaskInputError("Required tests must be nonempty and have unique IDs and selectors")
            (output_dir / "patch.diff").write_text(patch, encoding="utf-8")
            phase = "source"
            with tempfile.TemporaryDirectory(prefix="opbench-eval-") as temporary:
                workspace = (Path(temporary) / "workspace").resolve()
                cached = None
                if baseline_artifact is not None:
                    from op_bench.runtime.baseline import fork_baseline
                    cached = fork_baseline(task, baseline_artifact, workspace, expected_source=baseline_source)
                    result["source"] = cached["compatibility"]["source"]
                    result["baseline_artifact"] = {
                        "path": cached["artifact_path"], "compatibility": cached["compatibility"],
                        "producer_task_identity": cached["producer_task_identity"],
                        "build_required": True, "assurance": cached["assurance"],
                    }
                    age_cached_workspace(workspace)
                else:
                    result["source"] = materialize_source(task, workspace)
                (output_dir / "source.json").write_text(json.dumps(result["source"], indent=2) + "\n")
                phase = "build_environment"
                if cached is None:
                    execution_task = workspace_environment(task, workspace, result["artifact_contract"])
                else:
                    environment = cached["execution_environment"]
                    execution_task = replace(task, environment=replace(task.environment,
                        image=environment["image"], environment=environment["environment"]))
                    result["artifact_contract"]["runtime_directory"] = cached["artifact_contract"]["runtime_directory"]
                    result["artifact_contract"]["runtime_paths"] = cached["artifact_contract"].get(
                        "runtime_paths", [cached["artifact_contract"]["runtime_directory"]])
                    # The trusted artifact already completed preparation without
                    # a patch or grader. Apply on this independent host copy so
                    # its multi-GB workspace needs only one upload before build.
                    phase = "patch"
                    if not self._apply_patch(patch, workspace, output_dir, result):
                        result["status"] = "invalid_patch"
                        result["error"] = {"stage": phase, "type": "patch_rejected",
                                           "message": "Candidate patch could not be applied to the prepared baseline"}
                        return result
                    phase = "build_environment"
                # Untrusted preparation/build receives only the source workspace.
                # The separate grading session is not created until every build
                # process has stopped and the final workspace is captured.
                with create_session(execution_task, workspace, None) as build_session:
                    sessions["build"] = build_session
                    if cached is None:
                        phase = "prepare"
                        if not self._commands(task.environment.prepare, task.environment.prepare_timeout_sec,
                                              "prepare", build_session, output_dir, result):
                            result["status"] = "environment_error"
                            result["error"] = {"stage": phase, "type": "command_failed",
                                               "message": "Environment preparation failed before patch application"}
                            return result
                        phase = "environment_sync"
                        build_session.sync_from_container()
                        phase = "patch"
                        if not self._apply_patch(patch, workspace, output_dir, result):
                            result["status"] = "invalid_patch"
                            result["error"] = {"stage": phase, "type": "patch_rejected",
                                               "message": "Candidate patch could not be applied to the task source"}
                            return result
                        phase = "environment_sync"
                        build_session.sync_to_container()
                    # Cached artifacts skip preparation, but every candidate still builds.
                    phase = "build"
                    if not self._commands(task.environment.build, task.environment.build_timeout_sec,
                                          "build", build_session, output_dir, result):
                        result["status"] = "build_failed"
                        result["error"] = {"stage": phase, "type": "command_failed",
                                           "message": "Build failed after candidate patch application"}
                        return result
                    phase = "build_capture"
                if not build_session.closed or not build_session.workspace_capture_ready:
                    raise ExecutionError("Build workspace cannot be graded: stopping and final artifact capture did not complete")
                result["artifact_contract"]["build_capture_ready"] = True
                # Reuse the actual image identity, even if a mutable image tag
                # changes between the two sessions.
                if build_session.info.get("image_id"):
                    execution_task = replace(execution_task, environment=replace(
                        execution_task.environment, image=build_session.info["image_id"]))
                if any(case.kind != "numeric" for case in task.tests):
                    session_role = "grading"
                    phase = "grading_environment"
                    grading_workspace = Path(temporary) / "grading-workspace"
                    shutil.copytree(workspace, grading_workspace, symlinks=True)
                    grading_task = relocate_environment(execution_task, workspace, grading_workspace)
                    with create_session(grading_task, grading_workspace, task.grader_dir,
                                        capture_workspace=False) as grading_session:
                        sessions["grading"] = grading_session
                        phase = "tests"
                        self._tests(task, grading_session, output_dir, result)
                        phase = "grading_cleanup"
                # Each numeric case gets its own copy of the captured build and no
                # grader mount; only the oracle input is sent to the candidate.
                for index, case in enumerate(task.tests):
                    if case.kind != "numeric":
                        continue
                    session_role = f"numeric-{index:03d}"
                    phase = "numeric_oracle"
                    oracle = numeric.load_oracle(task.grader_dir, case.oracle)
                    oracle_log = f"logs/test-{index:03d}.oracle.json"
                    (output_dir / "logs").mkdir(exist_ok=True)
                    (output_dir / oracle_log).write_text(json.dumps(oracle, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
                    phase = "numeric_environment"
                    candidate_workspace = Path(temporary) / f"numeric-{index:03d}-workspace"
                    shutil.copytree(workspace, candidate_workspace, symlinks=True)
                    candidate_task = relocate_environment(execution_task, workspace, candidate_workspace)
                    log = f"logs/test-{index:03d}.log"
                    observation_log = f"logs/test-{index:03d}.observation.json"
                    record = result["cases"][index]
                    with create_session(candidate_task, candidate_workspace, None,
                                        capture_workspace=False) as numeric_session:
                        sessions[session_role] = numeric_session
                        phase = "numeric_execution"
                        evidence = numeric_session.observe(case.argv, case.timeout_sec,
                            json.dumps(oracle["input"], allow_nan=False).encode() + b"\n",
                            output_dir / observation_log, output_dir / log)
                        record.update(**evidence.to_dict(), log=log, oracle_log=oracle_log,
                                      observation_log=observation_log, session_role=session_role)
                        phase = "numeric_capture"
                    if not numeric_session.execution_stopped:
                        message = "Numeric execution could not be confirmed stopped before trusted judgment"
                        record.update(status="error", reason="numeric_execution_boundary_unavailable",
                                      error={"type": "ExecutionError", "message": message})
                        raise ExecutionError(message)
                    phase = "numeric_judgment"
                    with (output_dir / observation_log).open("rb") as stream:
                        observation = stream.read(numeric.MAX_DOCUMENT_BYTES + 1)
                    decision = numeric.process_judgment(oracle, observation, evidence.to_dict())
                    record.update(numeric=decision, status="passed" if decision["passed"] else "failed")
                    if decision["reason"] is not None:
                        record["reason"] = decision["reason"]
                resolved = all(case["status"] == "passed" for case in result["cases"])
                result["status"] = "resolved" if resolved else "test_failed"
                result["resolved"] = resolved
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                interrupted = exc
            elif not isinstance(exc, Exception):
                interrupted = exc
            result["resolved"] = False
            result["status"] = "environment_error" if phase in {
                "source", "build_environment", "grading_environment", "numeric_environment", "environment_sync", "prepare", "build_capture"
            } else "evaluation_error"
            result["error"] = {"stage": phase, "type": type(exc).__name__, "message": str(exc)}
            if phase == "numeric_judgment":
                record.update(status="error", reason="numeric_observation_collection_error",
                              error={"type": type(exc).__name__, "message": str(exc)})
            if getattr(exc, "execution_session_info", None) is not None:
                result["sessions"][session_role] = exc.execution_session_info
            if getattr(exc, "__notes__", None):
                result["error"]["notes"] = list(exc.__notes__)
        finally:
            for role, session in sessions.items():
                result["sessions"][role] = session.info
            for role, info in result["sessions"].items():
                if info is not None:
                    result["environment"] = info
                    result["cleanup_errors"].extend(f"{role}: {error}" for error in info.get("cleanup_errors", []))
            build_info = result["sessions"]["build"]
            if build_info is not None:
                result["artifact_contract"]["build_capture_ready"] = bool(
                    build_info.get("closed") and build_info.get("workspace_capture_ready"))
            result["duration_sec"] = time.monotonic() - started
            result["groups"] = {
                group: {"passed": sum(case["status"] == "passed" for case in result["cases"] if case["group"] == group),
                        "total": sum(case["group"] == group for case in result["cases"])}
                for group in ("fail_to_pass", "pass_to_pass")
            }
            write_json(output_dir / "result.json", result)
        if interrupted is not None:
            raise interrupted
        return result

    @staticmethod
    def _commands(commands, timeout, stage, session, output_dir, result) -> bool:
        for index, argv in enumerate(commands):
            relative_log = f"logs/{stage}-{index:03d}.log"
            evidence = session.run(argv, timeout, output_dir / relative_log)
            result["stages"].append({"stage": stage, "index": index, "argv": list(argv),
                                     "log": relative_log, **evidence.to_dict()})
            if evidence.timed_out or evidence.output_limited or evidence.exit_code != 0:
                return False
        return True

    @staticmethod
    def _apply_patch(patch: str, workspace: Path, output_dir: Path, result: dict) -> bool:
        if not patch.strip():
            result["stages"].append({"stage": "patch", "status": "empty_patch"})
            return True
        # git apply enforces relative path/symlink boundaries by default. No
        # --unsafe-paths, gold-derived scope, or allow-list of production files.
        # Run outside a repository so project-local Git metadata/config cannot
        # influence application, and never recreate historical Git objects.
        for name, options in (("patch-check", ["--check"]), ("patch-apply", [])):
            log = f"logs/{name}.log"
            command = ["git", "apply", "--whitespace=nowarn", *options, str(output_dir / "patch.diff")]
            evidence = run_process(command, cwd=workspace, timeout_sec=60, log_path=output_dir / log)
            result["stages"].append({"stage": name, "log": log, **evidence.to_dict()})
            if evidence.timed_out or evidence.output_limited or evidence.exit_code != 0:
                return False
        return True

    @staticmethod
    def _tests(task, session, output_dir, result) -> None:
        for index, case in enumerate(task.tests):
            if case.kind == "numeric":
                continue
            if session.closed:
                result["cases"][index]["reason"] = "execution_environment_stopped_after_resource_limit"
                continue
            log = f"logs/test-{index:03d}.log"
            witness_error = None
            if case.kind == "unittest":
                try:
                    evidence, witness = session.run_unittest(case.argv, case.timeout_sec, output_dir / log)
                except WitnessCollectionError as exc:
                    result["cases"][index].update({
                        **exc.command_result.to_dict(), "status": "error", "log": log, "witness": None,
                        "reason": "unittest_witness_collection_error",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    })
                    raise
                result["cases"][index]["witness"] = witness
                witness_error = unittest_witness_error(witness, case.expected_tests)
                if witness is not None:
                    witness_log = f"logs/test-{index:03d}.witness.json"
                    (output_dir / witness_log).write_text(json.dumps(witness, indent=2) + "\n", encoding="utf-8")
                    result["cases"][index]["witness_log"] = witness_log
                if witness_error is not None:
                    result["cases"][index]["reason"] = witness_error
            else:
                evidence = session.run(case.argv, case.timeout_sec, output_dir / log)
            result["cases"][index].update({
                **evidence.to_dict(), "log": log,
                "status": "passed" if evidence.exit_code == 0 and not evidence.timed_out and not evidence.output_limited and witness_error is None else "failed",
            })
