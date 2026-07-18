from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Callable

from op_bench.dataset import DatasetManifest
from op_bench.runtime.backends import (
    DockerRuntimeBackend,
    LocalProcessBackend,
    RemoteDockerRuntimeBackend,
    RuntimeAttemptContext,
    RuntimeBackend,
    RuntimeBackendUnavailable,
    RuntimeTargetBinding,
    load_runtime_target_binding,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import ContentIdentity, EvaluationSpec
from op_bench.runtime.evaluation import FreshEvaluator, ReplayEvaluationEvidence
from op_bench.runtime.legacy import (
    agent_spec_for_v1_adapter,
    full_task_spec_from_v05,
    runtime_bundle_from_v05_dataset,
)
from op_bench.runtime.resources import AttemptResourceLedger, RuntimeLeaseStore
from op_bench.runtime.runtime_evaluation import RuntimeFreshEvaluationBackend
from op_bench.runtime.validation import ContractError, require_int, require_str
from op_bench.runtime.workspace import (
    FrozenPatch,
    build_patch_artifact,
    patch_paths_from_bytes,
    raw_patch_identity,
)


LEGACY_ROOTS = (
    ("runs/v0.5_codex_legacy_cpu", 30),
    ("runs/v0.5_codex_legacy_cuda", 3),
    ("runs/v0.5_precision_codex_cpu", 9),
    ("runs/v0.5_precision_codex_gpu", 9),
)
_STATUS_OUTCOMES = {
    "resolved": "resolved",
    "fail_to_pass_failed": "f2p_failed",
    "pass_to_pass_regressed": "p2p_regression",
    "invalid_patch": "invalid_patch",
}
_EMPTY_PATCH_HASH = "sha256:" + hashlib.sha256(b"").hexdigest()
_PATCH_NAME = re.compile(
    r"(?P<task>.+)__codex_action_bridge\.patch$"
)
_ATTEMPT_DIRECTORY = re.compile(r"attempt_(?P<number>[0-9]{3})")


@dataclass(frozen=True)
class ReplayCase:
    replay_id: str
    case_kind: str
    task_id: str
    task_path: str
    task_verified: bool
    source_id: str
    environment_id: str
    runtime_profile_id: str
    runtime_profile_hash: str
    patch_path: str | None
    patch_hash: str
    expected_outcome: str
    attempt_number: int | None
    provenance_root: str
    provenance_line: int
    provenance_hash: str

    def __post_init__(self) -> None:
        require_str(self.replay_id, "replay_id", pattern=r"replay:v1:[0-9a-f]{64}")
        if self.case_kind not in {"baseline", "gold", "legacy"}:
            raise ContractError(f"case_kind: unsupported value {self.case_kind!r}")
        for value, path in (
            (self.task_id, "task_id"),
            (self.task_path, "task_path"),
            (self.source_id, "source_id"),
            (self.environment_id, "environment_id"),
            (self.runtime_profile_id, "runtime_profile_id"),
            (self.runtime_profile_hash, "runtime_profile_hash"),
            (self.patch_hash, "patch_hash"),
            (self.expected_outcome, "expected_outcome"),
            (self.provenance_root, "provenance_root"),
            (self.provenance_hash, "provenance_hash"),
        ):
            require_str(value, path)
        if not self.task_verified:
            raise ContractError("task_verified: replay cases require verified tasks")
        if self.patch_path is not None:
            path = Path(self.patch_path)
            if path.is_absolute() or ".." in path.parts:
                raise ContractError("patch_path: expected repository-relative path")
        if self.case_kind == "baseline":
            if self.patch_path is not None or self.attempt_number is not None:
                raise ContractError("baseline: patch and attempt must be absent")
        elif self.patch_path is None:
            raise ContractError("patch_path: patch case requires path")
        if self.case_kind == "legacy":
            if self.attempt_number is None:
                raise ContractError("legacy: attempt_number is required")
            require_int(self.attempt_number, "attempt_number", minimum=1)
            require_int(self.provenance_line, "provenance_line", minimum=1)

    @property
    def sort_key(self) -> tuple[object, ...]:
        order = {"baseline": 0, "gold": 1, "legacy": 2}
        return (
            order[self.case_kind],
            self.task_id,
            self.provenance_root,
            self.attempt_number or 0,
            self.patch_path or "",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "replay_id": self.replay_id,
            "case_kind": self.case_kind,
            "task_id": self.task_id,
            "task_path": self.task_path,
            "task_verified": self.task_verified,
            "source_id": self.source_id,
            "environment_id": self.environment_id,
            "runtime_profile_id": self.runtime_profile_id,
            "runtime_profile_hash": self.runtime_profile_hash,
            "patch_path": self.patch_path,
            "patch_hash": self.patch_hash,
            "expected_outcome": self.expected_outcome,
            "attempt_number": self.attempt_number,
            "provenance_root": self.provenance_root,
            "provenance_line": self.provenance_line,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class ReplayDifference:
    replay_id: str
    expected_outcome: str
    observed_outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "replay_id": self.replay_id,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
        }


@dataclass(frozen=True)
class ReplayResult:
    replay_id: str
    status: str
    expected_outcome: str
    observed_outcome: str | None
    reason_code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "replay_id": self.replay_id,
            "status": self.status,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class ReplaySummary:
    total: int
    passed: int
    failed: int
    blocked: int

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "blocked": self.blocked,
        }


@dataclass(frozen=True)
class ReplayRunReport:
    inventory_hash: str
    results: tuple[ReplayResult, ...]
    differences: tuple[ReplayDifference, ...]
    summary: ReplaySummary


ReplayBackendFactory = Callable[[object, RuntimeTargetBinding], RuntimeBackend]


class ExactReplayObserver:
    """Execute frozen replay cases on one explicitly configured Runtime target."""

    _GLOBAL_UNAVAILABLE = {
        "remote_workspace_create_failed",
        "remote_source_sync_failed",
    }

    def __init__(
        self,
        repository_root: Path,
        target_config: Path,
        *,
        dataset_path: Path | None = None,
        backend_factory: ReplayBackendFactory | None = None,
    ) -> None:
        self.repository_root = _real_directory(repository_root, "repository_root")
        self._temporary = tempfile.TemporaryDirectory(prefix="opbench-replay-")
        try:
            self._scratch = Path(self._temporary.name)
            local_workspaces = self._scratch / "workspaces"
            local_workspaces.mkdir()
            exact_config = Path(target_config).resolve(strict=True)
            self.target_binding = load_runtime_target_binding(
                exact_config,
                local_workspace_parent=local_workspaces,
            )
            selected_dataset = (
                self.repository_root / "datasets" / "pytorch_v0.5" / "dataset.json"
                if dataset_path is None
                else _within_root(
                    self.repository_root,
                    dataset_path,
                    "dataset_path",
                )
            )
            self._bundle = runtime_bundle_from_v05_dataset(
                selected_dataset,
                agents=(agent_spec_for_v1_adapter("scripted_canonical"),),
                repeat=1,
                created_at="1970-01-01T00:00:00Z",
            )
            self._tasks = {
                task.task.identifier: task for task in self._bundle.manifest.tasks
            }
            if any(
                task.runtime.backend != self.target_binding.backend
                for task in self._tasks.values()
            ):
                raise ContractError(
                    "target_config: backend does not match frozen replay Profiles"
                )
            self._backend_factory = backend_factory or _default_replay_backend
            if not callable(self._backend_factory):
                raise ContractError("backend_factory: expected callable")
            self._clock_value = 0
            self._global_unavailable_reason: str | None = None
            self._profile_unavailable_reasons: dict[str, str] = {}
            self._closed = False
        except Exception:
            self._temporary.cleanup()
            raise

    def __enter__(self) -> "ExactReplayObserver":
        if self._closed:
            raise ContractError("exact replay observer is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._temporary.cleanup()
            self._closed = True

    def __call__(self, case: ReplayCase) -> ReplayEvaluationEvidence:
        if self._closed:
            raise ContractError("exact replay observer is closed")
        if not isinstance(case, ReplayCase):
            raise ContractError("case: expected ReplayCase")
        task = self._tasks.get(case.task_id)
        if task is None:
            raise ContractError("case: Task is absent from frozen replay bundle")
        if (
            task.source.identifier != case.source_id
            or task.environment.identifier != case.environment_id
            or task.runtime.profile_id != case.runtime_profile_id
            or task.runtime.content_hash != case.runtime_profile_hash
        ):
            raise ContractError("case: frozen Task/Profile attribution mismatch")
        cached = self._global_unavailable_reason or self._profile_unavailable_reasons.get(
            task.runtime.profile_id
        )
        if cached is not None:
            raise RuntimeBackendUnavailable(cached)

        attempt_id = "attempt:v1:" + case.replay_id.removeprefix("replay:v1:")
        case_root = self._scratch / case.replay_id.removeprefix("replay:v1:")
        case_root.mkdir()
        ledger = AttemptResourceLedger(
            case_root / "runtime_resources.jsonl",
            attempt_id=attempt_id,
            retry_index=1,
            runtime_profile_hash=task.runtime.content_hash,
            clock_ms=self._clock_ms,
        )
        lease_store = RuntimeLeaseStore(
            case_root / "private_runtime_resources.json",
            attempt_id=attempt_id,
            retry_index=1,
            runtime_profile_hash=task.runtime.content_hash,
        )
        source = self._bundle.source_for(task)
        hidden_asset = self._bundle.hidden_asset_for(task)
        context = RuntimeAttemptContext(
            attempt_id=attempt_id,
            retry_index=1,
            runtime_profile_hash=task.runtime.content_hash,
            frozen_source_directory=source.repository,
            frozen_source_revision=source.revision,
            resource_ledger=ledger,
            lease_store=lease_store,
            target_binding=self.target_binding,
        )
        runtime_evaluator = RuntimeFreshEvaluationBackend(
            source=source,
            hidden_asset=hidden_asset,
            python_executable=(
                sys.executable if task.runtime.backend == "local" else "python"
            ),
            runtime_backend=self._backend_factory(task.runtime, self.target_binding),
            runtime_profile=task.runtime,
            attempt_context=context,
            source_overlay_paths=task.patch_scope,
        )
        frozen, patch_artifact = self._patch_inputs(case, task, source.revision)
        spec = EvaluationSpec(
            session_id="replay-session-" + case.replay_id.removeprefix("replay:v1:"),
            attempt_id=attempt_id,
            task=task.task,
            source=task.source,
            frozen_patch=None if frozen is None else frozen.patch,
            hidden_test_asset=task.hidden_test_asset,
            public_tests=task.public_tests,
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            runtime=task.runtime,
            timeout_ms=task.runtime.timeout_ms,
            evaluation=self._bundle.manifest.evaluation,
            scoring=self._bundle.manifest.scoring,
        )
        try:
            observed = FreshEvaluator(
                runtime_evaluator,
                clock_ms=self._clock_ms,
            ).evaluate_replay(spec, frozen, patch_artifact)
        finally:
            ledger.close()
            lease_store.close()
        unavailable = runtime_evaluator.last_backend_unavailable_reason
        if unavailable is not None:
            if unavailable in self._GLOBAL_UNAVAILABLE:
                self._global_unavailable_reason = unavailable
            else:
                self._profile_unavailable_reasons[task.runtime.profile_id] = unavailable
            raise RuntimeBackendUnavailable(unavailable)
        return observed

    def _patch_inputs(self, case, task, revision):
        if case.patch_path is None:
            return None, None
        patch_path = _within_root(
            self.repository_root,
            self.repository_root / case.patch_path,
            "replay patch",
        )
        patch_bytes = _read_regular(patch_path, "replay patch")
        patch = raw_patch_identity(
            patch_bytes,
            identifier=f"{case.replay_id}:patch",
        )
        if patch.digest != case.patch_hash:
            raise ContractError("replay patch identity changed")
        workspace = ContentIdentity(
            identity_type="workspace",
            identifier=f"{case.replay_id}:workspace",
            digest=canonical_sha256(
                {
                    "replay_id": case.replay_id,
                    "source": task.source.to_dict(),
                }
            ),
            digest_kind="canonical_config",
        )
        frozen = FrozenPatch(
            workspace=workspace,
            source=task.source,
            base_commit=revision,
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=patch_paths_from_bytes(patch_bytes),
            empty=False,
        )
        return frozen, build_patch_artifact(
            frozen,
            artifact_id=f"{case.replay_id}/replay.patch",
        )

    def _clock_ms(self) -> int:
        self._clock_value += 1
        return self._clock_value


def _default_replay_backend(profile, binding: RuntimeTargetBinding) -> RuntimeBackend:
    del binding
    backends = {
        "local": LocalProcessBackend,
        "docker": DockerRuntimeBackend,
        "remote_docker": RemoteDockerRuntimeBackend,
    }
    try:
        backend_type = backends[profile.backend]
    except KeyError as exc:
        raise ContractError("Runtime Profile backend is not executable") from exc
    return backend_type()


class ReplayRunner:
    def __init__(
        self,
        repository_root: Path,
        cases: tuple[ReplayCase, ...],
    ) -> None:
        self.repository_root = _real_directory(repository_root, "repository_root")
        if not isinstance(cases, tuple) or not cases:
            raise ContractError("cases: expected non-empty tuple")
        if any(not isinstance(case, ReplayCase) for case in cases):
            raise ContractError("cases: expected ReplayCase values")
        if tuple(sorted(cases, key=lambda item: item.sort_key)) != cases:
            raise ContractError("cases: expected frozen inventory order")
        if len({case.replay_id for case in cases}) != len(cases):
            raise ContractError("cases: duplicate replay_id")
        self.cases = cases

    def run(
        self,
        output_root: Path,
        *,
        observer: Callable[[ReplayCase], object] | None = None,
    ) -> ReplayRunReport:
        for case in self.cases:
            self._validate_case_authority(case)
        replay_root = Path(output_root).resolve() / "replay"
        self._validate_output_root(replay_root)
        replay_root.mkdir(parents=True, exist_ok=True)
        inventory_payload = [case.to_dict() for case in self.cases]
        inventory_hash = canonical_sha256(inventory_payload)
        results: list[ReplayResult] = []
        differences: list[ReplayDifference] = []
        for case in self.cases:
            status = "Blocked"
            observed_outcome: str | None = None
            reason_code: str | None = "exact_runtime_unavailable"
            if observer is not None:
                try:
                    observed = observer(case)
                    if observed is not None:
                        from op_bench.runtime.evaluation import ReplayEvaluationEvidence

                        if isinstance(observed, ReplayEvaluationEvidence):
                            observed_outcome = observed.observed_outcome
                            reason_code = observed.invalid_reason
                        else:
                            observed_outcome = require_str(
                                observed,
                                "observed_outcome",
                            )
                            reason_code = None
                        status = (
                            "Passed"
                            if observed_outcome == case.expected_outcome
                            else "Failed"
                        )
                except Exception as exc:  # noqa: BLE001 - one case cannot hide others.
                    from op_bench.runtime.backends import RuntimeBackendUnavailable
                    from op_bench.runtime.evaluation import EvaluationInfrastructureError

                    if isinstance(exc, RuntimeBackendUnavailable):
                        status = "Blocked"
                        reason_code = exc.reason_code
                    elif isinstance(exc, EvaluationInfrastructureError):
                        status = "Failed"
                        observed_outcome = "evaluation_error"
                        reason_code = exc.invalid_reason
                    else:
                        status = "Failed"
                        observed_outcome = "evaluation_error"
                        reason_code = "replay_runner_error"
            result = ReplayResult(
                replay_id=case.replay_id,
                status=status,
                expected_outcome=case.expected_outcome,
                observed_outcome=observed_outcome,
                reason_code=reason_code,
            )
            results.append(result)
            if status == "Failed":
                differences.append(
                    ReplayDifference(
                        replay_id=case.replay_id,
                        expected_outcome=case.expected_outcome,
                        observed_outcome=observed_outcome or "evaluation_error",
                    )
                )
        summary = ReplaySummary(
            total=len(results),
            passed=sum(item.status == "Passed" for item in results),
            failed=sum(item.status == "Failed" for item in results),
            blocked=sum(item.status == "Blocked" for item in results),
        )
        manifest = {
            "manifest_type": "legacy_replay_manifest",
            "schema_version": "v1",
            "inventory_hash": inventory_hash,
            "cases": inventory_payload,
        }
        _write_json(replay_root / "replay_manifest.json", manifest)
        _write_json_lines(
            replay_root / "replay_results.jsonl",
            [item.to_dict() for item in results],
        )
        _write_json_lines(
            replay_root / "replay_differences.jsonl",
            [item.to_dict() for item in differences],
        )
        _write_json(
            replay_root / "replay_summary.json",
            {"inventory_hash": inventory_hash, **summary.to_dict()},
        )
        return ReplayRunReport(
            inventory_hash=inventory_hash,
            results=tuple(results),
            differences=tuple(differences),
            summary=summary,
        )

    def _validate_case_authority(self, case: ReplayCase) -> None:
        if case.patch_path is None:
            if case.patch_hash != _EMPTY_PATCH_HASH:
                raise ContractError("baseline patch hash changed")
        else:
            patch = _within_root(
                self.repository_root,
                self.repository_root / case.patch_path,
                "replay patch",
            )
            if _sha256(_read_regular(patch, "replay patch")) != case.patch_hash:
                raise ContractError("replay patch bytes changed after inventory freeze")
        if case.case_kind == "legacy":
            results = self.repository_root / case.provenance_root / "results.jsonl"
            raw_lines = _read_regular(results, "replay provenance").splitlines()
            if case.provenance_line > len(raw_lines):
                raise ContractError("replay provenance line is missing")
            raw = raw_lines[case.provenance_line - 1]
        else:
            raw = _read_regular(
                self.repository_root / case.provenance_root,
                "replay provenance",
            )
        if _sha256(raw) != case.provenance_hash:
            raise ContractError("replay provenance changed after inventory freeze")

    def _validate_output_root(self, replay_root: Path) -> None:
        for relative_root, _ in LEGACY_ROOTS:
            historical = (self.repository_root / relative_root).resolve()
            try:
                replay_root.relative_to(historical)
            except ValueError:
                continue
            raise ContractError("output_root: cannot write under historical v0.5 run")


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_json_lines(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(canonical_json(value) + "\n" for value in values),
        encoding="utf-8",
    )


def build_replay_inventory(
    repository_root: Path,
    *,
    dataset_path: Path | None = None,
) -> tuple[ReplayCase, ...]:
    root = _real_directory(repository_root, "repository_root")
    selected_dataset = (
        root / "datasets" / "pytorch_v0.5" / "dataset.json"
        if dataset_path is None
        else _within_root(root, dataset_path, "dataset_path")
    )
    _regular_file(selected_dataset, "dataset_path")
    dataset = DatasetManifest.load(selected_dataset)
    entries = dataset.tasks
    if len(entries) != 17:
        raise ContractError("dataset: expected exactly 17 tasks")
    if any(
        entry.admission_status != "verified"
        or entry.environment_status != "ready"
        or entry.source_status != "ready"
        or entry.replay_status != "verified"
        for entry in entries
    ):
        raise ContractError("dataset: every replay task must be verified and ready")
    resolved_tasks = {
        task.task_id: task for task in dataset.load_tasks(verified_only=True)
    }
    if len(resolved_tasks) != 17:
        raise ContractError("dataset: duplicate or unresolved verified task")

    cases: list[ReplayCase] = []
    task_ids: set[str] = set()
    task_metadata: dict[str, tuple[object, str]] = {}
    for entry in entries:
        if entry.task_id in task_ids:
            raise ContractError(f"dataset: duplicate task_id {entry.task_id!r}")
        task_ids.add(entry.task_id)
        task_dir = _within_root(root, entry.task_path, "task_path")
        _real_directory(task_dir, "task_path")
        task = resolved_tasks.get(entry.task_id)
        if task is None or task.task_id != entry.task_id:
            raise ContractError("dataset: task identity mismatch")
        spec = full_task_spec_from_v05(task)
        task_relative = task_dir.relative_to(root).as_posix()
        evidence_path = entry.admission_evidence_path
        if evidence_path is None:
            raise ContractError("dataset: admission evidence is required")
        evidence_path = _within_root(root, evidence_path, "admission_evidence")
        evidence_raw = _read_regular(evidence_path, "admission_evidence")
        evidence = _json_object(evidence_raw, "admission_evidence")
        if (
            evidence.get("task_id") != entry.task_id
            or evidence.get("admission", {}).get("verified") is not True
            or evidence.get("baseline", {}).get("status") != "baseline_reproduced"
            or evidence.get("gold", {}).get("status") != "resolved"
        ):
            raise ContractError("admission evidence: baseline/gold verification mismatch")
        provenance_root = evidence_path.relative_to(root).as_posix()
        provenance_hash = _sha256(evidence_raw)
        common = {
            "task_id": entry.task_id,
            "task_path": task_relative,
            "task_verified": True,
            "source_id": spec.source.identifier,
            "environment_id": spec.environment.identifier,
            "runtime_profile_id": spec.runtime.profile_id,
            "runtime_profile_hash": spec.runtime.content_hash,
        }
        baseline = _case(
            case_kind="baseline",
            patch_path=None,
            patch_hash=_EMPTY_PATCH_HASH,
            expected_outcome="f2p_failed",
            attempt_number=None,
            provenance_root=provenance_root,
            provenance_line=0,
            provenance_hash=provenance_hash,
            **common,
        )
        gold_path = _within_root(root, task.gold_patch_path, "gold_patch")
        gold_raw = _read_regular(gold_path, "gold_patch")
        gold = _case(
            case_kind="gold",
            patch_path=gold_path.relative_to(root).as_posix(),
            patch_hash=_sha256(gold_raw),
            expected_outcome="resolved",
            attempt_number=None,
            provenance_root=provenance_root,
            provenance_line=0,
            provenance_hash=provenance_hash,
            **common,
        )
        cases.extend((baseline, gold))
        task_metadata[entry.task_id] = (spec, task_relative)

    for relative_root, expected_count in LEGACY_ROOTS:
        run_root = _within_root(root, root / relative_root, "legacy_root")
        patch_root = _real_directory(run_root / "patches", "legacy patch root")
        results_path = _regular_file(run_root / "results.jsonl", "legacy results")
        result_rows = _legacy_result_rows(root, results_path)
        patch_paths = _patch_files(patch_root)
        if len(patch_paths) != expected_count:
            raise ContractError(
                f"{relative_root}: expected {expected_count} legacy patches"
            )
        seen_attempts: set[tuple[str, int]] = set()
        for patch_path in patch_paths:
            relative_patch = patch_path.relative_to(root).as_posix()
            match = _PATCH_NAME.fullmatch(patch_path.name)
            attempt_match = _ATTEMPT_DIRECTORY.fullmatch(patch_path.parent.name)
            if match is None or attempt_match is None:
                raise ContractError("legacy patch: noncanonical task/attempt path")
            task_id = match.group("task")
            attempt = int(attempt_match.group("number"))
            key = (task_id, attempt)
            if key in seen_attempts:
                raise ContractError("legacy patch: duplicate task attempt")
            seen_attempts.add(key)
            if task_id not in task_metadata:
                raise ContractError("legacy patch: task is not in verified dataset")
            rows = result_rows.get(relative_patch)
            if not rows:
                raise ContractError("legacy patch: no matching historical result row")
            line_number, row, row_hash = rows[-1]
            if row.get("task_id") != task_id:
                raise ContractError("legacy result: task does not match patch")
            status = row.get("status")
            expected_outcome = _STATUS_OUTCOMES.get(status)
            if expected_outcome is None:
                raise ContractError(
                    f"legacy result: unsupported final status {status!r}"
                )
            spec, task_relative = task_metadata[task_id]
            patch_raw = _read_regular(patch_path, "legacy patch")
            cases.append(
                _case(
                    case_kind="legacy",
                    task_id=task_id,
                    task_path=task_relative,
                    task_verified=True,
                    source_id=spec.source.identifier,
                    environment_id=spec.environment.identifier,
                    runtime_profile_id=spec.runtime.profile_id,
                    runtime_profile_hash=spec.runtime.content_hash,
                    patch_path=relative_patch,
                    patch_hash=_sha256(patch_raw),
                    expected_outcome=expected_outcome,
                    attempt_number=attempt,
                    provenance_root=relative_root,
                    provenance_line=line_number,
                    provenance_hash=row_hash,
                )
            )
    ordered = tuple(sorted(cases, key=lambda item: item.sort_key))
    if len(ordered) != 85 or len({item.replay_id for item in ordered}) != 85:
        raise ContractError("replay inventory: expected 85 unique cases")
    return ordered


def _case(**values) -> ReplayCase:
    identity = {
        key: value
        for key, value in values.items()
        if key not in {"task_verified"}
    }
    replay_id = "replay:v1:" + canonical_sha256(
        {"identity_type": "legacy_replay_case", **identity}
    ).removeprefix("sha256:")
    return ReplayCase(replay_id=replay_id, **values)


def _legacy_result_rows(
    root: Path,
    path: Path,
) -> dict[str, list[tuple[int, dict[str, object], str]]]:
    raw = _read_regular(path, "legacy results")
    rows: dict[str, list[tuple[int, dict[str, object], str]]] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        row = _json_object(line, f"legacy results line {line_number}")
        patch_path = row.get("patch_path")
        if not isinstance(patch_path, str) or not patch_path:
            continue
        normalized = _published_patch_path(root, patch_path)
        rows.setdefault(normalized, []).append(
            (line_number, row, _sha256(line))
        )
    return rows


def _published_patch_path(root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        candidate = _within_root(root, root / path, "historical patch_path")
        return candidate.relative_to(root).as_posix()
    parts = path.parts
    try:
        index = parts.index("runs")
    except ValueError as exc:
        raise ContractError("historical patch_path: expected repository run path") from exc
    suffix = Path(*parts[index:])
    candidate = _within_root(root, root / suffix, "historical patch_path")
    return candidate.relative_to(root).as_posix()


def _patch_files(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            if (base / name).is_symlink():
                raise ContractError("legacy patch root: symlink directory is denied")
        for name in files:
            path = base / name
            if path.is_symlink():
                raise ContractError("legacy patch root: symlink file is denied")
            if path.suffix != ".patch":
                continue
            _regular_file(path, "legacy patch")
            paths.append(path)
    return tuple(sorted(paths))


def _real_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise ContractError(f"{label}: expected Path")
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label}: expected real directory")
    return path.resolve()


def _within_root(root: Path, path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_symlink():
        raise ContractError(f"{label}: symlink is denied")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label}: path escapes repository root") from exc
    return resolved


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ContractError(f"{label}: symlink is denied")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ContractError(f"{label}: missing file") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label}: expected regular file")
    return path


def _read_regular(path: Path, label: str) -> bytes:
    _regular_file(path, label)
    return path.read_bytes()


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label}: expected object")
    return value


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


__all__ = [
    "LEGACY_ROOTS",
    "ReplayCase",
    "ReplayDifference",
    "ReplayResult",
    "ReplayRunReport",
    "ReplayRunner",
    "ReplaySummary",
    "build_replay_inventory",
]
