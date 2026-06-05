from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from op_bench.evaluator import EvaluationResult, Evaluator
from op_bench.progress import Progress, noop_progress
from op_bench.task import TaskManifest


@dataclass(frozen=True)
class AdmissionEvidence:
    schema_version: str
    evidence_id: str
    task_id: str
    task_manifest_hash: str
    created_at: str
    source: dict[str, object]
    environment: dict[str, object]
    baseline: dict[str, object]
    gold: dict[str, object] | None
    decision: str
    verified: bool
    failure_classification: str | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["admission"] = {
            "decision": data.pop("decision"),
            "verified": data.pop("verified"),
            "failure_classification": data.pop("failure_classification"),
        }
        return data

    def summary_dict(self) -> dict[str, object]:
        data = self.to_dict()
        source = dict(data["source"])
        source.pop("snapshot_path", None)
        data["source"] = source
        environment = dict(data["environment"])
        environment.pop("observed", None)
        data["environment"] = environment
        data["baseline"] = self._evaluation_summary(self.baseline)
        data["gold"] = self._evaluation_summary(self.gold) if self.gold is not None else None
        return data

    def _evaluation_summary(self, evaluation: dict[str, object]) -> dict[str, object]:
        fields = (
            "task_id",
            "mode",
            "status",
            "fail_to_pass_total",
            "fail_to_pass_passed",
            "pass_to_pass_total",
            "pass_to_pass_passed",
            "duration_sec",
        )
        return {field: evaluation[field] for field in fields if field in evaluation}


class AdmissionRunner:
    def __init__(
        self,
        evaluator: Evaluator | None = None,
        now: Callable[[], datetime] | None = None,
        progress: Progress | None = None,
    ) -> None:
        self.evaluator = evaluator or Evaluator()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.progress = progress or noop_progress

    def run(self, task: TaskManifest) -> AdmissionEvidence:
        created_at = self._format_timestamp(self.now())
        task_manifest_hash = self._manifest_hash(task)
        self.progress(f"admission baseline start: task={task.task_id}")
        baseline = self.evaluator.evaluate_baseline(task)
        self.progress(f"admission baseline done: task={task.task_id}, status={baseline.status}")

        decision, failure = self._baseline_decision(baseline.status)
        gold: EvaluationResult | None = None
        if decision == "continue":
            self.progress(f"admission gold start: task={task.task_id}")
            gold = self.evaluator.evaluate_gold(task)
            self.progress(f"admission gold done: task={task.task_id}, status={gold.status}")
            decision, failure = self._gold_decision(gold.status)

        source = {
            "id": task.source_ref,
            "repo_url": task.repo_url,
            "base_commit": task.base_commit,
            "snapshot_path": str(task.source_snapshot_path) if task.source_snapshot_path else None,
            "snapshot_hash": task.source_snapshot_hash,
            "snapshot_method": task.source_snapshot_method,
        }
        environment = {
            "id": task.environment_ref,
            "runtime_tier": task.runtime_tier,
            "backend": task.environment_backend,
            "image": task.environment_image,
            "image_digest": task.environment_image_digest,
            "digest_kind": task.environment_digest_kind,
            "platform": task.environment_platform,
            "observed": baseline.environment,
        }
        evidence_id = f"{task.task_id}:{task_manifest_hash.removeprefix('sha256:')[:12]}:{created_at}"
        return AdmissionEvidence(
            schema_version="v1",
            evidence_id=evidence_id,
            task_id=task.task_id,
            task_manifest_hash=task_manifest_hash,
            created_at=created_at,
            source=source,
            environment=environment,
            baseline=baseline.to_dict(),
            gold=gold.to_dict() if gold is not None else None,
            decision=decision,
            verified=decision == "verified",
            failure_classification=failure,
        )

    def write_bundle(self, evidence: AdmissionEvidence, output_dir: Path | str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._write_json(output / "evidence.json", evidence.to_dict())
        self._write_json(output / "environment.json", evidence.environment)
        self._write_json(output / "source.json", evidence.source)
        self._write_json(output / "baseline.log", evidence.baseline)
        if evidence.gold is not None:
            self._write_json(output / "gold.log", evidence.gold)

    def write_task_evidence(self, task: TaskManifest, evidence: AdmissionEvidence) -> Path:
        output = task.task_dir / "admission" / "evidence.json"
        self._write_json(output, evidence.summary_dict())
        return output

    def _manifest_hash(self, task: TaskManifest) -> str:
        digest = hashlib.sha256(task.task_json_path.read_bytes()).hexdigest()
        return f"sha256:{digest}"

    def _baseline_decision(self, status: str) -> tuple[str, str | None]:
        if status == "baseline_reproduced":
            return "continue", None
        if status in {"environment_unavailable", "environment_error"}:
            return "blocked_environment", status
        if status in {"runner_error", "setup_failed", "patch_apply_failed", "timeout"}:
            return "blocked_test", status
        return "not_reproduced", status

    def _gold_decision(self, status: str) -> tuple[str, str | None]:
        if status == "resolved":
            return "verified", None
        if status in {"environment_unavailable", "environment_error"}:
            return "blocked_environment", status
        return "gold_failed", status

    def _format_timestamp(self, value: datetime) -> str:
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")

    def _write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
