"""Resume support for `run_experiment.py`.

Three primitives:

- `RunState`: writes `<output>/run_state.json` on start, validates CLI args
  when resuming. Prevents accidental cross-parameter continuation (e.g.
  restarting with `--agent-repeat 5` after a `--agent-repeat 3` batch).

- `ResultsStore`: append-mode writer for the unified `results.jsonl` event log.
  Every record is fsynced. Reader tolerates truncated/corrupt trailing lines.
  Exposes `completed_agents()` and `completed_baseline_task_ids()` for
  skip-lookup on resume.

- `BaselineCache`: shared cache under `runs/_baseline_cache/`. Key is a hash
  of `(task_id, source_snapshot_hash, hidden_test.patch content sha256)`. A
  hit means the identical baseline has been proven before, so `evaluate_baseline`
  can be skipped.

Idempotency key for the main experiment loop is
`(task_id, agent, attempt)` — attempt-level, not phase-level. If an attempt
dies mid-flight (before its record is fsynced), resume re-runs the whole
attempt. This is intentional: `median(45s)` per attempt, worst case
`~90min` for `cuda_kernel_build` — the additional complexity of phase-level
checkpoints is not worth the wall-clock savings.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


RUN_STATE_FILE = "run_state.json"
RESULTS_FILE = "results.jsonl"
BASELINE_CACHE_DIRNAME = "_baseline_cache"

TRANSIENT_STATUSES = {
    "environment_unavailable",
    "environment_error",
}


@dataclass(frozen=True)
class RunState:
    """Manifest of a run's CLI parameters. Persisted at `<output>/run_state.json`.

    On resume, the previous file is loaded and diffed against the current invocation.
    Non-critical fields (progress verbosity, output timestamps) are ignored; anything
    that would change what work gets scheduled must match.
    """

    dataset_signature: str
    task_ids: tuple[str, ...]
    agents: tuple[str, ...]
    agent_repeat: int
    only_tasks: tuple[str, ...] = field(default=())

    @classmethod
    def build(
        cls,
        task_ids: Iterable[str],
        agents: Iterable[str],
        agent_repeat: int,
        only_tasks: Iterable[str] = (),
        task_signatures: Iterable[str] | None = None,
    ) -> "RunState":
        ids = tuple(sorted(task_ids))
        signatures = tuple(sorted(task_signatures or ids))
        signature = hashlib.sha256("\n".join(signatures).encode("utf-8")).hexdigest()[:16]
        return cls(
            dataset_signature=signature,
            task_ids=ids,
            agents=tuple(agents),
            agent_repeat=int(agent_repeat),
            only_tasks=tuple(only_tasks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_signature": self.dataset_signature,
            "task_ids": list(self.task_ids),
            "agents": list(self.agents),
            "agent_repeat": self.agent_repeat,
            "only_tasks": list(self.only_tasks),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        _atomic_write_text(path, payload)

    @classmethod
    def load(cls, path: Path) -> "RunState | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return cls(
            dataset_signature=str(data.get("dataset_signature", "")),
            task_ids=tuple(data.get("task_ids", [])),
            agents=tuple(data.get("agents", [])),
            agent_repeat=int(data.get("agent_repeat", 0)),
            only_tasks=tuple(data.get("only_tasks", [])),
        )

    def is_compatible(self, other: "RunState") -> tuple[bool, str]:
        """Return (ok, reason). Extra tasks in-progress-run are fine (they'll just be
        pending). But changing agent_repeat, agents, or dataset_signature invalidates
        prior results."""
        if self.dataset_signature != other.dataset_signature:
            return False, (
                f"dataset changed (prev signature={other.dataset_signature}, "
                f"current={self.dataset_signature}); resume unsafe. "
                f"Use --fresh to start over."
            )
        if self.agent_repeat != other.agent_repeat:
            return False, (
                f"agent_repeat changed ({other.agent_repeat} -> {self.agent_repeat}); "
                f"resume unsafe. Use --fresh to start over."
            )
        if set(self.agents) != set(other.agents):
            return False, (
                f"agent list changed ({sorted(other.agents)} -> {sorted(self.agents)}); "
                f"resume unsafe. Use --fresh to start over."
            )
        return True, ""


class ResultsStore:
    """Append-mode writer for the unified `results.jsonl` event log.

    Every write is followed by `flush() + os.fsync()` on the file descriptor.
    A subsequent `kill -9` still preserves the most recent record.

    Reader tolerates malformed trailing lines (partial writes on crash) by
    skipping them silently.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        # Single authoritative sink. Baseline records (agent="baseline") and
        # agent records live together in insertion order, matching v0.3/v0.4
        # layout so downstream analysis tools keep working.
        self.results_path = output_dir / RESULTS_FILE

    def append_result(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.results_path, record)

    def append_baseline(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.results_path, record)

    def completed_agent_keys(self) -> set[tuple[str, str, int]]:
        """Return the set of (task_id, agent, attempt) triples already recorded.

        Transient infrastructure statuses are deliberately excluded so a resumed run
        retries them. The raw failed row remains in results.jsonl for auditability;
        summary readers use the latest row for each idempotency key.
        """
        keys: set[tuple[str, str, int]] = set()
        for record in _read_jsonl(self.results_path):
            if record.get("agent") == "baseline":
                continue
            task_id = record.get("task_id")
            agent = record.get("agent")
            attempt = record.get("attempt")
            status = str(record.get("status", ""))
            if task_id and agent and attempt is not None and status not in TRANSIENT_STATUSES:
                keys.add((str(task_id), str(agent), int(attempt)))
        return keys

    def completed_baseline_task_ids(self) -> dict[str, dict[str, Any]]:
        """Return baseline records already computed, keyed by task_id.

        On duplicates the latest record wins (would only happen if resume was
        forced and baseline re-run; not a real scenario in practice).
        """
        out: dict[str, dict[str, Any]] = {}
        for record in _read_jsonl(self.results_path):
            if record.get("agent") != "baseline":
                continue
            task_id = record.get("task_id")
            status = str(record.get("status", ""))
            if task_id and status not in TRANSIENT_STATUSES:
                out[str(task_id)] = record
        return out

    def load_all_records(self) -> list[dict[str, Any]]:
        """Latest logical rows in results.jsonl (baseline + agent).

        A transient record can be followed by a successful retry with the same
        idempotency key. Keep the raw JSONL append-only, but score only the latest
        row for each baseline task or agent attempt.
        """
        latest: dict[tuple[object, ...], dict[str, Any]] = {}
        unkeyed: list[dict[str, Any]] = []
        for record in _read_jsonl(self.results_path):
            task_id = record.get("task_id")
            agent = record.get("agent")
            if agent == "baseline" and task_id:
                latest[("baseline", str(task_id))] = record
                continue
            attempt = record.get("attempt")
            if task_id and agent and attempt is not None:
                latest[("agent", str(task_id), str(agent), int(attempt))] = record
                continue
            unkeyed.append(record)
        return [*latest.values(), *unkeyed]

    def load_all_results(self) -> list[dict[str, Any]]:
        """Agent-only records (excludes baseline rows)."""
        return [r for r in self.load_all_records() if r.get("agent") != "baseline"]

    def load_all_baselines(self) -> list[dict[str, Any]]:
        """Baseline-only records."""
        return [r for r in self.load_all_records() if r.get("agent") == "baseline"]


class BaselineCache:
    """Shared baseline cache under `<repo>/runs/_baseline_cache/`.

    A baseline result only depends on the source snapshot and the hidden test patch;
    it is independent of agent and attempt. Caching it across runs saves the baseline
    docker/rsync/build cycle on every re-run.

    Cache key: sha256 of `task_id + snapshot_hash + hidden_test.patch content sha`.
    If any component changes, the key changes and the cache is bypassed.

    The cache is opt-in: callers pass a cache directory (typically
    `<repo>/runs/_baseline_cache/`). Missing directory → cache disabled.
    """

    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, task_id: str, snapshot_hash: str | None, hidden_test_patch_path: Path) -> str | None:
        if self.cache_dir is None:
            return None
        if not hidden_test_patch_path.exists():
            patch_sha = "empty"
        else:
            content = hidden_test_patch_path.read_bytes()
            patch_sha = hashlib.sha256(content).hexdigest()[:16] if content.strip() else "empty"
        snapshot_part = snapshot_hash or "no-snapshot"
        material = f"{task_id}\n{snapshot_part}\n{patch_sha}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str | None) -> dict[str, Any] | None:
        if key is None or self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str | None, record: dict[str, Any]) -> None:
        if key is None or self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(path, payload)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            # Some filesystems (tmpfs) don't support fsync; ignore.
            pass


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                sys.stderr.write(f"[resume] skipping malformed jsonl line in {path}\n")
                continue


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
