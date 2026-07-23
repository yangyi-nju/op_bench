from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import EventRecord, IntegrityReport
from op_bench.runtime.integrity import load_run_manifest_artifact, verify_run_artifacts
from op_bench.runtime.mcp import McpAdapterTrace
from op_bench.runtime.resume import parse_attempt_ledger
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import ContractError, require_str
from op_bench.runtime.workspace import _patch_paths_from_bytes


_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_REPORT_FILES = (
    "experiment_index.json",
    "experiment_summary.json",
    "experiment_report.md",
)


@dataclass(frozen=True)
class McpExperimentCohortContract:
    profile_id: str
    task_repeats: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        require_str(self.profile_id, "profile_id")
        if not isinstance(self.task_repeats, tuple) or not self.task_repeats:
            raise ContractError("task_repeats: expected non-empty tuple")
        task_ids: list[str] = []
        for task_id, repeats in self.task_repeats:
            selected_task = require_str(task_id, "task_id")
            if not isinstance(repeats, tuple) or not repeats:
                raise ContractError("task repeats: expected non-empty tuple")
            if tuple(sorted(set(repeats))) != repeats or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in repeats
            ):
                raise ContractError("task repeats: expected sorted positive unique integers")
            task_ids.append(selected_task)
        if tuple(sorted(set(task_ids))) != tuple(sorted(task_ids)):
            raise ContractError("task_repeats: duplicate task ID")

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for task_id, _ in self.task_repeats)

    @property
    def expected_attempts(self) -> frozenset[tuple[str, int]]:
        return frozenset(
            (task_id, repeat)
            for task_id, repeats in self.task_repeats
            for repeat in repeats
        )


@dataclass(frozen=True)
class McpExperimentContract:
    dataset_identifier: str
    dataset_digest: str
    platform_version: str
    cohorts: tuple[McpExperimentCohortContract, ...]

    def __post_init__(self) -> None:
        require_str(self.dataset_identifier, "dataset_identifier")
        require_str(
            self.dataset_digest,
            "dataset_digest",
            pattern=r"sha256:[0-9a-f]{64}",
        )
        require_str(self.platform_version, "platform_version")
        if not isinstance(self.cohorts, tuple) or len(self.cohorts) != 4:
            raise ContractError("cohorts: expected exactly four cohort contracts")
        if not all(isinstance(item, McpExperimentCohortContract) for item in self.cohorts):
            raise ContractError("cohorts: expected MCP cohort contracts")
        all_tasks = [task_id for cohort in self.cohorts for task_id in cohort.task_ids]
        if len(all_tasks) != len(set(all_tasks)):
            raise ContractError("cohorts: task IDs must be partitioned exactly once")

    @property
    def expected_attempt_count(self) -> int:
        return sum(len(item.expected_attempts) for item in self.cohorts)


FORMAL_MCP_EXPERIMENT_CONTRACT = McpExperimentContract(
    dataset_identifier="pytorch_v0.5",
    dataset_digest="sha256:ff9d0c2999d1175a45165b387e0731dcaa211a190d994b176441ce81a0382abc",
    platform_version="opbench-v0.6.0",
    cohorts=(
        McpExperimentCohortContract(
            profile_id="remote-cpu-pytorch-2.6-py311-v1",
            task_repeats=tuple(
                (task_id, (1, 2, 3))
                for task_id in (
                    "pytorch__149693__lazylinear_init",
                    "pytorch__147599__lazylinear_state_forward",
                    "pytorch__160952__bilinear_lazy_check",
                    "pytorch__162340__nn_arg_length",
                    "pytorch__163961__dataloader_subset",
                    "pytorch__168295__autograd_create_graph",
                    "pytorch__161488__lbfgs_wolfe",
                    "pytorch__150975__autograd_backward_inputs",
                    "pytorch__124385__load_state_dict_prefix",
                    "pytorch__143455__set_submodule",
                    "pytorch__140557__layer_norm_decomp_precision",
                    "pytorch__139999__masked_mean_bool_upcast",
                )
            ),
        ),
        McpExperimentCohortContract(
            profile_id="remote-cpu-compile-pytorch-2.6-py311-v1",
            task_repeats=(("pytorch__129138__linear_add_bias_autocast", (1, 2, 3)),),
        ),
        McpExperimentCohortContract(
            profile_id="remote-cuda-overlay-pytorch-2.6-cu124-v1",
            task_repeats=(
                ("pytorch__132835__njt_sdpa_autocast", (1, 2, 3)),
                ("pytorch__132616__cuda_mem_get_info", (1, 2, 3)),
            ),
        ),
        McpExperimentCohortContract(
            profile_id="remote-cuda-kernel-pytorch-2.6-cu124-v1",
            task_repeats=(
                ("pytorch__144009__softmax_ilpreduce_size", (1, 2, 3)),
                ("pytorch__139372__histc_int8_cuda_bounds", (1, 2, 3)),
            ),
        ),
    ),
)


class _RunReader:
    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ContractError("run_root: expected Path")
        if root.is_symlink() or not root.is_dir():
            raise ContractError("run_root: expected real directory")
        self.root = root
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        try:
            self.descriptor = os.open(root, flags)
        except OSError as exc:
            raise ContractError("run_root: expected real directory") from exc
        metadata = os.fstat(self.descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(self.descriptor)
            raise ContractError("run_root: expected real directory")
        self.identity = metadata.st_dev, metadata.st_ino

    def close(self) -> None:
        descriptor = getattr(self, "descriptor", None)
        if descriptor is not None:
            os.close(descriptor)
            self.descriptor = None

    def read(self, *components: str, optional: bool = False) -> bytes | None:
        if self.descriptor is None:
            raise ContractError("run reader is closed")
        if not components:
            raise ContractError("artifact path: expected components")
        for component in components:
            if (
                not isinstance(component, str)
                or not component
                or component in {".", ".."}
                or "/" in component
                or "\\" in component
            ):
                raise ContractError("artifact path: invalid component")
        root_metadata = os.fstat(self.descriptor)
        if (root_metadata.st_dev, root_metadata.st_ino) != self.identity:
            raise ContractError("run_root: directory binding changed")
        opened: list[int] = []
        current = self.descriptor
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        try:
            for component in components[:-1]:
                try:
                    selected = os.open(component, directory_flags, dir_fd=current)
                except OSError as exc:
                    if optional and isinstance(exc, FileNotFoundError):
                        return None
                    raise ContractError("run artifact directory is missing or invalid") from exc
                opened.append(selected)
                current = selected
            try:
                descriptor = os.open(
                    components[-1],
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
            except OSError as exc:
                if optional and isinstance(exc, FileNotFoundError):
                    return None
                raise ContractError("run artifact is missing or invalid") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContractError("run artifact is not a regular file")
                if metadata.st_size > _MAX_ARTIFACT_BYTES:
                    raise ContractError("run artifact exceeds size limit")
                chunks: list[bytes] = []
                remaining = _MAX_ARTIFACT_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(65_536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                if len(raw) > _MAX_ARTIFACT_BYTES:
                    raise ContractError("run artifact exceeds size limit")
                return raw
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)


def _canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ContractError(f"{label}: invalid JSON") from None
    if not isinstance(value, dict):
        raise ContractError(f"{label}: expected object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise ContractError(f"{label}: expected canonical JSON")
    return value


def _canonical_lines(raw: bytes, label: str) -> list[dict[str, object]]:
    if raw and not raw.endswith(b"\n"):
        raise ContractError(f"{label}: missing final newline")
    result: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ContractError(f"{label} line {index}: invalid JSON") from None
        if not isinstance(value, dict):
            raise ContractError(f"{label} line {index}: expected object")
        if line != canonical_json(value).encode("utf-8"):
            raise ContractError(f"{label} line {index}: expected canonical JSON")
        result.append(value)
    return result


def _event_records(raw: bytes) -> tuple[EventRecord, ...]:
    return tuple(
        EventRecord.from_dict(value, path=f"events[{index}]")
        for index, value in enumerate(_canonical_lines(raw, "events.jsonl"))
    )


def _distribution(values: Sequence[int]) -> dict[str, object]:
    selected = tuple(values)
    if not selected:
        return {
            "count": 0,
            "sum": 0,
            "min": None,
            "max": None,
            "mean": {"numerator": 0, "denominator": 0},
        }
    return {
        "count": len(selected),
        "sum": sum(selected),
        "min": min(selected),
        "max": max(selected),
        "mean": {"numerator": sum(selected), "denominator": len(selected)},
    }


def build_mcp_experiment_report(
    run_roots: Sequence[Path],
    *,
    expected_adapter_id: str,
    expected_model_id: str,
    expected_codex_cli_version: str,
    experiment_contract: McpExperimentContract,
) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(run_roots, (str, bytes)) or not isinstance(run_roots, Sequence):
        raise ContractError("run_roots: expected a sequence of four Paths")
    roots = tuple(run_roots)
    if len(roots) != 4:
        raise ContractError("run_roots: expected exactly four roots")
    adapter_id = require_str(expected_adapter_id, "expected_adapter_id")
    model_id = require_str(expected_model_id, "expected_model_id")
    cli_version = require_str(
        expected_codex_cli_version,
        "expected_codex_cli_version",
    )
    if adapter_id != "codex_mcp_canonical":
        raise ContractError("expected_adapter_id: expected codex_mcp_canonical")
    if not isinstance(experiment_contract, McpExperimentContract):
        raise ContractError("experiment_contract: expected McpExperimentContract")

    cohort_rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []
    seen_attempts: set[str] = set()
    seen_comparability: set[str] = set()
    selected_counts: list[int] = []
    outcomes: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    action_calls: Counter[str] = Counter()
    action_errors: Counter[str] = Counter()
    mcp_terminals: Counter[str] = Counter()
    budgets = Counter()
    coverage = Counter()
    durations: list[int] = []
    patch_sizes: list[int] = []
    changed_files: list[int] = []
    retry_counts: list[int] = []
    trace_complete = 0
    provider_attribution = 0
    mcp_attribution = 0
    runtime_attribution = 0
    mcp_initialize = 0
    mcp_list = 0
    mcp_calls = 0
    mcp_protocol_errors = 0
    matched_contracts: set[int] = set()

    for root in roots:
        if not isinstance(root, Path):
            raise ContractError("run_roots: every root must be a Path")
        manifest = load_run_manifest_artifact(root)
        if (
            manifest.dataset.identifier != experiment_contract.dataset_identifier
            or manifest.dataset.digest != experiment_contract.dataset_digest
        ):
            raise ContractError("dataset identity does not match experiment contract")
        if manifest.platform_version != experiment_contract.platform_version:
            raise ContractError("platform identity does not match experiment contract")
        profile_ids = tuple(profile.profile_id for profile in manifest.runtime_profiles)
        if len(profile_ids) != 1:
            raise ContractError("each experiment root must use exactly one Runtime Profile")
        task_ids = frozenset(task.task.identifier for task in manifest.tasks)
        candidates = [
            (index, cohort)
            for index, cohort in enumerate(experiment_contract.cohorts)
            if cohort.profile_id == profile_ids[0]
            and frozenset(cohort.task_ids) == task_ids
        ]
        if len(candidates) != 1:
            raise ContractError("cohort profile/task partition does not match contract")
        contract_index, cohort_contract = candidates[0]
        if contract_index in matched_contracts:
            raise ContractError("duplicate experiment cohort contract")
        matched_contracts.add(contract_index)
        observed_attempts = frozenset(
            (item.task.identifier, item.repeat)
            for item in manifest.expected_attempts
        )
        if observed_attempts != cohort_contract.expected_attempts:
            raise ContractError("cohort repeat matrix does not match contract")
        fresh_integrity = verify_run_artifacts(root, manifest)
        if fresh_integrity.status != "passed" or any(
            check.status != "passed" for check in fresh_integrity.checks
        ):
            raise ContractError("Integrity verification failed")
        if manifest.comparability_key in seen_comparability:
            raise ContractError("duplicate Comparability Key")
        seen_comparability.add(manifest.comparability_key)
        for agent in manifest.agents:
            if agent.adapter.identifier != adapter_id:
                raise ContractError("adapter identity mismatch")
            if agent.model.identifier != model_id:
                raise ContractError("model identity mismatch")

        reader = _RunReader(root)
        try:
            persisted_integrity = IntegrityReport.from_dict(
                _canonical_object(
                    reader.read("integrity.json"),
                    "integrity.json",
                )
            )
            if persisted_integrity != fresh_integrity:
                raise ContractError("persisted Integrity report does not match verification")
            results = _canonical_lines(
                reader.read("results.jsonl"),
                "results.jsonl",
            )
            ledger_records = parse_attempt_ledger(reader.read("attempts.jsonl"))
            histories: dict[str, list[object]] = {}
            for record in ledger_records:
                histories.setdefault(record.attempt_id, []).append(record)
            expected_by_id = {
                item.attempt_id: item for item in manifest.expected_attempts
            }
            tasks = {item.task.identifier: item for item in manifest.tasks}
            result_by_id: dict[str, dict[str, object]] = {}
            for result in results:
                attempt_id = require_str(result.get("attempt_id"), "result.attempt_id")
                if attempt_id in result_by_id:
                    raise ContractError("duplicate selected result Attempt ID")
                result_by_id[attempt_id] = result
            if set(result_by_id) != set(expected_by_id):
                raise ContractError("run root has missing or blocked selected Attempts")

            selected_counts.append(len(results))
            cohort_rows.append(
                {
                    "cohort_id": manifest.cohort_id,
                    "comparability_key": manifest.comparability_key,
                    "runtime_profile_ids": [
                        profile.profile_id for profile in manifest.runtime_profiles
                    ],
                    "selected_attempts": len(results),
                }
            )
            for attempt_id in sorted(result_by_id):
                if attempt_id in seen_attempts:
                    raise ContractError("duplicate Attempt ID across run roots")
                seen_attempts.add(attempt_id)
                expected = expected_by_id[attempt_id]
                result = result_by_id[attempt_id]
                history = histories.get(attempt_id, [])
                valid = [item for item in history if item.attempt_validity == "valid"]
                if len(valid) != 1:
                    raise ContractError(
                        "selected Attempt is missing or infrastructure-invalid"
                    )
                selected = valid[0]
                if result.get("retry_index") != selected.retry_index:
                    raise ContractError("selected retry attribution mismatch")
                if result.get("attempt_validity") != "valid":
                    raise ContractError("selected Attempt is infrastructure-invalid")
                if result.get("evaluation_result_hash") != selected.evaluation_result_hash:
                    raise ContractError("selected Evaluation result binding mismatch")

                retry_name = f"retry-{selected.retry_index:04d}"
                prefix = (
                    "attempts",
                    attempt_id,
                    "retries",
                    retry_name,
                )
                trace = McpAdapterTrace.from_dict(
                    _canonical_object(
                        reader.read(*prefix, "adapter_trace.json"),
                        "adapter_trace.json",
                    )
                )
                if trace.adapter_id != adapter_id:
                    raise ContractError("adapter trace identity mismatch")
                if trace.model_id != model_id:
                    raise ContractError("model trace identity mismatch")
                if trace.codex_cli_version != cli_version:
                    raise ContractError("Codex CLI version identity mismatch")
                events = _event_records(reader.read(*prefix, "events.jsonl"))
                requests = [
                    record
                    for record in events
                    if record.event_type == "action_requested"
                ]
                observations = [
                    record
                    for record in events
                    if record.event_type == "action_observed"
                ]
                if trace.tools_call_count != len(requests):
                    raise ContractError("MCP trace pairing is incomplete")
                request_ids = {
                    record.to_dict()["public_payload"]["action_id"]
                    for record in requests
                }
                observation_ids = {
                    record.to_dict()["public_payload"]["action_id"]
                    for record in observations
                }
                if request_ids != observation_ids:
                    raise ContractError("Action request/observation pairing is incomplete")
                trace_complete += 1
                mcp_initialize += trace.initialize_count
                mcp_list += trace.tools_list_count
                mcp_calls += trace.tools_call_count
                mcp_protocol_errors += trace.protocol_error_count
                mcp_terminals[trace.server_terminal_status] += 1

                action_names: set[str] = set()
                for record in requests:
                    payload = record.to_dict()["public_payload"]
                    name = require_str(payload.get("action_name"), "action_name")
                    action_calls[name] += 1
                    action_names.add(name)
                for record in observations:
                    payload = record.to_dict()["public_payload"]
                    if payload.get("ok") is not True:
                        action_errors[
                            require_str(payload.get("error_code"), "error_code")
                        ] += 1
                    delta = payload.get("budget_delta")
                    if not isinstance(delta, dict):
                        raise ContractError("Action Budget delta is missing")
                    for name in (
                        "wall_clock_ms",
                        "actions",
                        "tests",
                        "commands",
                        "output_bytes",
                        "provider_tokens",
                    ):
                        value = delta.get(name)
                        if isinstance(value, bool) or not isinstance(value, int):
                            raise ContractError("Action Budget delta is invalid")
                        budgets[name] += value
                categories = {
                    "read": {"workspace_list", "workspace_search", "workspace_read"},
                    "edit": {"workspace_write", "workspace_apply_patch"},
                    "test": {"test_run"},
                    "diff": {"vcs_diff"},
                    "finish": {"session_finish"},
                }
                for category, names in categories.items():
                    if action_names & names:
                        coverage[category] += 1

                patch_bytes = reader.read(*prefix, "final.patch")
                paths = _patch_paths_from_bytes(patch_bytes)
                patch_size = len(patch_bytes)
                changed_count = len(paths)
                patch_sizes.append(patch_size)
                changed_files.append(changed_count)
                retry_count = selected.retry_index - 1
                retry_counts.append(retry_count)
                outcome = require_str(result.get("evaluation_outcome"), "evaluation_outcome")
                terminal = require_str(result.get("agent_terminal"), "agent_terminal")
                duration = result.get("duration_ms")
                if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                    raise ContractError("duration_ms: expected non-negative integer")
                outcomes[outcome] += 1
                terminals[terminal] += 1
                durations.append(duration)

                for prior in history:
                    if prior.retry_index >= selected.retry_index:
                        continue
                    if prior.attempt_validity != "infrastructure_invalid":
                        raise ContractError("retry history has invalid attribution")
                    if prior.session_result.terminal_reason == "provider_error":
                        provider_attribution += 1
                        continue
                    prior_trace_raw = reader.read(
                        "attempts",
                        attempt_id,
                        "retries",
                        f"retry-{prior.retry_index:04d}",
                        "adapter_trace.json",
                        optional=True,
                    )
                    prior_trace = (
                        None
                        if prior_trace_raw is None
                        else McpAdapterTrace.from_dict(
                            _canonical_object(prior_trace_raw, "adapter_trace.json")
                        )
                    )
                    if (
                        prior.session_result.terminal_reason == "runtime_error"
                        and prior_trace is not None
                        and prior_trace.server_terminal_status == "protocol_failed"
                    ):
                        mcp_attribution += 1
                    else:
                        runtime_attribution += 1

                task = tasks[expected.task.identifier]
                attempt_rows.append(
                    {
                        "cohort_id": manifest.cohort_id,
                        "comparability_key": manifest.comparability_key,
                        "runtime_profile_id": task.runtime.profile_id,
                        "task_id": expected.task.identifier,
                        "repeat": expected.repeat,
                        "attempt_id": attempt_id,
                        "retry_index": selected.retry_index,
                        "evaluation_result_hash": selected.evaluation_result_hash,
                        "terminal_reason": selected.session_result.terminal_reason,
                        "agent_terminal": terminal,
                        "evaluation_outcome": outcome,
                        "duration_ms": duration,
                        "patch_size_bytes": patch_size,
                        "changed_file_count": changed_count,
                        "action_count": len(requests),
                        "action_error_count": sum(
                            record.to_dict()["public_payload"].get("ok") is not True
                            for record in observations
                        ),
                        "trace_complete": True,
                    }
                )
        finally:
            reader.close()

    expected_counts = tuple(
        sorted(len(item.expected_attempts) for item in experiment_contract.cohorts)
    )
    if tuple(sorted(selected_counts)) != expected_counts:
        raise ContractError("four cohort selected counts do not match contract")
    if matched_contracts != set(range(len(experiment_contract.cohorts))):
        raise ContractError("experiment cohort contract is incomplete")
    attempt_rows.sort(
        key=lambda item: (
            item["cohort_id"],
            item["task_id"],
            item["repeat"],
            item["attempt_id"],
        )
    )
    cohort_rows.sort(key=lambda item: item["cohort_id"])
    total = len(attempt_rows)
    if total != experiment_contract.expected_attempt_count:
        raise ContractError("four cohort report has the wrong selected Attempt count")
    identities = {
        "adapter_id": adapter_id,
        "model_id": model_id,
        "codex_cli_version": cli_version,
        "dataset_identifier": experiment_contract.dataset_identifier,
        "dataset_digest": experiment_contract.dataset_digest,
        "platform_version": experiment_contract.platform_version,
    }
    index: dict[str, object] = {
        "report_type": "mcp_experiment_index",
        "schema_version": "v1",
        **identities,
        "cohorts": cohort_rows,
        "attempts": attempt_rows,
    }
    summary: dict[str, object] = {
        "report_type": "mcp_experiment_summary",
        "schema_version": "v1",
        **identities,
        "totals": {
            "cohorts": len(cohort_rows),
            "attempts": total,
            "trace_complete": trace_complete,
            "retries": sum(retry_counts),
            "actions": sum(action_calls.values()),
            "action_errors": sum(action_errors.values()),
        },
        "evaluation_outcomes": dict(sorted(outcomes.items())),
        "agent_terminals": dict(sorted(terminals.items())),
        "action_calls": dict(sorted(action_calls.items())),
        "action_errors": dict(sorted(action_errors.items())),
        "budget_totals": {
            key: budgets[key]
            for key in (
                "wall_clock_ms",
                "actions",
                "tests",
                "commands",
                "output_bytes",
                "provider_tokens",
            )
        },
        "action_coverage": {
            key: {"numerator": coverage[key], "denominator": total}
            for key in ("read", "edit", "test", "diff", "finish")
        },
        "duration_ms": _distribution(durations),
        "patch_size_bytes": _distribution(patch_sizes),
        "changed_file_count": _distribution(changed_files),
        "retry_count": _distribution(retry_counts),
        "attribution": {
            "provider": provider_attribution,
            "mcp": mcp_attribution,
            "runtime": runtime_attribution,
            "agent": total,
        },
        "mcp": {
            "initialize_count": mcp_initialize,
            "tools_list_count": mcp_list,
            "tools_call_count": mcp_calls,
            "protocol_error_count": mcp_protocol_errors,
            "server_terminals": dict(sorted(mcp_terminals.items())),
        },
    }
    assert_public_artifact_safe(index)
    assert_public_artifact_safe(summary)
    return index, summary


def render_mcp_experiment_markdown(
    index: dict[str, object],
    summary: dict[str, object],
) -> str:
    totals = summary["totals"]
    lines = [
        "# OpBench v0.6 MCP Agent Experiment",
        "",
        f"- Adapter: `{summary['adapter_id']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Codex CLI: `{summary['codex_cli_version']}`",
        f"- Cohorts: {totals['cohorts']}",
        f"- Selected Attempts: {totals['attempts']}",
        f"- Complete MCP traces: {totals['trace_complete']}",
        f"- Retries: {totals['retries']}",
        "",
        "## Evaluation outcomes",
        "",
    ]
    for name, count in summary["evaluation_outcomes"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(("", "## Cohorts", ""))
    for cohort in index["cohorts"]:
        profiles = ", ".join(f"`{item}`" for item in cohort["runtime_profile_ids"])
        lines.append(
            f"- `{cohort['cohort_id']}`: {cohort['selected_attempts']} Attempts; "
            f"profiles {profiles}"
        )
    return "\n".join(lines) + "\n"


def write_mcp_experiment_report(
    output_dir: Path,
    index: dict[str, object],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    if not isinstance(output_dir, Path):
        raise ContractError("output_dir: expected Path")
    if output_dir.is_symlink():
        raise ContractError("output_dir: symlink is denied")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ContractError("output_dir: expected real directory")
    payloads = {
        "experiment_index.json": (canonical_json(index) + "\n").encode("utf-8"),
        "experiment_summary.json": (canonical_json(summary) + "\n").encode("utf-8"),
        "experiment_report.md": render_mcp_experiment_markdown(index, summary).encode(
            "utf-8"
        ),
    }
    for filename, encoded in payloads.items():
        path = output_dir / filename
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise ContractError("output_dir contains a nonmatching report")
    for filename, encoded in payloads.items():
        path = output_dir / filename
        if path.exists():
            continue
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        try:
            view = memoryview(encoded)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return tuple(output_dir / filename for filename in _REPORT_FILES)


__all__ = [
    "FORMAL_MCP_EXPERIMENT_CONTRACT",
    "McpExperimentCohortContract",
    "McpExperimentContract",
    "build_mcp_experiment_report",
    "render_mcp_experiment_markdown",
    "write_mcp_experiment_report",
]
