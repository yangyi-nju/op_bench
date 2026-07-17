from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading

from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.contracts import (
    AGENT_TERMINALS,
    ATTEMPT_VALIDITIES,
    TERMINAL_REASONS,
    ActionObservation,
    ActionRequest,
    SessionResult,
    SessionSpec,
)
from op_bench.runtime.events import EventJournal
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_enum,
    require_int,
)
from op_bench.runtime.workspace import FrozenPatch


SESSION_STATES = (
    "created",
    "preparing",
    "ready",
    "running",
    "stopping",
    "freezing",
    "terminal",
)

TERMINATION_PRIORITY = (
    "platform_error",
    "workspace_error",
    "runtime_error",
    "provider_error",
    "cancelled",
    "timeout",
    "budget_exhausted",
    "agent_finished",
    "agent_exited",
)

_INFRASTRUCTURE_REASONS = frozenset(
    {"workspace_error", "runtime_error", "provider_error", "platform_error"}
)

_AGENT_TERMINALS = {
    "agent_finished": "finished",
    "agent_exited": "exited",
    "budget_exhausted": "budget",
    "timeout": "timeout",
    "cancelled": "cancelled",
}

_STOP_EVENTS = {
    "agent_finished": "finish_requested",
    "budget_exhausted": "budget_exhausted",
    "timeout": "timeout_requested",
    "cancelled": "cancel_requested",
}


class SessionStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminationAttribution:
    attempt_validity: str
    agent_terminal: str | None
    scorable: bool

    def __post_init__(self) -> None:
        require_enum(self.attempt_validity, "attempt_validity", ATTEMPT_VALIDITIES)
        if self.agent_terminal is not None:
            require_enum(self.agent_terminal, "agent_terminal", AGENT_TERMINALS)
        require_bool(self.scorable, "scorable")
        if self.attempt_validity == "infrastructure_invalid":
            if self.agent_terminal is not None or self.scorable:
                raise ContractError("infrastructure attribution cannot be scorable")
        elif self.agent_terminal is None or not self.scorable:
            raise ContractError("valid attribution requires a scorable Agent terminal")


def termination_attribution(reason: str) -> TerminationAttribution:
    selected = require_enum(reason, "terminal_reason", TERMINAL_REASONS)
    if selected in _INFRASTRUCTURE_REASONS:
        return TerminationAttribution(
            attempt_validity="infrastructure_invalid",
            agent_terminal=None,
            scorable=False,
        )
    return TerminationAttribution(
        attempt_validity="valid",
        agent_terminal=_AGENT_TERMINALS[selected],
        scorable=True,
    )


class AttemptSession:
    """Deterministic lifecycle and terminal authority for one v0.6 Attempt."""

    def __init__(
        self,
        *,
        spec: SessionSpec,
        action_service: CanonicalActionService,
        journal: EventJournal,
        freeze_patch: Callable[[], FrozenPatch],
        clock_ms: Callable[[], int],
    ) -> None:
        if not isinstance(spec, SessionSpec):
            raise ContractError("spec: expected SessionSpec")
        if not isinstance(action_service, CanonicalActionService):
            raise ContractError("action_service: expected CanonicalActionService")
        if not isinstance(journal, EventJournal):
            raise ContractError("journal: expected EventJournal")
        if not callable(freeze_patch):
            raise ContractError("freeze_patch: expected callable")
        if not callable(clock_ms):
            raise ContractError("clock_ms: expected callable")
        if spec.session_id != action_service.session_id:
            raise ContractError("action_service: session does not match spec")
        if journal.session_id != spec.session_id:
            raise ContractError("journal: session does not match spec")
        if action_service.event_journal is not journal:
            raise ContractError("journal: does not match action service journal")
        if spec.workspace != action_service.workspace_identity:
            raise ContractError("action_service: workspace does not match spec")
        if spec.capability_policy != action_service.capability_policy:
            raise ContractError("action_service: capability policy does not match spec")
        if spec.budget_policy != action_service.budget_policy:
            raise ContractError("action_service: budget policy does not match spec")
        if journal.records:
            raise ContractError("journal: expected empty journal for a new session")

        self.spec = spec
        self.action_service = action_service
        self._journal = journal
        self._freeze_patch = freeze_patch
        self._clock_ms = clock_ms
        self._condition = threading.Condition(threading.RLock())
        self._state = "created"
        self._created_at_ms = self._now()
        self._started_at_ms = self._created_at_ms
        self._agent_launched = False
        self._agent_exited = False
        self._stop_candidates: set[str] = set()
        self._sealed = False
        self._active_actions = 0
        self._pending_publishers = 0
        self._waiting_stop_requests = 0
        self._finalizing = False
        self._result: SessionResult | None = None
        self._freeze_started_emitted = False
        self._freeze_invoked = False
        self._freeze_outcome_emitted = False
        self._frozen_patch: FrozenPatch | None = None
        self._freeze_failed = False
        self._journal.append(
            "session_created",
            {
                "attempt_id": spec.attempt_id,
                "workspace": spec.workspace.to_dict(),
            },
        )

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def events(self):
        return self._journal.records

    def prepare(self) -> str:
        with self._condition:
            if self._state == "preparing":
                return self._state
            self._require_state("created", "prepare")
            self._state = "preparing"
            self._pending_publishers += 1
        self._publish_reserved([("session_prepared", {})])
        return "preparing"

    def mark_ready(self) -> str:
        with self._condition:
            self._wait_for_publishers_locked()
            if self._state == "ready":
                return self._state
            self._require_state("preparing", "mark ready")
            self._state = "ready"
            return self._state

    def start(self) -> str:
        with self._condition:
            self._wait_for_publishers_locked()
            if self._state == "running":
                return self._state
            self._require_state("ready", "start")
            self._state = "running"
            self._started_at_ms = self._now()
            self._pending_publishers += 1
        self._publish_reserved([("session_started", {})])
        return "running"

    def mark_agent_launched(self) -> str:
        with self._condition:
            self._wait_for_publishers_locked()
            self._require_state("running", "launch Agent")
            if self._agent_launched:
                return self._state
            self._agent_launched = True
            self._pending_publishers += 1
        self._publish_reserved([("agent_launched", {})])
        return "running"

    def mark_agent_exited(self, exit_code: int) -> str:
        require_int(exit_code, "exit_code")
        with self._condition:
            self._waiting_stop_requests += 1
            try:
                self._wait_for_publishers_locked(allow_after_seal=False)
                if self._state == "terminal":
                    return self._state
                if self._state not in {"running", "stopping"}:
                    raise SessionStateError(
                        f"state {self._state} cannot record Agent exit; expected running or stopping"
                    )
                if self._sealed:
                    return self._state
                first = not self._agent_exited
                self._agent_exited = True
                self._stop_candidates.add("agent_exited")
                self._state = "stopping"
                publications: list[tuple[str, dict[str, object]]] = []
                if first:
                    self._pending_publishers += 1
                    publications.append(("agent_exited", {"exit_code": exit_code}))
            finally:
                self._waiting_stop_requests -= 1
                self._condition.notify_all()
        self._publish_reserved(publications)
        return "stopping"

    def execute_action(self, request: ActionRequest) -> ActionObservation:
        if not isinstance(request, ActionRequest):
            raise ContractError("request: expected ActionRequest")
        with self._condition:
            while (
                self._pending_publishers
                and self._state == "running"
                and not self._sealed
            ):
                self._condition.wait()
            time_reasons, resource_exhausted = self._limit_reasons_locked()
            blocking_reasons = list(time_reasons)
            if request.action_name != "session_finish" and resource_exhausted:
                blocking_reasons.append("budget_exhausted")
            publications = self._admit_limit_reasons_locked(blocking_reasons)
            finish_after_budget = (
                request.action_name == "session_finish"
                and self._state == "stopping"
                and not self._sealed
                and self._stop_candidates <= {"budget_exhausted"}
            )
            if publications or (
                self._state != "running" and not finish_after_budget
            ):
                admitted = False
            else:
                self._active_actions += 1
                admitted = True
        self._publish_reserved(publications)
        if not admitted:
            if "timeout" in blocking_reasons:
                raise SessionStateError("session deadline has expired")
            if "budget_exhausted" in blocking_reasons:
                raise SessionStateError("session budget is exhausted")
            raise SessionStateError(
                f"state {self.state} cannot execute Action; running state is required"
            )

        try:
            observation = self.action_service.execute(request)
            if request.action_name == "session_finish" and observation.ok:
                self._request_stop(
                    "agent_finished", {}, emit_event=False, admitted=True
                )
            elif observation.error_code == "budget_exhausted":
                self._request_stop(
                    "budget_exhausted", {}, emit_event=False, admitted=True
                )
            elif observation.error_code == "timeout":
                self._request_stop("timeout", {}, emit_event=True, admitted=True)
            elif observation.error_code in {
                "workspace_error",
                "runtime_error",
                "platform_error",
            }:
                self._request_stop(
                    observation.error_code, {}, emit_event=False, admitted=True
                )
            self._poll_limits(admitted=True)
            return observation
        finally:
            with self._condition:
                self._active_actions -= 1
                self._condition.notify_all()

    def request_stop(
        self,
        reason: str,
        public_details: Mapping[str, object] | None = None,
    ) -> str:
        details = {} if public_details is None else dict(public_details)
        return self._request_stop(reason, details, emit_event=True)

    def _request_stop(
        self,
        reason: str,
        details: Mapping[str, object],
        *,
        emit_event: bool,
        admitted: bool = False,
    ) -> str:
        selected = require_enum(reason, "terminal_reason", TERMINAL_REASONS)
        with self._condition:
            self._waiting_stop_requests += 1
            try:
                self._wait_for_publishers_locked(allow_after_seal=admitted)
                if self._state == "terminal":
                    return self._state
                if self._sealed and not admitted:
                    return self._state
                first = selected not in self._stop_candidates
                self._stop_candidates.add(selected)
                if self._state != "stopping":
                    self._state = "stopping"
                current = self._state
                event_type = _STOP_EVENTS.get(selected)
                publications: list[tuple[str, dict[str, object]]] = []
                if first and emit_event and event_type is not None:
                    self._pending_publishers += 1
                    publications.append(
                        (event_type, {"reason": selected, **dict(details)})
                    )
            finally:
                self._waiting_stop_requests -= 1
                self._condition.notify_all()
        self._publish_reserved(publications)
        return current

    def poll_limits(self) -> str:
        return self._poll_limits(admitted=False)

    def _poll_limits(self, *, admitted: bool) -> str:
        with self._condition:
            self._waiting_stop_requests += 1
            try:
                self._wait_for_publishers_locked(allow_after_seal=admitted)
                if self._state not in {"running", "stopping", "terminal"}:
                    raise SessionStateError(
                        f"state {self._state} cannot poll limits; expected running"
                    )
                if self._state == "terminal" or (self._sealed and not admitted):
                    return self._state
                time_reasons, resource_exhausted = self._limit_reasons_locked()
                reasons = list(time_reasons)
                if resource_exhausted:
                    reasons.append("budget_exhausted")
                publications = self._admit_limit_reasons_locked(reasons)
                current = self._state
            finally:
                self._waiting_stop_requests -= 1
                self._condition.notify_all()
        self._publish_reserved(publications)
        return current

    def finalize(self) -> SessionResult:
        with self._condition:
            if self._result is not None:
                return self._result
            while self._finalizing:
                self._condition.wait()
                if self._result is not None:
                    return self._result
            if self._state != "stopping" or not self._stop_candidates:
                raise SessionStateError(
                    f"state {self._state} cannot finalize; a stop reason is required"
                )
            while self._waiting_stop_requests:
                self._condition.wait()
            self._sealed = True
            self._finalizing = True
            while self._active_actions or self._pending_publishers:
                self._condition.wait()
            self._state = "freezing"
        try:
            if not self._freeze_started_emitted:
                self._append_lifecycle("patch_freeze_started", {})
                self._freeze_started_emitted = True
            if not self._freeze_invoked:
                try:
                    candidate = self._freeze_patch()
                    if not isinstance(candidate, FrozenPatch):
                        raise ContractError("freeze_patch: expected FrozenPatch")
                    if candidate.workspace != self.spec.workspace:
                        raise ContractError(
                            "freeze_patch: workspace does not match session"
                        )
                    self._frozen_patch = candidate
                except Exception:  # noqa: BLE001 - fixed terminal attribution boundary.
                    self._freeze_failed = True
                    with self._condition:
                        self._stop_candidates.add("workspace_error")
                self._freeze_invoked = True
            if not self._freeze_outcome_emitted:
                if self._freeze_failed or self._frozen_patch is None:
                    self._append_lifecycle(
                        "patch_freeze_failed", {"error_code": "workspace_error"}
                    )
                else:
                    self._append_lifecycle(
                        "patch_freeze_completed",
                        {
                            "patch": self._frozen_patch.patch.to_dict(),
                            "empty": self._frozen_patch.empty,
                        },
                    )
                self._freeze_outcome_emitted = True

            with self._condition:
                reason = next(
                    item for item in TERMINATION_PRIORITY if item in self._stop_candidates
                )
                result = SessionResult(
                    session_id=self.spec.session_id,
                    attempt_id=self.spec.attempt_id,
                    terminal_reason=reason,
                    final_patch=(
                        None
                        if self._freeze_failed or self._frozen_patch is None
                        else self._frozen_patch.patch
                    ),
                    started_at_ms=self._started_at_ms,
                    ended_at_ms=self._now(),
                )
            self._append_lifecycle(
                "terminal_emitted",
                {
                    "attempt_id": self.spec.attempt_id,
                    "terminal_reason": reason,
                    "session_result_hash": result.content_hash,
                    "final_patch": (
                        None
                        if result.final_patch is None
                        else result.final_patch.to_dict()
                    ),
                    "attempt_validity": termination_attribution(reason).attempt_validity,
                },
            )
            with self._condition:
                self._result = result
                self._state = "terminal"
                return result
        finally:
            with self._condition:
                self._finalizing = False
                if self._result is None:
                    self._state = "stopping"
                self._condition.notify_all()

    def _append_lifecycle(
        self,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        try:
            self._journal.append(event_type, payload)
        except Exception:  # noqa: BLE001 - journal failure is platform invalidity.
            with self._condition:
                if self._state != "terminal":
                    self._stop_candidates.add("platform_error")
                    if self._state != "freezing":
                        self._state = "stopping"
            raise SessionStateError("session trajectory persistence failed") from None

    def _limit_reasons_locked(self) -> tuple[tuple[str, ...], bool]:
        now = self._now()
        time_reasons: list[str] = []
        if now >= self.spec.deadline_ms:
            time_reasons.append("timeout")
        if now - self._started_at_ms >= self.spec.budget_policy.wall_clock_ms:
            time_reasons.append("budget_exhausted")
        usage = self.action_service.usage
        policy = self.spec.budget_policy
        resource_exhausted = any(
            used > 0 and used >= limit
            for used, limit in (
                (usage.actions, policy.max_actions),
                (usage.tests, policy.max_tests),
                (usage.commands, policy.max_commands),
                (usage.output_bytes, policy.max_output_bytes),
            )
        )
        return tuple(time_reasons), resource_exhausted

    def _admit_limit_reasons_locked(
        self,
        reasons: list[str],
    ) -> list[tuple[str, dict[str, object]]]:
        publications: list[tuple[str, dict[str, object]]] = []
        for reason in reasons:
            if reason in self._stop_candidates:
                continue
            self._stop_candidates.add(reason)
            self._state = "stopping"
            event_type = _STOP_EVENTS[reason]
            self._pending_publishers += 1
            publications.append((event_type, {"reason": reason}))
        return publications

    def _publish_reserved(
        self,
        publications: list[tuple[str, dict[str, object]]],
    ) -> None:
        first_error: SessionStateError | None = None
        for event_type, payload in publications:
            try:
                self._append_lifecycle(event_type, payload)
            except SessionStateError as exc:
                if first_error is None:
                    first_error = exc
            finally:
                with self._condition:
                    self._pending_publishers -= 1
                    self._condition.notify_all()
        if first_error is not None:
            raise first_error

    def _wait_for_publishers_locked(self, *, allow_after_seal: bool = False) -> None:
        while (
            self._pending_publishers
            and self._state != "terminal"
            and (not self._sealed or allow_after_seal)
        ):
            self._condition.wait()

    def _require_state(self, expected: str, operation: str) -> None:
        if self._state != expected:
            raise SessionStateError(
                f"state {self._state} cannot {operation}; expected {expected}"
            )

    def _now(self) -> int:
        return require_int(self._clock_ms(), "clock_ms", minimum=0)


__all__ = [
    "AttemptSession",
    "SESSION_STATES",
    "SessionStateError",
    "TERMINATION_PRIORITY",
    "TerminationAttribution",
    "termination_attribution",
]
