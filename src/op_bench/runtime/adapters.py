from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import threading
from typing import Protocol
import uuid

from op_bench.runtime.task_view import AgentLaunchInput
from op_bench.runtime.contracts import ActionRequest
from op_bench.runtime.validation import ContractError, require_str


class ActionClient(Protocol):
    def execute(self, payload: object) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class _ChannelCall:
    call_id: str
    payload_json: str
    responses: queue.Queue[object]


@dataclass(frozen=True)
class _ChannelFailure:
    pass


@dataclass(frozen=True)
class ScriptedAdapterResult:
    terminal_reason: str
    observation_count: int
    finish_count: int


class ScriptedCanonicalAdapter:
    """A deterministic no-edit Adapter used for offline controller smoke runs."""

    def run(self, context: "AdapterContext") -> ScriptedAdapterResult:
        if not isinstance(context, AdapterContext):
            raise ContractError("context: expected AdapterContext")
        task_view = context.launch_input.task_view
        allowed = set(task_view.capability_policy.allowed_actions)
        actions: list[tuple[str, dict[str, object]]] = []
        if "workspace_list" in allowed:
            actions.append(("workspace_list", {"path": ".", "recursive": False}))
        if "test_run" in allowed:
            registered = set(task_view.capability_policy.registered_tests)
            selector = next(
                (
                    item.selector_id
                    for item in task_view.public_tests
                    if item.selector_id in registered
                ),
                None,
            )
            if selector is not None:
                actions.append(("test_run", {"selector_id": selector}))
        if "vcs_diff" in allowed:
            actions.append(("vcs_diff", {}))
        if "session_finish" not in allowed:
            raise ContractError("scripted Adapter requires session_finish capability")
        actions.append(("session_finish", {}))

        finish_count = 0
        for sequence, (action_name, arguments) in enumerate(actions, start=1):
            request = ActionRequest(
                session_id=context.session_id,
                action_id=f"scripted-{sequence:04d}",
                action_name=action_name,
                arguments=arguments,
                client_sequence=sequence,
                deadline_ms=task_view.budget_policy.wall_clock_ms,
            )
            observation = context.action_client.execute(request.to_dict())
            if not isinstance(observation, dict):
                raise ContractError("scripted Adapter received invalid observation")
            if action_name == "session_finish":
                finish_count += 1
        return ScriptedAdapterResult(
            terminal_reason="agent_finished",
            observation_count=len(actions),
            finish_count=finish_count,
        )


class AdapterActionClient:
    """Serialized queue endpoint with no callable or control-plane object reference."""

    __slots__ = ("_requests", "_timeout_sec")

    def __init__(
        self,
        requests: queue.Queue[object],
        *,
        timeout_sec: float,
    ) -> None:
        if not isinstance(requests, queue.Queue):
            raise ContractError("requests: expected Queue")
        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
            raise ContractError("timeout_sec: expected number")
        if timeout_sec <= 0:
            raise ContractError("timeout_sec: must be positive")
        self._requests = requests
        self._timeout_sec = float(timeout_sec)

    def execute(self, payload: object) -> dict[str, object]:
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            raise ContractError("adapter action payload is not JSON-safe") from None
        responses: queue.Queue[object] = queue.Queue(maxsize=1)
        call = _ChannelCall(
            call_id=uuid.uuid4().hex,
            payload_json=payload_json,
            responses=responses,
        )
        self._requests.put(call, timeout=self._timeout_sec)
        try:
            result = responses.get(timeout=self._timeout_sec)
        except queue.Empty as exc:
            raise TimeoutError("adapter action channel timed out") from exc
        if isinstance(result, _ChannelFailure):
            raise ContractError("adapter action channel failed") from None
        if not isinstance(result, str):
            raise ContractError("action client returned non-object response")
        try:
            decoded = json.loads(result)
        except (TypeError, ValueError):
            raise ContractError("action client returned invalid JSON response") from None
        if not isinstance(decoded, dict):
            raise ContractError("action client returned non-object response")
        return decoded


class AdapterActionChannel:
    """Control-plane API boundary; only its queue-only client reaches the Adapter.

    This in-process reference channel proves data minimization and serialization. It
    is not a security sandbox for adversarial Python code; untrusted adapters must
    use a process/IPC transport that preserves the same JSON-only client contract.
    """

    def __init__(self, execute_action: object, *, timeout_sec: float = 30.0) -> None:
        if not callable(execute_action):
            raise ContractError("execute_action: expected callable")
        self._execute_action = execute_action
        self._requests: queue.Queue[object] = queue.Queue()
        self._timeout_sec = timeout_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client = AdapterActionClient(
            self._requests,
            timeout_sec=timeout_sec,
        )

    def start(self) -> AdapterActionClient:
        if self._thread is not None:
            return self._client
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self._client

    def close(self) -> None:
        self._stop.set()
        self._requests.put(None)
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_sec)
            self._thread = None

    def __enter__(self) -> AdapterActionClient:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                continue
            if not isinstance(item, _ChannelCall):
                continue
            try:
                payload = json.loads(item.payload_json)
                response = self._execute_action(payload)
                if not isinstance(response, dict):
                    raise ContractError("control-plane action response is not an object")
                result = json.dumps(
                    response,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except BaseException:  # noqa: BLE001 - converted to a fixed boundary sentinel.
                result = _ChannelFailure()
            item.responses.put(result)


@dataclass(frozen=True)
class AdapterContext:
    """Complete control-plane input available to a standard Agent Adapter."""

    launch_input: AgentLaunchInput
    session_id: str
    action_client: AdapterActionClient

    def __post_init__(self) -> None:
        if not isinstance(self.launch_input, AgentLaunchInput):
            raise ContractError("launch_input: expected AgentLaunchInput")
        AgentLaunchInput(
            task_view=self.launch_input.task_view,
            task_view_identity=self.launch_input.task_view_identity,
        )
        require_str(self.session_id, "session_id")
        if not isinstance(self.action_client, AdapterActionClient):
            raise ContractError("action_client: expected AdapterActionClient")


class CanonicalAgentAdapter(Protocol):
    def run(self, context: AdapterContext) -> object:
        ...


__all__ = [
    "ActionClient",
    "AdapterActionChannel",
    "AdapterActionClient",
    "AdapterContext",
    "CanonicalAgentAdapter",
    "ScriptedAdapterResult",
    "ScriptedCanonicalAdapter",
]
