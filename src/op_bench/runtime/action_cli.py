from __future__ import annotations

from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.contracts import ActionRequest
from op_bench.runtime.validation import ContractError


class ActionCliTransport:
    """JSON-object CLI transport over the canonical in-process service."""

    def __init__(self, service: CanonicalActionService) -> None:
        if not isinstance(service, CanonicalActionService):
            raise ContractError("service: expected CanonicalActionService")
        self.service = service

    def execute(self, payload: object) -> dict[str, object]:
        try:
            request = ActionRequest.from_dict(payload)
        except ContractError as exc:
            action_id = payload.get("action_id", "invalid") if isinstance(payload, dict) else "invalid"
            return self.service.invalid_request_observation(
                action_id=action_id if isinstance(action_id, str) else "invalid",
                message=str(exc),
            ).to_dict()
        return self.service.execute(request).to_dict()

    def adapter_channel(self):
        from op_bench.runtime.adapters import AdapterActionChannel

        return AdapterActionChannel(self.execute)


__all__ = ["ActionCliTransport"]
