from __future__ import annotations

from collections.abc import Mapping

from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.contracts import ActionRequest
from op_bench.runtime.validation import ContractError


class CanonicalMcpTransport:
    """Minimal MCP tool dispatcher that delegates every call to one action service."""

    def __init__(self, service: CanonicalActionService) -> None:
        if not isinstance(service, CanonicalActionService):
            raise ContractError("service: expected CanonicalActionService")
        self.service = service

    def call_tool(self, action_name: str, envelope: object) -> dict[str, object]:
        action_id = "invalid"
        try:
            if not isinstance(envelope, Mapping):
                raise ContractError("MCP envelope: expected object")
            candidate_action_id = envelope.get("action_id")
            if isinstance(candidate_action_id, str) and candidate_action_id:
                action_id = candidate_action_id
            payload = {
                "contract_type": "action_request",
                "schema_version": "v1",
                "session_id": envelope.get("session_id"),
                "action_id": envelope.get("action_id"),
                "action_name": action_name,
                "arguments": envelope.get("arguments"),
                "client_sequence": envelope.get("client_sequence"),
                "deadline_ms": envelope.get("deadline_ms"),
            }
            request = ActionRequest.from_dict(payload)
        except ContractError as exc:
            return self.service.invalid_request_observation(
                action_id=action_id,
                message=str(exc),
            ).to_dict()
        return self.service.execute(request).to_dict()


__all__ = ["CanonicalMcpTransport"]
