"""The tool path gives EDI what governance rests on, never a claim of it.

EDI's ``can_do`` resolves governance from the live scope or a signed
capability bound to the action and its payload, and records a caller that
claims governance with neither. The tool path used to pass
``governed=<a live runtime is up>``; a running runtime is not a governance
decision, so every tool call on a live runtime was logged as a false claim
(swarm_debate, 6 October).
"""

from __future__ import annotations

import asyncio

from core.executive.execution_policy import canonical_authority_arguments


def test_edi_gets_the_context_and_payload_and_no_claim(monkeypatch) -> None:
    from core.orchestrator.mixins.tool_execution import ToolExecutionMixin

    class Host(ToolExecutionMixin):
        _current_objective = ""

    calls: list[dict] = []

    class Edi:
        def can_do(self, action, risk_level="low", **kwargs):
            calls.append({"action": action, **kwargs})
            return False, "held for the test"

    def get_service(name, default=None):
        return Edi() if name == "edi" else default

    monkeypatch.setattr("core.orchestrator.mixins.tool_execution.ServiceContainer.get", get_service)
    monkeypatch.setattr("core.orchestrator.mixins.tool_execution.ServiceContainer.has", lambda name: False)

    args = {"url": "https://example.com"}
    out = asyncio.run(Host().execute_tool("browser", dict(args), origin="desktop"))

    assert out == {"ok": False, "error": "EDI blocked: held for the test"}
    assert len(calls) == 1
    call = calls[0]
    assert "governed" not in call
    assert isinstance(call["governance_context"], dict)
    assert call["governance_context"]["origin"] == "desktop"
    assert call["governance_payload"] == canonical_authority_arguments("browser", args)
