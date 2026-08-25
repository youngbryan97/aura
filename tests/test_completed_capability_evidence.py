from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.response_generation import ResponseGenerationPhase
from core.utils.completed_capability import (
    any_capability_completed,
    completed_capabilities,
    make_completed_capability_evidence,
    remaining_capabilities,
)


def test_successful_same_process_receipt_removes_only_completed_work() -> None:
    receipt = make_completed_capability_evidence(
        ["web_search", "search_web"],
        ok=True,
        receipt_id="search-1",
    )

    assert completed_capabilities(receipt) == {"web_search", "search_web"}
    assert remaining_capabilities(["web_search", "code_repl"], receipt) == [
        "code_repl"
    ]
    assert any_capability_completed(receipt, {"grounded_search", "search_web"})
    assert remaining_capabilities("web_search", receipt) == []


def test_unstamped_failed_and_malformed_receipts_complete_nothing() -> None:
    unstamped = {
        "schema": "aura.completed_capability_evidence.v1",
        "ok": True,
        "completed_capabilities": ["web_search"],
    }
    failed = make_completed_capability_evidence(["web_search"], ok=False)
    malformed = make_completed_capability_evidence(["web_search"], ok=True)
    malformed["completed_capabilities"] = "web_search"

    for receipt in (unstamped, failed, malformed):
        assert completed_capabilities(receipt) == frozenset()
        assert remaining_capabilities(["web_search"], receipt) == ["web_search"]


@pytest.mark.asyncio
async def test_response_phase_reuses_completed_search_without_running_a_skill() -> None:
    class _Capability:
        async def execute(self, *_args, **_kwargs):
            raise AssertionError("completed search must not execute again")

    class _Container:
        def get(self, name, default=None):
            return _Capability() if name == "capability_engine" else default

    state = SimpleNamespace(response_modifiers={})
    receipt = make_completed_capability_evidence(["web_search"], ok=True)
    phase = ResponseGenerationPhase(_Container())

    ran = await phase._execute_required_search_evidence(
        state=state,
        objective=(
            "search for the release\n\n[WEB SEARCH EVIDENCE]\n"
            "source: https://example.test/release"
        ),
        contract=SimpleNamespace(tool_evidence_available=False, requires_search=True),
        origin="user",
        runtime_context={
            "visible_user_message": "search for the release",
            "completed_capability_evidence": receipt,
        },
    )

    assert ran is False
    assert state.response_modifiers["required_search_evidence_reused"] is True


def test_every_response_generation_path_carries_completion_evidence() -> None:
    import inspect

    source = inspect.getsource(ResponseGenerationPhase)
    assert source.count(
        'completed_capability_evidence=runtime_context.get('
    ) >= 3
    assert 'kw.setdefault(\n                            "completed_capability_evidence"' in source
