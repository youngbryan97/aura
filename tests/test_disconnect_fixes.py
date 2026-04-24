"""Tests for the four residual disconnect fixes.

1. Background enqueue gate admits trusted internal volition sources
   instead of silently dropping them.
2. macOS permission helper returns a structured report and opens the
   right Settings pane (without actually launching on non-macOS).
3. Neural-intent router converts action-schema matches into real skill
   dispatches via the Will, records LifeTrace, and blocks hallucinated
   or low-trust attempts.
4. VolitionEngine produces real initiatives from drive state when the
   idle cooldowns are met.
"""
from __future__ import annotations

import asyncio
import platform
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from core.agency.neural_intent_router import (
    NeuralIntentRouter,
    classify_neural_intent,
    reset_singleton_for_test as reset_router,
)
from core.security.permission_setup import (
    PermissionReport,
    PermissionStatus,
    check_all_permissions,
    format_report,
    open_settings_pane,
)


# ---------------------------------------------------------------------------
# 1. Background enqueue gate
# ---------------------------------------------------------------------------


def test_trusted_internal_origin_admitted_when_safe():
    """The fix: sensory_motor volition must not be silently dropped when
    authorization fails for infrastructure reasons.
    """
    from core.orchestrator.mixins.message_handling import MessageHandlingMixin

    class _Stub(MessageHandlingMixin):
        def __init__(self) -> None:
            self.state_repo = None

        def _background_enqueue_summary(self, message: Any, origin: str) -> str:
            return f"{origin}: {str(message)[:60]}"

    stub = _Stub()

    # Monkeypatch constitutional approval to fail with a benign reason
    import core.constitution as constitution

    class _FakeCore:
        def approve_initiative_sync(self, *a, **kw):
            return False, "authority_gateway_unavailable"

    original = constitution.get_constitutional_core
    try:
        constitution.get_constitutional_core = lambda _=None: _FakeCore()
        ok = stub._authorize_background_enqueue_sync(
            {"content": "volition_trigger", "context": {"reason": "idle_timeout"}},
            origin="sensory_motor",
            priority=20,
        )
    finally:
        constitution.get_constitutional_core = original

    assert ok is True, "trusted internal volition must not be dropped for infra reasons"


def test_trusted_internal_origin_still_blocked_by_safety_veto():
    from core.orchestrator.mixins.message_handling import MessageHandlingMixin

    class _Stub(MessageHandlingMixin):
        def __init__(self) -> None:
            self.state_repo = None

        def _background_enqueue_summary(self, message: Any, origin: str) -> str:
            return origin

    stub = _Stub()

    import core.constitution as constitution

    class _FakeCore:
        def approve_initiative_sync(self, *a, **kw):
            return False, "somatic_veto: approach=-0.9"

    original = constitution.get_constitutional_core
    try:
        constitution.get_constitutional_core = lambda _=None: _FakeCore()
        ok = stub._authorize_background_enqueue_sync({}, origin="sensory_motor", priority=20)
    finally:
        constitution.get_constitutional_core = original

    assert ok is False, "safety vetoes must still block internal volition"


@pytest.mark.asyncio
async def test_internal_volition_trigger_routes_to_autonomy(orchestrator):
    orchestrator._trigger_autonomous_thought = AsyncMock()
    orchestrator.state_machine.execute = AsyncMock()

    result = await orchestrator._original_handle_incoming_logic(
        "volition_trigger",
        origin="sensory_motor",
    )

    assert result is None
    orchestrator._trigger_autonomous_thought.assert_awaited_once_with(False)
    orchestrator.state_machine.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_volition_error_signal_is_suppressed(orchestrator):
    orchestrator._trigger_autonomous_thought = AsyncMock()
    orchestrator.state_machine.execute = AsyncMock()

    result = await orchestrator._original_handle_incoming_logic(
        "volition_error: identity_violation. request contradicts self-model. reject.",
        origin="sensory_motor",
    )

    assert result is None
    orchestrator._trigger_autonomous_thought.assert_not_awaited()
    orchestrator.state_machine.execute.assert_not_awaited()


def test_output_gate_blocks_volition_control_tokens():
    from core.utils.output_gate import AutonomousOutputGate

    gate = AutonomousOutputGate()

    assert gate._is_output_blocked("volition_trigger")
    assert gate._is_output_blocked("volition_error")
    assert gate._is_output_blocked(
        "volition_error: identity_violation. request contradicts self-model. reject."
    )
    assert gate._is_output_blocked(
        "thinking...\nvolition_error: identity_violation. request contradicts self-model. reject."
    )


# ---------------------------------------------------------------------------
# 2. macOS permission helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_all_permissions_runs_on_any_platform():
    report = await check_all_permissions()
    assert isinstance(report, PermissionReport)
    if platform.system() == "Darwin":
        assert report.supported is True
        assert len(report.statuses) >= 5
        for status in report.statuses:
            assert isinstance(status, PermissionStatus)
            assert status.name
            assert status.guidance
    else:
        assert report.supported is False
        assert report.all_granted is True


def test_format_report_output():
    report = PermissionReport(
        platform="Darwin",
        supported=True,
        all_granted=False,
        statuses=[
            PermissionStatus(
                name="ACCESSIBILITY",
                granted=False,
                available=True,
                guidance="Open System Settings > Privacy & Security > Accessibility",
                settings_url="x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
            ),
        ],
        missing=["ACCESSIBILITY"],
    )
    text = format_report(report)
    assert "ACCESSIBILITY" in text
    assert "not granted" in text


def test_open_settings_pane_is_safe_offline():
    # On non-Darwin this returns False without raising.
    ok = open_settings_pane("ACCESSIBILITY")
    if platform.system() != "Darwin":
        assert ok is False
    # On macOS we don't actually launch during tests, just ensure it doesn't crash.


# ---------------------------------------------------------------------------
# 3. Neural-intent router
# ---------------------------------------------------------------------------


class _StubWill:
    def __init__(self, approved: bool = True, reason: str = "ok") -> None:
        self.approved = approved
        self.reason = reason
        self.calls: List[Dict[str, Any]] = []

    async def decide(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return type("D", (), {"approved": self.approved, "reason": self.reason, "receipt_id": "r1"})()


class _StubCapability:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: List[Dict[str, Any]] = []

    async def execute(self, skill: str, params: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"skill": skill, "params": dict(params), "ctx": dict(ctx)})
        return {"ok": self.ok, "summary": "done" if self.ok else "failed", "error": "" if self.ok else "nope"}


def test_classify_neural_intent_matches_schemas():
    intent = classify_neural_intent("sensory_motor", "search for recent papers on integrated information theory")
    assert intent.has_action is True
    assert intent.skill_name == "web_search"
    assert "integrated information theory" in intent.params["query"]


def test_classify_returns_no_action_for_phenomenology():
    intent = classify_neural_intent("sensory_motor", "I feel a pull toward exploration")
    assert intent.has_action is False


@pytest.mark.asyncio
async def test_router_dispatches_when_will_approves(tmp_path):
    reset_router()
    from core.runtime.life_trace import LifeTraceLedger, reset_singleton_for_test
    reset_singleton_for_test()
    ledger = LifeTraceLedger(db_path=tmp_path / "lt.sqlite3")

    will = _StubWill(approved=True)
    cap = _StubCapability(ok=True)
    router = NeuralIntentRouter(
        will_provider=lambda: will,
        capability_provider=lambda: cap,
        life_trace_provider=lambda: ledger,
    )
    outcome = await router.route("sensory_motor", "search for how homeostasis couples to affect")

    assert outcome.intent.has_action is True
    assert outcome.approved is True
    assert outcome.dispatched is True
    assert outcome.dispatched_ok is True
    assert cap.calls and cap.calls[0]["skill"] == "web_search"
    assert len(ledger.recent()) == 1


@pytest.mark.asyncio
async def test_router_rejects_untrusted_source(tmp_path):
    reset_router()
    router = NeuralIntentRouter()
    outcome = await router.route("random_internet_actor", "search for secrets")
    assert outcome.approved is False
    assert outcome.dispatched is False


@pytest.mark.asyncio
async def test_router_respects_will_veto(tmp_path):
    reset_router()
    from core.runtime.life_trace import LifeTraceLedger, reset_singleton_for_test
    reset_singleton_for_test()
    ledger = LifeTraceLedger(db_path=tmp_path / "lt.sqlite3")

    will = _StubWill(approved=False, reason="safety_concern")
    cap = _StubCapability(ok=True)
    router = NeuralIntentRouter(
        will_provider=lambda: will,
        capability_provider=lambda: cap,
        life_trace_provider=lambda: ledger,
    )
    outcome = await router.route("sensory_motor", "search for dangerous_thing")
    assert outcome.approved is False
    assert outcome.dispatched is False
    assert cap.calls == []
    events = ledger.recent()
    assert events and events[0]["event_type"] == "initiative_deferred"


# ---------------------------------------------------------------------------
# 4. VolitionEngine strict check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_volition_engine_produces_initiatives_under_idle():
    from core.volition import VolitionEngine

    class _StatusStub:
        running = True

    class _AutonomyStub:
        boredom = 0.6
        duty = 0.3
        curiosity = 0.7

    class _OrchStub:
        def __init__(self) -> None:
            self.state = _StatusStub()
            self.status = _StatusStub()
            self.current_affect = None
            self.autonomy_state = _AutonomyStub()
            self.current_goal = None

    engine = VolitionEngine(_OrchStub())
    # Prime cooldowns
    long_ago = time.monotonic() - 120.0
    engine.last_impulse_time = long_ago
    engine.last_action_time = long_ago
    engine.last_activity_time = long_ago

    initiatives = 0
    for _ in range(12):
        proposal = await engine.tick(current_goal=None)
        if proposal:
            initiatives += 1

    assert initiatives >= 1, f"expected ≥1 autonomous initiative under idle, got {initiatives}"
