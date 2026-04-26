"""Phenomenal error map: every catalogued exception type produces a
non-empty user-facing message and a four-button recovery envelope.
Decorator must transform unhandled exceptions into PhenomenalRaise
without leaking the original traceback to the caller surface.
"""
from __future__ import annotations

import asyncio

import pytest

from core.resilience.phenomenal_error_map import (
    classify,
    build_envelope,
    phenomenal,
    PhenomenalRaise,
    PhenomenalContext,
    PHENOMENAL_STATES,
)


def test_classify_timeout_returns_cognitive_fog():
    state = classify(asyncio.TimeoutError("nope"))
    assert state.name == "cognitive_fog"


def test_classify_connection_refused_returns_network_offline():
    state = classify(ConnectionRefusedError("nope"))
    assert state.name == "network_offline"


def test_envelope_has_three_buttons():
    env = build_envelope(asyncio.TimeoutError("nope"))
    assert len(env.recovery_buttons) == 3
    labels = {b["label"] for b in env.recovery_buttons}
    assert {"Retry", "Use fallback", "Open diagnostics"} <= labels


def test_decorator_translates_exception_to_phenomenal_raise():
    @phenomenal()
    async def fn() -> None:
        raise ConnectionRefusedError("local cortex offline")

    async def runner():
        try:
            await fn()
        except PhenomenalRaise as exc:
            assert exc.envelope.phenomenal_state == "network_offline"
            return True
        return False

    assert asyncio.run(runner())


def test_context_manager_translates_exception():
    with pytest.raises(PhenomenalRaise) as ctx:
        with PhenomenalContext(scope="unit_test"):
            raise asyncio.TimeoutError("test_timeout")
    assert ctx.value.envelope.phenomenal_state == "cognitive_fog"


def test_unknown_exception_falls_back_to_unknown_state():
    state = classify(RuntimeError("garbage"))
    assert state.name in PHENOMENAL_STATES
