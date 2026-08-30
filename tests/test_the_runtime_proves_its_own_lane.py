"""A lane that is loaded is not a lane that is serving.

The runtime knew the difference. Optional background work is gated on the
conversation lane having produced at least one visible reply, and the executive
raises the threat level while it has not.

Nothing produced that reply. The gate blocked on the proof and never made one,
so the proof could only arrive from outside — a person typing something.

LIVE 2026-08-29: a transient memory blip deferred one recovery warmup, the lane
stayed unproven, the threat level went critical, and every desktop action was
refused for four hours — including the ones that would have produced a turn.
The lane was fine the whole time. One chat message healed it instantly, which
is the evidence that nothing inside was ever going to.
"""

from __future__ import annotations

import time

import pytest

from core.runtime.lane_reconciler import UNPROVEN_TOO_LONG_S, LaneReconciler


class Gate:
    """An inference gate reporting a given conversation-lane state."""

    def __init__(self, **lane):
        self.lane = lane

    def get_conversation_status(self):
        return dict(self.lane)


def reconciler(proof_result="proved", **kwargs):
    calls: list[int] = []

    async def prove():
        calls.append(1)
        return proof_result

    it = LaneReconciler(
        observe_lanes=lambda: [],
        primary_alive=lambda: True,
        primary_key=lambda: "primary",
        primary_age_s=lambda: 10.0,
        spawn_primary=_never,
        evict_lane=_never_lane,
        prove_lane=prove,
        **kwargs,
    )
    return it, calls


async def _never():
    return False


async def _never_lane(_path):
    return False


async def _asked(monkeypatch, gate):
    """Run the real proof against a gate reporting a given lane state."""
    import core.container as container
    from core.runtime.lane_reconciler import _default_prove_lane

    monkeypatch.setattr(
        container.ServiceContainer, "peek",
        staticmethod(lambda *a, **k: gate),
        raising=False,
    )
    return await _default_prove_lane()


# ── it proves the lane when nothing else will ────────────────────────────

@pytest.mark.asyncio
async def test_a_lane_nobody_has_proven_gets_proven():
    it, calls = reconciler(foreground_active=lambda: False)
    actions = await it.reconcile_once()
    assert calls, "the reconciler never tried to prove the lane"
    assert any(a.get("action") == "proof" for a in actions)


@pytest.mark.asyncio
async def test_and_it_says_what_happened():
    it, _calls = reconciler(proof_result="proof_timed_out", foreground_active=lambda: False)
    actions = await it.reconcile_once()
    proof = next(a for a in actions if a.get("action") == "proof")
    assert proof["detail"] == "proof_timed_out"


@pytest.mark.asyncio
async def test_nothing_is_noted_when_there_was_nothing_to_prove():
    it, _calls = reconciler(proof_result="", foreground_active=lambda: False)
    actions = await it.reconcile_once()
    assert not any(a.get("action") == "proof" for a in actions)


# ── and it never competes with a person ──────────────────────────────────

@pytest.mark.asyncio
async def test_a_turn_in_flight_is_left_alone():
    it, calls = reconciler(foreground_active=lambda: True)
    await it.reconcile_once()
    assert not calls


# ── what counts as needing proof ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_ready_lane_needs_no_proof(monkeypatch):
    assert await _asked(monkeypatch, Gate(conversation_ready=True)) == ""


@pytest.mark.asyncio
async def test_a_lane_answering_right_now_needs_no_proof(monkeypatch):
    lane = Gate(conversation_ready=False, active_generations=1)
    assert await _asked(monkeypatch, lane) == ""


@pytest.mark.asyncio
async def test_a_lane_still_warming_is_given_time(monkeypatch):
    lane = Gate(conversation_ready=False, warmup_in_flight=True)
    assert await _asked(monkeypatch, lane) == ""


@pytest.mark.asyncio
async def test_a_lane_proven_recently_is_not_proven_again(monkeypatch):
    lane = Gate(conversation_ready=False, last_visible_readiness_at=time.time())
    assert await _asked(monkeypatch, lane) == ""


def test_the_wait_is_long_enough_for_an_ordinary_turn_to_do_it_for_free():
    assert UNPROVEN_TOO_LONG_S >= 60.0
