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


# ── and the proof has to actually change what it proves ──────────────────

class Lane:
    """An MLX client whose readiness can be inspected after a probe."""

    def __init__(self, answers="ready", alive=True):
        self.answers = answers
        self.alive = alive
        self._last_visible_readiness_at = 0.0
        self._lane_state = "warming"

    def is_alive(self):
        return self.alive

    async def _generate_inner(self, *_a, **_k):
        return self.answers

    def _set_lane_state(self, state, error=""):
        self._lane_state = state


async def _prove(lane):
    from core.brain.llm.mlx_client import MLXLocalClient

    return await MLXLocalClient.prove_visible_readiness(lane, budget_s=5.0)


@pytest.mark.asyncio
async def test_a_proof_records_that_the_lane_was_seen_to_answer():
    """The recording IS the proof.

    Running a health probe and reading the text would succeed, report success,
    and leave readiness exactly as unproven as it found it: health_probe
    suppresses the user-facing mark by design, and conversation readiness
    blocks on visible_conversation_probe_missing until that mark exists. Both
    components sensible, the composition wrong.
    """
    lane = Lane()
    assert await _prove(lane) == "proved"
    assert lane._last_visible_readiness_at > 0.0
    assert lane._lane_state == "ready"


@pytest.mark.asyncio
async def test_an_answer_that_is_not_an_answer_proves_nothing():
    lane = Lane(answers="")
    assert await _prove(lane) == "no_text"
    assert lane._last_visible_readiness_at == 0.0


@pytest.mark.asyncio
async def test_nor_does_a_reply_that_does_not_respond_to_the_probe():
    lane = Lane(answers="the capital of Peru is Lima")
    assert await _prove(lane) == "answer_mismatch"
    assert lane._last_visible_readiness_at == 0.0


@pytest.mark.asyncio
async def test_a_dead_worker_proves_nothing_either():
    lane = Lane(alive=False)
    assert await _prove(lane) == "no_worker"
    assert lane._last_visible_readiness_at == 0.0


def test_only_one_place_records_that_the_lane_was_seen_to_answer():
    """So a third prover cannot reintroduce the gap by forgetting a line."""
    import inspect

    from core.brain.llm import mlx_client

    setting = [
        line
        for line in inspect.getsource(mlx_client).splitlines()
        if "_last_visible_readiness_at = time.time()" in line
        and not line.strip().startswith("#")
    ]
    assert len(setting) == 1
