"""Her resting rhythm (core/brain/resting_rhythm.py), from the third day run (6 November 2026).

On an unattended day her record advanced only through her journal's failed retries, every five minutes. With the
journal working it writes hourly; the rhythm keeps her record of an unattended day by running a quiet cycle of her own
when nothing has committed her state for five minutes. It says nothing, and never runs in a proof run.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from core.brain import cognitive_engine, resting_rhythm
from core.runtime import background_policy, response_policy


class _Brain:
    def __init__(self):
        self.calls = []

    async def think(self, objective=None, context=None, mode=None, origin=None, **kwargs):
        self.calls.append(origin)
        cognitive_engine.note_thought_committed()
        return SimpleNamespace(content="")


def _rhythm(monkeypatch, *, admitted=""):
    monkeypatch.setattr(background_policy, "background_activity_reason",
                        lambda *a, allow_no_user_anchor=False, **k: admitted if allow_no_user_anchor else "no_user_anchor")
    monkeypatch.delenv("AURA_PROOF_RUN", raising=False)
    brain = _Brain()
    rhythm = resting_rhythm.RestingRhythm(SimpleNamespace(cognitive_engine=brain, status=SimpleNamespace(is_processing=False)))
    rhythm._started_at = time.time() - 3600
    return rhythm, brain


def test_a_quiet_five_minutes_is_followed_by_one_resting_cycle(monkeypatch):
    rhythm, brain = _rhythm(monkeypatch)
    cognitive_engine._COGNITION_LEDGER["last_committed_at"] = time.time() - 400
    assert asyncio.run(rhythm.maybe_rest()) is True
    assert brain.calls == ["resting_cycle"]
    # It committed, so the next look finds her recently thinking and rests no more.
    assert asyncio.run(rhythm.maybe_rest()) is False
    assert rhythm.last_reason == "a thought committed recently"


def test_it_waits_for_a_turn_in_flight_and_for_admission(monkeypatch):
    rhythm, brain = _rhythm(monkeypatch, admitted="memory_pressure_91.0")
    cognitive_engine._COGNITION_LEDGER["last_committed_at"] = time.time() - 400
    assert asyncio.run(rhythm.maybe_rest()) is False and rhythm.last_reason == "memory_pressure_91.0"
    rhythm, brain = _rhythm(monkeypatch)
    rhythm.orchestrator.status.is_processing = True
    assert asyncio.run(rhythm.maybe_rest()) is False and rhythm.last_reason == "a turn is in flight"
    assert brain.calls == []


def test_never_in_a_proof_run(monkeypatch):
    rhythm, brain = _rhythm(monkeypatch)
    cognitive_engine._COGNITION_LEDGER["last_committed_at"] = time.time() - 400
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    assert asyncio.run(rhythm.maybe_rest()) is False and rhythm.last_reason == "proof_run_active"


def test_a_resting_cycle_asks_nothing_of_her_model():
    assert response_policy.background_response_suppression_reason(
        resting_rhythm.RESTING_OBJECTIVE, origin="resting_cycle") == "resting_cycle"
    assert not background_policy.is_background_origin("resting_cycle")
    assert not background_policy.is_user_facing_origin("resting_cycle")
