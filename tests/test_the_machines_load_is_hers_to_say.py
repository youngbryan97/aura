"""The machine's load is a reading beside every turn, and her reply about anything is hers, not a sentence put in
its place when a pattern over the question's words says it asked about her state.

A sentence of the runtime's replaced her reply five times when the patterns caught another question: a sourdough
experiment (28 August 2026), a sales analysis (27 August), her memory's design (7 September), a plan to test her
memory (8 September), and on 17 November "Your loop should meet the fault again", each answered "The machine is at N%
processor and M% memory right now."
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.brain import present_moment
from core.introspection import self_evidence


@pytest.fixture
def the_machine(monkeypatch):
    world = SimpleNamespace(_telemetry_measured=True, cpu_percent=9.7, memory_percent=60.0,
                            _thermal_measured=True, thermal_pressure=0.25, update=lambda: None)
    monkeypatch.setattr("core.world_state.get_world_state", lambda: world)
    return world


def test_her_present_moment_has_the_machine_as_measured(the_machine):
    block = present_moment.present_moment_block()
    assert "- Machine: processor 9.7%, memory 60.0%, thermal pressure 0.25 of 1." in block


def test_a_machine_that_did_not_read_is_left_out(monkeypatch):
    monkeypatch.setattr("core.world_state.get_world_state", lambda: SimpleNamespace(
        _telemetry_measured=False, update=lambda: None))
    assert "Machine:" not in present_moment.present_moment_block()


@pytest.mark.parametrize("asked", [
    "Back again. I took your priced() mend back out. Your loop should meet the fault again.",
    "How hard is the machine you run on working right now?",
])
def test_her_reply_is_served_as_she_wrote_it(the_machine, asked):
    from interface.routes.chat_recorded_answers import _recorded_answer_corrections

    assert self_evidence.asks_about_own_operational_state(asked) or "loop" in asked
    hers = "My loop will meet it again on its next pass; I'll tell you what it makes of it."
    served = asyncio.run(_recorded_answer_corrections(asked, hers))
    assert "The machine is at" not in str(served)
    assert hers in str(served)
