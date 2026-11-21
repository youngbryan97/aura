"""A step that stalls is done after the step it has only ever worked after, learned from her own runs.

A lesson from the first hour of Minecraft, until now used only by her on-screen play
(core/cognition/what_having_it_lets_her_do.py): a thing that does not work here would work given one other thing, and
the other thing is gettable. Her replanner could retry a stalled step, start over, or drop it; it had no way to see
that the step was waiting for another one later in the same plan.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.agency.replanning as replanning
from core.cognition.what_having_it_lets_her_do import WhatOpensWhat


def _run(*steps: tuple[str, bool]) -> SimpleNamespace:
    return SimpleNamespace(steps=[SimpleNamespace(name=name, ok=worked, verified=worked) for name, worked in steps],
                           verified_progress=sum(worked for _name, worked in steps))


@pytest.fixture
def opens(monkeypatch):
    fresh = WhatOpensWhat()
    monkeypatch.setattr(replanning, "_OPENS", [fresh])
    monkeypatch.setattr("core.runtime.what_she_learned.remember", lambda *_a, **_k: True)
    return fresh


def test_what_a_step_waits_for_is_learned_from_her_runs(opens) -> None:
    replanning.learn_from_the_run(_run(("write the file", False), ("make the folder", True)))
    replanning.learn_from_the_run(_run(("make the folder", True), ("write the file", True)))
    assert opens.what_opens("write the file") == ("make the folder",)


def test_a_stalled_step_is_offered_the_step_it_waits_for_first(opens) -> None:
    replanning.learn_from_the_run(_run(("write the file", False), ("make the folder", True)))
    replanning.learn_from_the_run(_run(("make the folder", True), ("write the file", True)))
    plan = [SimpleNamespace(name="write the file"), SimpleNamespace(name="make the folder"),
            SimpleNamespace(name="tell them")]
    stalled = _run(("write the file", False))
    options = replanning._repairs(plan, "write the file", replanning.done_before(stalled, "write the file"))
    first = [one for one in options if one.name == replanning.FIRST_WHAT_IT_WAITS_FOR]
    assert first and first[0].params["first"] == "make the folder"
    repaired = replanning.apply_repair(replanning.FIRST_WHAT_IT_WAITS_FOR, plan, "write the file", "make the folder")
    assert [step.name for step in repaired] == ["make the folder", "write the file", "tell them"]
    assert [step.name for step in plan] == ["write the file", "make the folder", "tell them"]  # not mutated


def test_nothing_is_offered_where_she_has_never_seen_the_step_open(opens) -> None:
    plan = [SimpleNamespace(name="write the file"), SimpleNamespace(name="make the folder")]
    options = replanning._repairs(plan, "write the file", [])
    assert all(one.name != replanning.FIRST_WHAT_IT_WAITS_FOR for one in options)
