"""A gap seen often enough becomes a capability, and something asks.

An external review traced this exactly: Aura closes the reactive loop — a
NAMED capability is missing, so synthesise it, test it, install it, retry —
and does not close the general one. `SkillSynthesizer` accumulates recurring
gaps and `synthesize_pending` turns the frequent ones into skills, and it had
no production caller. Its own docstring said "call this in a background loop"
and nothing did.

So gaps were recognised, counted, and never acted on: repeated unnamed
failure, infer the abstraction, forge it, deploy — that chain had a hole in
its middle.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest


def test_something_in_production_asks_the_forge():
    """The hole. Before this, only a test ever called it."""

    import core.learning.recursive_self_improvement as rsi

    source = inspect.getsource(rsi)
    assert "synthesize_pending" in source, "nothing in production drives the forge"
    assert "_forge_what_keeps_being_missing" in source


def test_it_is_gated_exactly_as_tool_creation_is_because_it_is_tool_creation():
    import core.learning.recursive_self_improvement as rsi

    source = inspect.getsource(
        rsi.RecursiveSelfImprovementLoop._ask_the_forge_about_recurring_gaps
    )
    assert "AURA_RSI_TOOL_CREATION" in source
    assert "if not allowed or not observed" in source


def test_it_does_nothing_when_it_is_not_allowed():
    import core.learning.recursive_self_improvement as rsi

    ask = rsi.RecursiveSelfImprovementLoop._ask_the_forge_about_recurring_gaps
    assert ask(allowed=False, observed=True) is False
    assert ask(allowed=True, observed=False) is False


def test_it_does_nothing_when_the_flag_is_off(monkeypatch):
    import core.learning.recursive_self_improvement as rsi

    monkeypatch.setenv("AURA_RSI_TOOL_CREATION", "0")
    ask = rsi.RecursiveSelfImprovementLoop._ask_the_forge_about_recurring_gaps
    assert ask(allowed=True, observed=True) is False


def test_when_it_is_allowed_it_asks_and_reports_what_came_back(monkeypatch):
    """Asked rather than awaited, so the plan does not wait for a build."""

    import core.learning.recursive_self_improvement as rsi

    monkeypatch.setenv("AURA_RSI_TOOL_CREATION", "1")
    asked = {"n": 0}

    class AForge:
        async def synthesize_pending(self):
            asked["n"] += 1
            return [type("Made", (), {"name": "a_forged_skill"})()]

    monkeypatch.setattr(
        "core.agi.skill_synthesizer.get_skill_synthesizer", lambda: AForge()
    )
    made = asyncio.run(rsi.RecursiveSelfImprovementLoop._forge_what_keeps_being_missing())
    assert made == ["a_forged_skill"]
    assert asked["n"] == 1

    async def through_the_scheduler() -> bool:
        return rsi.RecursiveSelfImprovementLoop._ask_the_forge_about_recurring_gaps(
            allowed=True, observed=True
        )

    assert asyncio.run(through_the_scheduler()) is True


def test_a_forge_that_raises_is_a_degradation_and_not_a_dead_cycle(monkeypatch):
    import core.learning.recursive_self_improvement as rsi

    monkeypatch.setenv("AURA_RSI_TOOL_CREATION", "1")

    class ABrokenForge:
        async def synthesize_pending(self):
            raise RuntimeError("the forge is down")

    monkeypatch.setattr(
        "core.agi.skill_synthesizer.get_skill_synthesizer", lambda: ABrokenForge()
    )
    assert asyncio.run(rsi.RecursiveSelfImprovementLoop._forge_what_keeps_being_missing()) == []


def test_the_ledger_still_records_the_gaps_it_counts():
    """The other half, already wired, held."""

    import core.learning.recursive_self_improvement as rsi

    source = inspect.getsource(rsi.RecursiveSelfImprovementLoop._remember_the_gaps)
    assert "log_gap" in source
