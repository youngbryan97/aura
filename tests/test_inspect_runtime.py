"""One place to ask what the runtime is.

Soar publishes uniform introspection commands, and the closure asked for the
same: a single API over topology, state owners, active phases, counters,
queues, resources and degradations, with machine-readable output.

Aura had all of those and they were in eight places.
"""
from __future__ import annotations

import json

import pytest

from tools.inspect_runtime import THE_SECTIONS, inspect


@pytest.fixture(scope="module")
def everything():
    return inspect()


def test_it_answers_every_question_the_closure_named(everything):
    for wanted in (
        "topology", "owners", "phases", "counters", "resources", "degradations"
    ):
        assert wanted in everything, wanted


def test_no_section_reports_an_error(everything):
    broken = {
        name: one["error"]
        for name, one in everything.items()
        if isinstance(one, dict) and "error" in one
    }
    assert broken == {}, broken


def test_the_answer_is_machine_readable(everything):
    """"Machine JSON output" was the ask, and prose is not that."""
    assert json.loads(json.dumps(everything, default=str))


def test_one_section_can_be_asked_for_on_its_own():
    said = inspect("resources")
    assert set(said) == {"resources"}
    assert set(said["resources"]) == {"held", "waiting", "record"}


def test_a_section_nobody_has_says_what_can_be_asked():
    said = inspect("the meaning of it all")
    assert "no such section" in said["the meaning of it all"]["error"]
    assert "topology" in said["the meaning of it all"]["error"]


def test_one_unhappy_subsystem_does_not_take_the_answer_down(monkeypatch):
    """A tool that fails entirely is useless exactly when it is needed."""
    def angry():
        raise RuntimeError("this subsystem is not well")

    monkeypatch.setitem(THE_SECTIONS, "resources", angry)
    said = inspect()
    assert "this subsystem is not well" in said["resources"]["error"]
    assert "topology" in said
    assert "error" not in said["topology"]


def test_the_phases_section_carries_the_seal_and_the_drawing(everything):
    phases = everything["phases"]
    assert set(phases) >= {"foreground", "background"}
    for mode, said in phases.items():
        assert said["seal"], mode
        assert said["drawing"].startswith("flowchart TD"), mode


def test_every_section_reads_from_the_module_that_owns_it():
    """Nothing here may drift from what the runtime actually reports."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "tools" / "inspect_runtime.py"
    ).read_text("utf-8")
    for name in THE_SECTIONS:
        assert f'"{name}": _' in source, name
    # Every reader is an import away, not a reimplementation.
    assert source.count("from core.") >= len(THE_SECTIONS) - 3


# ----------------------------------------------- everything is reachable


def test_every_module_this_session_added_is_reachable_from_here():
    """A module that is tested and not reachable is half-wired.

    Ten of them were: built, covered, and not askable from a running system.
    """
    for wanted in (
        "budgets_and_guardrails",
        "numbers",
        "observations",
        "interrupted",
        "action_history",
        "destinations",
    ):
        assert wanted in THE_SECTIONS, wanted


def test_the_new_sections_answer(everything):
    for name in (
        "budgets_and_guardrails",
        "numbers",
        "observations",
        "interrupted",
        "action_history",
        "destinations",
    ):
        assert "error" not in everything[name], everything[name]


def test_the_budget_section_shows_the_rules_rather_than_a_count(everything):
    """Nothing is spending yet; what a reader needs is what the rules are."""
    said = everything["budgets_and_guardrails"]
    assert said["an_empty_budget_refuses"] is True
    assert said["an_empty_chain_passes"] is True


def test_the_destinations_section_carries_all_seven(everything):
    assert everything["destinations"]["destinations"] == 7
    assert everything["destinations"]["not_really_a_destination"] == []
