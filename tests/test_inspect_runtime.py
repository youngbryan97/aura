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


def test_everything_the_maturity_pass_built_is_askable_from_here() -> None:
    """A primitive nobody can ask about is one nobody will find.

    Reachable is not governing — that is what `what_governs` measures — but a
    module that is not even in the inspector cannot become either.
    """
    from tools.inspect_runtime import THE_PRIMITIVES, THE_SECTIONS

    wanted = {
        "core.observability.which_clock_is_this": "clocks",
        "core.observability.does_one_trace_reach_the_end": "traces",
        "core.runtime.how_a_call_is_made": "calls",
        "core.runtime.how_a_task_should_end": "task_endings",
        "core.state.what_they_all_read": "supersteps",
        "core.memory.what_kind_of_memory_is_this": "memory_kinds",
        "core.brain.llm.who_got_the_room": "prompt_room",
        "core.runtime.what_is_on_its_way_out": "deprecations",
        "core.runtime.claiming_more_than_one": "multi_claims",
        "core.state.nothing_lands_before_its_writes": "write_drains",
        "core.runtime.what_she_decided_to_do_at_once": "action_batches",
        "core.runtime.cancelling_the_call_and_not_just_the_wait": "abandoned_calls",
        "core.runtime.which_parts_say_how_they_are": "lifecycles",
        "core.verify.a_promise_with_a_test": "promises",
    }
    missing = sorted(name for name, section in wanted.items() if section not in THE_SECTIONS)
    assert missing == [], f"no section reads {missing}"
    unasked = sorted(set(THE_PRIMITIVES) - set(wanted) - {
        "core.runtime.what_must_never_be_retried",
        "core.verify.is_this_async_code_correct",
    })
    assert unasked == [], f"{unasked} were built and are not askable"


def test_the_inspector_says_what_still_decides_nothing() -> None:
    from tools.inspect_runtime import inspect

    said = inspect("what_governs")["what_governs"]
    assert "proposals" in said and "governing" in said
    assert said["asked_about"] >= 16
