"""Improving her source and improving her language were two loops.

The generality review asked for native and source development in one verified
loop. They were separate: `RecursiveSelfImprovementLoop` planned weight
updates, code refinements and tool creation, while the developmental policy
changed the terms she thinks in — two loops, two records, and no cycle that
could weigh "change the code" against "change the language".

"Which of those is worth doing here" is the question a single improver has to
be able to answer, so the plan now asks the developmental policy too.
"""
from __future__ import annotations

import pytest

from core.learning.recursive_self_improvement import RecursiveSelfImprovementLoop


@pytest.fixture
def loop():
    return object.__new__(RecursiveSelfImprovementLoop)


def test_the_plan_can_choose_a_change_to_her_terms(loop, monkeypatch):
    import core.cognition.she_decides_to_develop as deciding
    import core.cognition.what_she_could_do_next as doing

    class AnAction:
        name = "a way of computing she wrote"

    monkeypatch.setattr(doing, "the_actions_she_has", lambda *a, **k: (AnAction(),))
    monkeypatch.setattr(
        deciding,
        "what_to_do_next",
        lambda *a, **k: type("D", (), {"action": AnAction()})(),
    )
    assert (
        loop._what_she_could_change_about_her_own_terms()
        == "a way of computing she wrote"
    )


def test_nothing_registered_is_an_answer_and_not_a_failure(loop, monkeypatch):
    """With no developmental actions there is no native change to weigh."""
    import core.cognition.what_she_could_do_next as doing

    monkeypatch.setattr(doing, "the_actions_she_has", lambda *a, **k: ())
    assert loop._what_she_could_change_about_her_own_terms() is None


def test_a_policy_that_declines_does_not_take_the_planner_down(loop, monkeypatch):
    import core.cognition.she_decides_to_develop as deciding
    import core.cognition.what_she_could_do_next as doing

    monkeypatch.setattr(doing, "the_actions_she_has", lambda *a, **k: ("something",))

    def angry(*a, **k):
        raise RuntimeError("the policy is not ready")

    monkeypatch.setattr(deciding, "what_to_do_next", angry)
    assert loop._what_she_could_change_about_her_own_terms() is None


def test_a_decision_with_no_action_is_none(loop, monkeypatch):
    import core.cognition.she_decides_to_develop as deciding
    import core.cognition.what_she_could_do_next as doing

    monkeypatch.setattr(doing, "the_actions_she_has", lambda *a, **k: ("something",))
    monkeypatch.setattr(
        deciding, "what_to_do_next", lambda *a, **k: type("D", (), {"action": None})()
    )
    assert loop._what_she_could_change_about_her_own_terms() is None


def test_the_planner_reads_the_choice_and_does_not_take_it(loop, monkeypatch):
    """A planner that acted while planning would make the plan a record."""
    import core.cognition.she_decides_to_develop as deciding
    import core.cognition.what_she_could_do_next as doing

    taken = []

    class AnAction:
        name = "an action"

        @staticmethod
        def do_it(*a, **k):
            taken.append(True)
            return "did it"

    monkeypatch.setattr(doing, "the_actions_she_has", lambda *a, **k: (AnAction(),))
    monkeypatch.setattr(
        deciding, "what_to_do_next", lambda *a, **k: type("D", (), {"action": AnAction()})()
    )
    loop._what_she_could_change_about_her_own_terms()
    assert taken == [], "the planner ran the action instead of reading the choice"


def test_the_plan_names_the_developmental_action_it_would_take():
    """Both kinds of change reach one plan, and each says which it is."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "learning" / "recursive_self_improvement.py"
    ).read_text("utf-8")
    assert 'actions.append("developmental")' in source
    assert "the cheapest thing worth changing about how she thinks is" in source
