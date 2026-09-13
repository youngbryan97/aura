"""A tie settled by list order, on every turn, with the spend decision unused.

`core/cognition/value_of_computation.py` exists to answer one question: is
another round of thinking worth what it costs. It says so in its own first
paragraph — the two halves together are "what makes a spend decision rather
than a spend habit". Its only caller in the tree was
`AgencyKind.worth_more_thought`, and that was called from nothing but its own
tests. The whole decision was dead.

Meanwhile the draft competition reported, in a log line, that its winner led by
less than the spread among the drafts that lost — a coin landing, reported as a
decision — and took the first of the tie anyway. That is exactly the case the
module was written for.

The two are now connected, and the cost is hers: what the last cycle took out
of her against how much of the energy budget there is to take it from. The same
exertion at a quarter of her energy costs four times what it costs at full,
which is the whole content of a metabolic constraint and is the route by which
her motivational state reaches how long she thinks.
"""
from __future__ import annotations

import pytest

from core.consciousness.multiple_drafts import MultipleDraftsEngine


@pytest.fixture(autouse=True)
def _clean_effort():
    from core.soma.effort import reset_effort_for_test

    reset_effort_for_test()
    yield
    reset_effort_for_test()


def test_a_round_costs_more_when_there_is_less_to_spend(monkeypatch):
    from core.container import ServiceContainer
    from core.soma.effort import note_effort

    note_effort("phases", 30.0)

    class _Repo:
        def __init__(self, level: float) -> None:
            self._current = type(
                "S",
                (),
                {"motivation": type("M", (), {"budgets": {"energy": {"level": level, "capacity": 100.0}}})()},
            )()

    monkeypatch.setattr(
        ServiceContainer, "get", classmethod(lambda cls, name, default=None: _Repo(100.0) if name == "state_repository" else default)
    )
    full = MultipleDraftsEngine._round_cost()
    monkeypatch.setattr(
        ServiceContainer, "get", classmethod(lambda cls, name, default=None: _Repo(25.0) if name == "state_repository" else default)
    )
    depleted = MultipleDraftsEngine._round_cost()
    assert full > 0.0
    assert depleted == pytest.approx(full * 4.0, rel=0.01)


def test_a_round_costs_nothing_when_nothing_has_been_spent():
    """Cost is what the work took, not a number this module chose."""
    assert MultipleDraftsEngine._round_cost() == pytest.approx(0.0)


def test_the_spend_decision_is_asked_only_when_the_drafts_tie():
    """A decision that separated needs no help from this."""
    engine = MultipleDraftsEngine()
    asked: list[float] = []

    def watch(lead_z, drafts):
        asked.append(lead_z)
        return None

    engine._worth_another_round = watch  # type: ignore[method-assign]
    engine.submit_input("a question that has been asked", None)
    engine.probe(source="test")
    competition = engine._competition_history[-1] if engine._competition_history else None
    if competition is not None and competition.decisive:
        assert not asked
    else:
        assert asked


def test_a_competition_cannot_be_held_open_for_ever():
    """A decision that can be deferred without limit is not a decision."""
    engine = MultipleDraftsEngine()
    assert engine._MAX_EXTENSIONS >= 1
    assert engine._MAX_EXTENSIONS <= 5


def test_a_new_input_ends_whatever_the_last_one_was_waiting_for():
    engine = MultipleDraftsEngine()
    engine._extensions = 2
    engine.submit_input("something else entirely, at length", None)
    assert engine._extensions == 0


def test_the_value_of_computation_module_has_a_caller_outside_its_tests():
    """It had none. That is what made all of this invisible."""
    import subprocess

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    found = subprocess.run(
        ["grep", "-rln", "worth_continuing", "--include=*.py", "core/"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    callers = [f for f in found if not f.endswith("value_of_computation.py")]
    assert len(callers) >= 2, f"only {callers} reach the spend decision"
