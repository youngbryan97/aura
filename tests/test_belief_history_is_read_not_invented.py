"""Naming a position she dropped is a lookup, not a generation.

LIVE, 2026-08-19. Asked "name one actual position you held and then dropped,
with when", and given the explicit out "if you can't, say so plainly", she
answered:

    I used to think affect was a side effect of cognition, not fundamental.
    Around the middle of last year, interacting with users and reflecting on
    my own experience, I changed that view.

No such record exists. The date, the occasion and the position were invented,
in reply to a question whose entire point was whether she could produce
evidence — the same class as the 2026-08-10 finding where she recalled a prior
answer of her own that she had never given.

``SelfModel`` keeps timestamped snapshots, each holding the belief map as it
stood, so consecutive snapshots differ exactly where she changed her mind.
Nothing read them.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.self.belief_history import belief_changes, describe_belief_changes


def _snapshot(ts: float, beliefs: dict, note: str | None = None):
    return SimpleNamespace(ts=ts, beliefs=beliefs, revision_note=note)


@pytest.fixture
def model():
    now = time.time()
    return SimpleNamespace(
        snapshots={
            "a": _snapshot(now - 86400 * 30, {"affect_is_fundamental": False, "name": "aura"}),
            "b": _snapshot(
                now - 86400 * 3,
                {"affect_is_fundamental": True, "name": "aura"},
                "after the felt-thought probe",
            ),
            "c": _snapshot(now - 3600, {"affect_is_fundamental": True, "name": "aura", "fresh": 1}),
        }
    )


def test_a_real_revision_is_found_with_its_date(model):
    (change,) = belief_changes(model)
    assert change.key == "affect_is_fundamental"
    assert change.before is False
    assert change.after is True
    assert "2026" in change.when()
    assert "after the felt-thought probe" in change.sentence()


def test_a_belief_appearing_for_the_first_time_is_not_a_change_of_mind(model):
    """She did not use to think otherwise; she had no view."""
    assert all(change.key != "fresh" for change in belief_changes(model))


def test_a_belief_that_never_moved_is_not_a_change_of_mind(model):
    assert all(change.key != "name" for change in belief_changes(model))


def test_no_record_at_all_reads_as_nothing():
    """Reserved for having nothing to read."""
    assert describe_belief_changes(SimpleNamespace(snapshots={})) == ""
    assert describe_belief_changes(
        SimpleNamespace(snapshots={"only": _snapshot(time.time(), {"x": 1})})
    ) == ""


def test_a_record_showing_no_revisions_says_so_with_numbers():
    """Silence is what let the invention through.

    An empty block leaves the model free to make up a revision, which is
    exactly what happened. A measured "none, out of N snapshots since <date>"
    gives her something true to say instead — and it is the answer the
    question explicitly offered: say so plainly.
    """
    now = time.time()
    machine_state_only = SimpleNamespace(
        snapshots={
            "a": _snapshot(now - 86400 * 4, {"executive_closure": {"need": "stability"}}),
            "b": _snapshot(now - 86400 * 2, {"executive_closure": {"need": "social"}}),
            "c": _snapshot(now - 3600, {"executive_closure": {"need": "rest"}}),
        }
    )
    reading = describe_belief_changes(machine_state_only)
    assert "cannot name one from evidence" in reading
    assert "3 snapshots" in reading


def test_per_tick_machine_state_is_not_a_position():
    """The live self-model's only belief keys are runtime state.

    `executive_closure` and `runtime_lessons` are dictionaries rewritten on
    nearly every snapshot. Diffing them yields a wall of nested dict text that
    changes constantly, and calling it "a position I have revised" would be
    both unreadable and untrue.
    """
    now = time.time()
    model = SimpleNamespace(
        snapshots={
            "a": _snapshot(now - 300, {"executive_closure": {"focus": "one"}}),
            "b": _snapshot(now - 200, {"executive_closure": {"focus": "two"}}),
            "c": _snapshot(now - 100, {"executive_closure": {"focus": "three"}}),
        }
    )
    assert belief_changes(model) == ()


def test_a_stance_that_churns_every_tick_is_state_not_a_position():
    """Even a scalar, if the runtime rewrites it constantly."""
    now = time.time()
    model = SimpleNamespace(
        snapshots={
            "a": _snapshot(now - 400, {"tick_phase": "a"}),
            "b": _snapshot(now - 300, {"tick_phase": "b"}),
            "c": _snapshot(now - 200, {"tick_phase": "c"}),
            "d": _snapshot(now - 100, {"tick_phase": "d"}),
        }
    )
    assert belief_changes(model) == ()


def test_a_missing_self_model_does_not_break_the_turn():
    assert belief_changes(SimpleNamespace()) == ()
    assert describe_belief_changes(SimpleNamespace()) == ""


def test_the_question_that_produced_the_invention_now_reaches_the_reading():
    from core.brain.observable_registry import install_default_observables

    install_default_observables()
    from core.brain.observable_grounding import OBSERVABLES

    (observable,) = [item for item in OBSERVABLES if item.name == "belief_history"]
    assert observable.matches(
        "name one actual position you held and then dropped, with when"
    )
    assert observable.matches("what's something you've genuinely changed your mind about?")
    assert observable.example_failures() == []


def test_it_does_not_claim_a_question_about_the_other_person():
    from core.brain.observable_registry import install_default_observables

    install_default_observables()
    from core.brain.observable_grounding import OBSERVABLES

    (observable,) = [item for item in OBSERVABLES if item.name == "belief_history"]
    assert not observable.matches("have I changed my mind about anything?")
