"""Machinery that cannot reach her is the defect, not the capability.

The emergent-goal engine was fed by production code for months and its
synthesise-and-adopt progression had no live caller. A measure she can invent
would have been the same story: a beautiful loop in a harness, and nothing in
her actual life that runs it.

So the invention happens in the run. Situations and how they turned out are
gathered where both exist, a property is proposed at the end of a run when the
outcomes are finally known, and it goes on TRIAL rather than straight into her
judgement — because nothing replays a life. It keeps its place only if things
actually go better with it.
"""

from __future__ import annotations

import pytest

from core.agency.how_good_is_this import (
    A_FAIR_TRIAL,
    INVENTED,
    ON_TRIAL,
    forget,
    how_the_trial_is_going,
    on_trial,
    what_it_was_like_before,
)
from core.agency.inventing_a_measure import Measure

SOMETHING = Measure("neighbours", "the gap between them", "on average", True)


@pytest.fixture(autouse=True)
def _nothing_left_behind():
    held, trials = dict(INVENTED), dict(ON_TRIAL)
    INVENTED.clear()
    ON_TRIAL.clear()
    yield
    INVENTED.clear()
    INVENTED.update(held)
    ON_TRIAL.clear()
    ON_TRIAL.update(trials)


def a_trial(*, was, becomes):
    name = on_trial(SOMETHING, 0.4)
    what_it_was_like_before(name, was)
    verdict = ""
    for _ in range(A_FAIR_TRIAL):
        verdict = how_the_trial_is_going(name, becomes)
    return name, verdict


# ── it is tried, not adopted ─────────────────────────────────────────────

def test_a_property_on_trial_is_already_being_judged_by():
    """It has to be in her judgement for the trial to mean anything."""
    name = on_trial(SOMETHING, 0.4)
    assert name in INVENTED
    assert name in ON_TRIAL


def test_one_that_makes_things_better_keeps_its_place():
    name, verdict = a_trial(was=1.0, becomes=2.0)
    assert verdict == "kept"
    assert name in INVENTED


def test_one_that_makes_things_worse_is_dropped():
    name, verdict = a_trial(was=5.0, becomes=1.0)
    assert verdict == "dropped"
    assert name not in INVENTED


def test_a_verdict_is_not_reached_early():
    name = on_trial(SOMETHING, 0.4)
    what_it_was_like_before(name, 1.0)
    for _ in range(A_FAIR_TRIAL - 1):
        assert how_the_trial_is_going(name, 9.0) == ""


def test_a_trial_that_is_over_says_nothing_more():
    name, _verdict = a_trial(was=1.0, becomes=2.0)
    assert how_the_trial_is_going(name, 2.0) == ""


def test_nothing_is_on_trial_until_something_is_put_on_one():
    assert ON_TRIAL == {}
    assert how_the_trial_is_going("never heard of it", 1.0) == ""


def test_what_it_was_like_before_is_recorded_once():
    name = on_trial(SOMETHING, 0.4)
    what_it_was_like_before(name, 1.0)
    what_it_was_like_before(name, 99.0)
    assert ON_TRIAL[name]["before"] == 1.0
    forget(name)


# ── and the run is where it happens ──────────────────────────────────────

def test_the_pursuit_gathers_what_it_cannot_explain():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    assert "cannot_explain.been_here(" in source


def test_and_proposes_something_when_the_run_ends():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    assert "cannot_explain.worth_trying(" in source
    assert "on_trial(measure" in source


def test_and_grades_the_trial_as_it_goes():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    assert "_how_the_trial_is_going(trying[" in source


def test_a_trial_is_long_enough_that_one_lucky_run_does_not_decide():
    assert A_FAIR_TRIAL >= 30
