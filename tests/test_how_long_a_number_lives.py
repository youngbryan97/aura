"""A count means one thing since boot and another this turn.

Soar makes reset semantics explicit and the closure asked for the same: every
metric declares a lifetime domain, and reset APIs operate by domain and never
silently mix them.

Aura accumulated many counters and almost none says what it counts over. A
reader who guesses wrong draws the opposite conclusion, and a reset that
clears everything takes the lifetime counters with it.
"""
from __future__ import annotations

import pytest

from core.observability.how_long_a_number_lives import (
    HowLong,
    declare_a_number,
    forget_everything,
    how_the_numbers_stand,
    reset_the_numbers_for,
    what_has_no_declared_lifetime,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_a_counter_says_what_it_counts_over():
    declare_a_number("routes.offered", HowLong.BOOT, "answer routes offered")
    stood = how_the_numbers_stand()
    assert stood["declared"] == 1
    assert stood["by_domain"] == {"boot": 1}


def test_a_counter_that_does_not_say_what_it_counts_is_refused():
    with pytest.raises(ValueError, match="does not say what it counts"):
        declare_a_number("x", HowLong.BOOT, "   ")


def test_a_lifetime_counter_may_not_be_given_a_way_to_clear_it():
    """That is the whole content of the word."""
    with pytest.raises(ValueError, match="not a lifetime counter"):
        declare_a_number("turns", HowLong.LIFETIME, "turns", clear=lambda: None)


def test_resetting_a_domain_touches_only_that_domain():
    held = {"boot": 5, "session": 7}
    declare_a_number(
        "b", HowLong.BOOT, "a boot count", clear=lambda: held.update(boot=0)
    )
    declare_a_number(
        "s", HowLong.SESSION, "a session count", clear=lambda: held.update(session=0)
    )

    assert reset_the_numbers_for(HowLong.BOOT) == ["b"]
    assert held == {"boot": 0, "session": 7}


def test_a_lifetime_counter_is_never_cleared():
    declare_a_number("t", HowLong.LIFETIME, "turns she has answered")
    declare_a_number("b", HowLong.BOOT, "since boot", clear=lambda: None)
    assert reset_the_numbers_for(HowLong.LIFETIME) == []


def test_resetting_a_domain_nobody_declared_is_refused():
    """One that silently matches nothing looks exactly like one that worked."""
    declare_a_number("b", HowLong.BOOT, "since boot")
    with pytest.raises(KeyError, match="nothing declared the turn domain"):
        reset_the_numbers_for(HowLong.TURN)


def test_the_refusal_says_what_was_declared():
    declare_a_number("b", HowLong.BOOT, "since boot")
    with pytest.raises(KeyError) as caught:
        reset_the_numbers_for(HowLong.SESSION)
    assert "boot" in str(caught.value)


def test_one_stuck_counter_does_not_stop_the_others():
    cleared = []

    def angry():
        raise RuntimeError("this one will not clear")

    declare_a_number("stuck", HowLong.BOOT, "a stuck count", clear=angry)
    declare_a_number(
        "fine", HowLong.BOOT, "a fine count", clear=lambda: cleared.append(True)
    )
    assert reset_the_numbers_for(HowLong.BOOT) == ["fine"]
    assert cleared == [True]


def test_declaring_twice_replaces_rather_than_duplicating():
    declare_a_number("b", HowLong.BOOT, "first")
    declare_a_number("b", HowLong.SESSION, "second")
    assert how_the_numbers_stand()["by_domain"] == {"session": 1}


def test_a_counter_records_how_often_it_was_cleared():
    declare_a_number("b", HowLong.BOOT, "since boot", clear=lambda: None)
    reset_the_numbers_for(HowLong.BOOT)
    reset_the_numbers_for(HowLong.BOOT)
    rows = how_the_numbers_stand()["numbers"]["boot"]
    assert rows[0]["cleared"] == 2


def test_undeclared_counters_can_be_named():
    declare_a_number("known", HowLong.BOOT, "a known count")
    assert what_has_no_declared_lifetime(["known", "a stranger"]) == ["a stranger"]
    assert what_has_no_declared_lifetime(None) == []


def test_the_four_domains_do_not_nest():
    assert {str(one) for one in HowLong} == {"turn", "session", "boot", "lifetime"}


def test_the_meaning_travels_with_the_counts():
    declare_a_number("b", HowLong.BOOT, "since boot")
    assert "opposite conclusion" in how_the_numbers_stand()["what_this_means"]
