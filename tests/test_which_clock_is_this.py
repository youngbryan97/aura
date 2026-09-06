"""Six clocks, and no arithmetic that mixes two of them."""
from __future__ import annotations

import time

import pytest

from core.observability.which_clock_is_this import (
    AStamp,
    ClockDomain,
    MixedClocks,
    advance,
    domains_with_no_reader,
    how_long_between,
    how_the_clocks_stand,
    now,
    set_the_rate_of,
    where_clocks_are_mixed,
)


def test_subtracting_across_domains_is_refused_not_silently_plausible() -> None:
    wall = now(ClockDomain.WALL)
    mono = now(ClockDomain.MONOTONIC)
    with pytest.raises(MixedClocks, match="plausible number and a wrong one"):
        _ = wall - mono
    with pytest.raises(MixedClocks):
        how_long_between(mono, wall)


def test_comparison_across_domains_is_refused_too() -> None:
    with pytest.raises(MixedClocks):
        _ = now(ClockDomain.WALL) < now(ClockDomain.SUBJECTIVE)


def test_within_one_domain_the_difference_is_a_plain_number() -> None:
    a = now(ClockDomain.MONOTONIC)
    time.sleep(0.01)
    assert (now(ClockDomain.MONOTONIC) - a) >= 0.005


def test_a_subjective_clock_runs_at_the_rate_cognition_sets() -> None:
    set_the_rate_of(ClockDomain.SUBJECTIVE, 4.0)
    try:
        a = now(ClockDomain.SUBJECTIVE)
        time.sleep(0.05)
        elapsed = now(ClockDomain.SUBJECTIVE) - a
        assert elapsed > 0.10, "four times 0.05s is 0.20s of felt time"
    finally:
        set_the_rate_of(ClockDomain.SUBJECTIVE, 1.0)


def test_an_hour_of_silence_is_no_distance_on_the_conversation_clock() -> None:
    a = now(ClockDomain.CONVERSATION)
    time.sleep(0.05)
    assert now(ClockDomain.CONVERSATION) - a == 0.0
    advance(ClockDomain.CONVERSATION, 1)
    assert now(ClockDomain.CONVERSATION) - a == 1.0


def test_the_model_budget_is_spent_not_elapsed() -> None:
    a = now(ClockDomain.MODEL_BUDGET)
    time.sleep(0.05)
    assert now(ClockDomain.MODEL_BUDGET) - a == 0.0
    advance(ClockDomain.MODEL_BUDGET, 512)
    assert now(ClockDomain.MODEL_BUDGET) - a == 512.0


def test_a_rate_change_does_not_rewrite_what_already_elapsed() -> None:
    set_the_rate_of(ClockDomain.SIMULATION, 1.0)
    a = now(ClockDomain.SIMULATION)
    time.sleep(0.05)
    at_one = now(ClockDomain.SIMULATION) - a
    set_the_rate_of(ClockDomain.SIMULATION, 100.0)
    try:
        assert (now(ClockDomain.SIMULATION) - a) >= at_one
    finally:
        set_the_rate_of(ClockDomain.SIMULATION, 1.0)


def test_the_machines_own_clocks_cannot_be_driven_by_hand() -> None:
    for domain in (ClockDomain.WALL, ClockDomain.MONOTONIC):
        with pytest.raises(ValueError):
            advance(domain, 1.0)
        with pytest.raises(ValueError):
            set_the_rate_of(domain, 2.0)


def test_no_expression_local_or_field_mixes_two_clocks() -> None:
    """The measured state of the tree. It is zero; this keeps it zero."""
    mixed = where_clocks_are_mixed()
    assert mixed["in_one_expression"] == [], mixed["in_one_expression"][:5]
    assert mixed["through_a_local"] == [], mixed["through_a_local"][:5]
    assert mixed["through_a_field"] == [], mixed["through_a_field"][:5]


def test_a_domain_nothing_reads_is_named() -> None:
    seen = how_the_clocks_stand()
    assert set(seen["domains"]) == {str(d) for d in ClockDomain}
    assert isinstance(seen["with_no_reader"], list)
    assert seen["clean"] is True


def test_a_stamp_says_which_clock_it_came_from() -> None:
    stamp = now(ClockDomain.WALL)
    assert isinstance(stamp, AStamp)
    assert stamp.domain is ClockDomain.WALL
    assert "@wall" in str(stamp)


def test_domains_with_no_reader_shrinks_once_something_reads_one() -> None:
    now(ClockDomain.SIMULATION)
    assert "simulation" not in domains_with_no_reader()
