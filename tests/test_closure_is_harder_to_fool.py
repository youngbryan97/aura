"""What the closure test can see, and what it says about what it cannot.

Every one-way column was dropped before the test ran, to be rid of the clocks.
A learning counter is one-way. So is accumulated evidence, a developmental
step, a depleting resource — all of them hidden state the core's future may
legitimately depend on, and all of them thrown away with the clocks. A closure
test that cannot see a leak cannot report one.

What a clock is is a question about provenance, not about shape. A stored
instant is dropped by its magnitude and by its name; a one-way quantity is
kept and entered as its increment, because what it carries about the next state
is how much it moved rather than where the run has got to.
"""

from __future__ import annotations

import numpy as np

from core.subject.closure import (
    CLOCK_NAMES,
    ClosureReport,
    _as_increments,
    _named_clock,
    _one_way,
    closure_gain,
    coverage,
)
from core.subject.nulls import architecture, toy_periphery, toy_recording


def test_a_name_that_says_it_is_a_moment_is_dropped() -> None:
    assert _named_clock("service.x._last_summary_refresh_at")
    assert _named_clock("organ.y.timestamp")
    assert _named_clock("service.z.last_update")
    assert CLOCK_NAMES


def test_a_counter_is_not_dropped_for_being_a_counter() -> None:
    """The four things Phase 28 names: a learning counter, accumulated
    evidence, a developmental step, a depleting resource."""
    for name in (
        "service.trainer.train_steps",
        "organ.ontogeny.steps",
        "service.evidence.total_observed",
        "organ.soma.energy_remaining",
    ):
        assert not _named_clock(name), f"{name} reads as a clock"


def test_a_one_way_column_is_entered_as_its_increment() -> None:
    rows = np.arange(200, dtype=float).reshape(-1, 1)
    rising = _one_way(rows)
    assert rising.all(), "a straight ramp is not recognised as one-way"
    stepped = _as_increments(rows, rising)
    assert np.allclose(stepped[1:], 1.0), "the increment of a unit ramp is not one"
    assert stepped.shape == rows.shape


def test_a_two_way_column_is_left_alone() -> None:
    rows = np.sin(np.linspace(0, 20, 200)).reshape(-1, 1)
    rising = _one_way(rows)
    assert not rising.any()
    assert np.allclose(_as_increments(rows, rising), rows)


def test_the_floor_is_read_from_several_permutations() -> None:
    """One draw of a permutation is one sample of the floor, and a floor read
    off a single draw is as likely to be lucky as the thing it measures."""
    system = architecture("hub", seed=5)
    report = closure_gain(
        toy_recording(system, steps=1200, seed=5),
        toy_periphery(system, steps=1200, seed=5),
        tuple(f"broker.{index}" for index in range(system.hub_width)),
        seed=5,
        draws=8,
    )
    assert report.shuffled_draws == 8
    assert report.leak > report.floor_high
    assert not report.closed


def test_a_core_is_closed_against_the_upper_tail_of_its_own_floor() -> None:
    """Not against the mean of it. A leak inside the floor's own spread has
    not established anything."""
    report = ClosureReport(
        loss_core=1.0,
        loss_core_and_periphery=0.99,
        loss_shuffled_periphery=1.0,
        leak=0.01,
        shuffled_leak=0.0,
        leak_over_shuffle=0.0,
        periphery_width=4,
        shuffled_draws=16,
        floor_high=0.02,
    )
    assert report.closed, "a leak under the floor's own upper tail read as open"


def test_the_walk_says_what_it_could_not_reach() -> None:
    """A closure result is a claim about everything outside the core."""
    seen = coverage()
    for key in ("numbers_read", "cap", "hit_the_cap", "max_depth", "reader_failures"):
        assert key in seen, f"the coverage report does not say {key}"


def test_the_report_names_what_it_differenced_and_what_it_dropped() -> None:
    system = architecture("hidden_broker", seed=5)
    report = closure_gain(
        toy_recording(system, steps=900, seed=5),
        toy_periphery(system, steps=900, seed=5),
        tuple(f"broker.{index}" for index in range(system.hub_width)),
        seed=5,
        draws=4,
    )
    as_dict = report.as_dict()
    assert "differenced" in as_dict
    assert "dropped_as_clocks" in as_dict
    assert "shuffled_draws" in as_dict
    assert "shuffled_leak_q95" in as_dict
