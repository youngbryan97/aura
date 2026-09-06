"""Time does not move unless somebody moves it, and a receipt says so.

Soar can be told to run for one phase, one decision or N cycles, and that is
why its experiments reproduce. Aura's runtime is continuous and event driven,
which is right for a thing living on a desktop and wrong for a measurement.

Three measurements in this repository have already been reversed by the clock
rather than by what was being measured: a search bounded by seconds gave a
family on an idle host and refused it on a loaded one, an ablation flipped
twice under adaptive depth, and one run of eighty-seven tasks took 414 seconds
while the next would not finish.
"""

from __future__ import annotations

import pytest

from core.runtime.the_laboratory import (
    in_the_laboratory,
    now,
    seeded,
    the_laboratory,
    under_the_laboratory,
    what_still_reads_the_wall_clock,
)


def test_time_does_not_move_on_its_own():
    with under_the_laboratory():
        first = now()
        for _ in range(1000):
            pass
        assert now() == first, "the clock moved without anybody moving it"


def test_advancing_runs_what_came_due_in_a_fixed_order():
    with under_the_laboratory() as lab:
        ran: list[str] = []
        lab.due_in(3, lambda: ran.append("three"), name="three")
        lab.due_in(1, lambda: ran.append("one"), name="one")
        lab.due_in(2, lambda: ran.append("two"), name="two")
        assert ran == [], "work ran before its time"
        assert lab.advance(2.5) == ("one", "two")
        assert ran == ["one", "two"]
        lab.advance(1)
        assert ran == ["one", "two", "three"]


def test_two_things_due_at_once_run_in_the_order_they_were_registered():
    """A tie broken on anything else is a run that does not reproduce."""

    with under_the_laboratory() as lab:
        ran: list[str] = []
        for name in ("first", "second", "third"):
            lab.due_in(1, lambda n=name: ran.append(n), name=name)
        lab.advance(1)
        assert ran == ["first", "second", "third"]


def test_a_step_does_not_move_the_clock():
    with under_the_laboratory() as lab:
        before = now()
        lab.step(lambda: None, name="a phase")
        lab.step(lambda: None, name="another phase")
        assert lab.steps == 2
        assert now() == before


def test_the_seed_is_fixed_and_salted_per_caller():
    with under_the_laboratory(seed=11):
        assert seeded("a").random() == seeded("a").random()
        assert seeded("a").random() != seeded("b").random(), (
            "two subsystems drawing from the seed drew the same numbers"
        )


def test_outside_the_laboratory_the_clock_is_the_wall_clock():
    assert not in_the_laboratory()
    assert the_laboratory() is None
    before = dict(what_still_reads_the_wall_clock())
    now(whose="a test")
    after = what_still_reads_the_wall_clock()
    assert after.get("a test", 0) == before.get("a test", 0) + 1


def test_nesting_is_refused_rather_than_shared():
    with under_the_laboratory():
        with pytest.raises(RuntimeError):
            with under_the_laboratory():
                pass


def test_a_gate_run_with_the_clocks_held_says_so_on_its_receipt():
    from tools.agi_gauntlet.gates import THE_GATES, run_a_gate
    from tools.agi_gauntlet.protocol import take_the_freeze

    freeze = take_the_freeze()
    gate = next(one for one in THE_GATES if one.number == 13)
    held = run_a_gate(gate, freeze, {"questions": 3, "hold_the_clocks": True})
    loose = run_a_gate(gate, freeze, {"questions": 3})
    assert held.ran and held.clocks_held is True
    assert loose.ran and loose.clocks_held is False
    assert "clocks_held" in held.to_dict()


def test_the_report_says_what_ran_and_what_is_waiting():
    with under_the_laboratory(seed=3) as lab:
        lab.due_in(5, lambda: None, name="later")
        lab.step(lambda: None, name="now")
        said = lab.report()
        assert said["in_the_laboratory"] is True
        assert said["seed"] == 3
        assert said["steps"] == 1
        assert [one["name"] for one in said["waiting"]] == ["later"]
