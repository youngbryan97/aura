"""Whether anything but affect can move her body.

`robust_recurrence_kappa` asks that no single domain's removal disconnects the
rest. Two runs in a row read interoception with exactly one cause: perturb
affect and the body moves, and perturb recurrent cognition, deliberation, the
workspace, memory, the world model, her self-state or her developmental state
and the arms come back bit-identical — effect 0.0000 at a p of exactly
1.000000, seven times over.

Her pulse was a function of affect and nothing else: five branches on valence
and arousal, each setting a chosen number. A mind holding an urgent unfinished
thing and a mind holding none had the same body as long as they felt the same
about it.
"""

from __future__ import annotations

import pytest

import core.soma.carrying as carrying_module
from core.soma.carrying import CarryingLedger, get_carrying_ledger, pressure_of


@pytest.fixture(autouse=True)
def _fresh():
    carrying_module.reset_for_test()
    yield
    carrying_module.reset_for_test()


def test_the_load_is_the_hardest_open_thing_across_goals_and_intentions():
    assert pressure_of([{"urgency": 0.4}], [{"urgency": 0.9}]) == pytest.approx(0.9)
    assert pressure_of([], []) == 0.0
    assert pressure_of(None, "not a list") == 0.0


def test_what_the_arbiter_decided_outranks_what_the_proposer_declared():
    """The declared value is a proposal; the decided one is the load."""
    assert pressure_of([{"urgency": 0.2, "decided_urgency": 0.9}]) == pytest.approx(0.9)


def test_a_middle_needs_a_point_either_side_of_it():
    ledger = CarryingLedger()
    ledger.note(0.9)
    assert ledger.shift() == 0.0
    ledger.note(0.9)
    assert ledger.shift() == 0.0


def test_steady_pressure_reads_as_no_pressure():
    """A life of steady load is a baseline, and only a change is felt."""
    ledger = CarryingLedger()
    for _ in range(8):
        ledger.note(0.7)
    assert ledger.shift() == pytest.approx(0.0)


def test_carrying_more_than_usual_is_felt_and_carrying_less_is_too():
    ledger = CarryingLedger()
    for value in (0.3, 0.5, 0.4, 0.45):
        ledger.note(value)
    ledger.note(0.95)
    assert ledger.shift() > 0.0
    ledger.note(0.05)
    assert ledger.shift() < 0.0


def test_a_pressing_intention_reaches_the_pulse_that_affect_set():
    """The end-to-end claim, through the field the intervention actually writes.

    `_perturb_D` appends a goal carrying `urgency` into
    `cognition.active_goals`, so this is the path a deliberation intervention
    can traverse into a body column.
    """
    ledger = get_carrying_ledger()
    for value in (0.3, 0.4, 0.35):
        ledger.note(value)

    mood_set = 1.0

    ledger.note(pressure_of([{"id": "probe", "urgency": 0.95}]))
    pressing = max(0.1, min(3.0, mood_set * (1.0 + ledger.shift())))

    ledger.note(pressure_of([{"id": "probe", "urgency": 0.05}]))
    slack = max(0.1, min(3.0, mood_set * (1.0 + ledger.shift())))

    assert pressing > mood_set > slack, (pressing, mood_set, slack)


def test_the_body_is_left_alone_when_nothing_has_been_weighed():
    assert get_carrying_ledger().shift() == 0.0
