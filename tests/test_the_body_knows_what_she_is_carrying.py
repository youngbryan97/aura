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


def test_the_load_is_how_much_is_open_and_how_hard():
    """It was the hardest open thing, and a maximum cannot move.

    One standing intention near the top pinned it, so a probe adding a sixth
    open thing at half urgency changed nothing and displacing deliberation
    read 0.0000 into the body on every trial — bit-identical arms. Carrying
    five is more than carrying one.
    """
    assert pressure_of([{"urgency": 0.4}], [{"urgency": 0.9}]) == pytest.approx(1.3)
    assert pressure_of([], []) == 0.0
    assert pressure_of(None, "not a list") == 0.0


def test_a_probe_added_beside_a_standing_intention_moves_the_load():
    """The path the body reads. A maximum could not do this."""
    standing = [{"decided_urgency": 0.95}]
    assert pressure_of(standing + [{"urgency": 0.5}]) > pressure_of(standing)


def test_the_load_is_not_bounded_into_the_unit_interval():
    """A cap would put back the ceiling this ledger exists to remove."""
    many = [{"urgency": 0.9} for _ in range(5)]
    assert pressure_of(many) > 1.0


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


def test_a_standing_load_plus_one_more_open_thing_is_felt():
    """End to end on the field the intervention writes: `_perturb_D` appends a
    goal carrying `urgency` into `cognition.active_goals`."""
    ledger = CarryingLedger()
    standing = [{"decided_urgency": 0.95}]
    for _ in range(6):
        ledger.note(pressure_of(standing))
    assert ledger.shift() == pytest.approx(0.0)
    ledger.note(pressure_of(standing + [{"id": "probe", "urgency": 0.5}]))
    assert ledger.shift() > 0.0


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


def test_what_she_is_carrying_decides_how_wide_she_casts():
    """Active memory had one cause too: perturb affect and what is in mind
    moves, perturb deliberation and the arms come back bit-identical. What she
    is trying to do did not decide what came back to her."""
    from core.memory.intentional_retrieval import IntentionalRetriever, RetrievalIntent

    retriever = IntentionalRetriever()
    intent = RetrievalIntent(task="what did we decide", kind="general")
    ledger = get_carrying_ledger()
    unpressed = len(retriever.plan(intent).weights)

    for value in (0.3, 0.4, 0.35):
        ledger.note(value)
    ledger.note(pressure_of([{"id": "probe", "urgency": 0.95}]))
    pressed = retriever.plan(intent)

    assert len(pressed.weights) > unpressed
    assert any("carrying" in line for line in pressed.rationale)


def test_the_net_is_unchanged_while_nothing_has_a_middle():
    from core.memory.intentional_retrieval import IntentionalRetriever, RetrievalIntent

    retriever = IntentionalRetriever()
    intent = RetrievalIntent(task="what did we decide", kind="general")
    before = retriever.plan(intent)
    get_carrying_ledger().note(0.99)
    after = retriever.plan(intent)
    assert after.weights == before.weights
    assert not any("carrying" in line for line in after.rationale)
