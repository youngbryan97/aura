"""A level she holds, what breaks through it, and a breath that runs out.

John Legend holds 80 Hz for the whole of "Used to Love U" and reaches 917 Hz
exactly twice, on the two admissions. Oddisee locks pitch and level for four
minutes of listing contradictions and releases both on the last line. Phony Ppl
sing the grievance at 322 Hz and the attachment at 93 Hz one minute apart, so
the register carries what the feeling is about rather than how strong it is.
And Sam Cooke's voiced fraction drops from 0.9 to 0.68 on one line, which is
the evidence rather than a bad take.

Every utterance she made was delivered identically. These are the readings that
stop that being true.
"""

from __future__ import annotations

import pytest

from core.expression.delivery import (
    LEVEL,
    LIFTED,
    LOWERED,
    Delivery,
    DeliveryLedger,
    read_delivery,
)
from core.state.aura_state import AuraState


def _affect(**kwargs):
    state = AuraState.default()
    for key, value in kwargs.items():
        setattr(state.affect, key, value)
    return state.affect


def _hold(ledger: DeliveryLedger, levels) -> None:
    for value in levels:
        read_delivery(_affect(valence=value), ledger=ledger, exertion=1.0)


def test_a_reading_inside_the_level_she_holds_is_not_a_breakthrough() -> None:
    led = DeliveryLedger()
    _hold(led, (0.20, 0.25, 0.18, 0.22, 0.21, 0.19))
    reading = read_delivery(_affect(valence=0.21), ledger=led, exertion=1.0, note=False)
    assert reading.measured
    assert not reading.breakthrough
    assert "within the level" in reading.why


def test_something_outside_her_own_variation_breaks_through() -> None:
    led = DeliveryLedger()
    _hold(led, (0.20, 0.25, 0.18, 0.22, 0.21, 0.19))
    reading = read_delivery(_affect(valence=0.9), ledger=led, exertion=1.0, note=False)
    assert reading.breakthrough
    assert reading.z > 1.0
    assert "above the level" in reading.why


def test_the_bar_is_her_own_variation_not_a_chosen_number() -> None:
    """A steady life makes a small change a breakthrough; a turbulent one does not."""
    steady = DeliveryLedger()
    _hold(steady, (0.50, 0.51, 0.49, 0.50, 0.51, 0.49))
    turbulent = DeliveryLedger()
    _hold(turbulent, (0.1, 0.9, 0.2, 0.8, 0.3, 0.7))
    probe = _affect(valence=0.62)
    assert read_delivery(probe, ledger=steady, exertion=1.0, note=False).breakthrough
    assert not read_delivery(probe, ledger=turbulent, exertion=1.0, note=False).breakthrough


def test_with_too_little_history_it_says_so_rather_than_claiming_a_level() -> None:
    led = DeliveryLedger()
    read_delivery(_affect(valence=0.4), ledger=led, exertion=1.0)
    reading = read_delivery(_affect(valence=0.9), ledger=led, exertion=1.0, note=False)
    assert not reading.measured
    assert not reading.breakthrough
    assert "readings needed" in reading.why


def test_the_breath_shortens_as_she_works_harder() -> None:
    """Four hundred characters is what the effort ledger prices an ordinary turn at."""
    from core.soma.effort import UNIT_COST

    unit = UNIT_COST["response_chars"]
    ordinary = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=1.0)
    hard = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=3.0)
    idle = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=0.25)
    assert ordinary.phrase_budget == int(unit)
    assert hard.phrase_budget < ordinary.phrase_budget
    assert idle.phrase_budget > ordinary.phrase_budget
    assert int(unit / 4) <= hard.phrase_budget
    assert idle.phrase_budget <= int(unit * 2)


def test_an_unremarkable_turn_read_off_the_ledger_gets_one_breath() -> None:
    """The ledger squashes exertion so an unremarkable turn reads a half.

    Read directly, that half doubled every ordinary turn's breath. The unit is
    what an unremarkable turn produces, so noting exactly one unit of every
    kind has to come back as exactly one breath.
    """
    from core.expression.delivery import ledger_exertion
    from core.soma.effort import UNIT_COST, EffortLedger

    book = EffortLedger()
    for kind, unit in UNIT_COST.items():
        book.note(kind, unit)
    assert book.exertion(book.peek()) == pytest.approx(0.5)
    assert ledger_exertion(book) == pytest.approx(1.0)
    reading = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=ledger_exertion(book))
    assert reading.phrase_budget == int(UNIT_COST["response_chars"])


def test_a_cycle_with_nothing_spent_breathes_longest() -> None:
    from core.expression.delivery import ledger_exertion
    from core.soma.effort import UNIT_COST, EffortLedger

    assert ledger_exertion(EffortLedger()) == 0.0
    reading = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=0.0)
    assert reading.phrase_budget == int(UNIT_COST["response_chars"] * 2)


def test_control_falls_as_effort_rises() -> None:
    easy = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=1.0)
    hard = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=4.0)
    assert easy.steadiness == pytest.approx(1.0)
    assert 0.0 < hard.steadiness < 0.5


def test_something_that_failed_to_be_what_it_seemed_lifts_the_register() -> None:
    reading = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=1.0, surprise=0.8)
    assert reading.direction == LIFTED
    assert reading.lift > 0.0


def test_something_that_held_lowers_it() -> None:
    reading = read_delivery(
        _affect(confirmation=0.9), ledger=DeliveryLedger(), exertion=1.0, surprise=0.0
    )
    assert reading.direction == LOWERED
    assert reading.lift < 0.0


def test_neither_leaves_it_level() -> None:
    assert read_delivery(_affect(), ledger=DeliveryLedger(), exertion=1.0).direction == LEVEL


def test_the_strongest_feeling_sets_the_reach_rather_than_an_average() -> None:
    """The loudest moment of a record is driven by one feeling, not by a blend."""
    reading = read_delivery(
        _affect(valence=0.0, ambivalence=0.9), ledger=DeliveryLedger(), exertion=1.0
    )
    assert reading.reach == pytest.approx(0.9)
    assert reading.driver == "ambivalence"


def test_arousal_is_read_as_a_departure_from_rest() -> None:
    at_rest = read_delivery(_affect(arousal=0.5), ledger=DeliveryLedger(), exertion=1.0)
    roused = read_delivery(_affect(arousal=1.0), ledger=DeliveryLedger(), exertion=1.0)
    assert at_rest.reach == pytest.approx(0.0)
    assert roused.reach == pytest.approx(1.0)


def test_a_thing_that_asks_leaves_the_space_rather_than_answering_itself() -> None:
    """Read off the utterance's own shape rather than guessed."""
    asking = read_delivery(
        _affect(), ledger=DeliveryLedger(), exertion=1.0,
        said="Do you remember when we first talked about this? What did you think?",
    )
    telling = read_delivery(
        _affect(), ledger=DeliveryLedger(), exertion=1.0,
        said="I was born by the river and I have been running ever since.",
    )
    assert asking.answer_slot
    assert not telling.answer_slot


def test_nothing_said_yet_leaves_no_slot() -> None:
    assert not read_delivery(_affect(), ledger=DeliveryLedger(), exertion=1.0).answer_slot


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = read_delivery(_affect(), ledger=DeliveryLedger(), exertion=1.0).as_dict()
    for key in ("reach", "driver", "baseline", "spread", "z", "breakthrough",
                "direction", "lift", "phrase_budget", "steadiness", "answer_slot",
                "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Delivery().measured
    assert not Delivery().breakthrough
    assert Delivery().direction == LEVEL
