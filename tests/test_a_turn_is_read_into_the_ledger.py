"""What a person's turn did is written where her appraisal reads it.

Measured 17 October: six turns as different as praise, an insult, a death and a
correction appraised identically (valence +0.112, relevance 0.0), because the
appraisal reads the relational ledger and nothing wrote a person's turn into it.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from core.interiority.appraisal import AppraisalEngine
from core.interiority.effects import LedgerWrite
from core.interiority.ledger import RelationalLedger
from core.interiority.text_features import a_person_said
from core.interiority.what_a_turn_did import LABELS, TurnReader, TurnReading, note_what_it_did, read_turn


def _reader_that_feels(**strengths: float) -> TurnReader:
    """A head whose logits are set so the named labels come out at the given probabilities."""
    weights = np.zeros((len(LABELS), 2), dtype=np.float32)
    bias = np.full(len(LABELS), -20.0, dtype=np.float32)
    for name, p in strengths.items():
        bias[LABELS.index(name)] = float(np.log(p / (1.0 - p)))
    return TurnReader(weights, bias, LABELS, embedding_model="test")


def _embed(_text: str):
    return np.array([1.0, 0.0], dtype=np.float32)


def test_reading_says_what_was_felt_and_to_whom():
    reading = read_turn("You got it wrong again, you're useless.",
                        reader=_reader_that_feels(anger=0.6, annoyance=0.8, disapproval=0.5), embed=_embed)
    assert reading.addressed_to_her
    assert reading.valence is not None and reading.valence < 0
    assert 0.5 < reading.hostile_share < 1.0
    unread = read_turn("You got it wrong.", reader=None, embed=None)
    assert unread.valence is None and not unread.read


def _frame(ledger: RelationalLedger, event):
    return AppraisalEngine(ledger).appraise(event)


def test_a_correction_blocks_her_answer_and_is_hers():
    ledger = RelationalLedger()
    event = note_what_it_did(ledger, a_person_said("bryan", "No, it's 1049.", at=time.time()),
                             TurnReading(corrected_her=True, confidence=1.0), person="bryan")
    assert event.object == "her_last_answer"
    frame = _frame(ledger, event)
    assert frame.value("congruence") == -1.0
    assert frame.value("agency_self") == 1.0


def test_praise_addressed_to_her_advances_her_answer():
    ledger = RelationalLedger()
    reading = TurnReading(feelings={"gratitude": 0.9, "admiration": 0.7}, addressed_to_her=True, confidence=0.9)
    event = note_what_it_did(ledger, a_person_said("bryan", "Thank you, you did brilliant work.", at=time.time()),
                             reading, person="bryan")
    assert _frame(ledger, event).value("congruence") > 0


def test_hostility_toward_her_is_theirs_and_breaks_a_norm():
    ledger = RelationalLedger()
    reading = TurnReading(feelings={"anger": 0.6, "annoyance": 0.8, "disapproval": 0.5}, addressed_to_her=True,
                          confidence=0.8)
    event = note_what_it_did(ledger, a_person_said("bryan", "You're useless.", at=time.time()), reading,
                             person="bryan")
    frame = _frame(ledger, event)
    assert frame.value("congruence") < 0
    assert frame.value("agency_other") > frame.value("agency_self")
    assert frame.value("norm_fit") < 0


def test_their_bad_news_is_about_their_wellbeing_and_carries_their_distress():
    ledger = RelationalLedger()
    ledger.bond("bryan", 0.8)
    reading = TurnReading(feelings={"sadness": 0.9, "grief": 0.7}, confidence=0.9)
    event = note_what_it_did(ledger, a_person_said("bryan", "My dog died this morning.", at=time.time()), reading,
                             person="bryan")
    assert event.object == "bryan:wellbeing"
    assert event.observations["context"].value > 0.0
    frame = _frame(ledger, event)
    assert frame.value("congruence") < 0
    assert frame.value("relevance") > 0  # her bond with him is the stake
    assert frame.value("agency_circumstance") == 1.0


def test_nothing_read_leaves_the_event_as_it_was():
    ledger = RelationalLedger()
    event = a_person_said("bryan", "What time is it?", at=time.time())
    assert note_what_it_did(ledger, event, TurnReading(), person="bryan") is event


def test_a_ledger_write_to_the_notes_is_not_dropped():
    from core.interiority.service import InteriorityService

    service = InteriorityService.__new__(InteriorityService)
    service.ledger = RelationalLedger()
    service._absorb_ledger_writes(SimpleNamespace(ledger=(
        LedgerWrite("note_expectation", {"kind": "compliance", "subject": "bryan", "value": 0.4}),
    )))
    assert service.ledger.notes.expectation("compliance", "bryan") == 0.4


def test_the_exchange_before_a_turn_is_found_in_her_history():
    from core.orchestrator.mixins.incoming_logic import _the_exchange_before

    history = [
        {"role": "user", "content": "How many days between March 14, 2019 and January 26, 2022?"},
        {"role": "assistant", "content": "1048 days."},
        {"role": "user", "content": "No, it's 1049."},
    ]
    assert _the_exchange_before(history, "No, it's 1049.") == (history[0]["content"], "1048 days.")
    assert _the_exchange_before([], "hello") == ("", "")
