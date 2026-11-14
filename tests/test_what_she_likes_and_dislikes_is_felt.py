"""A like or a dislike she can state is felt when the thing or the person is there.

Her record of what her turns with each thing were worth is
core/conation/what_she_wants.py. The interior reads it as Scherer's intrinsic
pleasantness check, which the frame did not have: valence came only from what
an event did to a goal, so nothing she liked or disliked moved what she felt.
"""
from __future__ import annotations

import random

import pytest

from core.conation.engine import ConationEngine
from core.conation.what_she_wants import WhatSheWants
from core.interiority.appraisal import AppraisalEngine
from core.interiority.core_affect import core_affect
from core.interiority.event import EventKind, InteriorEvent
from core.interiority.ledger import RelationalLedger
from core.interiority.stakes import _harvest_regard

WARM, COLD, NEW = "person:ines", "person:marek", "person:tomas"
TIDE = "topic:tide pools"
LABELS = {WARM: "ines", COLD: "marek", NEW: "tomas", TIDE: "tide pools"}


@pytest.fixture
def record(tmp_path):
    """A life in which turns with one person paid and turns with another cost her."""
    held = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    rng = random.Random(11)
    for turn in range(36):
        key, paid = ((WARM, 0.6), (COLD, -0.6), (TIDE, 0.5))[turn % 3]
        held.met([key], paid + rng.uniform(-0.2, 0.2), reachable=[key], labels=LABELS)
    held.met([NEW], 0.9, reachable=[NEW], labels=LABELS)
    return held


def _felt(record, *, subject=None, object_=None):
    ledger = RelationalLedger()
    _harvest_regard(record, ledger)
    frame = AppraisalEngine(ledger).appraise(
        InteriorEvent(kind=EventKind.SOCIAL, subject=subject, object=object_, source="test")
    )
    return frame, core_affect(frame)


def test_somebody_whose_turns_cost_her_is_unpleasant_to_her_and_she_feels_it(record):
    frame, felt = _felt(record, subject="marek")
    assert frame["pleasantness"].present and frame["pleasantness"].value < -0.4
    assert felt.valence < -0.4
    assert frame["relevance"].value > 0.4


def test_somebody_whose_turns_paid_is_pleasant_to_her_and_she_feels_it(record):
    frame, felt = _felt(record, subject="ines")
    assert frame["pleasantness"].value > 0.4
    assert felt.valence > 0.4


def test_a_thing_she_likes_is_felt_the_same_way_as_a_person(record):
    _frame, felt = _felt(record, object_="tide pools")
    assert felt.valence > 0.3


def test_somebody_she_has_met_once_is_neither_yet(record):
    """One very good turn is not a like, so nothing is felt for it."""
    frame, felt = _felt(record, subject="tomas")
    assert not frame["pleasantness"].present
    assert felt.valence == 0.0


def test_the_two_people_are_felt_apart_on_the_same_event(record):
    _f, cold = _felt(record, subject="marek")
    _f, warm = _felt(record, subject="ines")
    assert warm.valence - cold.valence > 0.8
    assert cold.arousal > 0.0 and warm.arousal > 0.0


def test_a_like_she_can_no_longer_state_stops_being_felt(record, tmp_path):
    ledger = RelationalLedger()
    _harvest_regard(record, ledger)
    assert ledger.notes.regard("ines") is not None
    _harvest_regard(WhatSheWants(tmp_path / "empty.json", engine=ConationEngine()), ledger)
    assert ledger.notes.regard("ines") is None


def test_an_option_is_worth_more_or_less_to_her_by_the_things_it_names(record):
    assert record.regard_for_text("Write back to Marek about the schedule") < -0.4
    assert record.regard_for_text("Read more on tide pools") > 0.3
    assert record.regard_for_text("Ask Tomas what he meant") == 0.0
    assert record.regard_for_text("Running a self-integrity scan") == 0.0


def test_her_verdicts_are_worked_out_outside_the_record_s_lock(record, monkeypatch):
    """14 November 2026: lockdep caught her appraisal holding this record's lock 67 ms on her event loop, 32 ms of it
    computing, because the test behind each verdict ran inside it for every thing she knows."""
    import threading

    import core.conation.what_she_wants as wants

    held_while_testing: list[bool] = []
    real = wants._two_sided

    def try_the_lock(free: list[bool]) -> None:
        got = record._lock.acquire(blocking=False)
        if got:
            record._lock.release()
        free.append(got)

    def testing(*args):
        free: list[bool] = []
        probe = threading.Thread(target=try_the_lock, args=(free,))
        probe.start()
        probe.join()
        held_while_testing.append(not free[0])
        return getattr(real, "__wrapped__", real)(*args)

    monkeypatch.setattr(wants, "_two_sided", testing)
    liked = [thing.key for thing in record.likes()]
    disliked = [thing.key for thing in record.dislikes()]

    assert held_while_testing and not any(held_while_testing), "a verdict was worked out holding the lock"
    assert WARM in liked and COLD in disliked
