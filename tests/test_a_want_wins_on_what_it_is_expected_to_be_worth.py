"""An option wins her slate on what her record says it will be worth, not on how loud it is.

Scored live on 13 October, a want, a self-scan, a drive alert and a
reflection read the same seven numbers in her arbiter and differed only in
the urgency each source wrote for itself.
"""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from core.agency.initiative_arbiter import InitiativeArbiter
from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from core.conation.what_she_wants import FIND_OUT, _conation, read_turn

LIMPET, SCAN = "topic:limpet", "act:running"


class _Worth:
    measured, payoff = True, {}

    def __init__(self, liking: float) -> None:
        self.worth, self.size = (1.0 if liking >= 0 else -1.0), abs(liking)


@pytest.fixture
def record(tmp_path):
    held = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    yield held
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def _a_life(record, turns: int = 24) -> None:
    """Turns with the limpet paid; turns in which she ran the scan cost her."""
    rng = random.Random(3)
    for turn in range(turns):
        key, paid = (LIMPET, 0.6) if turn % 2 == 0 else (SCAN, -0.5)
        record.met(["self", key], paid + rng.uniform(-0.2, 0.2), reachable=[LIMPET], labels={LIMPET: "limpet"})


def _state():
    return SimpleNamespace(
        cognition=SimpleNamespace(pending_initiatives=[], active_goals=[], working_memory=[]),
        affect=SimpleNamespace(valence=0.0, arousal=0.4, emotions={}),
        motivation=SimpleNamespace(budgets={}),
    )


def test_what_an_option_is_expected_to_be_worth_is_read_from_her_turns_with_it(record):
    _a_life(record)
    want = {"want": {"thing": LIMPET}}
    assert record.expected_worth("limpet", want) > 0.7
    assert record.expected_worth("Running a self-integrity scan") < 0.3
    assert record.expected_worth("Something she has never done") is None


def test_a_quiet_want_she_has_found_worth_it_outscores_a_loud_act_that_has_cost_her(record):
    _a_life(record)
    arbiter = InitiativeArbiter()
    want = {"goal": "limpet", "source": "want", "urgency": 0.2, "metadata": {"want": {"thing": LIMPET}}}
    scan = {"goal": "Running a self-integrity scan", "source": "motivation", "urgency": 0.5, "metadata": {}}

    async def scored():
        return [(await arbiter.score_initiative(option, _state())).final_score for option in (want, scan)]

    wanted, scanned = asyncio.run(scored())
    assert wanted > scanned


def test_with_no_record_the_scores_are_what_they_were(record):
    arbiter = InitiativeArbiter()
    option = {"goal": "Running a self-integrity scan", "source": "motivation", "urgency": 0.5, "metadata": {}}
    scored = asyncio.run(arbiter.score_initiative(option, _state())).to_dict()
    dimensions = scored.get("dimension_scores") or scored.get("scores")
    assert dimensions["expected_value"] == 0.5


def test_what_an_exchange_dwelt_on_pulls_harder_than_what_it_mentioned(record):
    from tests.fixtures.things_her_library_names import NamesThese

    record.talk.remember_talk([("What do you think of the plan?", "I think the plan is fine, and has been.")] * 60)
    record.about = NamesThese("limpet", "ledge")
    read_turn(
        ["self"],
        _Worth(0.3),
        record,
        said=(
            "The limpet left the scar and the limpet came back. A limpet knows its ledge, a limpet does.",
            "The limpet finds the scar by fit. On that ledge the limpet has one ledge and keeps the ledge.",
        ),
    )
    limpet, ledge = record.thing("topic:limpet"), record.thing("topic:ledge")
    assert limpet.stood_out > ledge.stood_out > 0.0
    for held in _conation(record).dynamics._satiation.values():
        held.last_update -= 600.0
    pull = {want.thing: want.utility for want in record.wants()}
    assert pull["topic:limpet"] > pull["topic:ledge"]


def test_nobody_is_offered_as_something_to_look_up(record):
    rng = random.Random(5)
    for _turn in range(12):
        record.met(["person:ines"], 0.5 + rng.uniform(-0.2, 0.2), reachable=["person:ines"], labels={"person:ines": "ines"})
    assert all(want.thing != "person:ines" for want in record.wants())
    record.went("person:ines", "speak", label="ines")
    record.met(["person:ines"], 0.6, reachable=["person:ines"], labels={"person:ines": "ines"})
    for held in _conation(record).dynamics._satiation.values():
        held.last_update -= 600.0
    ways = {want.thing: want.way for want in record.wants()}
    assert ways.get("person:ines") == "speak" and FIND_OUT not in ways.values()
