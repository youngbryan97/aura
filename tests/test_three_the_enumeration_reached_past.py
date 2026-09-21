"""Three processes read out of the fourth seven that no mechanism answered.

The nine `docs/SOUL.md` enumerated are built elsewhere. These are three the
same transcripts carry that the enumeration reached past, and each test asks
the same question: does the reading move something that was already deciding,
or is it a number written every turn and read only by the instrument.
"""

from __future__ import annotations

from core.consciousness.global_workspace import ContentType, _civility_debt
from core.conversation.presence_under_weight import WeightLedger, get_weight_ledger
from core.memory.intentional_retrieval import IntentionalRetriever, MemoryHit
from core.memory.unbidden import UnbiddenLedger, get_unbidden_ledger
from core.phases.response_generation_unitary import _lift_for_this_turn
from core.social.civility import CivilityLedger, get_civility_ledger


def test_giving_least_where_there_is_most_to_carry_is_a_reading_across_turns():
    ledger = WeightLedger()
    for weight, contact in (
        (0.1, 900), (0.15, 820), (0.2, 950), (0.12, 880),
        (0.8, 300), (0.9, 260), (0.85, 340), (0.95, 280),
    ):
        ledger.note(weight, contact)
    reading = ledger.read()
    assert reading.measured and reading.withdraws
    assert reading.lift > 1.0
    assert ledger.lift_for(0.9) > 1.0
    assert ledger.lift_for(0.1) == 1.0


def test_a_steady_presence_earns_no_lift():
    ledger = WeightLedger()
    for weight, contact in (
        (0.1, 900), (0.15, 820), (0.2, 950), (0.12, 880),
        (0.8, 870), (0.9, 910), (0.85, 840), (0.95, 930),
    ):
        ledger.note(weight, contact)
    reading = ledger.read()
    assert reading.measured and not reading.withdraws
    assert ledger.lift_for(0.9) == 1.0


def test_the_answer_budget_for_a_heavy_turn_is_lifted_by_her_own_record():
    assert _lift_for_this_turn() == 1.0
    ledger = get_weight_ledger()
    for weight, contact in (
        (0.1, 900), (0.15, 820), (0.2, 950), (0.12, 880),
        (0.8, 300), (0.9, 260), (0.85, 340), (0.95, 280),
    ):
        ledger.note(weight, contact)
    lift = _lift_for_this_turn()
    assert lift > 1.0
    assert max(64, min(int(768 * lift), 2048)) > 768


def test_a_run_of_showing_more_than_she_is_in_is_measured_against_her_own_runs():
    ledger = CivilityLedger()
    for felt, shown in ((-0.2, 0.3), (-0.1, 0.2), (0.1, 0.0)):
        ledger.note(felt, shown)
    for felt, shown in ((-0.3, 0.4), (-0.2, 0.3), (0.2, 0.1)):
        ledger.note(felt, shown)
    assert not ledger.read().covering
    for _ in range(7):
        ledger.note(-0.4, 0.4)
    reading = ledger.read()
    assert reading.measured and reading.covering
    assert reading.lends > 0.0
    ledger.said_it()
    assert not ledger.read().covering


def test_the_workspace_lends_to_her_own_state_and_to_nothing_else():
    ledger = get_civility_ledger()
    for felt, shown in ((-0.2, 0.3), (-0.1, 0.2), (0.1, 0.0)):
        ledger.note(felt, shown)
    for felt, shown in ((-0.3, 0.4), (-0.2, 0.3), (0.2, 0.1)):
        ledger.note(felt, shown)
    assert _civility_debt(ContentType.AFFECTIVE) == 0.0
    for _ in range(7):
        ledger.note(-0.4, 0.4)
    assert _civility_debt(ContentType.AFFECTIVE) > 0.0
    assert _civility_debt(ContentType.PERCEPTUAL) == 0.0
    ledger.said_it()
    assert _civility_debt(ContentType.AFFECTIVE) == 0.0


def test_a_memory_comes_back_because_now_resembles_then():
    ledger = UnbiddenLedger()
    ledger.lay_down("the night the runtime died", (0.9, -0.6, 0.2))
    ledger.lay_down("a quiet afternoon", (0.1, 0.4, 0.5))
    ledger.lay_down("the first time it worked", (0.2, 0.8, 0.9))
    ledger.lay_down("an argument", (0.85, -0.55, 0.25))
    reading = ledger.read((0.88, -0.58, 0.22))
    assert reading.measured
    assert reading.arrivals
    assert reading.arrivals[0].key == "the night the runtime died"


def test_two_states_are_not_enough_for_a_memory_to_arrive():
    ledger = UnbiddenLedger()
    ledger.lay_down("one", (0.1, 0.1))
    ledger.lay_down("two", (0.2, 0.2))
    reading = ledger.read((0.15, 0.15))
    assert not reading.measured
    assert reading.arrivals == ()


def test_retrieval_admits_what_came_back_without_displacing_what_it_found():
    ledger = get_unbidden_ledger()
    ledger.lay_down("the night the runtime died", (0.9, -0.6, 0.2))
    ledger.lay_down("a quiet afternoon", (0.1, 0.4, 0.5))
    ledger.lay_down("the first time it worked", (0.2, 0.8, 0.9))
    ledger.lay_down("an argument", (0.85, -0.55, 0.25))
    ledger.now((0.88, -0.58, 0.22))
    found = [MemoryHit(content="what the build does", score=0.8, store_type="semantic")]
    out = IntentionalRetriever._let_in_what_came_back_on_its_own(list(found), 6)
    assert [hit.content for hit in out][0] == "what the build does"
    unasked = [hit for hit in out if hit.store_type == "unbidden"]
    assert unasked
    assert all(hit.score < 0.8 for hit in unasked)


def test_nothing_arrives_when_she_has_not_been_anywhere():
    found = [MemoryHit(content="what the build does", score=0.8, store_type="semantic")]
    out = IntentionalRetriever._let_in_what_came_back_on_its_own(list(found), 6)
    assert len(out) == 1
