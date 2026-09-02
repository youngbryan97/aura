"""The readings, the causes, the evidence rules, and the governance.

Each of these holds one claim that the rest of the work rests on, and most of
them are the kind that is easy to assert and easy to get wrong: that a reading
reads something, that a lesion measures rather than correlates, that a gate can
say no, that a receipt cannot be quietly rewritten.
"""

from __future__ import annotations

import pytest

from core.cognition.how_a_change_is_promoted import (
    WHAT_A_TIER_WANTS,
    forget_the_receipts,
    how_far_it_reaches,
    nothing_installs_to_the_gate,
    promote,
    put_it_back,
    the_chain_holds,
    the_receipts,
    the_stack,
)
from core.cognition.how_sure_she_is import (
    after_the_winners_curse,
    enough_families_to_say,
    how_much_to_spend_on_developing,
    more_likely_than_not_better,
    sure_enough,
    tell_her_the_drives,
    the_bar_right_now,
    which_to_try,
)
from core.cognition.the_record_of_her_own_work import (
    forget_the_record,
    note_an_episode,
    other_families,
)
from core.cognition.the_shape_of_her_library import (
    how_long_the_library_is,
    recompress,
)
from core.cognition.what_she_expects_of_herself import (
    forget_what_she_expected,
    how_long_this_will_take,
    how_well_she_knows_herself,
    what_actually_happened,
    what_she_expects,
)
from core.cognition.what_she_is_made_of import (
    the_most_they_have_in_common,
    what_she_is_made_of,
)
from core.cognition.what_she_notices_about_herself import (
    THE_READINGS,
    a_reading_she_wrote,
    forget_the_agenda,
    what_she_notices,
)
from core.cognition.what_this_reminds_her_of import (
    a_fingerprint,
    how_much_they_share,
    what_keeps_coming_up,
)
from core.cognition.why_it_is_not_better import what_a_reader_would_call_it, ACause


@pytest.fixture(autouse=True)
def a_clean_slate():
    held = dict(THE_READINGS)
    forget_the_record()
    forget_the_agenda()
    forget_the_receipts()
    forget_what_she_expected()
    tell_her_the_drives(curiosity=0.5, energy=0.5)
    yield
    THE_READINGS.clear()
    THE_READINGS.update(held)
    forget_the_record()
    forget_the_agenda()
    forget_the_receipts()
    forget_what_she_expected()


# ── the readings ─────────────────────────────────────────────────────────


def test_a_reading_reads_something_off_the_record():
    for _ in range(4):
        note_an_episode("a shape that recurs", route=None, walked=900)
    noticed = {one.noticed_by for one in what_she_notices()}
    assert "recurrence" in noticed
    assert "silence" in noticed


def test_nothing_about_a_family_is_noticed_before_any_family_is_met():
    """The readings about her parts still fire; the ones about her work do not.

    Her library exists whether or not she has answered anything, so a reading
    about it has something to say from the start. A reading about a family she
    has met cannot, and one that spoke anyway would be reading its own
    defaults.
    """
    about_her_work = {
        "recurrence",
        "expense",
        "silence",
        "unevenness",
        "slack",
        "the slow route",
        "a flat improver",
        "yield",
    }
    noticed = what_she_notices()
    assert not [one for one in noticed if one.noticed_by in about_her_work]
    assert [one for one in noticed if one.about in {"the library", "herself"}]


def test_a_fourteenth_reading_needs_no_edit():
    from core.cognition.what_she_notices_about_herself import AnOpportunity

    a_reading_she_wrote(
        "something she thought of",
        lambda: [AnOpportunity("something she thought of", "herself", 1.0)],
    )
    assert any(
        one.noticed_by == "something she thought of" for one in what_she_notices()
    )


def test_a_reading_that_raises_costs_a_reading_and_not_a_tick():
    def it_breaks() -> list:
        raise ValueError("no")

    a_reading_she_wrote("a broken one", it_breaks)
    note_an_episode("f", route=None, walked=5)
    note_an_episode("f", route=None, walked=5)
    assert any(one.noticed_by == "recurrence" for one in what_she_notices())


# ── diagnosis ────────────────────────────────────────────────────────────


def test_the_label_is_retrospective_and_reaches_no_decision():
    made = ACause(at="word/here", change="without it", make_it=lambda: False)
    assert what_a_reader_would_call_it(made) == "a representation problem"
    order = ACause(
        at="the search/the order she tries them in",
        change="a different one",
        make_it=lambda: False,
    )
    assert what_a_reader_would_call_it(order) == "a search problem"


# ── the evidence rules ───────────────────────────────────────────────────


def test_the_number_of_probes_comes_from_the_size_of_the_claim():
    assert enough_families_to_say(at_least=0.1) > enough_families_to_say(
        at_least=0.5
    )


def test_a_decision_rule_can_be_met_where_a_confidence_level_cannot():
    """The defect this replaced: a gate that always says no is not a careful one."""
    assert sure_enough([1.0] * 4, better_by=0.5)[0] is False
    assert more_likely_than_not_better(4, 4, than=0.5)[0] is True
    assert more_likely_than_not_better(2, 4, than=0.5)[0] is False


def test_a_winning_estimate_is_shrunk_by_how_many_it_beat():
    assert after_the_winners_curse(5.0, 8, spread=1.0) < 5.0
    assert after_the_winners_curse(5.0, 0, spread=1.0) == 5.0


def test_the_uncertain_one_is_still_tried():
    import random

    counts = {"sure": (18, 20), "unknown": (0, 1)}
    rng = random.Random(7)
    picks = [
        which_to_try(
            ["sure", "unknown"],
            pays=lambda one: counts[one],
            gains=lambda one: 1.0,
            rng=rng,
        )
        for _ in range(300)
    ]
    assert 0 < picks.count("unknown") < 150


def test_the_drives_move_the_bar_and_choose_nothing():
    for _ in range(3):
        note_an_episode("f", route="an answer", walked=900)
    tell_her_the_drives(curiosity=0.9, energy=0.9)
    curious = the_bar_right_now(a_question_is_waiting=True)
    tell_her_the_drives(curiosity=0.1, energy=0.1)
    tired = the_bar_right_now(a_question_is_waiting=True)
    assert curious < tired


def test_the_bar_is_nothing_when_nothing_is_waiting():
    assert the_bar_right_now() == 0.0


# ── governance ───────────────────────────────────────────────────────────


def test_the_ladder_wants_more_evidence_for_what_everything_runs_through():
    assert WHAT_A_TIER_WANTS["the deciding"] > WHAT_A_TIER_WANTS["the search"]
    assert WHAT_A_TIER_WANTS["the search"] > WHAT_A_TIER_WANTS["word"]
    assert how_far_it_reaches("the deciding/x") < how_far_it_reaches("word/x")


def test_no_change_may_install_to_the_gate():
    assert nothing_installs_to_the_gate() == []


def test_a_receipt_chain_cannot_be_quietly_rewritten():
    promote("word/x", became="shadow", started_by="she", evidence="4 of 4")
    promote("word/y", became="canary", started_by="she", evidence="5 of 5")
    assert the_chain_holds()
    assert all(one.asked_from_outside is None for one in the_receipts())


def test_going_back_is_ordinary_and_the_predecessor_is_kept():
    promote(
        "the search/the order she tries them in",
        became="canary",
        started_by="she",
        evidence="sooner",
        replaced="the one before",
    )
    assert the_stack()
    assert put_it_back("the search/the order she tries them in") == "the one before"
    assert any(one.became == "rolled back" for one in the_receipts())


def test_an_external_command_is_recorded_as_one():
    promote(
        "word/x",
        became="shadow",
        started_by="asked",
        evidence="a caller named it",
        asked_from_outside="a test",
    )
    assert the_receipts()[-1].asked_from_outside == "a test"


# ── the self-model ───────────────────────────────────────────────────────


def test_a_prediction_is_fixed_before_the_outcome():
    said = what_she_expects("a new word", costs_now=400)
    assert not said.settled
    what_actually_happened("a new word", cost=420, kept=True, gained=300)
    assert said.settled
    assert how_well_she_knows_herself()["predictions"] == 1
    assert how_well_she_knows_herself()["cost out by"] < 0.1


def test_the_forecaster_refuses_where_nothing_resembles_it():
    note_an_episode("f", route="an answer", walked=7, about=[((1, 2), (2, 1))])
    known, why = how_long_this_will_take([((1, 2), (2, 1))])
    assert known == 7
    unknown, why = how_long_this_will_take(
        [((1, 2, 3, 4, 5, 6, 7, 8, 9), (9,) * 9)]
    )
    assert unknown is None
    assert "resembles" in why


# ── shape rather than surface ────────────────────────────────────────────


def test_two_terms_over_different_numbers_have_the_same_shape():
    from core.cognition.the_floor_she_stands_on import N, PLUS, TIMES, build

    here = build(PLUS(TIMES(N(2), N(3)), N(1)))
    there = build(PLUS(TIMES(N(9), N(8)), N(7)))
    assert a_fingerprint(here) == a_fingerprint(there)
    assert how_much_they_share(here, there) == 1.0
    assert how_much_they_share(here, build(N(4))) < 1.0


def test_what_recurs_across_the_corpus_is_not_what_a_pair_share():
    from core.cognition.the_floor_she_stands_on import N, PLUS, TIMES, build

    corpus = [
        build(PLUS(TIMES(N(1), N(2)), N(3))),
        build(TIMES(N(4), N(5))),
        build(PLUS(TIMES(N(6), N(7)), N(8))),
    ]
    found = dict(what_keeps_coming_up(corpus))
    assert found.get("times(.,.)") == 3


def test_the_generalisation_of_two_terms_keeps_only_what_they_agree_on():
    from core.cognition.the_floor_she_stands_on import N, PLUS, build, written_down

    got = the_most_they_have_in_common(
        build(PLUS(N(1), N(2))), build(PLUS(N(1), N(5)))
    )
    written = written_down(got)
    assert written["head"] == "plus"
    assert written["parts"][0]["value"] == 1
    # The half they disagree about is a hole, not the first of them.
    assert written["parts"][1]["head"] == "the one it was given"


# ── the library as one thing ─────────────────────────────────────────────


def test_the_library_has_a_length_and_recompressing_reads_it():
    assert how_long_the_library_is() > 0
    assert isinstance(recompress([], costs=lambda cases: 1), list)


def test_every_part_has_one_kind_of_address():
    parts = what_she_is_made_of()
    assert parts
    assert all("/" in one.at for one in parts)
    assert {one.kind for one in parts} <= {
        "word",
        "what is done",
        "way of building",
        "way of computing",
        "rule",
        "the search",
        "the deciding",
    }
