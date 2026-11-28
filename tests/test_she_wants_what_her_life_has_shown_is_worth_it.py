"""She wants what her own turns have shown is worth it, and can say what she likes.

Each test is a small life: turns with things in them and what each turn was
worth. The record under test is core/conation/what_she_wants.py.
"""
from __future__ import annotations

import random

import pytest

from core.conation.engine import ConationEngine
from core.conation.what_she_wants import FIND_OUT, WhatSheWants, _conation, liking_of
from core.runtime.atomic_writer import flush_writes_behind

TIDE, LOGS, LICHEN = "topic:tide pools", "topic:log files", "topic:lichen"
LABELS = {TIDE: "tide pools", LOGS: "log files", LICHEN: "lichen"}


@pytest.fixture
def record(tmp_path):
    return WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())


def _live(record, paid: dict[str, float], *, turns: int = 24, seed: int = 7, way: str = "") -> None:
    """Turns that take the things in `paid` one after another, each worth about what it says."""
    rng = random.Random(seed)
    keys = list(paid)
    for turn in range(turns):
        key = keys[turn % len(keys)]
        if way:
            record.went(key, way, label=LABELS.get(key, ""))
        record.met(["self", key], paid[key] + rng.uniform(-0.2, 0.2), reachable=[key], labels=LABELS)


def _rested(record) -> None:
    """Ten minutes on, so having just had a thing no longer holds the want down."""
    for held in _conation(record).dynamics._satiation.values():
        held.last_update -= 600.0


def test_a_thing_whose_turns_paid_is_wanted_and_one_whose_turns_cost_her_is_not(record):
    _live(record, {TIDE: 0.6, LOGS: -0.5})
    _rested(record)
    wanted = [want.thing for want in record.wants()]
    assert wanted[0] == TIDE
    assert LOGS not in wanted


def test_the_want_follows_the_record_and_not_the_name(tmp_path):
    """Two lives that differ only in which thing paid want different things."""
    wanted = []
    for paid in ({TIDE: 0.6, LOGS: -0.5}, {TIDE: -0.5, LOGS: 0.6}):
        record = WhatSheWants(tmp_path / f"{len(wanted)}.json", engine=ConationEngine())
        _live(record, paid)
        _rested(record)
        wanted.append(record.wants()[0].thing)
    assert wanted == [TIDE, LOGS]


def test_she_can_say_what_she_likes_and_dislikes_and_what_she_cannot_say_yet(record):
    _live(record, {TIDE: 0.6, LOGS: -0.5})
    record.met(["self", LICHEN], 0.9, reachable=[LICHEN], labels=LABELS)
    assert [thing.key for thing in record.likes()] == [TIDE]
    assert [thing.key for thing in record.dislikes()] == [LOGS]
    assert record.known(LICHEN) is None


def test_one_good_turn_is_not_a_like(record):
    record.met(["self", TIDE], 1.0, reachable=[TIDE], labels=LABELS)
    assert record.known(TIDE) is None
    assert record.likes() == []


def test_a_turn_with_no_reading_teaches_nothing(record):
    assert liking_of(None) is None
    assert record.met(["self", TIDE], None, reachable=[TIDE], labels=LABELS) == {"read": False}
    assert record.thing(TIDE) is None


def test_a_thing_that_only_arrives_with_one_she_already_likes_earns_nothing_of_its_own(record):
    for _ in range(20):
        record.met([TIDE], 0.6, reachable=[TIDE], labels=LABELS)
    for _ in range(20):
        record.met([TIDE, LICHEN], 0.6, reachable=[TIDE, LICHEN], labels=LABELS)
    assert record.thing(TIDE).liking > 0.5
    assert abs(record.thing(LICHEN).liking) < 0.05


def test_a_new_thing_beside_a_familiar_one_takes_most_of_a_surprise(record):
    for _ in range(20):
        record.met([TIDE], 0.0, reachable=[TIDE], labels=LABELS)
    read = record.met([TIDE, LICHEN], 0.8, reachable=[TIDE, LICHEN], labels=LABELS)
    assert read["shares"][LICHEN] > 4 * read["shares"][TIDE]


def test_having_just_had_it_holds_the_want_down_and_it_comes_back(record):
    _live(record, {TIDE: 0.6, LOGS: -0.5})
    just_after = record.wants()[0].wanting
    _rested(record)
    later = record.wants()[0].wanting
    assert just_after < later
    assert later > 0.1


def test_what_she_met_is_still_hers_after_a_restart(record, tmp_path):
    _live(record, {TIDE: 0.6, LOGS: -0.5})
    flush_writes_behind()
    again = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    assert [thing.key for thing in again.likes()] == [TIDE]
    assert [thing.key for thing in again.dislikes()] == [LOGS]
    assert again.thing(TIDE).liking == pytest.approx(record.thing(TIDE).liking)
    assert again.wants()[0].thing == TIDE


def test_she_goes_the_way_that_paid_and_tries_on_a_new_thing_a_way_that_paid_elsewhere(record):
    _live(record, {TIDE: 0.6}, way="web_search")
    assert record.thing(TIDE).ways["web_search"].times == 24
    record.met(["self", LICHEN], 0.1, reachable=[LICHEN], labels=LABELS)
    _rested(record)
    ways = {want.thing: want.way for want in record.wants()}
    assert ways[TIDE] == "web_search"
    assert ways[LICHEN] == "web_search"


def test_with_no_way_ever_taken_the_only_way_is_to_find_out(record):
    _live(record, {TIDE: 0.6})
    _rested(record)
    assert record.wants()[0].way == FIND_OUT


def test_what_she_left_unfinished_is_what_she_wants_to_finish(record):
    _live(record, {TIDE: 0.6})
    record.began(TIDE, "act-1", "Draw the zones of a tide pool from low water to the splash line")
    _rested(record)
    want = record.wants()[0]
    assert (want.kind, want.unfinished) == (
        "finish",
        "Draw the zones of a tide pool from low water to the splash line",
    )
    record.ended(TIDE, "act-1", finished=True)
    assert record.wants()[0].kind != "finish"


def test_a_thing_she_cannot_go_to_shares_the_credit_and_is_never_a_want(record):
    _live(record, {TIDE: 0.6})
    _rested(record)
    assert record.thing("self").contacts == 24
    assert all(want.thing != "self" for want in record.wants())


def test_every_want_says_what_it_was_read_from(record):
    _live(record, {TIDE: 0.6, LOGS: -0.5})
    _rested(record)
    want = record.wants()[0]
    assert "12 contacts, 12 of them worth more than she expected" in want.why
    assert want.contacts == 12 and want.known is True and want.liking > 0.4


def test_a_thing_in_every_one_of_her_turns_cannot_be_told_from_her_turns(record):
    """With nothing to set it against, she cannot say the thing is what paid."""
    _live(record, {TIDE: 0.6})
    assert record.known(TIDE) is None
    assert record.thing(TIDE).liking == pytest.approx(record.thing("self").liking, abs=0.05)


ORDINARY = [
    ("Can you check the time for me and tell me what you think of the plan?",
     "I think the plan is fine, and the time is what you would expect for a second look."),
    ("What do you make of the way that went? I would like to know what you think.",
     "I think it went the way these things go. There is more to say and I would rather say it plainly."),
    ("Tell me what surprised you more, the first one or the second one.",
     "The second one, though the first has held up better than anything has a right to."),
]


def _has_talked_before(record, exchanges: int = 45) -> None:
    """Enough ordinary talk behind her for a new subject to stand out against."""
    for turn in range(exchanges):
        record.talk.about(*ORDINARY[turn % len(ORDINARY)])


TIDE_TALK = (
    "Back to the tide pools for a second. Do you think an anemone has anything it prefers? "
    "The anemone let go of a small shell but held on to a mussel.",
    "The mussel is alive and the shell is not, so the anemone holds what smells of meat. "
    "In the tide pools that is most of what an anemone has to go on.",
)


def test_what_an_exchange_is_about_is_what_it_says_more_than_her_exchanges_have(record):
    _has_talked_before(record)
    about = record.talk.about(*TIDE_TALK)
    assert about == ["anemone"]


def test_a_word_said_in_passing_is_not_what_the_exchange_is_about(record):
    """The anemone is said four times; the mussel twice and the meat once."""
    _has_talked_before(record)
    about = record.talk.about(*TIDE_TALK)
    assert "anemone" in about
    assert not {"mussel", "meat", "shell", "tide pools"} & set(about)


def test_a_word_frequent_in_one_kind_of_talk_is_not_a_subject(record):
    """Spread thin across many exchanges, then used four times in one."""
    _has_talked_before(record)
    for turn in range(30):
        record.talk.about(f"Its number {turn} came up.", "Noted, and its place is kept.")
    about = record.talk.about(
        "The limpet left its scar. Its shell fits its scar and the limpet knows its scar.",
        "The limpet finds the scar by fit, and its foot settles in the scar.",
    )
    assert "limpet" in about and "scar" in about
    assert "its" not in about


def test_with_nothing_to_set_it_against_nothing_is_particular(record):
    assert record.talk.about(*TIDE_TALK) == []


def test_the_things_of_a_turn_are_its_particular_terms_in_place_of_the_topic_label(record):
    from core.conation.what_she_wants import read_turn

    class Worth:
        measured, worth, size, payoff = True, 1.0, 0.7, {}

    from tests.fixtures.things_her_library_names import NamesThese

    _has_talked_before(record)
    record.about = NamesThese("anemone")
    read_turn(["self", "topic:times", "attended:deliberation"], Worth(), record, said=TIDE_TALK)
    assert record.thing("topic:times") is None
    assert record.thing("topic:anemone").reachable
    assert record.thing("topic:anemone").liking > 0.0
    assert record.thing("self").contacts == 1


def test_what_her_exchanges_have_said_is_kept_across_a_restart(record, tmp_path):
    _has_talked_before(record)
    record.met(["self"], 0.1)
    flush_writes_behind()
    again = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    assert "anemone" in again.talk.about(*TIDE_TALK)


def test_a_turn_in_which_nothing_was_said_has_no_subject(record):
    from core.conation.what_she_wants import read_turn

    class Worth:
        measured, worth, size, payoff = True, -1.0, 0.4, {}

    read_turn(["self", "topic:times", "attended:drive_integrity"], Worth(), record)
    assert record.thing("topic:times") is None
    assert record.thing("self").contacts == 1
