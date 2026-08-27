"""Working a world out is not the same as getting good at it.

She could know exactly how a thing moves and still pay the full price of
deciding every single time — the same deliberation for the hundredth position
of a kind as for the first. A player who has met a shape a few times stops
deciding and starts recognising, and the deciding machinery is freed for the
positions that are actually new.

Nothing here knows what any world is. A position has a kind, kinds repeat, and
that is as true of a form or a queue as of a board.
"""

from __future__ import annotations

import pytest

from core.agency.what_worked_before import (
    KNOWN_WELL_ENOUGH,
    REMEMBERED_POSITIONS,
    WhatWorkedBefore,
)
from core.agency.worth_thinking_about import worth_a_pass

KIND = "4x4|full:6|largest:64@top-left"
ANOTHER = "4x4|full:11|largest:128@bottom-right"


def met(kind: str = KIND, move: str = "left", times: int = KNOWN_WELL_ENOUGH,
        worked: int | None = None) -> WhatWorkedBefore:
    skill = WhatWorkedBefore()
    helped = times if worked is None else worked
    for turn in range(times):
        skill.learned(kind, move, turn < helped)
    return skill


# ── a habit has to be earned ─────────────────────────────────────────────

def test_once_is_not_a_habit():
    assert met(times=1).suggests(KIND) == ""


def test_and_neither_is_twice():
    assert met(times=KNOWN_WELL_ENOUGH - 1).suggests(KIND) == ""


def test_a_move_that_keeps_working_becomes_one_she_can_make_on_sight():
    assert met().suggests(KIND) == "left"


def test_a_move_that_mostly_fails_is_a_gamble_rather_than_a_skill():
    assert met(times=6, worked=3).suggests(KIND) == ""


def test_and_it_stops_being_suggested_once_it_stops_working():
    skill = met(times=6)
    for _ in range(12):
        skill.learned(KIND, "left", False)
    assert skill.suggests(KIND) == ""


# ── recognition is about this position, not about any position ───────────

def test_a_kind_she_has_never_met_teaches_nothing():
    assert met().suggests(ANOTHER) == ""


def test_a_move_she_cannot_make_here_is_not_offered():
    assert met().suggests(KIND, among=("up", "down")) == ""
    assert met().suggests(KIND, among=("up", "left")) == "left"


# ── what recognition is FOR: not deciding again ──────────────────────────

def test_recognition_spends_no_words_when_the_arithmetic_agrees():
    asked, why = worth_a_pass(
        {"left": (0.9, ""), "up": (0.2, "")}, stakes=0.2, recognised="left"
    )
    assert asked is False
    assert "met" in why


def test_but_disagreement_is_exactly_what_thinking_is_for():
    """The surest sign a position is not the routine one it looked like."""
    asked, why = worth_a_pass(
        {"left": (0.2, ""), "up": (0.9, "")}, stakes=0.2, recognised="left"
    )
    assert asked is True
    assert "left" in why and "up" in why


def test_and_recognition_never_silences_a_moment_that_matters():
    asked, _why = worth_a_pass(
        {"left": (0.9, ""), "up": (0.2, "")}, stakes=0.2, recognised="left", unusual=True
    )
    assert asked is True


def test_nor_the_stakes():
    asked, _why = worth_a_pass(
        {"left": (0.9, ""), "up": (0.2, "")}, stakes=0.95, recognised="left"
    )
    assert asked is True


def test_nor_the_run_of_moves_without_a_word():
    asked, _why = worth_a_pass(
        {"left": (0.9, ""), "up": (0.2, "")},
        stakes=0.2, recognised="left", since_words=9, horizon=5,
    )
    assert asked is True


def test_without_recognition_nothing_about_the_old_behaviour_changes():
    quiet, why = worth_a_pass({"left": (0.9, ""), "up": (0.2, "")}, stakes=0.2)
    assert quiet is False
    assert "clear" in why


# ── what she can say about it ────────────────────────────────────────────

def test_she_can_say_what_she_has_got_good_at():
    skill = met()
    skill.took(KIND)
    said = skill.says()
    assert "1 of 1" in said and "1 move(s)" in said


def test_and_says_plainly_when_she_has_not():
    assert "taught her anything yet" in WhatWorkedBefore().says()


def test_fluency_is_the_share_of_kinds_she_answers_on_sight():
    skill = met()
    skill.learned(ANOTHER, "up", True)
    assert skill.fluency() == pytest.approx(0.5)


# ── it stays bounded, and it survives the run ────────────────────────────

def test_a_world_with_endless_kinds_does_not_grow_without_limit():
    skill = WhatWorkedBefore()
    for n in range(REMEMBERED_POSITIONS + 50):
        skill.learned(f"kind-{n}", "left", True)
    assert len(skill.known) == REMEMBERED_POSITIONS


def test_the_oldest_kind_is_the_one_that_goes():
    skill = WhatWorkedBefore()
    for n in range(REMEMBERED_POSITIONS + 1):
        skill.learned(f"kind-{n}", "left", True)
    assert "kind-0" not in skill.known
    assert f"kind-{REMEMBERED_POSITIONS}" in skill.known


def test_what_she_got_good_at_survives_the_process():
    back = WhatWorkedBefore.from_memory(met().as_memory())
    assert back.suggests(KIND) == "left"


def test_but_comes_back_light_enough_to_be_overturned():
    """A habit from yesterday is evidence about today, not a fact about it."""
    skill = met(times=8)
    back = WhatWorkedBefore.from_memory(skill.as_memory(), trust=0.5)
    for _ in range(4):
        back.learned(KIND, "left", False)
    assert back.suggests(KIND) == ""


def test_nothing_carried_back_claims_more_than_it_was():
    back = WhatWorkedBefore.from_memory(met(times=6, worked=6).as_memory(), trust=0.5)
    tried, worked = back.known[KIND]["left"]
    assert worked <= tried


def test_rubbish_is_not_a_memory():
    assert WhatWorkedBefore.from_memory("not a memory").known == {}
    assert WhatWorkedBefore.from_memory({"known": {"k": "bad"}}).known == {}
