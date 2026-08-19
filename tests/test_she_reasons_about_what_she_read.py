"""Retrieving advice is not applying it.

"Keep your largest tile in a corner" is a fact about the game. What it means
depends on where the tiles actually are, and working that out is the step
between reading something and playing differently. Live, she collected
sentences and kept pressing the same keys.

Three things had to be added, and all of them are general:

  * a question has a kind. What a thing IS goes to her offline encyclopedia,
    which holds a full Wikipedia snapshot and answers instantly. What to DO
    about it goes to the web, because an article about a game does not say
    what to do when the board is locked;
  * a snippet is an advertisement for an answer. Being stuck needs the
    answer, so the best-matching page is opened and read;
  * and what it means HERE is worked out against the position she is in —
    reasoned when language is reachable, derived structurally when it is not,
    so losing language costs the quality of the step and not the step.
"""
from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption
from core.agency.task_knowledge import (
    BACKGROUND,
    TACTIC,
    Finding,
    Implication,
    TaskKnowledge,
    _favoured_option,
    kind_of_question,
    work_out_what_it_means,
)

MOVES = [ActionOption(name=name, detail=f"press {name}") for name in ("up", "down", "left", "right")]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("what is the 2048 video game", BACKGROUND),
        ("history of the game", BACKGROUND),
        ("who is Gabriele Cirulli", BACKGROUND),
        ("play 2048 stuck, up and left do nothing, what to do", TACTIC),
        ("how to do this well: play 2048", TACTIC),
        ("best way to get past a locked board", TACTIC),
    ],
)
def test_a_question_has_a_kind_and_the_kind_picks_the_source(question, expected):
    assert kind_of_question(question) == expected


def test_an_unclassifiable_question_is_treated_as_needing_a_tactic():
    """Mid-task, what to do is the useful default."""
    assert kind_of_question("2048 board") == TACTIC


def test_the_move_a_finding_names_is_the_move_it_favours():
    assert _favoured_option("keep the largest tile in a corner and never move up", MOVES) == "up"
    assert _favoured_option("slide left to consolidate", MOVES) == "left"


def test_a_short_move_name_is_not_lost_to_a_word_filter():
    """A filter that drops words under three letters drops "up" with "the"."""
    assert _favoured_option("never move up", MOVES) == "up"


def test_a_finding_that_names_no_move_favours_none():
    assert _favoured_option("the game was released in 2014", MOVES) == ""


@pytest.mark.asyncio
async def test_without_language_she_still_works_out_what_it_means():
    known = TaskKnowledge(
        goal="play 2048",
        findings=[Finding(says="Keep your largest tile in a corner and never move up", source="read on x")],
    )
    meant = await work_out_what_it_means(known, "2 4 8 16 in the left column", MOVES)
    assert meant and meant[0].favours == "up"
    assert "available now" in meant[0].means


@pytest.mark.asyncio
async def test_with_language_the_meaning_is_reasoned_and_still_names_a_move():
    async def think(objective, evidence):
        think.seen = list(evidence)
        return "The big tile is bottom-left, so pressing up would dislodge it — go left instead."

    think.seen = None
    known = TaskKnowledge(
        goal="play 2048", findings=[Finding(says="Keep your largest tile in a corner", source="read on x")]
    )
    meant = await work_out_what_it_means(known, "16 in the bottom left", MOVES, think=think)
    assert meant[0].means.startswith("The big tile is bottom-left")
    assert meant[0].favours == "left"
    assert any("What is on screen" in line for line in think.seen)


@pytest.mark.asyncio
async def test_language_failing_mid_thought_falls_back_rather_than_dropping_the_step():
    async def broken(objective, evidence):
        raise RuntimeError("worker_not_alive")

    known = TaskKnowledge(goal="play 2048", findings=[Finding(says="never move up", source="read on x")])
    meant = await work_out_what_it_means(known, "a board", MOVES, think=broken)
    assert meant and meant[0].favours == "up"


@pytest.mark.asyncio
async def test_knowing_nothing_means_there_is_nothing_to_apply():
    assert await work_out_what_it_means(TaskKnowledge(goal="x"), "a board", MOVES) == []


def test_an_implication_reads_back_as_advice_about_now():
    line = Implication(finding="keep it in the corner", means="it names up, which is available now", favours="up").as_evidence()
    assert line.startswith("What that means here —")
    assert line.endswith("(so: up)")
