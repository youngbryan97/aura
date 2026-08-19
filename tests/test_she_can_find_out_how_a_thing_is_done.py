"""A loop that only reads the screen can play badly forever.

It has the board, the moves available, and what its own last few moves led
to. None of that contains the thing a person would go and look up: that this
kind of task has a known way of being done well. Live, she played 2048 to a
score of 280 and stalled, never once asking how the game is played.

So a pursuit asks two questions the screen cannot answer. Have I done this
before, and what happened? And if that is thin, what does anyone know about
doing it? The first comes from her own record, the second from the search
skills she already has, and both arrive as attributed evidence beside the
reading rather than as instructions above it.

Being stuck is the trigger to ask again: a run of broken predictions means
what she is doing is not working, whatever the reason.
"""
from __future__ import annotations

import pytest

from core.agency.deliberate_action import Attempt, Verdict
from core.agency.task_knowledge import (
    Finding,
    TaskKnowledge,
    forget_everything,
    how_is_this_done,
    learn_about,
    stuck,
    usable_sentences,
)


class _Engine:
    def __init__(self, text=""):
        self.text = text
        self.asked = []

    async def execute(self, name, params, context=None):
        self.asked.append((name, dict(params)))
        return {"summary": self.text}


class _Graph:
    def __init__(self, rows=None):
        self.rows = rows or []

    def query_consequences(self, action, params=None):
        return self.rows


PAGE = (
    "2048 is a sliding puzzle game created in 2014 by Gabriele Cirulli. "
    "The key is to keep your largest tile in a corner and never move in the direction "
    "that would dislodge it. "
    "It has been cloned many times. "
    "Players should build tiles in a snake order along one row before merging."
)


@pytest.fixture(autouse=True)
def _fresh():
    forget_everything()
    yield
    forget_everything()


def test_a_sentence_that_says_how_beats_one_that_says_what():
    found = usable_sentences(PAGE)
    assert any("largest tile in a corner" in line for line in found)
    assert not any("created in 2014" in line for line in found)


def test_the_question_is_built_from_the_goal_itself():
    assert how_is_this_done("play 2048 until you get a 128 tile").endswith(
        "play 2048 until you get a 128 tile"
    )
    assert how_is_this_done("") == ""


@pytest.mark.asyncio
async def test_she_looks_it_up_and_can_say_what_she_found():
    engine = _Engine(PAGE)
    known = await learn_about("play 2048 until 128", engine=engine, graph=_Graph())
    assert known.known
    assert any("corner" in finding.says for finding in known.findings)
    spoken = known.narrate()
    assert spoken.startswith("I read that")
    assert "going to try that" in spoken


@pytest.mark.asyncio
async def test_what_she_found_is_attributed_rather_than_folded_in():
    engine = _Engine(PAGE)
    known = await learn_about("play 2048 until 128", engine=engine, graph=_Graph())
    evidence = known.as_evidence()
    assert all(line.startswith("Known about this task —") for line in evidence)
    assert any("what I read" in line for line in evidence)


@pytest.mark.asyncio
async def test_her_own_record_comes_first_and_can_make_searching_unnecessary():
    """A run should not go to the network to be told what it learned the hard way."""
    graph = _Graph(
        [
            {"outcome": "board locked up after repeated left", "success": False},
            {"outcome": "kept the big tile in the corner and it went well", "success": True},
            {"outcome": "merged upward twice", "success": True},
            {"outcome": "ran out of space", "success": False},
        ]
    )
    engine = _Engine(PAGE)
    known = await learn_about("play 2048 until 128", engine=engine, graph=graph)
    assert known.from_memory == 4
    assert engine.asked == [], "she searched for something she already knew"


@pytest.mark.asyncio
async def test_a_thin_record_sends_her_looking():
    engine = _Engine(PAGE)
    known = await learn_about("play 2048 until 128", engine=engine, graph=_Graph())
    assert engine.asked, "she never looked it up"
    name, params = engine.asked[0]
    assert name == "web_search"
    assert "play 2048" in params["query"]
    assert known.searched


@pytest.mark.asyncio
async def test_looking_it_up_twice_for_one_goal_is_not_needed():
    engine = _Engine(PAGE)
    await learn_about("play 2048 until 128", engine=engine, graph=_Graph())
    await learn_about("play 2048 until 128", engine=engine, graph=_Graph())
    assert len(engine.asked) == 1


@pytest.mark.asyncio
async def test_finding_nothing_is_said_plainly():
    engine = _Engine("2048 was created in 2014. It has been cloned many times.")
    known = await learn_about("play 2048 until 128", engine=engine, graph=_Graph())
    assert not known.known
    assert "could not find anything" in known.narrate()


@pytest.mark.asyncio
async def test_research_can_be_turned_off():
    engine = _Engine(PAGE)
    known = await learn_about("play 2048", engine=engine, graph=_Graph(), search=False)
    assert engine.asked == []
    assert not known.known


def test_a_run_of_broken_predictions_is_what_being_stuck_means():
    broke = Attempt(option="up", expected="a shift", verdict=Verdict(held=False, observed_change=False, stalled=True))
    held = Attempt(option="up", expected="a shift", verdict=Verdict(held=True, observed_change=True))
    assert stuck([broke, broke, broke])
    assert not stuck([broke, broke])
    assert not stuck([broke, broke, held])


def test_findings_read_back_without_a_source_too():
    assert Finding(says="keep it in the corner").as_evidence().endswith("keep it in the corner")


def test_knowledge_with_nothing_in_it_says_nothing_useful():
    assert not TaskKnowledge(goal="x").known


def _broke(option):
    return Attempt(option=option, expected="a change", verdict=Verdict(held=False, observed_change=False, stalled=True))


def test_being_stuck_asks_about_the_position_not_about_the_game():
    """"How is this played" returns the beginner's answer she already has."""
    from core.agency.task_knowledge import why_is_this_stuck

    question = why_is_this_stuck(
        "play 2048 until you get a 128 tile", "2 4 8 16 32 64", [_broke("up"), _broke("left")]
    )
    assert "stuck" in question
    assert "up and left do nothing" in question
    assert "64" in question
    assert question.endswith("what to do")


def test_the_question_names_only_the_moves_that_actually_failed():
    from core.agency.task_knowledge import why_is_this_stuck

    worked = Attempt(option="right", expected="a change", verdict=Verdict(held=True, observed_change=True))
    question = why_is_this_stuck("play a board game", "", [_broke("up"), worked])
    assert "up do nothing" in question
    assert "right" not in question


def test_a_position_is_characterised_by_its_values():
    from core.agency.task_knowledge import _salient

    assert _salient("2 4 8 16 32 64") == "64 32 16 8 4 2"
    assert _salient("the installer is waiting") == "the installer is waiting"
    assert _salient("") == ""


@pytest.mark.asyncio
async def test_being_stuck_does_not_re_read_her_own_record():
    """Her record is what got her here."""
    graph = _Graph([{"outcome": "pressing up worked once", "success": True}] * 4)
    engine = _Engine(PAGE)
    known = await learn_about(
        "play 2048 until 128",
        engine=engine,
        graph=graph,
        because_stuck=True,
        situation="2 4 8 16",
        history=[_broke("up")],
    )
    assert known.from_memory == 0
    assert engine.asked, "being stuck must send her looking"
    _name, params = engine.asked[0]
    assert "stuck" in params["query"]


@pytest.mark.asyncio
async def test_a_stuck_finding_is_narrated_as_a_change_of_approach():
    engine = _Engine(PAGE)
    known = await learn_about(
        "play 2048 until 128",
        engine=engine,
        graph=_Graph(),
        because_stuck=True,
        situation="2 4 8",
        history=[_broke("up")],
    )
    spoken = known.narrate()
    assert spoken.startswith("This stopped working, so I looked it up")
    assert "going to try that" in spoken


@pytest.mark.asyncio
async def test_being_stuck_and_finding_nothing_says_what_was_asked():
    engine = _Engine("nothing useful here at all, just a description")
    known = await learn_about(
        "play 2048 until 128", engine=engine, graph=_Graph(), because_stuck=True, history=[_broke("up")]
    )
    assert "could not find out why" in known.narrate()
    assert "stuck" in known.narrate()
