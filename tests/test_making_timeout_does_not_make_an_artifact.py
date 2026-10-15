"""Stopping an unchanged making surface cannot produce a success or a claimed artifact."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.skills import sovereign_browser_drawing as drawing
from core.skills.screen_pursuit_as_it_happens import (
    MAKING_STALLED,
    TRYING_OUT_MOST_S,
    PlayingAsItHappens,
)

pytestmark = pytest.mark.unit


def test_the_making_timeout_canary_requires_an_incomplete_outcome():
    assert drawing._stalled_making_invariant() == ()


@pytest.mark.parametrize("instructions", ["Create your own character.", "Design your own card.", "Draw your own picture."])
def test_a_real_making_run_has_a_distinct_stalled_outcome(monkeypatch, instructions):
    monkeypatch.setattr("time.monotonic", lambda: TRYING_OUT_MOST_S + 1)
    run = PlayingAsItHappens(None, (0, 0, 1, 1), "make one", 1000)
    run.words = [instructions]
    run.for_making_since = 0
    assert run.run_is_over({"text": instructions})
    assert run.over_because == MAKING_STALLED


@pytest.mark.asyncio
async def test_the_real_completion_caller_does_not_describe_the_untouched_surface_as_made(monkeypatch):
    said = []

    async def stock(*args):
        return None

    async def one(page, band, goal, url, deadline, keep):
        reflexes = SimpleNamespace(
            keep=keep, words=["Create your own card."], ending_words="CARD EDITOR", ending_parts=["CARD EDITOR"],
            over_because=MAKING_STALLED, stretches=[], new_screens=0, what_it_came_to=lambda: "No result observed.",
        )
        return {"moves": [{"click": "Open editor"}], "last_seen": "CARD EDITOR", "completed": True}, reflexes

    async def should_not_describe(*args):
        pytest.fail("no produced artifact is available to describe")

    monkeypatch.setattr(drawing, "_a_stocktaking", stock)
    monkeypatch.setattr(drawing, "_one_run", one)
    monkeypatch.setattr(drawing, "_what_was_made_as_seen", should_not_describe)
    monkeypatch.setattr(drawing, "_tell", said.append)
    goal = "Play this game. If it has neither a winning goal nor a score, do it once and tell me what you made and why."
    result = await drawing._played(None, (0, 0, 1, 1), goal, "https://example.org/card", {})
    assert not result["completed"] and not result["ok"] and not result["finished"]
    assert result["completed_runs"] == 0
    assert result["stopped_because"] == MAKING_STALLED
    assert all("what i made shows" not in line.lower() for line in said)


@pytest.mark.asyncio
@pytest.mark.parametrize('making', [True, False])
async def test_an_outer_deadline_cannot_bypass_the_creation_handoff(monkeypatch, making):
    from core.cognition.a_guide_to_a_place import SCREEN, Guide
    from core.perception.reading_a_program import ProgramRead

    clock, said = [10.], []
    words = ('Create your own picture.' if making else
             'Find the hidden tools. Use them to reach the robot and win the game.')
    guide = Guide()
    guide.take_in(SCREEN, words)
    guide.take_in_program(ProgramRead(text='outfit outfit outfit', kind='script'))
    assert 'making' in guide.mechanics  # A source mechanic alone cannot establish the task's end.

    async def stock(*args):
        return None

    async def one(page, band, goal, url, deadline, keep):
        keep['guide'] = guide
        clock[0] = deadline+10.
        reflexes = SimpleNamespace(keep=keep, words=[words], ending_words='INSTRUCTIONS', ending_parts=[],
                                  over_because='', stretches=[], new_screens=0, what_it_came_to=lambda: '')
        return {'moves': [{'key': 'click "Instructions"'}], 'last_seen': 'INSTRUCTIONS', 'completed': True}, reflexes

    async def should_not_describe(*args):
        pytest.fail('there is no creation handoff to describe')

    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    monkeypatch.setattr(drawing, '_a_stocktaking', stock)
    monkeypatch.setattr(drawing, '_one_run', one)
    monkeypatch.setattr(drawing, '_what_was_made_as_seen', should_not_describe)
    monkeypatch.setattr(drawing, '_tell', said.append)
    goal = 'Play this game. If it has neither a winning goal nor a score, do it once and tell me what you made and why.'
    result = await drawing._played(None, (0, 0, 1, 1), goal, f'https://example.org/task-{making}', {})
    assert not result['completed'] and not result['ok'] and not result['finished']
    assert result['completed_runs'] == 0 and result['stopped_because'] != drawing.MADE_AS_ASKED
    assert not any('I made it' in line for line in said)
    assert drawing._a_place_for_making({'guide': guide}) is making
    assert drawing._making_handoff_invariant() == ()
