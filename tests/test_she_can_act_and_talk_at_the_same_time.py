"""Saying what she is doing used to be a step inside doing it.

The pursuit loop spoke its own line between deciding and acting, which meant
the next move waited on language. That is backwards twice over: a loop that
cannot talk should still play, and a faculty that can talk should be able to
talk about anything, not only about the loop it was written into.

So a decision is offered to the global workspace — where a faculty's content
already becomes available to every other faculty — and the narrator listens
there. The actor never waits. Silence is the absence of a narrator rather
than a different code path, and the same narrator speaks about a chosen move,
a perception, or a plan that changed, without knowing what any of them are.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.agency.narrator import Narrator, line_for


class _Winner:
    def __init__(self, source="somewhere", metadata=None, content="", age_s=0.0):
        self.source = source
        self.metadata = metadata or {}
        self.content = content
        self.submitted_at = time.time() - age_s


class _Event:
    def __init__(self, *winners):
        self.winners = list(winners)


class _Workspace:
    def __init__(self):
        self._processors = []

    def register_processor(self, fn):
        self._processors.append(fn)

    async def broadcast(self, *winners):
        for fn in list(self._processors):
            await fn(_Event(*winners))


def _decision(**fields):
    base = {"chose": "up: press up", "because": "the corner holds", "expected": "the board to shift"}
    base.update(fields)
    return {"payload": {"decision": base}}


async def _run_with(narrator, workspace, *winners):
    narrator.start()
    await workspace.broadcast(*winners)
    await narrator.stop()


def _spoken():
    said = []

    async def say(line):
        said.append(line)

    return said, say


def test_a_decision_reads_back_as_a_sentence_about_the_reasoning():
    line = line_for(_decision()["payload"])
    assert line == "up: press up — the corner holds. I expect the board to shift"


def test_a_decision_made_without_language_still_says_so():
    line = line_for(_decision(spoke=False)["payload"])
    assert "deciding without words" in line


def test_a_prediction_that_held_is_not_worth_saying():
    assert line_for({"outcome": {"chose": "up", "held": True}}) == ""


def test_a_prediction_that_broke_is():
    line = line_for({"outcome": {"chose": "up", "held": False, "why": "nothing changed"}})
    assert line == "up did not work — nothing changed"


def test_it_narrates_whatever_reached_her_not_only_moves():
    """The point of subscribing to the workspace rather than to one loop."""
    assert line_for("a face appeared in the camera") == "a face appeared in the camera"


@pytest.mark.asyncio
async def test_the_narrator_speaks_about_a_broadcast():
    said, say = _spoken()
    workspace = _Workspace()
    await _run_with(Narrator(say=say, workspace=workspace), workspace, _Winner(metadata=_decision()))
    assert said == ["up: press up — the corner holds. I expect the board to shift"]


@pytest.mark.asyncio
async def test_a_narrator_can_be_scoped_to_one_thing_she_is_doing():
    said, say = _spoken()
    workspace = _Workspace()
    narrator = Narrator(say=say, workspace=workspace, about="screen_pursuit.next_move")
    await _run_with(
        narrator,
        workspace,
        _Winner(source="somewhere_else", metadata=_decision()),
        _Winner(source="screen_pursuit.next_move", metadata=_decision(chose="left: press left")),
    )
    assert len(said) == 1
    assert "left" in said[0]


@pytest.mark.asyncio
async def test_falling_behind_drops_lines_rather_than_holding_anything_up():
    """Deciding must never wait on talking."""
    said, say = _spoken()
    workspace = _Workspace()
    narrator = Narrator(say=say, workspace=workspace)
    narrator.start()
    for index in range(40):
        narrator.offer(_Winner(metadata=_decision(chose=f"move {index}")))
    assert narrator.dropped > 0, "a full queue must drop, not block"
    await narrator.stop()


@pytest.mark.asyncio
async def test_old_news_is_not_narrated_as_though_it_were_now():
    said, say = _spoken()
    workspace = _Workspace()
    await _run_with(
        Narrator(say=say, workspace=workspace),
        workspace,
        _Winner(metadata=_decision(), age_s=120.0),
    )
    assert said == []


@pytest.mark.asyncio
async def test_a_broken_surface_does_not_break_the_narrator():
    workspace = _Workspace()

    async def broken(_line):
        raise RuntimeError("no surface")

    narrator = Narrator(say=broken, workspace=workspace)
    await _run_with(narrator, workspace, _Winner(metadata=_decision()))
    assert narrator.spoken == 0


@pytest.mark.asyncio
async def test_no_workspace_at_all_is_not_a_failure():
    """Nothing to listen to is a state, not an error."""
    said, say = _spoken()
    narrator = Narrator(say=say, workspace=None)
    narrator.start()
    await narrator.stop()
    assert said == []
