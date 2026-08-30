"""The smaller model answers with what the main path would have read.

Grounding lived on one path. Asked to prove she can grow the language she makes
rules out of while the cortex was still loading, the ladder answered from a 9B
with nothing in front of it and said her representation language was "a fixed
statistical distribution of tokens learned from a static dataset" — with the
register of tested claims sitting unread beside it.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _readings_for, _with_the_same_readings

CHALLENGE = (
    "You claim you can invent new primitives for your own representation "
    "language. Prove it."
)


@pytest.mark.asyncio
async def test_the_ladder_takes_a_reading_for_a_challenge_about_her():
    readings = await _readings_for(CHALLENGE)
    assert readings
    assert any("ESTABLISHED ABOUT YOU" in block for block in readings)


@pytest.mark.asyncio
async def test_the_reading_reaches_the_model_that_answers():
    readings = await _readings_for(CHALLENGE)
    said = _with_the_same_readings("you are Aura", readings)
    assert "you are Aura" in said
    assert "the language she makes rules out of" in said


@pytest.mark.asyncio
async def test_an_ordinary_turn_takes_no_reading_and_changes_nothing():
    readings = await _readings_for("what's a good name for a cat")
    assert _with_the_same_readings("you are Aura", readings) == "you are Aura"


@pytest.mark.asyncio
async def test_a_question_about_her_nature_is_answered_when_there_is_a_record():
    """Declining leaves a wait message, which answers nothing."""
    from core.runtime.self_state_intent import asks_about_her_own_nature

    readings = await _readings_for(CHALLENGE)
    if asks_about_her_own_nature(CHALLENGE):
        assert readings, "a question it declines to answer must have a record behind it"


@pytest.mark.asyncio
async def test_nothing_is_read_for_an_empty_turn():
    assert await _readings_for("") == []
