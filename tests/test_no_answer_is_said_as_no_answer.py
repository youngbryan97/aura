"""A deliberation with no model behind it does not return a sentence.

Measured live on 2026-08-26: "No model was available to answer, but the
question has been sharpened" came back where her thinking should have been,
and was held as her plan. No caller could tell it from an answer.
"""

from __future__ import annotations

import pytest

from core.brain.deep_deliberation import DeepDeliberationEngine


class _NoBrain:
    """A container with nothing in it that can think."""

    def get(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_a_deliberation_with_no_model_answers_nothing():
    engine = DeepDeliberationEngine(orchestrator=_NoBrain())
    result = await engine.deliberate("how should I play this", timeout_s=1.0)
    assert not str(result.answer or "").strip()


@pytest.mark.asyncio
async def test_the_sharpened_question_is_still_there_for_whoever_wants_it():
    engine = DeepDeliberationEngine(orchestrator=_NoBrain())
    result = await engine.deliberate("how should I play this", timeout_s=1.0)
    assert str(result.refined_question or "").strip()


@pytest.mark.asyncio
async def test_a_run_with_no_model_is_not_counted_as_one_with_a_model():
    engine = DeepDeliberationEngine(orchestrator=_NoBrain())
    await engine.deliberate("how should I play this", timeout_s=1.0)
    assert engine._model_backed == 0
