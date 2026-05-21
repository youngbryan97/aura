from __future__ import annotations

import pytest

from core.brain.reasoning_strategies import (
    _REASONING_FAILURE_MESSAGE,
    ReasoningStrategies,
    StrategyType,
)


@pytest.mark.asyncio
async def test_direct_generation_failure_returns_honest_degraded_result():
    async def generate(prompt, **_kwargs):
        if prompt:
            raise RuntimeError("backend offline")
        return ""

    strategies = ReasoningStrategies(generate)

    result = await strategies.execute(
        "Explain the current runtime health contract.",
        strategy=StrategyType.DIRECT,
    )

    assert result.content == _REASONING_FAILURE_MESSAGE
    assert result.confidence == 0.0
    assert result.metadata["degraded"] is True


@pytest.mark.asyncio
async def test_debate_continues_when_one_perspective_fails():
    async def generate(prompt, **_kwargs):
        if prompt.startswith("You are Perspective A"):
            raise RuntimeError("left branch failed")
        if prompt.startswith("You are Perspective B"):
            return "The safer alternative is to validate the runtime first."
        if "You are Aura" in prompt:
            return "Validate the runtime first, then proceed with the smallest safe action."
        return "direct answer"

    strategies = ReasoningStrategies(generate)
    strategies._get_tree_of_thoughts = lambda: None

    result = await strategies.execute(
        "Compare quick deployment versus validating the runtime first.",
        strategy=StrategyType.DEBATE,
    )

    assert result.strategy_used is StrategyType.DEBATE
    assert "Validate the runtime first" in result.content
    assert result.metadata["perspectives"] == 2


@pytest.mark.asyncio
async def test_tree_of_thought_failure_falls_back_to_legacy_strategy():
    class FailingTree:
        async def deliberate(self, **_kwargs):
            if _kwargs is not None:
                raise RuntimeError("deliberation failed")
            return None

    async def generate(prompt, **_kwargs):
        if "You are Aura" in prompt:
            return "Use the legacy strategy after the failed deliberation."
        return "legacy branch"

    strategies = ReasoningStrategies(generate)
    strategies._tree_of_thoughts = FailingTree()

    result = await strategies.execute(
        "Weigh the trade-off between speed and resilience.",
        strategy=StrategyType.DEBATE,
    )

    assert result.strategy_used is StrategyType.DEBATE
    assert result.content == "Use the legacy strategy after the failed deliberation."


@pytest.mark.asyncio
async def test_confidence_estimation_failure_returns_neutral_floor():
    async def generate(prompt, **_kwargs):
        if prompt:
            raise RuntimeError("confidence backend offline")
        return ""

    strategies = ReasoningStrategies(generate)

    assert await strategies.estimate_confidence("question", "answer") == 0.5
