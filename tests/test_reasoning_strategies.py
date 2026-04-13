from core.brain.aura_persona import AURA_IDENTITY
from core.brain.reasoning_strategies import ReasoningStrategies, StrategyType


def test_reasoning_strategies_treat_identity_anchor_as_direct():
    strategies = ReasoningStrategies(lambda *args, **kwargs: "")

    assert strategies.classify(AURA_IDENTITY) == StrategyType.DIRECT


def test_reasoning_strategies_still_allow_real_debate_queries():
    strategies = ReasoningStrategies(lambda *args, **kwargs: "")

    assert (
        strategies.classify("Give me the pros and cons of remote work versus office work.")
        == StrategyType.DEBATE
    )
