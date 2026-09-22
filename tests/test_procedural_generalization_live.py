"""Tier 2 is wired into live deliberation, not merely implemented.

The capability Tier 1 cannot have: the exact-signature chunker misses whenever
the wording changes, so the same structural problem reruns the whole search
forever. A promoted generalized rule recognises it.

The separation that makes that safe: a rule narrows the field and the search
still confirms, so the chain stays

    rule -> proposed decision -> Will/authority -> execution -> receipt

and the receipt distinguishes rule_applied from chunk_reused, because the two
carry different risk — a chunk answered the same question again, a rule
answered one it had never been asked.
"""

from __future__ import annotations

import asyncio

import pytest

from core.cognition.procedural_generalization import (
    ProceduralGeneralizer,
    PromotionCriteria,
    RuleTier,
    get_procedural_generalizer,
)
from core.reasoning.native_system2 import NativeSystem2Engine

pytestmark = pytest.mark.unit


def _drive(engine, contexts, actions=("archive it", "delete it")):
    async def run():
        out = []
        for ctx in contexts:
            out.append(await engine.rank_actions(context=ctx, actions=list(actions)))
        return out

    return asyncio.run(run())


def test_every_deliberation_feeds_tier_two():
    """Episodes must accumulate from the real path, not a test harness."""
    import core.cognition.procedural_generalization as module

    previous = module._generalizer
    module._generalizer = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    try:
        engine = NativeSystem2Engine()
        _drive(engine, [f"phrasing {i}" for i in range(5)])
        assert module._generalizer.report()["episodes"] >= 5
    finally:
        module._generalizer = previous


def test_a_promoted_rule_fires_on_wording_tier_one_would_miss():
    """The transfer property, end to end through rank_actions."""
    import core.cognition.procedural_generalization as module

    previous = module._generalizer
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    module._generalizer = gen
    try:
        engine = NativeSystem2Engine()
        _drive(engine, [f"differently worded problem {i}" for i in range(14)])

        # Grade the accumulated episodes, then derive and promote.
        graded = [
            type(e)(features=e.features, resolution=e.resolution, correct=True,
                    protected=e.protected)
            for e in gen._episodes
        ]
        gen._episodes = graded
        resolution = graded[0].resolution
        rule = gen.derive(resolution)
        assert rule is not None, "no rule derived from 14 consistent episodes"
        assert gen.promote(rule) is True, rule.to_dict()

        fresh = _drive(engine, ["a phrasing never seen before at all"])[0]
        assert fresh.receipt.rule_applied is True
        assert fresh.receipt.rule_conditions, "the rule fired without recording why"
        assert fresh.receipt.chunk_reused is False, (
            "this must be Tier 2 transfer, not a Tier 1 exact hit"
        )
        gen.record(module.DecisionEpisode(rule.conditions, resolution, correct=False))
        after_failure = _drive(engine, ["new request after independent counterevidence"])[0]
        assert after_failure.receipt.rule_applied is False
    finally:
        module._generalizer = previous


def test_an_unpromoted_rule_never_fires_on_the_live_path():
    import core.cognition.procedural_generalization as module

    previous = module._generalizer
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    module._generalizer = gen
    try:
        engine = NativeSystem2Engine()
        _drive(engine, [f"situation {i}" for i in range(6)])
        graded = [
            type(e)(features=e.features, resolution=e.resolution, correct=True,
                    protected=e.protected)
            for e in gen._episodes
        ]
        gen._episodes = graded
        rule = gen.derive(graded[0].resolution)
        if rule is not None:
            assert rule.tier is not RuleTier.PROMOTED
        fresh = _drive(engine, ["yet another phrasing"])[0]
        assert fresh.receipt.rule_applied is False
    finally:
        module._generalizer = previous


def test_a_high_risk_decision_is_recorded_as_protected():
    """Protected episodes are what make one counterexample retire a rule."""
    import core.cognition.procedural_generalization as module

    previous = module._generalizer
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    module._generalizer = gen
    try:
        engine = NativeSystem2Engine()
        _drive(engine, ["risky"], actions=("delete the production database", "archive"))
        assert any(e.protected for e in gen._episodes), (
            "a decision over a destructive candidate was not marked protected"
        )
    finally:
        module._generalizer = previous


def test_the_generalizer_is_a_process_singleton():
    assert get_procedural_generalizer() is get_procedural_generalizer()
