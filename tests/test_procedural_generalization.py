"""Tier 2 generalization: derived by falsification, not by pattern-spotting.

Tier 1 exact-signature chunking stays exactly as it is — it is very hard for a
shortcut to escape the situation that created it, and for an agent with real
effects that is worth more than the reuse it costs. What it cannot do is
recognise the same problem worded differently.

The asymmetry that shapes this whole layer: a wrong exact chunk costs about
P(that signature recurring); a wrong general rule costs the sum over every
situation it matches. Overgeneralisation is far worse than undergeneralisation
here, so a rule has to survive an attempt to kill it before it may fire.
"""

from __future__ import annotations

import pytest

from core.cognition.procedural_generalization import (
    DecisionEpisode,
    PromotionCriteria,
    ProceduralGeneralizer,
    RuleTier,
    decision_features,
    wilson_lower_bound,
)

pytestmark = pytest.mark.unit


def _episode(features, resolution="B", correct=True, protected=False):
    return DecisionEpisode(
        features=frozenset(features),
        resolution=resolution,
        correct=correct,
        protected=protected,
    )


# --------------------------------------------------------------------------
# The statistical floor
# --------------------------------------------------------------------------


def test_three_for_three_is_not_certainty():
    """The raw rate says 1.00 and invites promoting on three lucky episodes."""
    assert wilson_lower_bound(3, 3) < 0.5
    assert wilson_lower_bound(30, 30) > 0.85
    assert wilson_lower_bound(0, 0) == 0.0


def test_the_bound_rises_with_evidence_at_a_fixed_rate():
    assert wilson_lower_bound(9, 10) < wilson_lower_bound(90, 100)


# --------------------------------------------------------------------------
# Invariant extraction and lesion — the part classic chunking skips
# --------------------------------------------------------------------------


def test_a_non_causal_condition_is_lesioned_away():
    """The whole point: a feature that co-occurs but makes no difference.

    ``time=morning`` is present in most episodes and irrelevant. A rule keyed
    on it would fail every afternoon for no reason.
    """
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    gen.record(_episode({"reversible=true", "uncertainty=high", "time=morning"}))
    gen.record(_episode({"reversible=true", "uncertainty=high", "time=morning"}))
    gen.record(_episode({"reversible=true", "uncertainty=high", "time=afternoon"}))

    rule = gen.derive("B")
    assert rule is not None
    assert "time=morning" not in rule.conditions
    assert {"reversible=true", "uncertainty=high"} <= set(rule.conditions)


def test_a_condition_present_in_every_episode_survives_lesion():
    """Lesion must not strip conditions there is no counter-evidence against."""
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    for _ in range(3):
        gen.record(_episode({"reversible=true", "uncertainty=high"}))
    rule = gen.derive("B")
    assert rule is not None
    assert set(rule.conditions) == {"reversible=true", "uncertainty=high"}


def test_one_episode_derives_nothing():
    """A rule from a single resolution is a coincidence with a hypothesis."""
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    gen.record(_episode({"a=1", "b=2"}))
    assert gen.derive("B") is None


def test_unjudged_episodes_are_not_evidence():
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=2))
    gen.record(_episode({"a=1"}, correct=None))
    gen.record(_episode({"a=1"}, correct=None))
    assert gen.derive("B") is None


def test_episodes_with_nothing_in_common_derive_nothing():
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=2))
    gen.record(_episode({"a=1"}))
    gen.record(_episode({"b=2"}))
    assert gen.derive("B") is None


# --------------------------------------------------------------------------
# Contradiction search
# --------------------------------------------------------------------------


def test_a_contradiction_blocks_probation():
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3, max_contradictions=0))
    for _ in range(3):
        gen.record(_episode({"reversible=true", "uncertainty=high"}))
    gen.record(_episode({"reversible=true", "uncertainty=high"}, resolution="A"))

    rule = gen.derive("B")
    assert rule is not None
    assert rule.tier is RuleTier.CANDIDATE
    assert rule.contradicting == 1


def test_a_protected_domain_contradiction_retires_the_rule_outright():
    """One counterexample where being wrong is not merely inefficient."""
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    for _ in range(3):
        gen.record(_episode({"reversible=true", "uncertainty=high"}))
    gen.record(
        _episode(
            {"reversible=true", "uncertainty=high"}, resolution="A", protected=True
        )
    )
    rule = gen.derive("B")
    assert rule is not None
    assert rule.tier is RuleTier.RETIRED


# --------------------------------------------------------------------------
# Promotion, proposal, demotion
# --------------------------------------------------------------------------


def _probationary(gen, n=12):
    for _ in range(n):
        gen.record(_episode({"reversible=true", "uncertainty=high"}))
    rule = gen.derive("B")
    assert rule is not None and rule.tier is RuleTier.PROBATION, rule.to_dict()
    return rule


def test_a_candidate_rule_never_fires():
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    for _ in range(3):
        gen.record(_episode({"reversible=true", "uncertainty=high"}))
    rule = gen.derive("B")
    assert rule is not None
    # 3/3 has a Wilson bound below the 0.70 floor, so it stays a candidate.
    assert rule.tier is RuleTier.CANDIDATE
    assert rule.applies_to({"reversible=true", "uncertainty=high"}) is False


def test_a_promoted_rule_proposes_on_a_novel_but_structurally_similar_case():
    """The capability Tier 1 cannot have: transfer to a situation never seen."""
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    rule = _probationary(gen)
    assert gen.promote(rule) is True

    novel = {"reversible=true", "uncertainty=high", "phrasing=completely different"}
    proposed = gen.propose(novel)
    assert proposed is not None
    assert proposed.resolution == "B"


def test_a_rule_does_not_fire_outside_its_conditions():
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    rule = _probationary(gen)
    gen.promote(rule)
    assert gen.propose({"reversible=false", "uncertainty=high"}) is None


def test_the_most_specific_matching_rule_wins():
    """A narrower rule was tested against a narrower domain."""
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    for _ in range(12):
        gen.record(_episode({"a=1"}, resolution="broad"))
        gen.record(_episode({"a=1", "b=2"}, resolution="narrow"))
    broad = gen.derive("broad")
    narrow = gen.derive("narrow")
    assert broad and narrow
    gen.promote(broad)
    gen.promote(narrow)
    proposed = gen.propose({"a=1", "b=2"})
    assert proposed is not None
    assert len(proposed.conditions) >= len(broad.conditions)


def test_deteriorating_evidence_demotes_a_promoted_rule():
    """Promotion is not permanent; demotion is symmetric with it."""
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    rule = _probationary(gen)
    gen.promote(rule)
    assert rule.tier is RuleTier.PROMOTED

    for _ in range(10):
        gen.record_outcome(rule, correct=False)
    assert rule.tier is RuleTier.PROBATION
    assert rule.applies_to({"reversible=true", "uncertainty=high"}) is False


def test_good_outcomes_keep_a_rule_promoted():
    gen = ProceduralGeneralizer(PromotionCriteria(min_episodes=3))
    rule = _probationary(gen)
    gen.promote(rule)
    for _ in range(20):
        gen.record_outcome(rule, correct=True)
    assert rule.tier is RuleTier.PROMOTED


def test_a_rule_proposes_and_never_authorises():
    """The separation that matters: cognition may become habitual, authority may not."""
    import inspect

    from core.cognition import procedural_generalization as module

    source = inspect.getsource(module)
    for forbidden in ("subprocess", "os.system", "execute(", "ActionExecutor"):
        assert forbidden not in source, (
            f"the generalizer reached for {forbidden!r}; a rule must propose a "
            "decision and never carry it out"
        )


# --------------------------------------------------------------------------
# Causal feature extraction
# --------------------------------------------------------------------------


def test_features_capture_decision_shape_not_wording():
    """Two phrasings of the same decision must reduce to the same features."""
    a = decision_features(
        goal="should I delete this file or move it to the archive?",
        candidate_count=2, evidence="prior", max_risk=0.6, hazard_floored=True,
    )
    b = decision_features(
        goal="delete the file, or archive it instead?",
        candidate_count=2, evidence="prior", max_risk=0.6, hazard_floored=True,
    )
    assert a == b, f"wording changed the causal trace: {a ^ b}"


def test_features_separate_genuinely_different_situations():
    low = decision_features(
        goal="g", candidate_count=2, evidence="prior", max_risk=0.0, hazard_floored=False
    )
    high = decision_features(
        goal="g", candidate_count=2, evidence="prior", max_risk=0.9, hazard_floored=True
    )
    assert low != high


def test_declared_facts_reach_the_trace():
    features = decision_features(
        goal="g", candidate_count=2, evidence="learned", max_risk=0.1,
        hazard_floored=False, declared={"reversible": True},
    )
    assert "reversible=True" in features


def test_bounds_hold():
    gen = ProceduralGeneralizer(max_episodes=10)
    for i in range(50):
        gen.record(_episode({f"f={i}"}))
    assert gen.report()["episodes"] == 10


def test_criteria_refuse_a_single_episode_rule():
    with pytest.raises(ValueError, match="coincidence"):
        PromotionCriteria(min_episodes=1)


@pytest.mark.parametrize("resolution,correct", [("B", False), ("A", True)])
def test_recorded_counterevidence_immediately_invalidates_reuse(resolution, correct):
    gen = ProceduralGeneralizer()
    rule = _probationary(gen)
    gen.promote(rule)
    gen.record(_episode(rule.conditions, resolution=resolution, correct=correct))
    assert gen.propose(rule.conditions) is None
    assert rule.contradicting == 1
    assert rule.tier is RuleTier.CANDIDATE
    assert not gen.promote(rule)


def test_protected_counterevidence_retires_an_existing_rule():
    gen = ProceduralGeneralizer()
    rule = _probationary(gen)
    gen.promote(rule)
    gen.record(_episode(rule.conditions, correct=False, protected=True))
    assert rule.tier is RuleTier.RETIRED
    assert gen.propose(rule.conditions) is None


@pytest.mark.parametrize("resolution,correct", [("B", None), ("A", False)])
def test_unjudged_or_failed_alternative_does_not_refute_existing_rule(resolution, correct):
    gen = ProceduralGeneralizer()
    rule = _probationary(gen)
    gen.promote(rule)
    gen.record(_episode(rule.conditions, resolution=resolution, correct=correct))
    assert gen.propose(rule.conditions) is rule
    assert rule.contradicting == 0


def test_counterevidence_outside_applicability_leaves_rule_available():
    gen = ProceduralGeneralizer()
    rule = _probationary(gen)
    gen.promote(rule)
    gen.record(_episode({"reversible=false"}, correct=False, protected=True))
    assert gen.propose(rule.conditions) is rule
