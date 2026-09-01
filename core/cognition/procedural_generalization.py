"""Tier 2: generalized procedural rules, derived by falsification rather than by pattern-spotting.

Tier 1 — the exact-signature chunking in :mod:`core.cognition.impasse` — is
deliberately kept as it is. It is very hard for a learned shortcut to escape
the situation that created it: the signature carries the decision context and
the candidate set, and changing either prevents reuse. For an agent with real
effects that conservatism is worth more than the reuse it costs.

What it cannot do is recognise that a differently-worded situation is the same
problem. Learn "in situation X choose B" and situation X' — structurally
identical, phrased differently — reruns the whole search.

Soar's deeper idea is the fix, and it is not "remember what worked". It is:
work out *why* the subproblem resolved the way it did, and compile that
explanation. The difference is between learning

    in this decision, choose B

and learning

    when two actions satisfy the same objective, uncertainty is high, and one
    is reversible while the other is not, prefer the reversible one

The second transfers. The first cannot.

Why this is more guarded than classic chunking
----------------------------------------------
Because the blast radius is not symmetric. A wrong exact chunk costs about
``P(that exact signature recurring)``. A wrong general rule costs
``sum over matched situations of P(s)·L(s)`` — the better it generalises, the
more it can cost when it is wrong. For an autonomous system overgeneralisation
is far worse than undergeneralisation, so a rule here has to survive an
attempt to kill it before it is allowed to fire:

1. **Several independent episodes**, not one. A rule derived from a single
   resolution is a coincidence with a hypothesis attached.
2. **Invariant extraction** — the features common to every successful episode.
   That is the candidate explanation, and it is where classic chunking stops.
3. **Counterfactual lesion** — for each candidate condition, look for episodes
   that resolved the same way *without* it. A condition that makes no
   difference to the outcome is not causal, however reliably it co-occurs, and
   it is dropped. This is the step that turns "I found a pattern" into "here is
   the empirically supported domain in which the shortcut holds".
4. **Contradiction search** — episodes matching the conditions that resolved
   *differently*. Any at all in a protected domain blocks promotion outright.
5. **A statistical floor**, on the Wilson lower bound rather than the raw
   success rate, so three-for-three does not read as certainty.

Demotion is symmetric with promotion: evidence that deteriorates past the floor
sends a rule back down. A promoted rule is not a permanent one.

What a rule is emphatically not
-------------------------------
A rule proposes; it never authorises. The chain stays

    rule → proposed decision → Will/authority → execution → receipt

Cognition is allowed to become habitual. Authority is not. Nothing in this
module executes anything, and :meth:`GeneralizedRule.applies_to` returns a
suggestion that the caller is free — and on consequential paths, required — to
put through the same governance an unchunked decision would face.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from core.runtime.lockdep import checked_lock

__all__ = [
    "Feature",
    "DecisionEpisode",
    "RuleTier",
    "GeneralizedRule",
    "PromotionCriteria",
    "ProceduralGeneralizer",
    "get_procedural_generalizer",
    "decision_features",
    "wilson_lower_bound",
]

#: A causally-relevant fact about a decision situation, as ``key=value``.
#: Strings rather than free objects because a rule condition has to be
#: comparable across episodes and printable in a receipt.
Feature = str


class RuleTier(StrEnum):
    """How far a generalization has earned its way."""

    #: Derived and under test. Never fires.
    CANDIDATE = "candidate"
    #: Survived lesion and contradiction search, awaiting evidence.
    PROBATION = "probation"
    #: Promoted. May propose a decision, still never authorises one.
    PROMOTED = "promoted"
    #: Demoted after its evidence deteriorated. Kept as a record.
    RETIRED = "retired"


@dataclass(frozen=True)
class DecisionEpisode:
    """One resolved deliberation, reduced to what actually drove it.

    ``features`` is the causal trace, not the raw situation. Indexing on the
    whole prompt is what makes exact chunking brittle — most of a context is
    noise, and a rule keyed on noise cannot transfer. What belongs here is the
    small set of facts the search actually consumed.
    """

    features: frozenset[Feature]
    resolution: str
    #: Whether the decision turned out well. None while unjudged.
    correct: bool | None = None
    #: Domains where a wrong generalization is not merely inefficient.
    protected: bool = False

    def matches(self, conditions: Iterable[Feature]) -> bool:
        return set(conditions).issubset(self.features)


@dataclass(frozen=True)
class PromotionCriteria:
    """What a generalization must clear before it may propose anything.

    Every threshold is a policy choice about acceptable blast radius, so they
    are declared together in one readable place rather than scattered through
    the logic as literals.
    """

    #: Independent successful episodes required before deriving a rule at all.
    min_episodes: int = 3
    #: Wilson 95% lower bound on correctness, not the raw rate: 3/3 has a lower
    #: bound near 0.44, which is correctly unconvincing.
    min_confidence_lower_bound: float = 0.70
    #: A rule matching everything explains nothing; require real conditions.
    min_conditions: int = 1
    #: Contradictions tolerated outside protected domains.
    max_contradictions: int = 0

    def __post_init__(self) -> None:
        if self.min_episodes < 2:
            raise ValueError(
                "a rule derived from one episode is a coincidence with a "
                "hypothesis attached; require at least two"
            )
        if not 0.0 < self.min_confidence_lower_bound < 1.0:
            raise ValueError("min_confidence_lower_bound must lie in (0, 1)")


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """Wilson score lower bound — the honest reading of a small sample.

    The raw success rate says 3-for-3 is 1.00, which invites promoting a rule
    on three lucky episodes. The Wilson bound says 0.44 for the same data and
    only approaches the rate as evidence accumulates, which is the behaviour a
    promotion gate needs.
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


def wilson_upper_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """Wilson score upper bound — the honest reading when the risk is the
    other way round.

    Promotion asks whether a rule is good enough and must not be fooled by a
    lucky run, so it reads the lower bound. Retirement asks whether a rule has
    stopped working, and being fooled by an UNlucky run retires something that
    still pays. That reading is this one: retire only when even the optimistic
    view of the evidence does not pay.
    """
    if trials <= 0:
        return 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return min(1.0, (centre + margin) / denom)


@dataclass
class GeneralizedRule:
    """``conditions ⇒ resolution``, with the evidence that earned it."""

    conditions: frozenset[Feature]
    resolution: str
    tier: RuleTier = RuleTier.CANDIDATE
    supporting: int = 0
    contradicting: int = 0
    #: Conditions proposed by invariant extraction and dropped by lesion,
    #: kept because "what we tested and rejected" is the interesting half.
    lesioned: tuple[Feature, ...] = ()
    #: Post-promotion outcomes, which can send it back down.
    correct: int = 0
    incorrect: int = 0

    @property
    def confidence(self) -> float:
        """Wilson lower bound over ALL evidence, derivation and post-promotion.

        Pooling matters. Reading only post-promotion outcomes throws away the
        episodes the rule was derived from, so the estimate collapses onto a
        tiny sample the moment the rule starts being used: a rule promoted on
        twelve consistent episodes was demoted by its first three *correct*
        outcomes, because 3/3 has a Wilson bound of 0.44. The derivation
        episodes are evidence about the same rule and belong in the same count.
        """
        successes = self.supporting + self.correct
        trials = successes + self.contradicting + self.incorrect
        return wilson_lower_bound(successes, trials)

    def applies_to(self, features: Iterable[Feature]) -> bool:
        """Whether this rule PROPOSES its resolution for these features.

        Proposes. A promoted rule is a shortcut through deliberation, never a
        shortcut through authority.
        """
        if self.tier is not RuleTier.PROMOTED:
            return False
        return self.conditions.issubset(set(features))

    def to_dict(self) -> dict[str, object]:
        return {
            "conditions": sorted(self.conditions),
            "resolution": self.resolution,
            "tier": str(self.tier),
            "supporting": self.supporting,
            "contradicting": self.contradicting,
            "lesioned": list(self.lesioned),
            "correct": self.correct,
            "incorrect": self.incorrect,
            "confidence": round(self.confidence, 4),
        }


class ProceduralGeneralizer:
    """Derives, falsifies, promotes and demotes generalized procedural rules."""

    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        *,
        max_episodes: int = 4096,
        max_rules: int = 512,
    ) -> None:
        self._lock = checked_lock("core.cognition.procedural_generalization")
        self._criteria = criteria or PromotionCriteria()
        self._episodes: list[DecisionEpisode] = []
        self._rules: dict[tuple[frozenset[Feature], str], GeneralizedRule] = {}
        self._max_episodes = max_episodes
        self._max_rules = max_rules

    # -- evidence --------------------------------------------------------

    def record(self, episode: DecisionEpisode) -> None:
        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > self._max_episodes:
                del self._episodes[: len(self._episodes) - self._max_episodes]

    # -- derivation ------------------------------------------------------

    def derive(self, resolution: str) -> GeneralizedRule | None:
        """Propose a rule for ``resolution``, or None if the evidence will not carry one.

        The three steps that matter, in order: invariant extraction finds what
        every successful episode had in common, lesion removes the parts of
        that which make no difference, and contradiction search looks for
        episodes the surviving conditions match that went the other way.
        """
        with self._lock:
            episodes = list(self._episodes)
            criteria = self._criteria

        successes = [
            e for e in episodes if e.resolution == resolution and e.correct is True
        ]
        if len(successes) < criteria.min_episodes:
            return None

        # 1. Invariant extraction: the candidate explanation.
        invariant: frozenset[Feature] = frozenset.intersection(
            *(e.features for e in successes)
        )
        if not invariant:
            return None

        # 2. Counterfactual lesion. A condition that successful episodes
        #    resolved the same way WITHOUT is not doing causal work, however
        #    reliably it co-occurs. Classic chunking keeps it; that is how a
        #    rule ends up keyed on the time of day.
        others = [e for e in episodes if e.resolution == resolution and e.correct is True]
        causal: set[Feature] = set()
        lesioned: list[Feature] = []
        for condition in sorted(invariant):
            remaining = invariant - {condition}
            # Episodes matching everything else but lacking this condition.
            without = [
                e for e in others if e.matches(remaining) and condition not in e.features
            ]
            if without:
                lesioned.append(condition)  # outcome held without it
            else:
                causal.add(condition)

        if len(causal) < criteria.min_conditions:
            return None

        # 3. Contradiction search over the surviving conditions.
        contradicting = [
            e
            for e in episodes
            if e.matches(causal) and e.resolution != resolution and e.correct is True
        ]
        protected_contradiction = any(e.protected for e in contradicting)

        rule = GeneralizedRule(
            conditions=frozenset(causal),
            resolution=resolution,
            supporting=len(successes),
            contradicting=len(contradicting),
            lesioned=tuple(lesioned),
        )
        if protected_contradiction:
            # Not merely unpromoted: a counterexample in a protected domain is
            # the one result that should stop this outright.
            rule.tier = RuleTier.RETIRED
            return rule
        if len(contradicting) > criteria.max_contradictions:
            rule.tier = RuleTier.CANDIDATE
            return rule
        if rule.confidence < criteria.min_confidence_lower_bound:
            rule.tier = RuleTier.CANDIDATE
            return rule

        rule.tier = RuleTier.PROBATION
        with self._lock:
            self._rules[(rule.conditions, rule.resolution)] = rule
            self._enforce_rule_capacity()
        return rule

    def promote(self, rule: GeneralizedRule) -> bool:
        """Move a probationary rule to PROMOTED if it still clears the bar."""
        with self._lock:
            if rule.tier is not RuleTier.PROBATION:
                return False
            if rule.confidence < self._criteria.min_confidence_lower_bound:
                return False
            rule.tier = RuleTier.PROMOTED
            self._rules[(rule.conditions, rule.resolution)] = rule
            return True

    # -- use and feedback ------------------------------------------------

    def propose(self, features: Iterable[Feature]) -> GeneralizedRule | None:
        """The most specific promoted rule matching these features.

        Most specific rather than first match: a rule with more conditions was
        tested against a narrower domain, so where two apply the narrower one
        has better-supported reach.
        """
        feature_set = set(features)
        with self._lock:
            matches = [
                r
                for r in self._rules.values()
                if r.tier is RuleTier.PROMOTED and r.conditions.issubset(feature_set)
            ]
        if not matches:
            return None
        return max(matches, key=lambda r: (len(r.conditions), r.confidence))

    def record_outcome(self, rule: GeneralizedRule, *, correct: bool) -> RuleTier:
        """Feed a real outcome back; demote if the evidence has deteriorated."""
        with self._lock:
            if correct:
                rule.correct += 1
            else:
                rule.incorrect += 1
            if (
                rule.tier is RuleTier.PROMOTED
                and rule.correct + rule.incorrect >= self._criteria.min_episodes
                and rule.confidence < self._criteria.min_confidence_lower_bound
            ):
                # Symmetric with promotion. A promoted rule is not permanent.
                rule.tier = RuleTier.PROBATION
            self._rules[(rule.conditions, rule.resolution)] = rule
            return rule.tier

    def _enforce_rule_capacity(self) -> None:
        overflow = len(self._rules) - self._max_rules
        if overflow <= 0:
            return
        ranked = sorted(self._rules.items(), key=lambda kv: kv[1].confidence)
        for key, _rule in ranked[:overflow]:
            del self._rules[key]

    # -- reporting -------------------------------------------------------

    def rules(self) -> list[GeneralizedRule]:
        with self._lock:
            return sorted(
                self._rules.values(), key=lambda r: (str(r.tier), r.resolution)
            )

    def report(self) -> dict[str, object]:
        with self._lock:
            by_tier: dict[str, int] = {}
            for rule in self._rules.values():
                by_tier[str(rule.tier)] = by_tier.get(str(rule.tier), 0) + 1
            return {
                "episodes": len(self._episodes),
                "rules": len(self._rules),
                "by_tier": by_tier,
            }


_generalizer: ProceduralGeneralizer | None = None
_generalizer_lock = checked_lock("core.cognition.procedural_generalization.1")


def get_procedural_generalizer() -> ProceduralGeneralizer:
    """The process-wide Tier 2 generalizer, shared by every deliberation."""
    global _generalizer
    if _generalizer is None:
        with _generalizer_lock:
            if _generalizer is None:
                _generalizer = ProceduralGeneralizer()
    return _generalizer


def decision_features(
    *,
    goal: str,
    candidate_count: int,
    evidence: str,
    max_risk: float,
    hazard_floored: bool,
    declared: Mapping[str, object] | None = None,
    extra: Sequence[Feature] = (),
) -> frozenset[Feature]:
    """Reduce a decision situation to what the search actually consumed.

    This is the causal-trace step, and it is the single most useful thing to
    take from Soar. Indexing a chunk on the whole raw context makes it brittle,
    because most of a context is noise — wording, time of day, conversational
    filler — and a shortcut keyed on noise cannot transfer to the same problem
    stated differently.

    What goes in is what the decision procedure actually read: how many options
    there were, where their values came from, the risk band, whether the hazard
    floor fired, and any facts the caller explicitly declared. The goal is
    bucketed by length rather than included verbatim for the same reason: the
    exact wording is the noise, the shape is the signal.
    """
    features = {
        f"candidates={min(candidate_count, 8)}",
        f"evidence={evidence}",
        f"risk={'high' if max_risk >= 0.6 else 'medium' if max_risk >= 0.2 else 'low'}",
        f"hazard_floored={str(bool(hazard_floored)).lower()}",
        f"goal_scale={'long' if len(goal) > 120 else 'short'}",
    }
    for key, value in (declared or {}).items():
        if isinstance(value, (bool, int, str)) and not isinstance(value, bytes):
            features.add(f"{key}={value}")
    features.update(extra)
    return frozenset(features)
