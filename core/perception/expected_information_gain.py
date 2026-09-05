"""core/perception/expected_information_gain.py — deciding what to look at.

Aura's perception is receptive: a camera sends an image and something
describes it. The observations she gets are the ones that happened to arrive.

The version worth having chooses. She is unsure whether X is true; some
observations would discriminate between the possibilities and most would not;
she takes the one that would. That is active sensing, and it is the difference
between a system that has senses and one that uses them.

The quantity is expected information gain — how much the entropy over her
hypotheses would fall, averaged over what the observation might return. It is
computable from three things she already has: a hypothesis set with
probabilities (``core/perception/belief_state.py`` keeps them), a model of
what each observation would show under each hypothesis, and the cost of
making it.

Two properties matter more than the arithmetic.

**An observation that cannot discriminate has zero gain however interesting it
looks.** If every hypothesis predicts the same reading, taking it changes
nothing, and a system that scores observations by salience rather than by
discrimination will keep taking it.

**Gain is not worth.** An observation that would settle the question and costs
more than the question is worth should not be made, which is the same
subtraction :mod:`core.cognition.value_of_computation` does for thinking. The
two are the same decision about different resources, and both come back
"already settled" when nothing further could change the answer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Perception.EIG")

#: Probability floor. A hypothesis at exactly zero contributes nothing to
#: entropy and cannot be revived by evidence, which is not a belief state
#: anybody should be in.
EPSILON = 1e-9

#: Bits of expected gain below which an observation is not discriminating.
#: Small rather than zero: floating-point noise on a non-discriminating
#: observation is not information.
MIN_INFORMATIVE_BITS = 1e-6

#: Entropy below which the question is settled and no observation is worth
#: making. A different quantity from the one above and it needs its own
#: number: reusing the noise floor here made SETTLED unreachable, because a
#: belief at 99.9999% still carries more entropy than floating-point error.
#: 0.02 bits is about 99.8% on one hypothesis.
SETTLED_BELOW_BITS = 0.02


class Recommendation(StrEnum):
    """What to do about an observation."""

    #: It discriminates and is worth what it costs.
    TAKE = "take"
    #: It discriminates and costs more than the question is worth.
    TOO_EXPENSIVE = "too_expensive"
    #: Every hypothesis predicts the same thing. Taking it changes nothing.
    UNINFORMATIVE = "uninformative"
    #: There is nothing left to be unsure about.
    SETTLED = "settled"


@dataclass(frozen=True)
class Observation:
    """Something she could do, and what it would show.

    ``likelihoods[outcome][hypothesis]`` is how likely that outcome is if that
    hypothesis is true. This is the model of the instrument, and writing it
    down is most of the work — an observation nobody can say that much about
    cannot be scored, which is a better answer than scoring it anyway.
    """

    name: str
    likelihoods: Mapping[str, Mapping[str, float]]
    #: What making it costs, in the same unit as the value of the question.
    cost: float = 0.0
    description: str = ""

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(sorted(self.likelihoods))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcomes": list(self.outcomes),
            "cost": self.cost,
            "description": self.description,
        }


@dataclass(frozen=True)
class Score:
    """What one observation is worth, and whether to make it."""

    observation: str
    #: Bits the entropy would fall by, averaged over what it might show.
    expected_bits: float
    #: Entropy over the hypotheses now.
    prior_bits: float
    cost: float
    value: float
    recommendation: Recommendation
    because: str

    @property
    def take(self) -> bool:
        return self.recommendation is Recommendation.TAKE

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "expected_bits": round(self.expected_bits, 5),
            "prior_bits": round(self.prior_bits, 5),
            "cost": round(self.cost, 5),
            "value": round(self.value, 5),
            "recommendation": str(self.recommendation),
            "take": self.take,
            "because": self.because,
        }


def _normalise(beliefs: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(EPSILON, float(p)) for p in beliefs.values())
    if total <= 0.0:
        count = max(1, len(beliefs))
        return {k: 1.0 / count for k in beliefs}
    return {k: max(EPSILON, float(p)) / total for k, p in beliefs.items()}


def entropy(beliefs: Mapping[str, float]) -> float:
    """Shannon entropy in bits. Zero when one hypothesis has all the mass."""
    normalised = _normalise(beliefs)
    return -sum(p * math.log2(p) for p in normalised.values() if p > 0.0)


def posterior(
    beliefs: Mapping[str, float], likelihoods: Mapping[str, float]
) -> dict[str, float]:
    """Beliefs after seeing one outcome."""
    prior = _normalise(beliefs)
    weighted = {
        h: prior[h] * max(EPSILON, float(likelihoods.get(h, EPSILON))) for h in prior
    }
    return _normalise(weighted)


def expected_information_gain(
    beliefs: Mapping[str, float], observation: Observation
) -> float:
    """Bits the entropy is expected to fall by if this observation is made.

    Averaged over the outcomes weighted by how likely they are *under her
    current beliefs*, which is what makes it an expectation she can act on
    rather than a best case she cannot.
    """
    prior = _normalise(beliefs)
    before = entropy(prior)
    expected_after = 0.0
    for outcome in observation.outcomes:
        per_hypothesis = observation.likelihoods[outcome]
        marginal = sum(
            prior[h] * max(EPSILON, float(per_hypothesis.get(h, EPSILON)))
            for h in prior
        )
        if marginal <= 0.0:
            continue
        expected_after += marginal * entropy(posterior(prior, per_hypothesis))
    total = sum(
        sum(
            prior[h] * max(EPSILON, float(observation.likelihoods[o].get(h, EPSILON)))
            for h in prior
        )
        for o in observation.outcomes
    )
    if total > 0.0:
        expected_after /= total
    return max(0.0, before - expected_after)


def score(
    beliefs: Mapping[str, float],
    observation: Observation,
    *,
    value_per_bit: float = 1.0,
) -> Score:
    """What this observation is worth, and whether to make it."""
    prior_bits = entropy(beliefs)
    gain = expected_information_gain(beliefs, observation)
    value = gain * float(value_per_bit) - float(observation.cost)

    if prior_bits <= SETTLED_BELOW_BITS:
        return Score(
            observation=observation.name,
            expected_bits=gain,
            prior_bits=prior_bits,
            cost=observation.cost,
            value=value,
            recommendation=Recommendation.SETTLED,
            because=(
                f"{prior_bits:.4f} bits of uncertainty is under the "
                f"{SETTLED_BELOW_BITS} that would make looking worthwhile"
            ),
        )
    if gain <= MIN_INFORMATIVE_BITS:
        return Score(
            observation=observation.name,
            expected_bits=gain,
            prior_bits=prior_bits,
            cost=observation.cost,
            value=value,
            recommendation=Recommendation.UNINFORMATIVE,
            because=(
                "every hypothesis predicts the same reading, so making it "
                "changes nothing about what she believes"
            ),
        )
    if value <= 0.0:
        return Score(
            observation=observation.name,
            expected_bits=gain,
            prior_bits=prior_bits,
            cost=observation.cost,
            value=value,
            recommendation=Recommendation.TOO_EXPENSIVE,
            because=(
                f"{gain:.3f} bits at {value_per_bit:.3f} a bit does not repay "
                f"{observation.cost:.3f}"
            ),
        )
    return Score(
        observation=observation.name,
        expected_bits=gain,
        prior_bits=prior_bits,
        cost=observation.cost,
        value=value,
        recommendation=Recommendation.TAKE,
        because=(
            f"it would cut {gain:.3f} of {prior_bits:.3f} bits, which repays "
            f"{observation.cost:.3f}"
        ),
    )


def choose(
    beliefs: Mapping[str, float],
    observations: Sequence[Observation],
    *,
    value_per_bit: float = 1.0,
) -> tuple[Score, ...]:
    """Rank what she could look at, best first.

    Ties break by name rather than by list order, so the choice does not
    depend on which observation somebody happened to write down first.
    """
    scored = [score(beliefs, o, value_per_bit=value_per_bit) for o in observations]
    scored.sort(key=lambda s: (-s.value, -s.expected_bits, s.observation))
    return tuple(scored)


def best(
    beliefs: Mapping[str, float],
    observations: Sequence[Observation],
    *,
    value_per_bit: float = 1.0,
) -> Score | None:
    """The one worth making, or None. None is a real answer."""
    for candidate in choose(beliefs, observations, value_per_bit=value_per_bit):
        if candidate.take:
            return candidate
    return None


__all__ = [
    "EPSILON",
    "MIN_INFORMATIVE_BITS",
    "SETTLED_BELOW_BITS",
    "Observation",
    "Recommendation",
    "Score",
    "best",
    "choose",
    "entropy",
    "expected_information_gain",
    "posterior",
]
