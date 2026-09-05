"""core/cognition/agent_model.py — what she has learned about someone, from them.

Aura's social ability is largely the cortex's: a pretrained model that knows a
great deal about people in general and nothing about the particular person in
front of it beyond what is in the context window. A model of an individual that
is really a language prior is indistinguishable from one that was learned,
right up until it is wrong about someone unusual.

An :class:`AgentModel` is beliefs about one agent, held as revisable
hypotheses with the interactions that support them. Two things make it more
than a profile:

* **A perspective is not a fact.** ``believes`` holds what Aura thinks THEY
  think, separately from what she thinks. :meth:`AgentModel.false_belief`
  answers the question a theory of mind is for - what will they do given what
  they know, which is not what they would do given what is true. Nesting is
  bounded at depth two, because "she thinks that he thinks that she thinks" is
  where the representation stops paying and starts hallucinating.

* **The prior is a control, not a floor.** Every prediction records what the
  language prior alone would have said. :meth:`AgentModel.beats_the_prior` is
  the number that says whether interaction taught anything, and without it a
  social model is a place to keep text the cortex could have generated.

Reliability is per topic
------------------------
Someone can be exact about dates and vague about names. One reliability number
per agent averages those into something true of neither, so reliability is
tracked per topic and the aggregate is reported as a range rather than a mean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Belief",
    "Prediction",
    "AgentModel",
    "AgentRegistry",
    "MAX_PERSPECTIVE_DEPTH",
    "get_agent_registry",
    "reset_agent_registry_for_test",
]

#: How deep nested perspective goes. Two is "she thinks he thinks"; three is
#: where the representation stops paying and starts inventing.
MAX_PERSPECTIVE_DEPTH = 2


@dataclass
class Belief:
    """Something this agent is thought to hold, and what supports it."""

    proposition: str
    strength: float = 0.5
    supporting: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    #: Whose belief this is a belief about. Empty is the agent themselves;
    #: a name is Aura's model of their model of that person.
    about: str = ""
    depth: int = 1
    updated_at: float = field(default_factory=time.time)

    def revise(self, *, supports: bool, evidence: str) -> None:
        if supports:
            self.supporting = (*self.supporting, evidence)
        else:
            self.contradicting = (*self.contradicting, evidence)
        total = len(self.supporting) + len(self.contradicting)
        self.strength = len(self.supporting) / total if total else 0.5
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition,
            "strength": self.strength,
            "supporting": len(self.supporting),
            "contradicting": len(self.contradicting),
            "about": self.about,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class Prediction:
    """One prediction about an agent, beside what the prior would have said."""

    topic: str
    predicted: str
    prior_predicted: str
    actual: str = ""

    @property
    def correct(self) -> bool | None:
        return None if not self.actual else self.predicted == self.actual

    @property
    def prior_correct(self) -> bool | None:
        return None if not self.actual else self.prior_predicted == self.actual


@dataclass
class AgentModel:
    """One other agent, as Aura has come to understand them."""

    name: str
    beliefs: dict[str, Belief] = field(default_factory=dict)
    goals: dict[str, float] = field(default_factory=dict)
    reliability: dict[str, tuple[int, int]] = field(default_factory=dict)
    predictions: list[Prediction] = field(default_factory=list)
    interactions: int = 0

    def observe_belief(
        self, proposition: str, *, supports: bool, evidence: str, about: str = "", depth: int = 1
    ) -> Belief:
        if depth > MAX_PERSPECTIVE_DEPTH:
            raise ValueError(
                f"perspective depth {depth} exceeds {MAX_PERSPECTIVE_DEPTH}; past this the "
                "representation stops paying and starts inventing"
            )
        key = f"{about}:{proposition}" if about else proposition
        belief = self.beliefs.setdefault(
            key, Belief(proposition=proposition, about=about, depth=depth)
        )
        belief.revise(supports=supports, evidence=evidence)
        return belief

    def observe_reliability(self, topic: str, *, accurate: bool) -> None:
        hits, total = self.reliability.get(topic, (0, 0))
        self.reliability[topic] = (hits + (1 if accurate else 0), total + 1)

    def reliability_range(self) -> dict[str, Any]:
        """Per-topic reliability, reported as a spread rather than a mean.

        Averaging exactness about dates with vagueness about names produces a
        number true of neither, and it is the number a caller will threshold on.
        """
        rates = {t: h / n for t, (h, n) in self.reliability.items() if n}
        if not rates:
            return {"measured": False}
        return {
            "measured": True,
            "by_topic": dict(sorted(rates.items())),
            "lowest": min(rates.items(), key=lambda kv: kv[1]),
            "highest": max(rates.items(), key=lambda kv: kv[1]),
            "spread": max(rates.values()) - min(rates.values()),
        }

    def false_belief(self, proposition: str, *, world_truth: bool) -> dict[str, Any]:
        """What they will act on, which is what they believe, not what is true.

        The classic test: when their belief and the world disagree, a model
        that predicts from the world has no theory of mind, whatever it says.
        """
        belief = self.beliefs.get(proposition)
        if belief is None:
            return {"known": False, "acts_on": None}
        they_believe = belief.strength >= 0.5
        return {
            "known": True,
            "they_believe": they_believe,
            "world_truth": world_truth,
            "acts_on": they_believe,
            "diverges_from_reality": they_believe != world_truth,
        }

    def predict(self, topic: str, predicted: str, prior_predicted: str) -> Prediction:
        prediction = Prediction(topic=topic, predicted=predicted, prior_predicted=prior_predicted)
        self.predictions.append(prediction)
        return prediction

    def resolve(self, index: int, actual: str) -> Prediction:
        prediction = self.predictions[index]
        resolved = Prediction(
            topic=prediction.topic, predicted=prediction.predicted,
            prior_predicted=prediction.prior_predicted, actual=actual,
        )
        self.predictions[index] = resolved
        return resolved

    def beats_the_prior(self) -> dict[str, Any]:
        """Did interaction teach anything the language prior did not already have."""
        resolved = [p for p in self.predictions if p.actual]
        if not resolved:
            return {"measurable": False, "reason": "no prediction has been resolved"}
        model_hits = sum(1 for p in resolved if p.correct)
        prior_hits = sum(1 for p in resolved if p.prior_correct)
        return {
            "measurable": True,
            "n": len(resolved),
            "model_accuracy": model_hits / len(resolved),
            "prior_accuracy": prior_hits / len(resolved),
            "delta": (model_hits - prior_hits) / len(resolved),
            "learned_something": model_hits > prior_hits,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interactions": self.interactions,
            "beliefs": [b.to_dict() for b in self.beliefs.values()],
            "goals": dict(self.goals),
            "reliability": self.reliability_range(),
            "beats_the_prior": self.beats_the_prior(),
        }


class AgentRegistry:
    """Every other agent Aura has a model of."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.agent_model.AgentRegistry", reentrant=True)
        self._agents: dict[str, AgentModel] = {}

    def model(self, name: str) -> AgentModel:
        with self._lock:
            return self._agents.setdefault(name, AgentModel(name=name))

    def interacted(self, name: str) -> None:
        with self._lock:
            self.model(name).interactions += 1

    def report(self) -> dict[str, Any]:
        with self._lock:
            agents = list(self._agents.values())
        learned = [a for a in agents if a.beats_the_prior().get("learned_something")]
        return {
            "agents": len(agents),
            "with_resolved_predictions": sum(
                1 for a in agents if a.beats_the_prior().get("measurable")
            ),
            "beating_the_prior": [a.name for a in learned],
            "by_agent": {a.name: a.to_dict() for a in agents},
        }


_lock = checked_lock("core.cognition.agent_model.singleton")
_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = AgentRegistry()
        return _registry


def reset_agent_registry_for_test() -> AgentRegistry:
    global _registry
    with _lock:
        _registry = AgentRegistry()
        return _registry
