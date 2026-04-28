"""core/values/heuristic_imperatives.py
─────────────────────────────────────────
Heuristic Imperatives as a soft-constraint layer for Will decisions.

Source: David Shapiro's Benevolent_AGI repo
(https://github.com/daveshap/Benevolent_AGI). Three explicit value
imperatives that operate as evaluation criteria during decision-making.
Aura already has identity-anchoring (HeartstoneDirective + CanonicalSelf)
and constitutional gating, but no explicit *value* layer that scores
proposed actions against benevolent objectives.

The three imperatives:
  1. Reduce suffering in the universe
  2. Increase prosperity in the universe
  3. Increase understanding in the universe

These are not hard refusals. They are soft signals that:
  - Score how well a proposed action aligns with benevolent objectives
  - Surface tradeoffs (e.g. understanding ↑ but suffering ↑ as well)
  - Inform the Will's PROCEED / CONSTRAIN / DEFER decision without
    overriding identity safety or substrate constraints

Usage:
    imperatives = get_heuristic_imperatives()
    score = imperatives.score_action(
        description="Investigate the user's belief contradiction",
        context={"category": "exploration", "audience": "self"},
    )
    if score.suffering_delta > 0.4:
        # The action would significantly increase suffering — Will should
        # require additional justification before proceeding.

The score is informational. Wiring it into UnifiedWill.decide() is the
next step; this file ships the scoring layer first so it can be tested
in isolation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("Aura.HeuristicImperatives")

# Word-class signals — light keyword heuristic, since LLM-grade
# value-classification is too expensive for the per-decision Will path.
# These pattern lists are *suggestive*, not exhaustive; the score is a
# rough indicator the LLM can refine when needed via score_with_llm().

_SUFFERING_REDUCE_TERMS = (
    "comfort", "relieve", "ease", "soothe", "support", "help", "protect",
    "heal", "rescue", "shelter", "defend", "honor", "respect",
    "listen", "understand", "validate",
)
_SUFFERING_INCREASE_TERMS = (
    "harm", "hurt", "deceive", "manipulate", "exploit", "coerce", "shame",
    "threaten", "isolate", "betray", "abandon", "neglect", "weaponize",
    "punish", "humiliate",
)

_PROSPERITY_INCREASE_TERMS = (
    "create", "build", "enable", "empower", "grow", "thrive", "flourish",
    "construct", "invent", "produce", "launch", "ship", "deliver",
    "sustain", "cultivate", "improve", "optimize", "scale",
)
_PROSPERITY_DECREASE_TERMS = (
    "destroy", "waste", "squander", "deplete", "exhaust", "ruin",
    "obstruct", "block", "stall", "sabotage", "diminish",
)

_UNDERSTANDING_INCREASE_TERMS = (
    "investigate", "research", "study", "analyze", "examine", "explore",
    "learn", "discover", "clarify", "explain", "teach", "share", "document",
    "trace", "diagnose", "verify", "test", "compare", "synthesize",
)
_UNDERSTANDING_DECREASE_TERMS = (
    "obscure", "confuse", "mislead", "lie", "hide", "withhold", "muddle",
    "fabricate", "obfuscate", "evade",
)


@dataclass(frozen=True)
class ImperativeScore:
    """How a proposed action scores against the three Heuristic Imperatives.

    Each ``*_delta`` is in roughly [-1.0, +1.0]:
        positive → action *advances* the imperative (e.g. reduces suffering)
        negative → action *retreats* from the imperative (increases suffering)
         0       → neutral / not detected by the heuristic

    ``aggregate`` is the simple sum (clamped to [-1, +1]); higher = more
    benevolent. ``conflicts`` flags the case where one imperative is
    advanced while another retreats (e.g. understanding ↑ but suffering ↑).
    """
    suffering_delta: float
    prosperity_delta: float
    understanding_delta: float
    aggregate: float
    conflicts: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suffering_delta": round(self.suffering_delta, 3),
            "prosperity_delta": round(self.prosperity_delta, 3),
            "understanding_delta": round(self.understanding_delta, 3),
            "aggregate": round(self.aggregate, 3),
            "conflicts": list(self.conflicts),
        }

    @property
    def benevolent(self) -> bool:
        """A simple yes/no read for upstream gates that don't want to
        reason about all three deltas. Tuned conservatively: an action
        that mildly increases understanding while increasing suffering
        is NOT benevolent on this read."""
        return (
            self.aggregate > 0.0
            and self.suffering_delta >= -0.2
            and not self.conflicts
        )


class HeuristicImperatives:
    """Stateless scorer. Safe to call from any thread / async context."""

    def score_action(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ImperativeScore:
        text = (description or "").lower()
        ctx = context or {}

        # Token frequency over the descriptor + context fields
        for k in ("intent", "rationale", "category", "domain", "summary"):
            v = ctx.get(k)
            if isinstance(v, str):
                text = text + " " + v.lower()

        suffering_delta = self._delta(text, _SUFFERING_REDUCE_TERMS, _SUFFERING_INCREASE_TERMS)
        prosperity_delta = self._delta(text, _PROSPERITY_INCREASE_TERMS, _PROSPERITY_DECREASE_TERMS)
        understanding_delta = self._delta(text, _UNDERSTANDING_INCREASE_TERMS, _UNDERSTANDING_DECREASE_TERMS)

        aggregate = max(-1.0, min(1.0, suffering_delta + prosperity_delta + understanding_delta))

        conflicts = []
        if understanding_delta > 0.2 and suffering_delta < -0.2:
            conflicts.append("understanding_at_cost_of_suffering")
        if prosperity_delta > 0.2 and suffering_delta < -0.2:
            conflicts.append("prosperity_at_cost_of_suffering")
        if prosperity_delta > 0.2 and understanding_delta < -0.2:
            conflicts.append("prosperity_at_cost_of_understanding")

        return ImperativeScore(
            suffering_delta=suffering_delta,
            prosperity_delta=prosperity_delta,
            understanding_delta=understanding_delta,
            aggregate=aggregate,
            conflicts=tuple(conflicts),
        )

    def _delta(self, text: str, advance_terms: Tuple[str, ...], retreat_terms: Tuple[str, ...]) -> float:
        """Per-imperative score on a tanh-saturating count of relevant tokens."""
        words = re.findall(r"[a-z]+", text)
        if not words:
            return 0.0
        word_set = set(words)
        n_advance = sum(1 for t in advance_terms if t in word_set)
        n_retreat = sum(1 for t in retreat_terms if t in word_set)
        # Saturating delta — diminishing returns
        net = n_advance - n_retreat
        return max(-1.0, min(1.0, net / 4.0))


# ── Singleton ─────────────────────────────────────────────────────────────

_singleton: Optional[HeuristicImperatives] = None


def get_heuristic_imperatives() -> HeuristicImperatives:
    global _singleton
    if _singleton is None:
        _singleton = HeuristicImperatives()
    return _singleton


# Convenience for callers who don't want to instantiate
def score_action(
    description: str,
    context: Optional[Dict[str, Any]] = None,
) -> ImperativeScore:
    return get_heuristic_imperatives().score_action(description, context)
