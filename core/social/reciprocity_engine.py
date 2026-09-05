"""core/social/reciprocity_engine.py — you treat me well and I will treat you well.

The plainest social rule there is, and it has a threshold. Cooperating costs
``c`` and gives the other party ``b``. In a single meeting, defecting wins.
Across a relationship that continues with probability ``w`` each round, the
expected number of further meetings is ``1 / (1 - w)``, and conditional
cooperation beats defection exactly when

    w > c / b

That is Nowak's condition for direct reciprocity, and it is a fact about the
relationship rather than about anyone's character. Below the line, holding to
the rule is a losing strategy against anyone who notices. Above it, the rule
wins without needing anybody to be good. What the rule needs is a future.

## Strict repayment breaks under noise, and generosity is the repair

Tit-for-tat is the obvious way to hold the rule: do what they did last time.
Against a perfect partner it cooperates forever. Add any chance of a mistake —
a message lost, help offered and not noticed, a bad day — and two tit-for-tat
players fall into an alternating echo of retaliation that neither of them
wants and neither can stop, because each is correctly returning what they just
got.

The fix is to forgive at a rate. Nowak and Sigmund's generous tit-for-tat
cooperates after a defection with probability

    min(1 - c/b, (b - c) / (b + c))

which is the largest forgiveness that is still not exploitable. This is worth
being precise about: the generous strategy is not the nice one losing
gracefully. It earns strictly more than strict tit-for-tat in any world with
mistakes in it, and ``compare`` runs the two so the difference is a number.

## What the engine estimates rather than assumes

``w`` comes from how often this relationship has in fact continued, ``b`` and
``c`` come from the recorded value of what was given and what it cost, and the
error rate comes from actions that did not match the intention. All three are
measured, because the threshold is only meaningful against real values and a
threshold checked against invented ones is decoration.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Social.Reciprocity")

#: Exchanges kept per relationship.
MAX_EXCHANGES = 512

#: Exchanges needed before the continuation probability means anything. Below
#: this the estimate is one relationship's worth of accident.
MIN_FOR_ESTIMATE = 5

ALWAYS_DEFECT = "always_defect"
TIT_FOR_TAT = "tit_for_tat"
GENEROUS = "generous_tit_for_tat"
ALWAYS_COOPERATE = "always_cooperate"


@dataclass(frozen=True)
class Exchange:
    """One round of a relationship, from both sides."""

    at: float
    they_cooperated: bool
    we_cooperated: bool
    benefit_received: float = 0.0
    """What their action was worth to us."""

    cost_borne: float = 0.0
    """What our action cost us."""

    intended_cooperation: bool | None = None
    """What we meant to do. Different from ``we_cooperated`` when it went wrong."""


@dataclass
class Relationship:
    """One counterpart, and what the numbers say about holding the rule."""

    person: str
    exchanges: list[Exchange] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    last_at: float | None = None

    def record(self, exchange: Exchange) -> None:
        if self.last_at is not None:
            self.gaps.append(max(0.0, exchange.at - self.last_at))
            if len(self.gaps) > MAX_EXCHANGES:
                del self.gaps[: len(self.gaps) - MAX_EXCHANGES]
        self.last_at = exchange.at
        self.exchanges.append(exchange)
        if len(self.exchanges) > MAX_EXCHANGES:
            del self.exchanges[: len(self.exchanges) - MAX_EXCHANGES]

    def benefit(self) -> float | None:
        """Mean worth of what they do for us."""
        values = [e.benefit_received for e in self.exchanges if e.they_cooperated]
        return float(sum(values) / len(values)) if values else None

    def cost(self) -> float | None:
        """Mean cost to us of doing it for them."""
        values = [e.cost_borne for e in self.exchanges if e.we_cooperated]
        return float(sum(values) / len(values)) if values else None

    def error_rate(self) -> float:
        """Share of our actions that came out other than intended."""
        marked = [e for e in self.exchanges if e.intended_cooperation is not None]
        if not marked:
            return 0.0
        wrong = sum(1 for e in marked if e.we_cooperated != e.intended_cooperation)
        return wrong / len(marked)

    def continuation(self, *, horizon_s: float | None = None,
                     at: float | None = None) -> float | None:
        """Probability this relationship has another round in it.

        Estimated from the record rather than declared. A relationship that
        has continued through many rounds has a high one by construction; a
        long silence since the last round pulls it down, because the shadow of
        the future is shorter for someone who has stopped answering.
        """
        n = len(self.exchanges)
        if n < MIN_FOR_ESTIMATE:
            return None
        # Rounds observed imply a per-round survival of (n - 1) / n before any
        # correction for how long it has been quiet: the one round that has
        # not been followed is the evidence about stopping.
        survival = (n - 1) / n
        if horizon_s and self.last_at is not None:
            moment = at if at is not None else time.time()
            silence = max(0.0, moment - self.last_at)
            typical = (
                sum(self.gaps) / len(self.gaps) if self.gaps else horizon_s
            )
            if typical > 0:
                overdue = silence / typical
                # One typical gap of silence is nothing; several is a signal.
                survival *= 1.0 / (1.0 + max(0.0, overdue - 1.0))
        return min(max(survival, 0.0), 1.0)


def generosity(benefit: float, cost: float) -> float:
    """Nowak and Sigmund's forgiveness rate. The most that is not exploitable."""
    if benefit <= 0:
        return 0.0
    return max(0.0, min(1.0 - cost / benefit, (benefit - cost) / (benefit + cost)))


@dataclass(frozen=True)
class Stance:
    """What to do with this counterpart, and why that."""

    person: str
    strategy: str
    cooperation_stable: bool | None
    continuation: float | None
    threshold: float | None
    """``c / b``. Cooperation holds when continuation clears it."""

    forgiveness: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "strategy": self.strategy,
            "stable": self.cooperation_stable,
            "continuation": None if self.continuation is None else round(self.continuation, 4),
            "threshold": None if self.threshold is None else round(self.threshold, 4),
            "forgiveness": round(self.forgiveness, 4),
            "reason": self.reason,
        }


def play(
    strategy: str,
    *,
    rounds: int,
    error_rate: float,
    benefit: float,
    cost: float,
    opponent: str = TIT_FOR_TAT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run two strategies against each other with mistakes in the channel.

    The simulation exists so ``compare`` can produce a number rather than an
    assertion. Errors are applied to the action actually taken, not to the
    intention, which is what makes the echo appear.
    """
    rng = random.Random(seed)
    forgive = generosity(benefit, cost)

    def act(kind: str, last_other: bool | None) -> bool:
        if kind == ALWAYS_DEFECT:
            return False
        if kind == ALWAYS_COOPERATE:
            return True
        if last_other is None:
            return True
        if last_other:
            return True
        if kind == GENEROUS:
            return rng.random() < forgive
        return False

    a_last: bool | None = None
    b_last: bool | None = None
    a_score = 0.0
    b_score = 0.0
    retaliations = 0
    for _ in range(rounds):
        a_intent = act(strategy, b_last)
        b_intent = act(opponent, a_last)
        a_move = (not a_intent) if rng.random() < error_rate else a_intent
        b_move = (not b_intent) if rng.random() < error_rate else b_intent
        if a_move:
            a_score -= cost
            b_score += benefit
        if b_move:
            b_score -= cost
            a_score += benefit
        if not a_move and not b_move:
            retaliations += 1
        a_last, b_last = a_move, b_move
    return {
        "strategy": strategy,
        "opponent": opponent,
        "score": round(a_score / max(rounds, 1), 4),
        "opponent_score": round(b_score / max(rounds, 1), 4),
        "mutual_defections": retaliations,
        "forgiveness": round(forgive, 4),
    }


def compare(*, rounds: int = 2000, error_rate: float = 0.05,
            benefit: float = 3.0, cost: float = 1.0,
            seed: int | None = 0) -> dict[str, Any]:
    """Strict repayment against generous repayment, in a world with mistakes."""
    strict = play(TIT_FOR_TAT, rounds=rounds, error_rate=error_rate,
                  benefit=benefit, cost=cost, opponent=TIT_FOR_TAT, seed=seed)
    generous = play(GENEROUS, rounds=rounds, error_rate=error_rate,
                    benefit=benefit, cost=cost, opponent=GENEROUS, seed=seed)
    return {
        "strict": strict,
        "generous": generous,
        "gain": round(generous["score"] - strict["score"], 4),
        "error_rate": error_rate,
    }


class ReciprocityEngine:
    """Whether the rule holds with this person, and what to do about it.

    Everything the engine says about a relationship comes from what has been
    recorded about it. A relationship with too little history gets no verdict,
    which is a better answer than a number the caller cannot tell apart from a
    measured one.
    """

    def __init__(self, *, horizon_s: float = 30 * 86400.0) -> None:
        self.horizon_s = float(horizon_s)
        self._relationships: dict[str, Relationship] = {}

    def relationship(self, person: str) -> Relationship:
        record = self._relationships.get(person)
        if record is None:
            record = Relationship(person=person)
            self._relationships[person] = record
        return record

    def record_exchange(
        self,
        person: str,
        *,
        they_cooperated: bool,
        we_cooperated: bool,
        benefit_received: float = 0.0,
        cost_borne: float = 0.0,
        intended_cooperation: bool | None = None,
        at: float | None = None,
    ) -> None:
        self.relationship(person).record(
            Exchange(
                at=at if at is not None else time.time(),
                they_cooperated=they_cooperated, we_cooperated=we_cooperated,
                benefit_received=float(benefit_received),
                cost_borne=float(cost_borne),
                intended_cooperation=intended_cooperation,
            )
        )

    def stance(self, person: str, *, at: float | None = None) -> Stance:
        """What the numbers say to do with this person."""
        record = self.relationship(person)
        w = record.continuation(horizon_s=self.horizon_s, at=at)
        b = record.benefit()
        c = record.cost()
        if w is None or b is None or c is None or b <= 0:
            return Stance(
                person=person, strategy=TIT_FOR_TAT, cooperation_stable=None,
                continuation=w, threshold=None, forgiveness=0.0,
                reason=(
                    "too little history to say; returning what was given is the "
                    "opening move with the least to lose"
                ),
            )
        threshold = c / b
        stable = w > threshold
        forgive = generosity(b, c)
        if not stable:
            return Stance(
                person=person, strategy=ALWAYS_DEFECT, cooperation_stable=False,
                continuation=w, threshold=threshold, forgiveness=0.0,
                reason=(
                    "what it costs to help exceeds what the relationship is "
                    "likely to return; the rule does not hold here"
                ),
            )
        if record.error_rate() > 0:
            return Stance(
                person=person, strategy=GENEROUS, cooperation_stable=True,
                continuation=w, threshold=threshold, forgiveness=forgive,
                reason="things go wrong between us, so a strict ledger would echo",
            )
        return Stance(
            person=person, strategy=TIT_FOR_TAT, cooperation_stable=True,
            continuation=w, threshold=threshold, forgiveness=0.0,
            reason="the relationship is long enough for returning in kind to pay",
        )

    def balance(self, person: str) -> float:
        """Net of what they have done for us against what we have done for them."""
        record = self.relationship(person)
        given = sum(e.cost_borne for e in record.exchanges if e.we_cooperated)
        received = sum(e.benefit_received for e in record.exchanges if e.they_cooperated)
        return float(received - given)

    # The two names the earlier stub exposed, kept working. The index is now
    # the balance in the units the exchanges were recorded in rather than a
    # score with no dimension.
    def get_reciprocity_index(self, person: str) -> float:
        return self.balance(person)

    def record_transaction(self, person: str, agent_helped: bool,
                           human_helped: bool) -> None:
        self.record_exchange(
            person, they_cooperated=human_helped, we_cooperated=agent_helped,
            benefit_received=1.0 if human_helped else 0.0,
            cost_borne=1.0 if agent_helped else 0.0,
        )

    def status(self, *, at: float | None = None) -> dict[str, Any]:
        return {
            person: {
                "exchanges": len(r.exchanges),
                "balance": round(self.balance(person), 4),
                "error_rate": round(r.error_rate(), 4),
                "stance": self.stance(person, at=at).as_dict(),
            }
            for person, r in sorted(self._relationships.items())
        }


_ENGINE: ReciprocityEngine | None = None


def get_reciprocity_engine() -> ReciprocityEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ReciprocityEngine()
    return _ENGINE


def reset_reciprocity_engine_for_test() -> None:
    global _ENGINE
    _ENGINE = None
