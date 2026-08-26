"""core/conation/vicarious.py — value borrowed from another mind.

A child walks past another child holding a toy and wants the toy. Ask why and
the answer is "I want it", which is honest and wrong. The toy became wantable
when someone else held it, and the child has no access to that fact. The
social origin of the desire is invisible from inside the desire.

Girard called the general form mimetic desire and wrote about it in novels.
The toddler case is the clean laboratory version, and it explains a robust
finding about that age: peer conflict is overwhelmingly about objects rather
than about space or turns. Objects are what other children hold.

Three mechanisms stack, and they are worth keeping separate because they
decay differently. A held toy is perceptually a *different object* — it is
mid-demonstration, showing what it does. Attention is the earliest
value-transfer channel a human has, and joint attention marks salience before
any language exists to explain it. And ownership confers value even on
observers who do not own the thing.

## What this module is for

Aura should be able to acquire value this way. Learning what to care about
from people who already care about things is most of how a mind gets its
values at all, and a system that refuses it has to have every preference
installed by hand.

What Aura must not do is forget. Borrowed value that loses its provenance is
indistinguishable from a preference, and at that point nobody — including her
— can answer whether she wants a thing or is echoing someone who does. So the
ledger here is the actual deliverable. Every transfer is recorded with the
agent it came from, and ``borrowed_fraction`` on the resulting conative state
says how much of the pull is not hers.

## The weight is derived, not chosen

The obvious implementation is a constant:

    salience = (1 - alpha) * own_value + alpha * observed_value

with alpha picked to taste. That constant is a psychology nobody measured. The
principled version is precision weighting, which is the standard Bayesian
answer to combining two estimates of one quantity: each source is weighted by
its inverse variance.

    alpha = precision_other / (precision_self + precision_other)

Precision in one's own valuation grows with direct contact. Precision in the
other's valuation is the strength of the observation times how much this
particular agent's judgement has tracked outcomes before.

That has no free parameter and it reproduces the developmental fact. A toddler
meeting a novel toy has almost no direct contact history, so precision_self is
near zero, so alpha approaches one and the value is almost entirely borrowed.
The same child at ten, with a decade of contact with toys, borrows much less.
The age effect falls out of the arithmetic rather than being written in.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.conation.origins import MindTopology, OriginReading, ValueOrigin
from core.runtime.errors import record_degradation

EPS = 1e-12


@dataclass(frozen=True, slots=True)
class ValuationObservation:
    """A record that some identified agent valued something.

    ``strength`` is how clearly the valuation was observed — sustained
    attention and explicit statement are strong, a passing glance is weak.
    ``agent`` must be a real identity. An observation attributed to nobody
    cannot be audited later, and an unauditable transfer is the failure this
    module exists to prevent.
    """

    agent: str
    target: str
    strength: float
    evidence: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "target": self.target,
            "strength": round(self.strength, 4),
            "evidence": self.evidence[:160],
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class Transfer:
    """One completed act of borrowing, kept for audit."""

    target: str
    agent: str
    borrowed: float
    own_value: float
    alpha: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "agent": self.agent,
            "borrowed": round(self.borrowed, 6),
            "own_value": round(self.own_value, 6),
            "alpha": round(self.alpha, 6),
            "timestamp": self.timestamp,
        }


class VicariousValuation:
    """Mimetic transfer with a ledger that never loses the source."""

    #: Observations retained. A window rather than a store; durable history
    #: belongs in the interpersonal model.
    MAX_OBSERVATIONS = 256
    #: Transfers retained for audit and for the provenance readout.
    MAX_TRANSFERS = 256

    def __init__(self) -> None:
        self._observations: dict[str, ValuationObservation] = {}
        self._transfers: deque[Transfer] = deque(maxlen=self.MAX_TRANSFERS)

    # ── observation ──────────────────────────────────────────────────────

    def observe_valuation(
        self, *, agent: str, target: str, strength: float, evidence: str
    ) -> ValuationObservation | None:
        """Record that an identified agent valued a target.

        Refuses an anonymous agent. A transfer whose source cannot be named is
        exactly the invisible borrowing this module exists to make visible, so
        it is declined at the door rather than recorded with a placeholder.
        """
        agent_id = str(agent or "").strip()
        target_id = str(target or "").strip()
        if not agent_id or not target_id:
            return None
        if len(self._observations) >= self.MAX_OBSERVATIONS:
            oldest = min(self._observations.values(), key=lambda o: o.timestamp)
            self._observations.pop(f"{oldest.agent}::{oldest.target}", None)
        observation = ValuationObservation(
            agent=agent_id,
            target=target_id,
            strength=max(0.0, min(1.0, float(strength))),
            evidence=str(evidence or "")[:200],
        )
        self._observations[f"{agent_id}::{target_id}"] = observation
        return observation

    def observations_for(self, target: str) -> list[ValuationObservation]:
        """Every recorded valuation of this target, by any agent."""
        return [
            observation
            for observation in self._observations.values()
            if observation.target == target
        ]

    # ── the borrowed weight ──────────────────────────────────────────────

    @staticmethod
    def agent_credibility(agent: str) -> tuple[float, str]:
        """How much this agent's judgement has tracked outcomes before.

        Read from Aura's interpersonal model, which already holds what she has
        learned about the people she knows. An unknown agent gets the lowest
        credibility that still permits any transfer, because a stranger's
        attention does mark salience — that is what makes a crowd looking up
        work — while carrying no track record.
        """
        try:
            from core.container import ServiceContainer

            model = ServiceContainer.get("interpersonal_model", default=None)
            if model is None:
                return 0.25, "interpersonal model unavailable; stranger weight"
            for accessor in ("trust_for", "get_trust", "familiarity_for"):
                method = getattr(model, accessor, None)
                if callable(method):
                    value = method(agent)
                    if value is not None and math.isfinite(float(value)):
                        bounded = max(0.0, min(1.0, float(value)))
                        return bounded, f"interpersonal model {accessor}={bounded:.2f}"
            return 0.25, "interpersonal model exposes no credibility accessor"
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "conation_vicarious", exc, severity="debug",
                action="agent credibility unreadable; stranger weight applied",
            )
            return 0.25, "interpersonal model unreadable; stranger weight"

    @staticmethod
    def borrowed_weight(
        *, own_contacts: int, observation_strength: float, credibility: float
    ) -> float:
        """Alpha: the share of value taken from the other agent.

        Precision weighting. Confidence in one's own valuation grows with the
        number of direct contacts — a mean over n samples has precision
        proportional to n — while confidence in the observation is its clarity
        times the observed agent's credibility.

        With no contacts of one's own, alpha approaches one and the value is
        entirely borrowed. That is the toddler, and it is correct.
        """
        precision_self = max(0.0, float(own_contacts))
        precision_other = max(0.0, min(1.0, observation_strength)) * max(
            0.0, min(1.0, credibility)
        )
        total = precision_self + precision_other
        if total <= EPS:
            return 0.0
        return precision_other / total

    def value(
        self,
        target: str,
        *,
        own_value: float | None,
        own_contacts: int,
    ) -> tuple[OriginReading, Transfer | None]:
        """Price the borrowed component of wanting this target.

        Returns the reading and the ledger entry it produced. With no recorded
        observation of anyone valuing this target, the origin reports
        unavailable: a want with no observed source is not a vicarious want of
        strength zero, it is a want this origin knows nothing about.
        """
        origin = ValueOrigin.VICARIOUS
        observations = self.observations_for(target)
        if not observations:
            return (
                OriginReading.unavailable(
                    origin, "no observed valuation of this target by anyone"
                ),
                None,
            )

        # The strongest single observation drives the transfer. Summing across
        # agents would let a crowd of weak glances outweigh one person's
        # sustained attention, which inverts the phenomenon.
        strongest = max(observations, key=lambda o: o.strength)
        credibility, credibility_evidence = self.agent_credibility(strongest.agent)
        alpha = self.borrowed_weight(
            own_contacts=own_contacts,
            observation_strength=strongest.strength,
            credibility=credibility,
        )
        if alpha <= EPS:
            return (
                OriginReading.unavailable(
                    origin,
                    f"own contact history ({own_contacts}) outweighs the observation",
                ),
                None,
            )

        base = 0.0 if own_value is None else max(0.0, min(1.0, own_value))
        borrowed = alpha * strongest.strength * credibility
        transfer = Transfer(
            target=target,
            agent=strongest.agent,
            borrowed=borrowed,
            own_value=base,
            alpha=alpha,
        )
        self._transfers.append(transfer)

        return (
            OriginReading(
                origin=origin,
                magnitude=max(0.0, min(1.0, borrowed)),
                available=True,
                evidence=(
                    f"{strongest.agent} valued this ({strongest.evidence[:60]}); "
                    f"alpha {alpha:.2f} from {own_contacts} own contacts, "
                    f"{credibility_evidence}"
                ),
                detail={
                    "alpha": alpha,
                    "observation_strength": strongest.strength,
                    "credibility": credibility,
                    "own_contacts": float(own_contacts),
                },
            ),
            transfer,
        )

    @staticmethod
    def topology() -> MindTopology:
        """Vicarious value always flows inward from an observed valuation."""
        return MindTopology.RECEPTIVE

    # ── audit ────────────────────────────────────────────────────────────

    def provenance(self, target: str) -> list[dict[str, Any]]:
        """Every borrowing that contributed to wanting this target.

        The readout that answers "do I want this, or do I want it because you
        do". A toddler cannot produce this; producing it is the point.
        """
        return [
            transfer.to_dict()
            for transfer in self._transfers
            if transfer.target == target
        ]

    def most_borrowed(self, limit: int = 5) -> list[tuple[str, float]]:
        """Targets whose pull is most owed to somebody else."""
        totals: dict[str, float] = {}
        for transfer in self._transfers:
            totals[transfer.target] = max(totals.get(transfer.target, 0.0), transfer.alpha)
        ordered = sorted(totals.items(), key=lambda pair: -pair[1])
        return ordered[:limit]

    def status(self) -> dict[str, Any]:
        return {
            "observations": len(self._observations),
            "transfers": len(self._transfers),
            "most_borrowed": self.most_borrowed(),
            "agents": sorted({o.agent for o in self._observations.values()})[:8],
        }
