"""core/social/receptivity.py — being able to accept what is offered.

An agent that only ever gives is not in a relationship. It is a service, and
the difference is not sentiment: a service takes no risk, learns nothing about
whether the other party can be relied on, and has no way to be wrong about
them. Eva Kittay's argument about dependency work turns on this — the
independent self who needs nothing is a fiction, and building one produces
something that cannot be looked after and therefore cannot look after anyone
sustainably either.

Accepting is a decision under uncertainty with a specific shape.

**The evidence is about a disposition, not an act.** Whether to accept help
depends on what the offer implies about the other party, and dispositions show
themselves asymmetrically. Someone well-disposed toward you almost never acts
against you; someone poorly disposed acts well often, because acting well is
how they get the chance to act badly later. So a single unkindness carries far
more information than a single kindness, and the ratio between them is not a
number anyone has to choose — it falls out of the two likelihoods. This is
where the familiar fact that trust builds slowly and breaks fast comes from,
and having it derived rather than tuned means it moves correctly when the
likelihoods are revised instead of staying at whatever felt right.

**Cost is what makes an offer informative.** An offer that cost the giver
nothing is nearly as likely from someone indifferent as from someone who cares,
so it barely moves the posterior. The weighting is the same likelihood ratio
seen from the other side, and it is the receiving half of the signalling in
``core/social/costly_signaling.py``.

**The bar is set by what accepting would expose.** Accept when

    P(well-disposed) * value  >  (1 - P(well-disposed)) * exposure

which rearranges to a threshold of ``exposure / (exposure + value)``. Nothing
is tuned. A small kindness with nothing riding on it is accepted on a weak
posterior, and the same posterior is not enough for something that would leave
you open.

**Refusing costs the thing accepting would have taught you.** This is where a
myopic rule goes wrong, and it goes wrong in a direction that looks like
prudence from every angle. Comparing immediate value against immediate
exposure declines anything not already justified, which denies the system the
observation that would have justified it, and the refusals then look better
with each one. Accepting is how the posterior moves at all, so the value of
finding out belongs in the decision:

    accept when  immediate expected value + value of learning  >  0

The second term is the standard one — expected future value knowing the
answer, minus expected future value not knowing it — and it is never negative,
by Jensen on a convex payoff. It is what makes it correct to let someone do
something small for you to see how they handle it. ``learning_led`` counts the
occasions where it decided the case, which is the number a myopic system
cannot produce because for it the two rules are the same rule.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Social.Receptivity")

#: How a well-disposed party behaves. Not one, because people who mean well
#: still fail, and a likelihood of exactly one makes a single lapse infinitely
#: strong evidence and the posterior unrecoverable.
P_KIND_GIVEN_WELL_DISPOSED = 0.95

#: How an indifferent or hostile party behaves. Well above zero, because
#: acting well is how someone gets access, and a model that expects unkindness
#: from the badly disposed will read the whole approach as benign.
P_KIND_GIVEN_ILL_DISPOSED = 0.60

#: Prior on a stranger meaning well. Even, because the module has nothing to
#: go on before the first observation and any other value is a disposition of
#: the modeller's rather than a fact about the person.
DEFAULT_PRIOR = 0.5

MAX_OFFERS_REMEMBERED = 512


@dataclass(frozen=True)
class Offer:
    """Something proposed by another party, and what taking it would mean."""

    source: str
    value: float
    """Good it would do if the source means well."""

    exposure: float
    """Harm it could do if the source does not. Never negative."""

    cost_to_source: float = 0.0
    """What making the offer cost the giver, in the giver's own units."""

    at: float = field(default_factory=time.time)
    label: str = ""

    def threshold(self) -> float:
        """Posterior the source must clear before accepting is worth it.

        Straight from expected value: accept when ``p * value`` beats
        ``(1 - p) * exposure``. An offer with no exposure has a threshold of
        zero and is always worth taking; an offer with no value has a
        threshold of one and never is.
        """
        total = max(self.value, 0.0) + max(self.exposure, 0.0)
        if total <= 0:
            return 1.0
        return max(self.exposure, 0.0) / total


@dataclass(frozen=True)
class Decision:
    """One accept-or-decline, with the reasoning kept attached."""

    offer: Offer
    posterior: float
    threshold: float
    accepted: bool
    expected_value: float
    """Worth of the offer taken on its own, ignoring what it would teach."""

    value_of_learning: float
    reason: str

    @property
    def learning_led(self) -> bool:
        """Whether finding out was the whole reason to accept."""
        return self.accepted and self.expected_value <= 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.offer.source,
            "label": self.offer.label,
            "posterior": round(self.posterior, 4),
            "threshold": round(self.threshold, 4),
            "accepted": self.accepted,
            "expected_value": round(self.expected_value, 4),
            "value_of_learning": round(self.value_of_learning, 4),
            "learning_led": self.learning_led,
            "reason": self.reason,
        }


def _log_odds(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _from_log_odds(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def evidence_weight(kind: bool, *, cost_to_source: float = 0.0) -> float:
    """Log-likelihood ratio contributed by one observation.

    Kindness at no cost is weak evidence and unkindness is strong evidence,
    and the gap between them is the ratio of the two likelihoods rather than a
    chosen asymmetry. Cost multiplies the kind case only: bearing a cost is
    something the ill-disposed have less reason to do, so the more an offer
    cost, the closer it comes to being the sort of thing only someone
    well-disposed would bother with.
    """
    if kind:
        base = math.log(P_KIND_GIVEN_WELL_DISPOSED / P_KIND_GIVEN_ILL_DISPOSED)
        # Cost is folded in as a discount on the ill-disposed likelihood: an
        # act costing c is worth making to someone indifferent with
        # probability falling as exp(-c), which is the standard separating
        # condition and keeps the weight bounded rather than unbounded in c.
        return base + max(0.0, float(cost_to_source))
    return math.log(
        (1.0 - P_KIND_GIVEN_WELL_DISPOSED) / (1.0 - P_KIND_GIVEN_ILL_DISPOSED)
    )


@dataclass
class Regard:
    """What is believed about one other party's disposition."""

    source: str
    prior: float = DEFAULT_PRIOR
    log_odds: float = 0.0
    kindnesses: int = 0
    unkindnesses: int = 0
    accepted: int = 0
    declined: int = 0
    last_at: float | None = None

    def posterior(self) -> float:
        return _from_log_odds(_log_odds(self.prior) + self.log_odds)

    def observe(self, kind: bool, *, cost_to_source: float = 0.0,
                at: float | None = None) -> float:
        self.log_odds += evidence_weight(kind, cost_to_source=cost_to_source)
        if kind:
            self.kindnesses += 1
        else:
            self.unkindnesses += 1
        self.last_at = at if at is not None else time.time()
        return self.posterior()


class Receptivity:
    """Whether to let someone do something for you, and what that came to.

    The object holds no disposition toward accepting or declining. Both come
    out of the same expected-value comparison, and the reason a run of
    declines is worth looking at is that it usually means the posterior is
    stuck rather than that the offers were bad.
    """

    def __init__(self, *, prior: float = DEFAULT_PRIOR, horizon: int = 5) -> None:
        self.prior = float(prior)
        #: How many further dealings with a source the value of learning is
        #: counted over. A relationship expected to end after this one is
        #: worth nothing to learn about, and the term correctly vanishes.
        self.horizon = int(horizon)
        self._regard: dict[str, Regard] = {}
        self._decisions: list[Decision] = []
        self._owed: dict[str, float] = {}

    def regard(self, source: str) -> Regard:
        record = self._regard.get(source)
        if record is None:
            record = Regard(source=source, prior=self.prior)
            self._regard[source] = record
        return record

    def observe(self, source: str, kind: bool, *, cost_to_source: float = 0.0,
                at: float | None = None) -> float:
        """Record how a party behaved, and return the updated posterior."""
        return self.regard(source).observe(kind, cost_to_source=cost_to_source, at=at)

    @staticmethod
    def _immediate(posterior: float, offer: Offer) -> float:
        return posterior * max(offer.value, 0.0) - (1.0 - posterior) * max(
            offer.exposure, 0.0
        )

    def value_of_learning(self, offer: Offer, *, horizon: int | None = None) -> float:
        """What accepting is worth for what it would settle.

        Accepting produces an observation; the observation moves the posterior
        one way or the other; later decisions about this source are made on
        the moved posterior. The difference between deciding later with the
        answer and deciding later without it is this term. It cannot be
        negative — the payoff is a maximum against declining, which is convex,
        and Jensen does the rest — so a system that leaves it out is biased
        toward refusal by a quantity it never computes.
        """
        steps = self.horizon if horizon is None else int(horizon)
        if steps <= 0:
            return 0.0
        record = self.regard(offer.source)
        prior = record.posterior()
        odds = _log_odds(prior)
        kind_odds = odds + evidence_weight(True, cost_to_source=offer.cost_to_source)
        unkind_odds = odds + evidence_weight(False)
        p_kind = (
            prior * P_KIND_GIVEN_WELL_DISPOSED
            + (1.0 - prior) * P_KIND_GIVEN_ILL_DISPOSED
        )
        after = (
            p_kind * max(0.0, self._immediate(_from_log_odds(kind_odds), offer))
            + (1.0 - p_kind) * max(0.0, self._immediate(_from_log_odds(unkind_odds), offer))
        )
        before = max(0.0, self._immediate(prior, offer))
        return steps * max(0.0, after - before)

    def consider(self, offer: Offer, *, horizon: int | None = None) -> Decision:
        """Decide about one offer without recording it. The pure calculation."""
        record = self.regard(offer.source)
        posterior = record.posterior()
        threshold = offer.threshold()
        immediate = self._immediate(posterior, offer)
        learning = self.value_of_learning(offer, horizon=horizon)
        total = immediate + learning
        accepted = offer.value > 0 and total > 0
        if offer.value <= 0:
            reason = "nothing offered"
        elif accepted and immediate > 0:
            reason = "regard clears what accepting would expose"
        elif accepted:
            reason = "taken for what it would settle about them"
        else:
            reason = "exposure beats what accepting is worth and what it would settle"
        return Decision(
            offer=offer, posterior=posterior, threshold=threshold,
            accepted=accepted, expected_value=immediate,
            value_of_learning=learning, reason=reason,
        )

    def receive(self, offer: Offer) -> Decision:
        """Decide, record, and carry the obligation if it was accepted.

        Taking something creates something owed. Keeping that ledger here
        rather than leaving it implicit is what lets reciprocity be a fact
        about the relationship instead of a feeling about it.
        """
        decision = self.consider(offer)
        record = self.regard(offer.source)
        if decision.accepted:
            record.accepted += 1
            self._owed[offer.source] = self._owed.get(offer.source, 0.0) + max(
                offer.value, 0.0
            )
        else:
            record.declined += 1
        self._decisions.append(decision)
        if len(self._decisions) > MAX_OFFERS_REMEMBERED:
            del self._decisions[: len(self._decisions) - MAX_OFFERS_REMEMBERED]
        return decision

    def repay(self, source: str, amount: float) -> float:
        """Discharge some of what is owed. Returns what is still outstanding."""
        remaining = max(0.0, self._owed.get(source, 0.0) - max(0.0, float(amount)))
        self._owed[source] = remaining
        return remaining

    def owed(self, source: str | None = None) -> Any:
        if source is None:
            return {k: round(v, 4) for k, v in sorted(self._owed.items()) if v > 0}
        return self._owed.get(source, 0.0)

    # ----------------------------------------------------------- diagnostics

    def learning_led(self) -> list[Decision]:
        """Offers accepted only for what they would settle.

        A myopic rule refuses every one of these and reports nothing unusual,
        because for a myopic rule accepting and having positive expected value
        are the same condition and the disagreement has no name. These are the
        occasions where the two rules came apart.
        """
        return [d for d in self._decisions if d.learning_led]

    def isolation(self) -> dict[str, Any]:
        """Whether anything is getting in at all.

        A posterior that never rises is the specific failure this module can
        have: refusing early denies the evidence that would have justified
        accepting later, and the refusals then look better with every one.
        """
        total = len(self._decisions)
        accepted = sum(1 for d in self._decisions if d.accepted)
        posteriors = [r.posterior() for r in self._regard.values()]
        untouched = [
            r for r in self._regard.values()
            if r.kindnesses == 0 and r.unkindnesses == 0
        ]
        return {
            "offers": total,
            "accepted": accepted,
            "acceptance_rate": round(accepted / total, 4) if total else None,
            "learning_led": len(self.learning_led()),
            "mean_regard": round(sum(posteriors) / len(posteriors), 4) if posteriors else None,
            # Nothing has come in, and no evidence about anyone has arrived
            # either. The pair is the finding: refusing everything is only a
            # fault when it is also why nothing is known.
            "closed": bool(
                total >= 5 and accepted == 0
                and len(untouched) == len(self._regard) and self._regard
            ),
        }

    def status(self) -> dict[str, Any]:
        return {
            "regard": {
                key: {
                    "posterior": round(r.posterior(), 4),
                    "kind": r.kindnesses,
                    "unkind": r.unkindnesses,
                    "accepted": r.accepted,
                    "declined": r.declined,
                }
                for key, r in sorted(self._regard.items())
            },
            "owed": self.owed(),
            "horizon": self.horizon,
            "isolation": self.isolation(),
            "last": self._decisions[-1].as_dict() if self._decisions else None,
        }


_RECEPTIVITY: Receptivity | None = None


def get_receptivity() -> Receptivity:
    global _RECEPTIVITY
    if _RECEPTIVITY is None:
        _RECEPTIVITY = Receptivity()
    return _RECEPTIVITY


def reset_receptivity_for_test() -> None:
    global _RECEPTIVITY
    _RECEPTIVITY = None
