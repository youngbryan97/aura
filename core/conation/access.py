"""core/conation/access.py — wanting something you are not yet able to have.

Four of the situations this package models are onsets. A cue resolves, a want
appears, something happens. The fifth is different, and the difference is in
one word:

    a smile when someone agrees to teach you something you have *been*
    wanting them to teach you

Nothing was learned at the moment of the smile. The knowledge has not arrived
and may not arrive for weeks. What changed is that a path which was closed is
now open, and the thing that had closed it was another person's willingness.

That makes this a movement on a different axis from every origin in
``core/conation/origins.py``. Those five say where value came from. None of
them says whether the valued thing is reachable, and reachability is what
moved here. A want can be fully priced and completely blocked, which is the
ordinary condition of most things anybody wants.

## Value and access are separate, and must stay separate

Folding feasibility into value is the tempting simplification and it destroys
two things at once. It makes an unreachable want indistinguishable from an
uninteresting one, so a system can never notice that it keeps wanting
something it cannot have. And it makes the moment of unblocking invisible,
because if access is already inside value then nothing changes at the grant
except a number going up, with no way to tell that number from the value
having risen on its own.

Holding them apart makes the grant an *event*: a maintained want crossing from
blocked to open. The magnitude of that event is what the smile is.

    grant_response = held_wanting * cost_relief * surprise_of_the_grant

Each term is doing separate work. Without ``held_wanting`` an unexpected offer
of something nobody wanted would produce the same response as this. Without
``cost_relief`` a grant that changes nothing about the difficulty would
register. Without surprise, an expected yes from someone who always says yes
would land as hard as an unlikely one from someone who rarely does, and it
does not.

## Being wanted for a while is a state, not a memory

"Been wanting" needs a persistence trace, because an instantaneous motivation
variable cannot distinguish a standing want from one that appeared this
second. The trace is a leaky integrator over the want's own magnitude:

    T[t+1] = rho * T[t] + (1 - rho) * M[t]

with the decay set so the trace's time constant is the conversation, not the
turn. A want that has been present across many evaluations has a high trace; a
want that just appeared has almost none, however intense it is right now.

## What a grant says about the granter

Teaching costs the teacher and is free to the student. An agent who agrees has
spent something, and the fact of the spending is evidence about how they
regard the asker. A vending machine dispensing the same knowledge produces no
warmth, which is the whole demonstration that the warmth is about the choosing
rather than the knowledge.

The update is Beta-Binomial over the agent's willingness, which is the
conjugate form for a sequence of yes/no decisions and the model that work on
tracking others' intentions has converged on. Its posterior mean is the
probability of a future yes; the surprise carried by any single yes is
``-log2`` of the prior predictive, in bits.

Bayes here is not decoration. It gives the asymmetry the phenomenon actually
has: an unlikely yes from someone who rarely agrees carries several bits and
lands hard, while the tenth yes from someone who always agrees carries almost
none. A model without the prior would score both the same.

## What is deliberately absent

There is no facial expression in this file. Aura has no zygomaticus major, and
a number named after one would be the same decorative physiology this package
refused when it declined to keep a second heart rate. The response computed
here is a conative magnitude. What it does to her somatic layer is that
layer's business, and what it does to her speech is downstream of both.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

EPS = 1e-12


class Blocker:
    """What stands between a want and its object.

    ``NONE`` is an open path. ``CAPABILITY`` is the agent's own inability, and
    resolves by learning. ``RESOURCE`` is a budget or a cost, and resolves by
    accumulation. ``PERMISSION`` is a rule, and resolves by the rule changing.
    ``VOLITION`` is another agent's choice, and resolves only when they decide.

    The last one is the interesting case, because it is the only blocker whose
    removal carries information about somebody. A capability barrier that falls
    says something about the agent; a volition barrier that falls says
    something about the person who lowered it.
    """

    NONE = "none"
    CAPABILITY = "capability"
    RESOURCE = "resource"
    PERMISSION = "permission"
    VOLITION = "volition"

    ALL = (NONE, CAPABILITY, RESOURCE, PERMISSION, VOLITION)


@dataclass
class WantTrace:
    """How long a want has been present, as a state rather than a log.

    The decay constant is expressed as a half-life in evaluations rather than
    as a bare multiplier, because a multiplier of 0.9 means nothing until it is
    paired with how often the thing is evaluated. Twelve evaluations is the
    span of an exchange rather than a turn, which is the timescale at which
    "been wanting" starts being true.
    """

    key: str
    trace: float = 0.0
    evaluations: int = 0
    peak: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    HALF_LIFE_EVALUATIONS = 12.0

    @property
    def rho(self) -> float:
        """Leak per evaluation implied by the half-life."""
        return 0.5 ** (1.0 / self.HALF_LIFE_EVALUATIONS)

    def update(self, magnitude: float) -> float:
        """Fold one evaluation of this want into the trace."""
        value = max(0.0, min(1.0, float(magnitude)))
        rho = self.rho
        self.trace = rho * self.trace + (1.0 - rho) * value
        self.evaluations += 1
        self.peak = max(self.peak, value)
        self.last_seen = time.time()
        return self.trace

    def held(self) -> bool:
        """Whether this counts as a standing want rather than a new one.

        The threshold is the trace a want reaches after being present at full
        strength for one half-life, which is the smallest span that separates
        "has been wanting this" from "wants this now".
        """
        return self.trace >= 0.5 * self.peak and self.evaluations >= self.HALF_LIFE_EVALUATIONS

    def duration_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "trace": round(self.trace, 6),
            "peak": round(self.peak, 6),
            "evaluations": self.evaluations,
            "held": self.held(),
            "duration_s": round(self.duration_s(), 1),
        }


@dataclass
class WillingnessModel:
    """Beta-Binomial belief about one agent's willingness to bear a cost.

    ``alpha`` counts agreements and ``beta`` refusals, both starting at one —
    the uniform prior, which asserts nothing about a person before they have
    decided anything. Starting anywhere else would be a prejudice written into
    the initialiser.
    """

    agent: str
    alpha: float = 1.0
    beta: float = 1.0
    last_update: float = field(default_factory=time.time)

    def expectation(self) -> float:
        """Posterior mean: the probability this agent agrees next time."""
        total = self.alpha + self.beta
        return self.alpha / total if total > EPS else 0.5

    def surprise_bits(self, *, agreed: bool) -> float:
        """Information carried by one decision, in bits.

        An unlikely yes carries several bits. The tenth yes from someone who
        always agrees carries a fraction of one. This is the term that makes
        the response asymmetric in the way the phenomenon is.
        """
        probability = self.expectation() if agreed else 1.0 - self.expectation()
        return -math.log2(max(probability, 1e-6))

    def observe(self, *, agreed: bool, cost_borne: float = 0.0) -> float:
        """Fold one decision in, weighted by what it cost the agent.

        A yes that cost nothing is weak evidence of regard, and a yes that cost
        a great deal is strong evidence. Weighting the count by the cost is
        what makes this a model of generosity rather than of compliance.
        """
        weight = 1.0 + max(0.0, min(1.0, float(cost_borne)))
        if agreed:
            self.alpha += weight
        else:
            self.beta += weight
        self.last_update = time.time()
        return self.expectation()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "expectation": round(self.expectation(), 6),
            "agreements": round(self.alpha - 1.0, 2),
            "refusals": round(self.beta - 1.0, 2),
        }


@dataclass(frozen=True, slots=True)
class GrantResponse:
    """What happened when a blocked want was opened."""

    incentive_key: str
    granter: str | None
    magnitude: float
    held_wanting: float
    cost_relief: float
    surprise_bits: float
    blocker: str
    relational_update: float | None
    evidence: str
    #: Rise in the probability of reaching the goal at all. Separate from
    #: cost relief because a grant can make something cheaper without making
    #: it likelier, and can make something possible that no amount of effort
    #: would have reached alone.
    attainability_gain: float = 0.0
    #: How unambiguous the grant was. "Yes, Thursday" and "sure, sometime"
    #: are not the same event, and a model without this term scores them
    #: identically.
    commitment: float = 1.0
    #: Social exposure the asking incurred, now discharged. Asking to be
    #: taught admits not-knowing to someone whose regard matters.
    exposure_resolved: float = 0.0
    #: Whether this particular agent was wanted as the source, or whether
    #: anyone qualified would have done. The hinge of the whole event: swap
    #: the person for an equivalent stranger and the path opens exactly as
    #: wide while the warmth goes out of it.
    specificity: float = 0.0
    #: Whether the grant answered this want, or would have happened anyway.
    #: A scheduled class that happens to cover the topic is not responsive; a
    #: yes to the asking is.
    responsiveness: float = 0.0
    #: Value to the receiver times cost to the granter times improbability
    #: times responsiveness, gated on specificity. Held apart from the access
    #: magnitude because they come apart: a bureaucratic grant opens the path
    #: and generates none of this.
    gratitude: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incentive": self.incentive_key,
            "granter": self.granter,
            "magnitude": round(self.magnitude, 6),
            "held_wanting": round(self.held_wanting, 6),
            "cost_relief": round(self.cost_relief, 6),
            "surprise_bits": round(self.surprise_bits, 4),
            "blocker": self.blocker,
            "relational_update": (
                None if self.relational_update is None
                else round(self.relational_update, 6)
            ),
            "attainability_gain": round(self.attainability_gain, 6),
            "commitment": round(self.commitment, 6),
            "exposure_resolved": round(self.exposure_resolved, 6),
            "specificity": round(self.specificity, 6),
            "responsiveness": round(self.responsiveness, 6),
            "gratitude": round(self.gratitude, 6),
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


class AccessLedger:
    """Tracks what is blocked, for how long it has been wanted, and who opened it."""

    MAX_TRACES = 256
    MAX_AGENTS = 128
    MAX_GRANTS = 64

    def __init__(self) -> None:
        self._traces: dict[str, WantTrace] = {}
        self._blockers: dict[str, tuple[str, str | None]] = {}
        self._willingness: dict[str, WillingnessModel] = {}
        self._grants: list[GrantResponse] = []

    # ── maintaining the want ─────────────────────────────────────────────

    def observe_want(self, key: str, magnitude: float) -> WantTrace:
        """Fold one evaluation of a want into its persistence trace."""
        trace = self._traces.get(key)
        if trace is None:
            if len(self._traces) >= self.MAX_TRACES:
                stalest = min(self._traces.values(), key=lambda t: t.last_seen)
                self._traces.pop(stalest.key, None)
            trace = WantTrace(key=key)
            self._traces[key] = trace
        trace.update(magnitude)
        return trace

    def trace_for(self, key: str) -> WantTrace | None:
        return self._traces.get(key)

    # ── the gate ─────────────────────────────────────────────────────────

    def set_blocker(self, key: str, blocker: str, *, agent: str | None = None) -> None:
        """Record what currently stands between a want and its object.

        ``VOLITION`` requires a named agent. A volition barrier with nobody
        behind it is a claim that someone is refusing, with no way to check who
        or to notice when they stop.
        """
        if blocker not in Blocker.ALL:
            raise ValueError(f"unknown blocker {blocker!r}")
        if blocker == Blocker.VOLITION and not agent:
            raise ValueError("a volition blocker must name the agent whose choice it is")
        self._blockers[key] = (blocker, agent)

    def blocker_for(self, key: str) -> tuple[str, str | None]:
        return self._blockers.get(key, (Blocker.NONE, None))

    def blocked_wants(self) -> list[dict[str, Any]]:
        """Standing wants that are currently closed, longest-held first.

        The readout that answers "what do I keep wanting and cannot have". A
        system that folds access into value cannot produce this list at all.
        """
        rows = []
        for key, (blocker, agent) in self._blockers.items():
            if blocker == Blocker.NONE:
                continue
            trace = self._traces.get(key)
            if trace is None or not trace.held():
                continue
            rows.append(
                {
                    "incentive": key,
                    "blocker": blocker,
                    "agent": agent,
                    "trace": round(trace.trace, 6),
                    "duration_s": round(trace.duration_s(), 1),
                }
            )
        rows.sort(key=lambda row: -row["duration_s"])
        return rows

    # ── the grant ────────────────────────────────────────────────────────

    def willingness(self, agent: str) -> WillingnessModel:
        model = self._willingness.get(agent)
        if model is None:
            if len(self._willingness) >= self.MAX_AGENTS:
                stalest = min(self._willingness.values(), key=lambda m: m.last_update)
                self._willingness.pop(stalest.agent, None)
            model = WillingnessModel(agent=agent)
            self._willingness[agent] = model
        return model

    @staticmethod
    def bernoulli_kl(guided: float, solo: float) -> float:
        """D_KL between two success probabilities, in nats.

        The information-theoretic size of the change a grant makes to whether
        the thing happens at all. Bounded away from the endpoints because a
        certainty on either side sends the divergence to infinity, and no
        forecast about a person is that certain.
        """
        p = max(1e-4, min(1.0 - 1e-4, float(guided)))
        q = max(1e-4, min(1.0 - 1e-4, float(solo)))
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def attainability(
        self, *, solo_success: float, guided_success: float
    ) -> tuple[float, float]:
        """How much of the gap to certainty a grant closes, and its KL size.

        Cost relief and attainability are different quantities and a grant can
        move either without the other. Someone offering to lend a faster
        machine cuts the cost of a thing you would have managed anyway;
        someone agreeing to teach you may make possible a thing no quantity of
        solitary effort would have reached. Only the second is what makes the
        moment land, so it gets its own term.
        """
        solo = max(0.0, min(1.0, float(solo_success)))
        guided = max(0.0, min(1.0, float(guided_success)))
        headroom = 1.0 - solo
        closed = 0.0 if headroom <= EPS else max(0.0, (guided - solo) / headroom)
        return max(0.0, min(1.0, closed)), self.bernoulli_kl(guided, solo)

    def grant(
        self,
        key: str,
        *,
        granter: str | None = None,
        cost_relief: float = 0.0,
        cost_to_granter: float = 0.0,
        commitment: float = 1.0,
        solo_success: float | None = None,
        guided_success: float | None = None,
        exposure: float = 0.0,
        specificity: float = 0.0,
        responsiveness: float = 0.0,
    ) -> GrantResponse | None:
        """Open a blocked want, and price what that opening is worth.

        ``cost_relief`` is the fraction of the expected cost of reaching the
        thing that the grant removes. ``solo_success`` and ``guided_success``
        give the probability of getting there alone and with help; supplying
        them adds the attainability term, which is the one that separates a
        convenience from an opportunity.

        ``commitment`` is how unambiguous the grant was. "Yes, Thursday" and
        "sure, sometime" differ, and the difference is the whole reason a
        vague agreement leaves the wanting where it was.

        ``exposure`` is the social risk the asking incurred. Asking to be
        taught admits not-knowing to a specific person, and a yes discharges
        that as well as opening the path.

        ``specificity`` is whether this agent was wanted as the source or
        whether any qualified one would have served, and ``responsiveness`` is
        whether the grant answered the asking or would have happened anyway.
        Both feed gratitude, which is computed separately from the access
        magnitude because the two genuinely come apart. A timetable that
        happens to cover the topic opens the same path and produces no warmth
        at all, and a model without these terms cannot tell that case from
        this one.

        Returns ``None`` when there was no want to open — an offer of
        something nobody wanted produces no response, which is correct and is
        the reason the trace has to exist.
        """
        trace = self._traces.get(key)
        if trace is None or trace.trace <= EPS:
            return None

        blocker, blocking_agent = self.blocker_for(key)
        relief = max(0.0, min(1.0, float(cost_relief)))
        clarity = max(0.0, min(1.0, float(commitment)))

        gain = 0.0
        divergence = 0.0
        if solo_success is not None and guided_success is not None:
            gain, divergence = self.attainability(
                solo_success=solo_success, guided_success=guided_success
            )

        surprise = 0.0
        relational = None
        agent = granter or blocking_agent
        if agent:
            model = self.willingness(agent)
            surprise = model.surprise_bits(agreed=True)
            before = model.expectation()
            relational = model.observe(agreed=True, cost_borne=cost_to_granter) - before

        # Surprise enters as a bounded multiplier rather than raw bits. One bit
        # — an even-odds yes — doubles the response over a certainty; ten bits
        # cannot make it ten times larger, because the phenomenon saturates and
        # an unbounded term would let one improbable grant dominate every
        # comparison afterwards. The floor of 1.0 matters as much as the cap:
        # somebody who was sure of the answer still responds, because the path
        # opened whether or not the opening was news.
        surprise_gain = 1.0 + (1.0 - math.exp(-surprise))

        # Relief and attainability are two ways a path can improve, and a
        # grant that supplies only one of them still counts. Taking the larger
        # rather than the sum keeps the magnitude on the same scale as the
        # trace, so a want held at 0.6 cannot be opened into a response of 2.
        path_improvement = max(relief, gain)
        discharged = max(0.0, min(1.0, float(exposure)))

        magnitude = trace.trace * clarity * surprise_gain * (
            path_improvement + discharged * (1.0 - path_improvement)
        )
        magnitude = max(0.0, min(1.0, magnitude))

        # Gratitude, on Algoe's conditions: the benefit is valuable to the
        # receiver, costly to the giver, better than expected, and responsive
        # to what the receiver actually wanted. Every one of the four is
        # necessary, so they multiply — a costly benefit nobody wanted, or a
        # wanted benefit that cost nothing, generates none of this.
        wanted_from_them = max(0.0, min(1.0, float(specificity)))
        answered = max(0.0, min(1.0, float(responsiveness)))
        improbability = 1.0 - (
            self.willingness(agent).expectation() if agent else 1.0
        )
        gratitude = (
            trace.trace
            * max(0.0, min(1.0, float(cost_to_granter)))
            * max(0.0, improbability)
            * answered
            * wanted_from_them
        )

        if clarity > 0.0:
            self.set_blocker(key, Blocker.NONE)
        response = GrantResponse(
            incentive_key=key,
            granter=agent,
            magnitude=magnitude,
            held_wanting=trace.trace,
            cost_relief=relief,
            surprise_bits=surprise,
            blocker=blocker,
            relational_update=relational,
            attainability_gain=gain,
            commitment=clarity,
            exposure_resolved=discharged,
            specificity=wanted_from_them,
            responsiveness=answered,
            gratitude=max(0.0, min(1.0, gratitude)),
            evidence=(
                f"held {trace.evaluations} evaluations over {trace.duration_s():.0f}s; "
                f"{blocker} barrier opened at commitment {clarity:.2f}"
                + (f" by {agent} ({surprise:.2f} bits)" if agent else "")
                + (f"; attainability +{gain:.2f} ({divergence:.3f} nats)" if gain else "")
                + (f"; gratitude {gratitude:.2f} at specificity {wanted_from_them:.2f}"
                   if gratitude > 0.0 else "")
            ),
        )
        self._grants.append(response)
        if len(self._grants) > self.MAX_GRANTS:
            self._grants.pop(0)
        return response

    def refuse(self, key: str, *, agent: str, cost_to_granter: float = 0.0) -> float:
        """Record a refusal, which is evidence about the agent too.

        A model that only updates on yes drifts toward believing everyone
        agrees, and would then report a routine yes as surprising forever.
        """
        model = self.willingness(agent)
        return model.observe(agreed=False, cost_borne=cost_to_granter)

    def recent_grants(self, limit: int = 5) -> list[dict[str, Any]]:
        return [response.to_dict() for response in self._grants[-limit:]]

    def status(self) -> dict[str, Any]:
        blocked = self.blocked_wants()
        return {
            "traces": len(self._traces),
            "blocked_wants": blocked[:5],
            "blocked_count": len(blocked),
            "agents_modelled": len(self._willingness),
            "willingness": [
                model.to_dict()
                for model in sorted(
                    self._willingness.values(), key=lambda m: -m.last_update
                )[:5]
            ],
            "recent_grants": self.recent_grants(),
        }
