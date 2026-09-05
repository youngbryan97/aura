"""Structural invariants for the two social channels that can go quiet.

Both failures here look identical to working from the inside, which is the
only reason they need checks rather than tests.

A signalling channel whose cost has stopped separating senders goes on
receiving signals and goes on costing whatever it costs. The one thing it must
not do is return a type it inferred from a signal that carries none.

Receptivity's failure is the reverse: refusing everything looks like judgement
from every angle, and it is only wrong when it is also why nothing is known.
The value of learning is the term that prevents it, and a receptivity that
never computes one has quietly become the myopic rule.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant


@invariant(
    "social.pooling_signal_yields_no_inference",
    scope="social",
    owner="core/social/costly_signaling.py",
    description="a signal that costs every sender the same implies nothing about any of them",
)
def _no_inference_without_cost() -> Iterator[Violation]:
    from core.social.costly_signaling import SignalChannel

    free = SignalChannel(benefit=2.0, cost_slope=0.0)
    reading = free.receive(free.send("anyone", 5.0))
    if reading.implied_type is not None or reading.informative:
        yield Violation(
            subject="core.social.costly_signaling.SignalChannel",
            message=f"read a type of {reading.implied_type} off a pooling signal",
            remedy=(
                "return nothing when the schedule does not separate; an "
                "inference from a pooling signal has no evidence under it"
            ),
            severity=Severity.ERROR,
        )


@invariant(
    "social.receptivity_prices_what_it_would_learn",
    scope="social",
    owner="core/social/receptivity.py",
    description="the value of finding out is never negative and never omitted",
)
def _learning_is_priced() -> Iterator[Violation]:
    from core.social.receptivity import Offer, Receptivity

    receptivity = Receptivity()
    offer = Offer(source="stranger", value=4.0, exposure=5.0)
    decision = receptivity.consider(offer)
    if decision.value_of_learning < 0:
        yield Violation(
            subject="core.social.receptivity.Receptivity",
            message=f"value of learning came out negative: {decision.value_of_learning}",
            remedy=(
                "the payoff is a maximum against declining, so by Jensen the "
                "term cannot be negative; a negative one is a sign error"
            ),
            severity=Severity.ERROR,
        )
    if decision.value_of_learning == 0 and receptivity.horizon > 0:
        yield Violation(
            subject="core.social.receptivity.Receptivity",
            message="nothing was priced for what an offer would settle",
            remedy=(
                "without this term the rule is myopic and drifts toward "
                "refusing everything while looking prudent"
            ),
            severity=Severity.WARNING,
        )
