"""Structural invariants for arbitrating between a feeling and an argument.

The failure worth guarding is the quiet one. An arbiter with no measured
skill in a domain has to say so; one that falls back on a built-in preference
produces a confident number out of two guesses, and — because the fallback is
almost always toward the channel that can explain itself — it looks like good
engineering every time it happens.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant


@invariant(
    "affect.arbitration_has_no_default_channel",
    scope="affect",
    owner="core/affect/dual_process_arbiter.py",
    description="with no measured skill the arbiter abstains instead of preferring one",
)
def _abstains_without_evidence() -> Iterator[Violation]:
    from core.affect.dual_process_arbiter import DualProcessArbiter

    arbiter = DualProcessArbiter()
    result = arbiter.arbitrate("untested", 0.9, 0.1, record=False)
    if not result.abstained or result.probability is not None:
        yield Violation(
            subject="core.affect.dual_process_arbiter.DualProcessArbiter",
            message=(
                f"answered {result.probability} in a domain with no resolved outcomes"
            ),
            remedy=(
                "abstain until a channel has shown skill here; averaging two "
                "uncalibrated estimates makes a confident number out of guesses"
            ),
            severity=Severity.ERROR,
        )
    if result.weight_affective != result.weight_deliberate:
        yield Violation(
            subject="core.affect.dual_process_arbiter.DualProcessArbiter",
            message=(
                f"unequal weights with no evidence: {result.weight_affective} "
                f"against {result.weight_deliberate}"
            ),
            remedy="weights come from measured skill scores and from nowhere else",
            severity=Severity.ERROR,
        )


@invariant(
    "affect.empathy_keeps_a_return_path",
    scope="affect",
    owner="core/affect/empathic_coupling.py",
    description="a field with no anchors reports no rest rather than a consensus",
)
def _unanchored_field_reports_no_rest() -> Iterator[Violation]:
    from core.affect.empathic_coupling import EmpathicField

    field = EmpathicField()
    for who in ("a", "b", "c"):
        field.add_person(who, setpoint=1.0, anchor=0.0)
    for a in ("a", "b", "c"):
        for b in ("a", "b", "c"):
            if a != b:
                field.couple(a, b, 1.0)
    if field.rest() is not None:
        yield Violation(
            subject="core.affect.empathic_coupling.EmpathicField",
            message="returned a rest state for a field nobody is anchored in",
            remedy=(
                "the matrix is singular there; report it, because a plausible "
                "vector is exactly what such a system goes on producing"
            ),
            severity=Severity.ERROR,
        )
