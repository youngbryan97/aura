"""Item 32 — emotional upheaval.

Distinct from strong emotion, and the difference decides what to do
about it. Upheaval is reorganisation: several appraisals change at once,
or one changes that many others depended on, and the set of readiness
states becomes temporarily inconsistent. Being pulled several ways is
the accurate report of a system with no dominant action tendency.

Dynamically it is a phase transition, and the signatures are the
standard ones: rising variance, rising autocorrelation, and critical
slowing — the state takes longer to return after a perturbation. There
is real work applying exactly these as early warnings for mood
transitions (van de Leemput et al. 2014).

The reason it earns its own mechanism is that the correct response
differs. Strong emotion wants regulation. Upheaval wants time, fewer
commitments, and no irreversible decisions, precisely because the
objective function is currently unstable. A system that detects
instability in its own affect should lower its own authority to make
binding choices until it settles — and nothing in this runtime does that
today.
"""

from __future__ import annotations

import math
import statistics

from core.interiority.effects import AffectDelta, BudgetDelta, Effects
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)


@register
class Upheaval(Faculty):
    id = "f32_upheaval"
    number = 32
    question = "Emotional upheaval"
    mechanism = (
        "Detects a phase transition in its own affect by rising variance and "
        "critical slowing, and withdraws its own authority to make "
        "irreversible choices while it lasts"
    )
    requires = ()
    optional = ("novelty", "certainty", "urgency")
    counterfactuals = (
        Counterfactual(
            "stable_history",
            {"novelty": 0.0},
            Direction.UNCHANGED,
            "Upheaval is a property of the affect trajectory, not of the "
            "current event. A frame variable moving it would mean instability "
            "is being inferred from the stimulus rather than measured.",
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "It reports upheaval while the recent affect trace is flat, or "
            "stays quiet through a genuine transition. The signatures are "
            "measurable on the trace, so either is a straightforward "
            "refutation rather than a matter of interpretation."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        trace = ctx.interior.get("affect_trace")
        if not isinstance(trace, (list, tuple)) or len(trace) < 6:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="not enough affect history to measure a transition",
            )

        values = [float(x) for x in trace if isinstance(x, (int, float))]
        if len(values) < 6:
            return Activation(
                faculty=self.id, intensity=0.0, declined="affect history is not numeric"
            )

        # Rising variance.
        half = len(values) // 2
        early = statistics.pstdev(values[:half]) if half > 1 else 0.0
        late = statistics.pstdev(values[half:]) if len(values) - half > 1 else 0.0
        variance_rise = max(0.0, late - early) / max(1e-6, late + early)

        # Critical slowing: lag-1 autocorrelation approaching one.
        mean = statistics.fmean(values)
        num = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(len(values) - 1))
        den = sum((v - mean) ** 2 for v in values)
        autocorrelation = num / den if den > 1e-12 else 0.0
        slowing = max(0.0, min(1.0, autocorrelation))

        # Conflicting readiness: several action tendencies at once, which is
        # what being pulled several ways is.
        conflict = ctx.interior_value("tendency_conflict", 0.0)

        intensity = max(0.0, min(1.0, (variance_rise * slowing) ** 0.5 * (0.6 + 0.4 * conflict)))

        effects = Effects(
            affect=AffectDelta(arousal=0.4 * intensity, engagement=-0.3 * intensity),
            # The whole point: while the objective is unstable, the agent
            # lowers its own authority rather than acting decisively on a
            # function that is currently moving.
            budget=BudgetDelta(
                depth=1.0 + 0.3 * intensity,
                deadline=1.0 + 0.6 * intensity,
                irreversibility_ceiling=max(0.0, 1.0 - intensity),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="inhibit",
            effects=effects,
            receipt={
                "variance_rise": variance_rise,
                "lag1_autocorrelation": autocorrelation,
                "critical_slowing": slowing,
                "tendency_conflict": conflict,
                "authority_withdrawn_to": max(0.0, 1.0 - intensity),
            },
        )
