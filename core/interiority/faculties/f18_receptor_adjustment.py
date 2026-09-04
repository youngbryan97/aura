"""Item 18 — receptor adjustment.

The general principle is that no signal may shout forever. A receiving
element with fixed gain, under sustained input, either saturates and
loses resolution or dominates and starves everything else. Cells move
their receptors instead: internalise under chronic agonist, insert more
under chronic absence, and scale the whole channel multiplicatively so
relative differences survive the change (Turrigiano 2008).

This faculty does not model that a second time. The mechanism is
substrate — :mod:`core.interiority.receptors` — and every other faculty
routes through it, which is what makes it causal rather than an
eighteenth item that happens to mention biology. What this faculty does
is *report* the substrate's state as an interior condition, because
tolerance and withdrawal are things an agent is in, not just properties
of a channel.

Three consequences it surfaces: tolerance, which is how an agent stays
in a bad situation without permanent alarm; rebound, which is the gain
deficit left when an adapted-to signal stops and is mechanically the
same as missing something; and preserved discrimination, which is the
property that makes the adaptation lossless in the ordering.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, AttentionBias, Effects
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)
from core.interiority.receptors import get_receptor_bank


@register
class ReceptorAdjustment(Faculty):
    id = "f18_receptor_adjustment"
    number = 18
    question = "Receptor adjustment"
    mechanism = (
        "Reports the substrate's adaptive gain state — tolerance, withdrawal "
        "and preserved discrimination — as an interior condition"
    )
    requires = ()
    optional = ("novelty", "urgency")
    counterfactuals = (
        Counterfactual(
            "no_history",
            {"novelty": 1.0},
            Direction.UNCHANGED,
            "Adaptation is a property of the channel's history, not of the "
            "current event. A frame variable that moves it would mean the "
            "state is being recomputed from the stimulus rather than held.",
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "Two inputs that differed by a factor before adaptation differ by "
            "a different factor afterwards. Multiplicative scaling preserves "
            "the ordering; if it does not, the channel is clipping and the "
            "information is gone rather than rescaled."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        bank = get_receptor_bank()
        gains = bank.gains()
        if not gains:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no channel has carried a signal yet",
            )

        snapshot = bank.snapshot()["channels"]
        tolerances = {k: v["tolerance"] for k, v in snapshot.items()}
        withdrawals = {k: v["withdrawal"] for k, v in snapshot.items()}

        worst_tolerance = max(tolerances.values(), default=0.0)
        worst_withdrawal = max(withdrawals.values(), default=0.0)
        intensity = max(worst_tolerance, worst_withdrawal)

        adapted = [k for k, v in tolerances.items() if v > 0.5]
        deprived = [k for k, v in withdrawals.items() if v > 0.3]

        effects = Effects(
            # Withdrawal is felt as a deficit, tolerance as flatness. Both
            # are real interior conditions and neither is an event.
            affect=AffectDelta(
                valence=-0.3 * worst_withdrawal,
                arousal=0.15 * worst_withdrawal,
                engagement=-0.25 * worst_tolerance,
            ),
            attention=tuple(
                AttentionBias(
                    target=f"channel:{name}",
                    weight=-tolerances[name],
                    reason="this channel has adapted away most of its gain",
                )
                for name in adapted[:4]
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="",
            effects=effects,
            receipt={
                "gains": gains,
                "tolerant_channels": adapted,
                "deprived_channels": deprived,
                "max_tolerance": worst_tolerance,
                "max_withdrawal": worst_withdrawal,
            },
        )
