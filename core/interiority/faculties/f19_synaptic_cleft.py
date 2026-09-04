"""Item 19 — the synaptic cleft interface.

The cleft is where a signal changes kind: electrical to chemical to
electrical, across a gap of a few tens of nanometres. Four of its
properties are computational rather than incidental. Release is quantal
and probabilistic, so transmission is unreliable by design and the rate
rather than the event is the carrier. The gap is shared, so transmitter
spills to neighbours and a modulator can act on a region instead of a
wire. Clearance sets the time constant, and so what counts as "now" for
the receiver. And the postsynaptic side decides what it hears: the same
molecule excites at one receptor and inhibits at another.

Read as an architectural rule this says subsystems should not be wired
by direct calls with guaranteed delivery. In this package they are not:
:mod:`core.interiority.cleft` is the transport every faculty publishes
into and every consumer reads from, which is why this item is substrate
rather than a nineteenth scorer. What this faculty reports is the
medium's own condition — fidelity, facilitation, spillover, and where a
state failed to cross.

That last is the useful one. A state that was real and did not transmit
is a specific failure this runtime has had in other forms, and it is
invisible to any design where publishing is a function call.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects
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
class SynapticCleftInterface(Faculty):
    id = "f19_synaptic_cleft"
    number = 19
    question = "Synaptic cleft interface"
    mechanism = (
        "Reports the transmission medium's condition: release fidelity, "
        "facilitation, spillover, and states that failed to cross"
    )
    requires = ()
    optional = ("urgency",)
    counterfactuals = (
        Counterfactual(
            "event_content_irrelevant",
            {"urgency": 1.0},
            Direction.UNCHANGED,
            "The medium's condition is a property of what has crossed it, not "
            "of the current event's meaning. Movement here would mean the "
            "transport is being recomputed from content.",
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "Fidelity of exactly 1.0 across many releases. Quantal release is "
            "probabilistic; perfect delivery would mean the medium is a "
            "function call with a biological docstring."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        snapshot = ctx.medium().snapshot()
        channels = snapshot["channels"]
        if not channels:
            return Activation(
                faculty=self.id, intensity=0.0, declined="nothing has been published yet"
            )

        reliabilities = {k: v["reliability"] for k, v in channels.items()}
        worst = min(reliabilities.values(), default=1.0)
        failed = [k for k, v in reliabilities.items() if v < 0.5]
        loss = 1.0 - worst

        effects = Effects(
            # A state that is not reaching its consumers is a real interior
            # condition, and the honest report of it is reduced engagement
            # rather than a louder signal.
            affect=AffectDelta(engagement=-0.2 * loss)
        )
        return Activation(
            faculty=self.id,
            intensity=loss,
            tendency="",
            effects=effects,
            receipt={
                "reliability": reliabilities,
                "channels_failing_to_cross": failed,
                "modulators": snapshot["modulators"],
                "neighbourhoods": snapshot["neighbourhoods"],
            },
        )
