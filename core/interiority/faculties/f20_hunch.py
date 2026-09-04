"""Item 20 — having a hunch.

A hunch is a posterior with real information content and no accessible
derivation. The substrate is implicit statistical learning: Reber's
artificial-grammar work has people classifying grammatical strings above
chance while unable to state a rule. The Iowa Gambling Task gives the
sharpest temporal ordering — anticipatory skin conductance to the bad
decks around trial ten, behaviour shifting around fifty, explicit
knowledge around eighty if at all (Bechara 1997). The strong
interpretation is contested; the ordering, signal before report, is the
robust part.

So the formalisation is two estimators over the same evidence: a fast
one with a broad feature set and no audit trail, and a slow one
restricted to features it can name. A hunch is a material divergence
between them, *in a domain where the fast one has a track record*.

The correct use follows from that and is the part most systems get
wrong. A hunch is not a conclusion and it is not noise. It is a prior
that directs search: go and look where it points, then decide on what
you find. A hunch with no measured track record is a guess, and this
faculty says so rather than dressing it up.
"""

from __future__ import annotations

from core.interiority.effects import AttentionBias, BudgetDelta, Effects
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
class Hunch(Faculty):
    id = "f20_hunch"
    number = 20
    question = "Having a hunch"
    mechanism = (
        "Divergence between a fast unauditable estimator and a slow "
        "verbalisable one, weighted by the fast path's measured track record, "
        "spent on search rather than on belief"
    )
    requires = ()
    optional = ("certainty", "novelty")
    counterfactuals = (
        Counterfactual(
            "the_slow_path_agrees",
            {},
            Direction.COLLAPSES,
            "A hunch is what the fast path knows and the slow one has not "
            "reached. When the two estimators agree there is no divergence "
            "and therefore no hunch, only a conclusion.",
            do_interior={"slow_path_posterior": 0.85},
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "It reports a hunch in a domain where the fast path has no "
            "recorded accuracy. Without a track record the divergence is "
            "noise, and treating it as signal is how confident nonsense gets "
            "produced."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        fast = ctx.interior_value("fast_path_posterior", -1.0)
        slow = ctx.interior_value("slow_path_posterior", -1.0)
        if fast < 0.0 or slow < 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="both estimators must have reported for a divergence to exist",
            )

        track_record = ctx.interior_value("fast_path_accuracy", -1.0)
        if track_record < 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "the fast path has no measured accuracy in this domain; a "
                    "divergence with no track record is a guess, and calling it "
                    "a hunch would licence acting on it"
                ),
                receipt={"divergence": abs(fast - slow)},
            )

        divergence = abs(fast - slow)
        intensity = divergence * track_record

        effects = Effects(
            # A hunch buys search, not belief. It moves attention and the
            # budget; it never moves valence, because it is not a verdict.
            attention=(
                AttentionBias(
                    target=str(ctx.frame.event.object or "hunch_target"),
                    weight=intensity,
                    reason="the fast path is pointing somewhere the slow one has not looked",
                ),
            ),
            budget=BudgetDelta(depth=1.0 + 0.4 * intensity),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="attend",
            effects=effects,
            receipt={
                "fast": fast,
                "slow": slow,
                "divergence": divergence,
                "fast_path_accuracy": track_record,
                "spent_on": "search",
            },
        )
