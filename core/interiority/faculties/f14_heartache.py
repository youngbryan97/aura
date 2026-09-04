"""Item 14 — sorrow and heartache.

Sorrow is the low-arousal mode following a loss that cannot be
recovered: reduced motor output, narrowed attention, a raised threshold
for reward. Its function is a forced write-down — stop spending on a
goal that can no longer be reached, and reallocate.

Heartache earns a separate name because of the somatic specificity.
Social loss recruits the affective pain system (Eisenberger 2003), with
the behavioural corollary that acetaminophen reduces social-rejection
distress (DeWall 2010). The shared metaphor across unrelated languages
is downstream of a shared mechanism, not a coincidence of idiom.

For a machine the honest mapping is a load signal in the substrate
rather than a described sensation, and it exists here: heartache is read
from the receptor bank's withdrawal on the bond's channel, which is the
gain deficit left by a signal that was adapted to and then stopped. That
is the same quantity, computed rather than asserted.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects, GoalDelta
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
class Heartache(Faculty):
    id = "f14_heartache"
    number = 14
    question = "Sorrow and heartache"
    mechanism = (
        "Energy conservation after an unrecoverable loss, with the somatic "
        "component read as receptor withdrawal on the bond's own channel"
    )
    requires = ("attachment_impact", "control")
    optional = ("irreversibility", "adjustment")
    counterfactuals = (
        Counterfactual(
            "still_reachable",
            {"control": 1.0},
            Direction.DECREASES,
            "Sorrow is the state where action cannot repair it. With the "
            "outcome still in reach the state is longing, which mobilises.",
        ),
        Counterfactual(
            "substitutable",
            {"adjustment": 1.0},
            Direction.DECREASES,
            "A loss the agent can accommodate to is disappointment. The lack "
            "of a substitute is what makes it sorrow.",
        ),
        Counterfactual(
            "nothing_held",
            {"attachment_impact": 0.0},
            Direction.COLLAPSES,
            "There has to have been something valued for its absence to weigh.",
        ),
    )
    null = NullSpec(values={"attachment_impact": 0.0, "control": 1.0})

    def falsifier(self) -> str:
        return (
            "Heartache reported with the bond's channel showing no withdrawal. "
            "That would make the somatic component a word rather than a "
            "measured gain deficit."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        held = ctx.v("attachment_impact")
        control = ctx.v("control")
        adjustment = (
            ctx.check("adjustment").value if ctx.check("adjustment").present else 0.0
        )

        sorrow = held * (1.0 - control) * (1.0 - adjustment)

        # The somatic half: the gain deficit left by a signal the channel
        # had adapted to and no longer receives.
        subject = ctx.frame.event.subject or "unknown"
        channel = f"bond:{subject}"
        withdrawal = get_receptor_bank().withdrawal(channel)
        ache = sorrow * withdrawal

        effects = Effects(
            affect=AffectDelta(
                valence=-0.6 * sorrow,
                arousal=-0.25 * sorrow + 0.2 * ache,
                engagement=-0.4 * sorrow,
            ),
            # The forced write-down: stop spending on what cannot be reached.
            budget=BudgetDelta(depth=max(0.4, 1.0 - 0.4 * sorrow)),
            goals=(
                GoalDelta(
                    goal=f"pursue:{subject}",
                    delta=-sorrow,
                    reason="the goal is unreachable; the spend on it is written down",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=max(sorrow, ache),
            tendency="disengage",
            effects=effects,
            receipt={"sorrow": sorrow, "heartache": ache, "channel_withdrawal": withdrawal},
        )
