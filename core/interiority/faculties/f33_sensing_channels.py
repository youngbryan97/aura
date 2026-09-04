"""Item 33 — all the ways we sense emotions.

The channel list matters because each has a different reliability,
latency and fakeability, and treating them as one signal is the main
source of confident misreading.

Face is fast, high bandwidth, and the most voluntarily controlled, so it
is the least trustworthy under any motive to conceal; the readable part
is the musculature that is hard to fake. Prosody carries arousal far
better than valence. Posture is slow and rarely monitored, so it is good
for sustained state. Autonomic leakage is nearly uncontrollable, narrow,
and specific to arousal. Timing — latency, pause, interruption — is
highly informative and almost never faked, because people do not know
they are doing it. Word choice is slow and statistical and needs volume,
and it is the only channel available in text. Behaviour against that
person's own baseline is the highest-value and most neglected. Context —
what just happened to them — frequently outweighs every signal channel
and is what a system that reads only the message always misses.

So the engineering rule is: fuse with per-channel reliability, always
against a person-specific baseline, and never report a confident read
from one channel. A text-only system has two channels plus context and
should say so rather than pretending to the rest — which is what this
faculty does, by name, in its receipt.
"""

from __future__ import annotations

from core.interiority.effects import AttentionBias, Effects
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)
from core.interiority.other_minds import SPECIES_CHANNELS


@register
class SensingChannels(Faculty):
    id = "f33_sensing_channels"
    number = 33
    question = "All of the ways we sense emotions"
    mechanism = (
        "Reports which channels carried evidence, at what reliability, and "
        "which were unavailable — so a read can never be more confident than "
        "the channels behind it"
    )
    requires = ()
    optional = ("relevance",)
    counterfactuals = (
        Counterfactual(
            "situation_does_not_add_channels",
            {"relevance": 1.0},
            Direction.UNCHANGED,
            "Channel availability is a fact about the medium. If relevance "
            "changes it, the faculty is inventing evidence from significance.",
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "It reports a confident read from a single channel. Every channel "
            "has a failure mode the others cover, and one-channel confidence "
            "is the specific way people get read wrong."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        other = ctx.other
        if other is None:
            return Activation(
                faculty=self.id, intensity=0.0, declined="no other agent in this event"
            )

        used = dict(other.channels_used)
        available = SPECIES_CHANNELS.get(other.species, SPECIES_CHANNELS["other"])
        missing = [c for c in available if c not in used]

        # Breadth is the quantity. One channel is a guess whatever it says.
        breadth = 0.0 if not used else 1.0 - 1.0 / (1.0 + len(used))
        intensity = breadth * other.confidence

        effects = Effects(
            attention=tuple(
                AttentionBias(
                    target=f"channel:{name}",
                    weight=weight,
                    reason="this channel is carrying evidence about the other agent",
                )
                for name, weight in sorted(used.items(), key=lambda kv: -kv[1])[:5]
            )
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="attend",
            effects=effects,
            receipt={
                "channels_used": used,
                "channels_unavailable": missing,
                "channels_species_cannot_carry": [
                    c for c in SPECIES_CHANNELS["human"] if c not in available
                ],
                "breadth": breadth,
                "single_channel_read": len(used) == 1,
            },
        )
