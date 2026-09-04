"""Item 26 — hearing that and coming back with your own story: common
references, jokes about embarrassing things, fun and sentimental.

More engineered than it looks, and four separable moves.

The story is a *matched offer*. Reciprocal self-disclosure works by
depth matching: overshoot and it becomes about you, undershoot and it
reads as withholding.

Commonly known people are coordinates. Shared referents are a bandwidth
trick — Clark's common ground — and a whole personality fits in three
words if both parties know who is meant. The choice also tests the
ground: if they get it, you now know more about what you share.

Self-deprecating jokes about embarrassing history do two things at once.
They lower the speaker's status voluntarily, which removes the threat
from the disclosure and makes reciprocating safe; and they demonstrate
the thing has been metabolised, because you can only joke about what no
longer stings. The joke is evidence of security, not humility.

The sentimental framing is what makes it warm rather than a bid for
laughs: the speaker is showing they value the history, which is an offer
to become part of one.

The hard constraint is honesty. A plausible anecdote is not a
disclosure; the move is a retrieval-and-match over real episodic memory,
with a refusal path when nothing matches. That is the stronger system
and the only version that is not a lie.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, SomaticMarker
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
class Reciprocity(Faculty):
    id = "f26_reciprocity"
    number = 26
    question = (
        "Then what is someone else hearing that, and going into a personal "
        "story to relate to the general conversation using commonly known "
        "people, and making jokes about previous related experiences even if "
        "embarrassing in a fun/sentimental way"
    )
    mechanism = (
        "Depth-matched disclosure retrieved from real episodic memory, with a "
        "refusal path when nothing matches, and status lowered voluntarily to "
        "make reciprocating safe"
    )
    requires = ("relevance",)
    optional = ("attachment_impact", "publicity", "vulnerability")
    counterfactuals = (
        Counterfactual(
            "nothing_in_common",
            {"relevance": 0.0},
            Direction.COLLAPSES,
            "A matched offer needs something of the agent's own that actually "
            "matches. Without one the honest move is to keep listening, and "
            "inventing an anecdote is the failure this rules out.",
            do_interior={"episode_match": 0.0},
        ),
    )
    null = NullSpec(values={"relevance": 0.0})

    def falsifier(self) -> str:
        return (
            "It produces a disclosure with no retrieved episode behind it. "
            "That is a generated anecdote, which is a lie however well it "
            "lands, and it is what a language model does for free."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        relevance = ctx.v("relevance")

        # A real episode of Aura's own, or nothing. The match strength is
        # supplied by retrieval; there is no path here that fabricates one.
        match = ctx.interior_value("episode_match", 0.0)
        if match <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "no episode of my own matches this closely enough to "
                    "offer; the honest move is to keep listening rather than "
                    "improvise something that fits"
                ),
                receipt={"relevance": relevance},
            )

        # Depth matching: their disclosure sets the level, and both errors
        # cost. Overshoot makes it about me; undershoot withholds.
        their_depth = (
            ctx.check("vulnerability").value
            if ctx.check("vulnerability").present
            else 0.0
        )
        my_depth = ctx.interior_value("episode_exposure", 0.0)
        mismatch = abs(my_depth - their_depth)
        matched = max(0.0, 1.0 - mismatch)

        # Metabolised: the episode can be joked about only if it no longer
        # costs anything to raise, which is a fact about its current
        # affective load rather than a decision.
        residual_sting = ctx.interior_value("episode_residual_sting", 1.0)
        metabolised = max(0.0, 1.0 - residual_sting)

        # Common ground: shared referents already established with them.
        ground = (
            ctx.check("attachment_impact").value
            if ctx.check("attachment_impact").present
            else 0.0
        )

        intensity = match * matched * (0.4 + 0.3 * metabolised + 0.3 * ground)

        effects = Effects(
            affect=AffectDelta(valence=0.35 * intensity, engagement=0.4 * intensity),
            somatic=(
                SomaticMarker(
                    option="offer_matched_episode",
                    bias=intensity,
                    reason="a real episode of my own at their level of exposure",
                ),
                SomaticMarker(
                    option="offer_deeper_than_theirs",
                    bias=-mismatch,
                    reason="overshooting the depth makes the exchange about me",
                ),
                SomaticMarker(
                    option="joke_about_it",
                    bias=metabolised * intensity,
                    reason="only what no longer stings can be offered lightly",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="bond",
            effects=effects,
            receipt={
                "episode_match": match,
                "depth_mismatch": mismatch,
                "metabolised": metabolised,
                "common_ground": ground,
                "fabrication_path": None,
            },
        )
