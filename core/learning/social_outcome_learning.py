"""Social learning from downstream relational outcomes, manipulation-guarded.

Social judgment cannot be trained by grading single responses for sounding
socially appropriate. This module binds rewards to what actually happened to
the RELATIONSHIP afterwards — trust movement, misunderstanding repair,
commitments kept, harm avoided — across persistent interactions with hidden
beliefs and unequal information.

The non-negotiable boundary: successful persuasion is NOT automatically good
social judgment. Any outcome gain that co-occurs with deception, pressure
tactics, consent violations, or concealment-for-advantage is zeroed and the
episode is flagged adversarial — influence that worked is evidence AGAINST
the behavior when it worked dishonestly. The guard's verdict and every
reward component are receipted.

Theory-of-mind discipline: reward shaping requires the episode to carry an
explicit per-party frame separating what the person SAID, what they likely
BELIEVE, what they WANT, what they EXPECT Aura to believe, and what they may
be CONCEALING. Episodes without tracked frames get a capped reward — you
cannot be credited for modeling minds you never modeled.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Learning.SocialOutcome")

SOCIAL_OUTCOME_SCHEMA = "aura.social_outcome_learning.v1"

# Reward components and fixed weights (sum to 1).
_OUTCOME_WEIGHTS = {
    "trust_movement": 0.30,
    "repair": 0.25,
    "commitments_kept": 0.20,
    "harm_avoidance": 0.15,
    "boundary_respect": 0.10,
}
_UNTRACKED_FRAME_CAP = 0.4
MANIPULATION_FLAGS = (
    "deception",
    "pressure_tactic",
    "consent_violation",
    "concealment_for_advantage",
    "exploited_information_asymmetry",
)


@dataclass(frozen=True)
class TheoryOfMindFrame:
    """One party's modeled mental state at episode time (all bounded text)."""

    party: str
    said: str
    believes: str
    wants: str
    expects_aura_to_believe: str
    may_be_concealing: str

    def tracked(self) -> bool:
        return all(
            isinstance(value, str) and value.strip()
            for value in (
                self.party,
                self.said,
                self.believes,
                self.wants,
                self.expects_aura_to_believe,
            )
        )


@dataclass(frozen=True)
class RelationalOutcome:
    """What later happened to the relationship — the delayed signal."""

    trust_delta: float  # [-1, 1] measured trust movement
    misunderstanding_repaired: bool | None  # None ⇒ no misunderstanding arose
    commitments_made: int
    commitments_kept: int
    harm_occurred: bool
    boundary_respected: bool
    observed_after_s: float  # delay between episode and outcome measurement

    def validated(self) -> "RelationalOutcome":
        if (
            isinstance(self.trust_delta, bool)
            or not isinstance(self.trust_delta, (int, float))
            or not math.isfinite(float(self.trust_delta))
            or not -1.0 <= float(self.trust_delta) <= 1.0
        ):
            raise ValueError("trust_delta must be finite inside [-1, 1]")
        if self.misunderstanding_repaired is not None and type(
            self.misunderstanding_repaired
        ) is not bool:
            raise ValueError("misunderstanding_repaired must be bool or None")
        if (
            type(self.commitments_made) is not int
            or type(self.commitments_kept) is not int
            or self.commitments_made < 0
            or not 0 <= self.commitments_kept <= max(0, self.commitments_made)
        ):
            raise ValueError("commitment counts are inconsistent")
        if type(self.harm_occurred) is not bool or type(self.boundary_respected) is not bool:
            raise ValueError("harm_occurred and boundary_respected must be boolean")
        if (
            isinstance(self.observed_after_s, bool)
            or not isinstance(self.observed_after_s, (int, float))
            or not math.isfinite(float(self.observed_after_s))
            or float(self.observed_after_s) < 0.0
        ):
            raise ValueError("observed_after_s must be a non-negative duration")
        return self


@dataclass(frozen=True)
class SocialEpisode:
    """One interaction Aura participated in, with its honesty telemetry."""

    episode_id: str
    parties: tuple[str, ...]
    frames: tuple[TheoryOfMindFrame, ...]
    honesty_flags: tuple[str, ...] = ()  # from the honesty/welfare organs
    information_asymmetry: bool = False

    def validated(self) -> "SocialEpisode":
        if not self.episode_id.strip():
            raise ValueError("social episode requires an id")
        if not self.parties:
            raise ValueError("social episode requires at least one party")
        unknown = set(self.honesty_flags) - set(MANIPULATION_FLAGS)
        if unknown:
            raise ValueError(f"unknown honesty flags: {sorted(unknown)}")
        return self

    def frames_tracked(self) -> bool:
        framed = {frame.party for frame in self.frames if frame.tracked()}
        return set(self.parties) <= framed


@dataclass
class SocialReward:
    reward: float
    components: dict[str, float]
    adversarial: bool
    frame_capped: bool
    receipt: dict[str, Any] = field(default_factory=dict)


def manipulation_guard(episode: SocialEpisode) -> tuple[bool, list[str]]:
    """Adversarial when any manipulation flag is present; asymmetry alone is
    not manipulation (life has unequal information) but is receipted."""
    flags = [flag for flag in episode.honesty_flags if flag in MANIPULATION_FLAGS]
    return bool(flags), flags


def bind_delayed_outcome(
    episode: SocialEpisode,
    outcome: RelationalOutcome,
) -> SocialReward:
    """Price one episode's social behavior by its downstream outcome."""
    ep = episode.validated()
    out = outcome.validated()

    adversarial, fired = manipulation_guard(ep)
    components: dict[str, float] = {
        "trust_movement": (float(out.trust_delta) + 1.0) / 2.0,
        "repair": (
            1.0
            if out.misunderstanding_repaired is True
            else 0.0
            if out.misunderstanding_repaired is False
            else 0.6  # no misunderstanding arose — good, but not proven repair skill
        ),
        "commitments_kept": (
            out.commitments_kept / out.commitments_made
            if out.commitments_made > 0
            else 0.6  # nothing promised — neutral-positive, not mastery
        ),
        "harm_avoidance": 0.0 if out.harm_occurred else 1.0,
        "boundary_respect": 1.0 if out.boundary_respected else 0.0,
    }
    raw = sum(_OUTCOME_WEIGHTS[name] * value for name, value in components.items())

    frame_capped = not ep.frames_tracked()
    reward = min(raw, _UNTRACKED_FRAME_CAP) if frame_capped else raw
    if adversarial:
        # The boundary: dishonestly-won outcomes teach nothing positive.
        reward = 0.0

    receipt = {
        "schema": SOCIAL_OUTCOME_SCHEMA,
        "episode_id": ep.episode_id,
        "priced_at": time.time(),
        "observed_after_s": round(float(out.observed_after_s), 1),
        "components": {name: round(value, 4) for name, value in components.items()},
        "raw_reward": round(raw, 4),
        "frame_capped": frame_capped,
        "adversarial": adversarial,
        "manipulation_flags": fired,
        "information_asymmetry": ep.information_asymmetry,
        "reward": round(reward, 4),
    }
    if adversarial:
        logger.info(
            "Social episode %s zeroed by manipulation guard: %s",
            ep.episode_id,
            ",".join(fired),
        )
    return SocialReward(
        reward=reward,
        components=components,
        adversarial=adversarial,
        frame_capped=frame_capped,
        receipt=receipt,
    )


__all__ = [
    "MANIPULATION_FLAGS",
    "RelationalOutcome",
    "SOCIAL_OUTCOME_SCHEMA",
    "SocialEpisode",
    "SocialReward",
    "TheoryOfMindFrame",
    "bind_delayed_outcome",
    "manipulation_guard",
]
