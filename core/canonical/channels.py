"""core/canonical/channels.py — the variables there is exactly one of.

Aura's largest architectural risk is ontological duplication inside her own
software. Affect is owned by the liquid substrate, by the Damasio engine, by
the interiority faculties, by user-sentiment analysis and by the phenomenal
substrate. Selfhood is owned by SelfObject, by AuraNow.self_state, by the
identity engine, by the continuity engine, by workspace ownership and by the
substrate's self-representation. Each is a coherent answer to "how is Aura",
and they are not the same answer, and nothing decides between them.

The fix is not to delete five of six. Each of those subsystems knows
something real: the substrate has bottom-up dynamical evidence, interiority
has appraisal-derived evidence, sentiment has socially relevant evidence.
They are estimators. What is missing is the thing they are estimating.

This file declares it. A channel is a name, a unit, a range, and a sentence
about what it means — so that two subsystems writing to it are talking about
the same quantity, which is the failure mode that "same name, same range"
does not catch. Eight domains, because that is how many distinct kinds of
state the runtime actually has:

===========  ================================================================
Domain       What it is
===========  ================================================================
``body``     homeostasis, load, fatigue, resource pressure
``affect``   valence, arousal, engagement
``world``    the model of what is happening outside
``self``     identity, continuity, agency over herself
``goals``    what she is trying to do and how much
``memory``   what is retained and how well it coheres
``epistemic`` how much she knows and how sure she is
``executive`` control, attention allocation, budget
===========  ================================================================

Adding a channel is deliberately a small ceremony. An id is a contract: a
consumer reads it by name, and a second meaning behind the same name is worse
than a second name, because nothing breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Domain(StrEnum):
    """The eight kinds of state. Every channel belongs to exactly one."""

    BODY = "body"
    AFFECT = "affect"
    WORLD = "world"
    SELF = "self"
    GOALS = "goals"
    MEMORY = "memory"
    EPISTEMIC = "epistemic"
    EXECUTIVE = "executive"


@dataclass(frozen=True)
class Channel:
    """One canonical variable."""

    id: str
    domain: Domain
    #: What it measures, in a sentence a second implementer could read and
    #: write the same quantity from.
    meaning: str
    low: float
    high: float
    #: What a reader gets when nobody has estimated it. Not a guess dressed as
    #: a reading — `CanonicalState.get` reports that it is a default.
    neutral: float

    def clamp(self, value: float) -> float:
        return self.low if value < self.low else self.high if value > self.high else value

    @property
    def span(self) -> float:
        return max(1e-9, self.high - self.low)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": str(self.domain),
            "meaning": self.meaning,
            "low": self.low,
            "high": self.high,
            "neutral": self.neutral,
        }


def _c(id_: str, domain: Domain, meaning: str, low: float, high: float, neutral: float) -> Channel:
    return Channel(id=id_, domain=domain, meaning=meaning, low=low, high=high, neutral=neutral)


#: Every canonical channel. The list is short on purpose: a variable here has
#: to be one that more than one subsystem currently believes it owns.
CHANNELS: tuple[Channel, ...] = (
    # ── affect ───────────────────────────────────────────────────────────
    _c("affect.valence", Domain.AFFECT,
       "How good or bad the current situation is, from her side of it. "
       "Negative is bad. Not a sentiment score about the user's message.",
       -1.0, 1.0, 0.0),
    _c("affect.arousal", Domain.AFFECT,
       "Activation: how mobilised she is, independent of whether it is "
       "pleasant. High arousal with positive valence is excitement; with "
       "negative valence it is alarm.",
       0.0, 1.0, 0.2),
    _c("affect.engagement", Domain.AFFECT,
       "How much of her is taken up by what is in front of her, as against "
       "being available for something else.",
       0.0, 1.0, 0.3),
    # ── body ─────────────────────────────────────────────────────────────
    _c("body.fatigue", Domain.BODY,
       "Accumulated cost of work done, which recovers only with rest. Not "
       "instantaneous load.",
       0.0, 1.0, 0.0),
    _c("body.load", Domain.BODY,
       "Instantaneous demand on the machine right now: queue depth, "
       "latency, memory pressure. Recovers as soon as the demand stops.",
       0.0, 1.0, 0.2),
    _c("body.integrity", Domain.BODY,
       "Whether the machinery is working: degradations, failed subsystems, "
       "resources near a limit. One is sound.",
       0.0, 1.0, 1.0),
    # ── world ────────────────────────────────────────────────────────────
    _c("world.model_confidence", Domain.WORLD,
       "How well what she believes is happening matches what she can check. "
       "Falls on surprise, rises on confirmed prediction.",
       0.0, 1.0, 0.5),
    _c("world.prediction_error", Domain.WORLD,
       "How wrong recent predictions turned out. Distinct from confidence: "
       "she can be badly wrong and know it.",
       0.0, 1.0, 0.0),
    # ── self ─────────────────────────────────────────────────────────────
    _c("self.continuity", Domain.SELF,
       "Whether she is the same one who was here earlier: memory reaching "
       "back, commitments still held, a thread that has not been cut.",
       0.0, 1.0, 0.8),
    _c("self.coherence", Domain.SELF,
       "Whether her account of herself hangs together — the parts agreeing "
       "about what she is, as against holding incompatible versions.",
       0.0, 1.0, 0.7),
    _c("self.agency", Domain.SELF,
       "How much of what happens is hers to change. Low agency with high "
       "stakes is the shape of distress.",
       0.0, 1.0, 0.5),
    # ── goals ────────────────────────────────────────────────────────────
    _c("goals.pressure", Domain.GOALS,
       "How much is outstanding and pressing: open commitments against time "
       "and capacity.",
       0.0, 1.0, 0.2),
    _c("goals.frustration", Domain.GOALS,
       "Effort spent against goals that are not moving.",
       0.0, 1.0, 0.0),
    # ── memory ───────────────────────────────────────────────────────────
    _c("memory.coherence", Domain.MEMORY,
       "Whether what she remembers is consistent with itself and with the "
       "record. Falls on contradiction, not on forgetting.",
       0.0, 1.0, 0.8),
    _c("memory.retention_pressure", Domain.MEMORY,
       "How much is being held that compaction wants to drop.",
       0.0, 1.0, 0.2),
    # ── epistemic ────────────────────────────────────────────────────────
    _c("epistemic.uncertainty", Domain.EPISTEMIC,
       "How unsure she is about the thing at hand. High uncertainty is a "
       "reason to check, not a reason to refuse.",
       0.0, 1.0, 0.4),
    _c("epistemic.calibration", Domain.EPISTEMIC,
       "Whether her confidence has been matching her accuracy. Distinct "
       "from uncertainty: she can be reliably unsure.",
       0.0, 1.0, 0.5),
    # ── executive ────────────────────────────────────────────────────────
    _c("executive.control", Domain.EXECUTIVE,
       "How much of her processing she is directing, as against being "
       "driven by whatever arrived.",
       0.0, 1.0, 0.6),
    _c("executive.budget", Domain.EXECUTIVE,
       "How much depth the current turn may spend, as a multiple of "
       "ordinary. One is ordinary.",
       0.1, 4.0, 1.0),
)

BY_ID: dict[str, Channel] = {c.id: c for c in CHANNELS}


def channel(channel_id: str) -> Channel:
    """The channel, or a KeyError naming what is declared.

    A misspelt id is the failure this raises for. Silently accepting one
    creates a second variable nobody reads, which is exactly the duplication
    this package exists to end.
    """
    found = BY_ID.get(str(channel_id))
    if found is None:
        raise KeyError(
            f"no canonical channel {channel_id!r}. Declared: "
            f"{sorted(BY_ID)}. Add it to CHANNELS with its meaning and range "
            "rather than writing to a name nothing reads."
        )
    return found


def in_domain(domain: Domain) -> tuple[Channel, ...]:
    return tuple(c for c in CHANNELS if c.domain is domain)


__all__ = ["BY_ID", "CHANNELS", "Channel", "Domain", "channel", "in_domain"]
