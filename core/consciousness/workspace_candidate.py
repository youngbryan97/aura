"""What a bid for the global workspace is, and the kinds of content it carries.

Lifted whole out of `global_workspace`, which imports them straight back, so
every caller that names them there still finds them. Fourteen of the modules
that imported the workspace took these two and nothing else, and each depended
on the whole competition to say what a bid is.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ContentType(Enum):
    """Types of cognitive content for workspace processing."""
    UNKNOWN = auto()
    PERCEPTUAL = auto()
    AFFECTIVE = auto()
    MEMORIAL = auto()
    INTENTIONAL = auto()
    LINGUISTIC = auto()
    SOMATIC = auto()
    SOCIAL = auto()
    META = auto()


@dataclass
class CognitiveCandidate:
    """A bid for the global workspace broadcast slot.
    Any subsystem can submit one each tick.
    """

    content: str                       # What wants to be broadcast
    source: str                        # e.g. "drive_curiosity", "affect_distress", "memory"
    priority: float                    # 0.0–1.0 base weight
    content_type: ContentType = ContentType.UNKNOWN
    affect_weight: float = 0.0        # Emotional urgency boost (from AffectEngine)
    focus_bias: float = 0.0           # Priority boost for focused attention (from AttentionSchema)
    submitted_at: float = field(default_factory=lambda: time.time())
    gate_instance_id: str = field(default="", repr=False)
    gate_checked_at: float = field(default=0.0, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    #: Who is bidding, when that is not what the label says. Affect names its
    #: bid after whichever emotion is on top, so one subsystem entered the
    #: competition under forty-four different labels — and fatigue, which is
    #: what gives another source a turn, is per label. Affect was therefore
    #: never fatigued: the channel that had just won handed off to a fresh
    #: sibling. Over twenty-four competitions affect took thirty-three wins and
    #: ten sources bid and never once won. Empty means the label is the bidder.
    bidder_id: str = ""

    @property
    def bidder(self) -> str:
        """Who the refractory period applies to. The label, unless declared."""
        return self.bidder_id or self.source

    @property
    def salience(self) -> float:
        """Alias for effective_priority for downstream compatibility."""
        return self.effective_priority

    @property
    def effective_priority(self) -> float:
        """Priority as of now. Prefer :meth:`priority_at` inside a competition."""
        return self.priority_at(time.time())

    @property
    def cognitive_priority(self) -> float:
        """Everything about this bid EXCEPT when it arrived.

        What a competition is supposed to be settled by: base salience, how
        urgent affect makes it, where attention already is, and whether it
        aligns with the dominant action. Recency is a real cognitive factor
        and it is applied on top of this; sub-microsecond arrival order is
        not, and separating them is what lets a tie be recognised as a tie.
        """
        return self.priority_at(self.submitted_at)

    def priority_at(self, now: float) -> float:
        """Priority evaluated against ONE instant.

        The property used to call `time.time()` itself, which had two
        consequences and the smaller one was the known flake.
        
        It was used as a sort key, so the comparator re-read the clock during
        the sort and the ordering was not guaranteed to be consistent — a
        comparison function that changes between comparisons can produce an
        arbitrary permutation, not merely a jittered one.
        
        And a competition is one cognitive moment. Ageing each candidate from
        the instant its own comparison happened to run meant identical bids
        came out microseconds apart, and the workspace settled the choice by
        arrival order while presenting it as a priority difference.
        """
        from .global_workspace_supply import priority_of

        return priority_of(self, now)
