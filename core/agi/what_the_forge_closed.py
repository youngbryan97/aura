"""Where the forge's chain stops, named by the step that is empty.

Six arrows have to land for a forged skill to have improved anything: a gap
is seen often enough to be worth answering, a candidate is drafted for it, the
candidate passes probes that were fixed before it was redrafted, the artifact
is installed, something takes it, and the gap stops being logged.

Each of those exists and each is reported somewhere. What was not reported is
which one is empty, and that is the only thing worth knowing about a loop that
is running and paying nothing. A forge with 40 gaps, 3 candidates, 3 verified,
3 installed and 0 ever taken is not the same organism as one with 0 gaps, and
both used to read as "closed_the_gap: 0".

The measure is the first step whose count is zero. Steps after it are not
failures — they never had input — and reporting them as failures is how a
chain with one broken link reads as five.

Nothing here decides anything. It says where to look.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.WhatTheForgeClosed")

__all__ = [
    "AForgeReading",
    "THE_CHAIN",
    "how_the_forge_closes",
    "where_the_forge_stops",
]

#: The six arrows, in the order they have to land.
THE_CHAIN: tuple[str, ...] = (
    "gaps seen",
    "gaps worth answering",
    "candidates drafted",
    "candidates verified",
    "skills installed",
    "skills taken",
    "gaps that stopped",
)


@dataclass(frozen=True, slots=True)
class AForgeReading:
    """Every step of the chain, and the first one that is empty."""

    counts: dict[str, int]
    stops_at: str
    turns_over: bool
    targets_held: dict[str, Any]

    @property
    def reached(self) -> int:
        """How many arrows landed before the chain ran out of input."""
        if self.turns_over:
            return len(THE_CHAIN)
        return THE_CHAIN.index(self.stops_at) if self.stops_at in THE_CHAIN else 0

    @property
    def why_it_matters(self) -> str:
        if self.turns_over:
            return "a gap was answered and stopped being logged"
        if self.stops_at == THE_CHAIN[0]:
            return "nothing has failed often enough to be worth a skill"
        if self.stops_at == "skills taken":
            return (
                "forged, verified, installed and never once taken: the forge "
                "is running and paying nothing"
            )
        return f"the chain has no input at {self.stops_at!r}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.forge.chain.v1",
            "chain": list(THE_CHAIN),
            "counts": dict(self.counts),
            "stops_at": self.stops_at,
            "arrows_that_landed": self.reached,
            "turns_over": self.turns_over,
            "why_it_matters": self.why_it_matters,
            "the_target_held": self.targets_held,
        }


def _counts_from(forge: Any) -> dict[str, int]:
    from core.agi.skill_synthesizer import GAP_FORGE_THRESHOLD

    produced = forge.what_the_forge_has_produced()
    gap_counts = dict(getattr(forge, "_gap_counts", None) or {})
    rows = produced.get("skills") or []
    return {
        "gaps seen": len(gap_counts),
        "gaps worth answering": sum(
            1 for count in gap_counts.values() if count >= GAP_FORGE_THRESHOLD
        ),
        "candidates drafted": int(produced.get("forged", 0)),
        "candidates verified": int(produced.get("verified", 0)),
        "skills installed": int(produced.get("installed", 0)),
        "skills taken": int(produced.get("taken_at_least_once", 0)),
        "gaps that stopped": sum(
            1 for one in rows if (one.get("the_gap_after") or {}).get("stopped")
        ),
    }


def how_the_forge_closes(
    forge: Any = None, *, targets_held: dict[str, Any] | None = None
) -> AForgeReading:
    """The chain as it stands, and the first step with nothing in it."""
    if forge is None:
        from core.agi.skill_synthesizer import get_skill_synthesizer

        forge = get_skill_synthesizer()
    try:
        counts = _counts_from(forge)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("could not read the forge: %s", exc)
        counts = dict.fromkeys(THE_CHAIN, 0)

    stops_at = ""
    for step in THE_CHAIN:
        if counts.get(step, 0) <= 0:
            stops_at = step
            break

    return AForgeReading(
        counts=counts,
        stops_at=stops_at,
        turns_over=not stops_at,
        # Whether a redraft moved its own target is the forge's own record and
        # core/agi may not reach into core/skill_management to read it. The
        # caller that can see both — the inspector — puts them side by side,
        # and passing it in keeps this module's answer about the six arrows.
        targets_held=dict(targets_held or {}),
    )


def where_the_forge_stops(forge: Any = None) -> str:
    """The one name worth reading off a loop that is paying nothing."""
    return how_the_forge_closes(forge).stops_at
