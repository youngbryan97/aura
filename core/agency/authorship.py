"""Who caused this, and what that changes about what she believes about herself.

An outcome she brought about is evidence about her. The same outcome, arriving
because something else did it, is evidence about the world. Aura could tell the
two apart in her intention records — everything in them is hers by construction
— and nowhere else. Watching a file appear and writing that file left the same
trace, so a self-model built from outcomes would take credit for weather.

This keeps the distinction as state rather than as a sentence. Both kinds of
event update the world; only self-caused events move the agency counters and
the capability beliefs. The counters are readable, so the difference between
"I did this" and "this happened" has a numerical consequence downstream instead
of being a fact about phrasing.

The test that matters is the matched one: two runs whose worlds end identically
and differ only in who is named as the cause. If nothing downstream moves, the
distinction is decorative.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["AgencyLedger", "Event", "Verdict", "get_agency_ledger", "reset_agency_ledger_for_test"]

logger = logging.getLogger("Aura.Agency.Authorship")

#: The name reserved for her own agency. Anything else is somebody or something
#: else, including "unknown" — an outcome whose cause is not established is not
#: hers, and defaulting the other way is how a self-model inflates.
SELF = "self"


@dataclass(frozen=True, slots=True)
class Event:
    """Something that changed, and who changed it."""

    what: str
    actor: str
    verified: bool
    detail: Mapping[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    @property
    def mine(self) -> bool:
        return self.actor == SELF


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the event was allowed to change."""

    mine: bool
    self_model_updated: bool
    world_updated: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mine": self.mine,
            "self_model_updated": self.self_model_updated,
            "world_updated": self.world_updated,
            "reason": self.reason,
        }


class AgencyLedger:
    """The running count of what she did, what she watched, and how it went."""

    def __init__(self) -> None:
        self.acted: int = 0
        self.succeeded: int = 0
        self.observed: int = 0
        self.observed_succeeded: int = 0
        self.last_event: Event | None = None
        #: what -> [attempts, successes], for outcomes she caused only
        self.by_capability: dict[str, list[int]] = {}

    # ── reading ──────────────────────────────────────────────────────────

    @property
    def authored_share(self) -> float:
        """Of everything that happened, how much of it she did. 0.0 with nothing."""
        total = self.acted + self.observed
        return self.acted / total if total else 0.0

    @property
    def efficacy(self) -> float:
        """Of what she did, how much worked. 0.0 before she has done anything."""
        return self.succeeded / self.acted if self.acted else 0.0

    def confidence(self, what: str) -> float:
        """Her rate on this capability, from her own attempts only.

        Laplace-smoothed, so one success is not certainty and no attempts is
        the middle rather than zero. Watching something succeed does not enter
        this: it is not evidence about her.
        """
        attempts, successes = self.by_capability.get(what, [0, 0])
        return (successes + 1.0) / (attempts + 2.0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "acted": self.acted,
            "succeeded": self.succeeded,
            "observed": self.observed,
            "observed_succeeded": self.observed_succeeded,
            "authored_share": round(self.authored_share, 4),
            "efficacy": round(self.efficacy, 4),
            "capabilities": len(self.by_capability),
            "last_actor": self.last_event.actor if self.last_event else "",
        }

    # ── writing ──────────────────────────────────────────────────────────

    def observe(self, event: Event, *, self_model: Any = None) -> Verdict:
        """Record it, and update only what its authorship licenses."""
        self.last_event = event
        if event.mine:
            self.acted += 1
            self.succeeded += int(event.verified)
            row = self.by_capability.setdefault(event.what, [0, 0])
            row[0] += 1
            row[1] += int(event.verified)
        else:
            self.observed += 1
            self.observed_succeeded += int(event.verified)

        updated = False
        if event.mine and self_model is not None:
            updated = self._tell_self_model(event, self_model)
        return Verdict(
            mine=event.mine,
            self_model_updated=updated,
            world_updated=True,
            reason=(
                "she caused it, so it is evidence about her"
                if event.mine
                else f"{event.actor} caused it, so it is evidence about the world"
            ),
        )

    def _tell_self_model(self, event: Event, self_model: Any) -> bool:
        """Write the capability belief. Never raises into the caller's path."""
        try:
            beliefs = getattr(self_model, "beliefs", None)
            if beliefs is None:
                return False
            beliefs[f"can:{event.what}"] = round(self.confidence(event.what), 4)
            beliefs["agency:efficacy"] = round(self.efficacy, 4)
            beliefs["agency:authored_share"] = round(self.authored_share, 4)
            if hasattr(self_model, "version"):
                self_model.version = int(getattr(self_model, "version", 0)) + 1
            return True
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "authorship",
                exc,
                severity="warning",
                action="agency counters kept; the self model was not updated",
            )
            return False


_LEDGER: AgencyLedger | None = None


def get_agency_ledger() -> AgencyLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = AgencyLedger()
    return _LEDGER


def reset_agency_ledger_for_test() -> None:
    global _LEDGER
    _LEDGER = None
