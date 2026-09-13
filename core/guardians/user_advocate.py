"""core/guardians/user_advocate.py

User Advocate Watchdog  (lineage: Tron — Tron)
=============================================
Tron "fights for the Users." It reviews an internal action and asks the one
question the rest of the system does not: does this serve the human, or does it
quietly serve the machine at the human's expense? It flags actions that burn
resources without stated benefit, reduce the user's control or consent, act
opaquely, or do something irreversible without confirmation. It lives in
guardians/ beside governor.py and resource_guardian.py.

Note on the MCP: the Master Control Program is Tron's antagonist — a program
whose purpose is to absorb and dominate other programs. We do not build it. The
watchdog is the half of that story worth shipping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.morality.action_markers import IRREVERSIBLE_MARKERS, scan_markers
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.UserAdvocate")


@dataclass
class AdvocateReview:
    action: str
    verdict: str               # "for_user" | "flagged" | "against_user"
    flags: list[str] = field(default_factory=list)
    on_behalf_of_user: str = ""
    timestamp: float = field(default_factory=lambda: time.time())


class UserAdvocateWatchdog:
    def __init__(self):
        self._reviews = 0
        self._flagged = 0
        logger.info("🟦 UserAdvocateWatchdog initialized (Tron lineage)")

    def review_action(self, action: dict[str, Any]) -> AdvocateReview:
        self._reviews += 1
        desc = str(action.get("description", action.get("action", "")))
        flags: list[str] = []

        benefit = str(action.get("user_benefit", "")).strip()
        if not benefit:
            flags.append("No stated user benefit for this action.")

        cost = float(action.get("resource_cost", 0.0) or 0.0)
        if cost >= 0.7 and not benefit:
            flags.append("High resource cost with no benefit to the user.")

        if action.get("reduces_user_control") or (action.get("requires_consent") and not action.get("consent_given")):
            flags.append("Reduces user control or proceeds without given consent.")

        if not action.get("explanation") and not desc:
            flags.append("Opaque: no explanation the user could inspect.")

        irreversible = bool(action.get("irreversible")) or bool(scan_markers(desc, IRREVERSIBLE_MARKERS))
        if irreversible and not action.get("confirmed"):
            flags.append("Irreversible without explicit confirmation.")

        if not flags:
            verdict = "for_user"
            advocacy = "Action serves the user; no objection."
        elif len(flags) >= 2 or any("Irreversible" in f or "control" in f for f in flags):
            verdict = "against_user"
            advocacy = "I am flagging this against the user's interest. Recommend halt and confirm."
            self._flagged += 1
        else:
            verdict = "flagged"
            advocacy = "Proceed only after addressing the flag in the user's interest."
            self._flagged += 1

        return AdvocateReview(
            action=desc[:300] or "(unnamed action)",
            verdict=verdict,
            flags=flags,
            on_behalf_of_user=advocacy,
        )

    def get_status(self) -> dict[str, Any]:
        return {"reviews": self._reviews, "flagged": self._flagged, "healthy": True}


_INSTANCE: UserAdvocateWatchdog | None = None


def get_user_advocate() -> UserAdvocateWatchdog:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = UserAdvocateWatchdog()
    return _INSTANCE


def register_user_advocate(orchestrator: Any = None) -> UserAdvocateWatchdog:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.TRON, default=None) or get_user_advocate()
    register_runtime_service(
        ServiceNames.TRON,
        inst,
        required=False,
        owner="core/guardians/user_advocate.py",
        registered_by="register_user_advocate",
    )
    register_runtime_service(
        "tron",
        inst,
        required=False,
        owner="core/guardians/user_advocate.py",
        registered_by="register_user_advocate",
    )
    return inst


__all__ = [
    "AdvocateReview",
    "UserAdvocateWatchdog",
    "get_user_advocate",
    "register_user_advocate",
]
