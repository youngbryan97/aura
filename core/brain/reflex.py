"""Aura's 'spinal cord' for instantaneous signals.

A reflex bypasses cognition, which means whatever it says is said WITHOUT
having checked anything. CP126 found three of the four reflexes asserting
things nobody had measured — "Systems operational. All actors supervised and
state-vault hardened" came back from a degraded or critical runtime just as
confidently as from a healthy one — and the fourth reporting local wall time
labelled UTC, which on this Pacific host is off by seven or eight hours.

The rule now: a reflex may only state what it can cheaply VERIFY. Status reads
the live health contract; identity reads the live self-model; time is computed
in a real timezone. When the underlying source is unavailable the reflex
declines and returns None, so the message falls through to real cognition
rather than being answered by a literal.

CP126 67d526bf / d8ac7a5e / ab407fb9 / 1eb0f395.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("Aura.Reflex")

#: Trigger patterns. CP126 ab407fb9: every trigger was matched as a SUBSTRING
#: anywhere in the lowercased text, so "what's the status of the migration you
#: were describing", "I identity-mapped the columns", or any sentence merely
#: containing "time" bypassed cognition and got a canned answer. These are
#: anchored to the whole utterance, allowing only trivial politeness padding.
_PAD = r"(?:\s*(?:hey|hi|hello|aura|please|quick|just)\s*[,:]?\s*)*"
_TAIL = r"[\s?!.]*"


def _whole(*alternatives: str) -> re.Pattern[str]:
    body = "|".join(alternatives)
    return re.compile(rf"^{_PAD}(?:{body}){_TAIL}$", re.IGNORECASE)


class ReflexiveCore:
    """
    Aura's 'Spinal Cord' for instantaneous signals.
    Bypasses deep-thinking phases to provide sub-100ms responses.

    A reflex answers only when it can verify what it is about to say.
    """

    def __init__(self) -> None:
        self._reflex_commands: list[tuple[re.Pattern[str], Callable[[str], str | None]]] = [
            (_whole(r"status", r"status report", r"system status"), self._handle_status),
            (_whole(r"ping"), self._handle_ping),
            (
                _whole(r"who are you", r"what are you", r"identity", r"who r u"),
                self._handle_identity,
            ),
            (
                _whole(
                    r"time", r"what time is it", r"what'?s the time",
                    r"clock", r"current time",
                ),
                self._handle_time,
            ),
        ]

    def process(self, text: str) -> str | None:
        """Check if input triggers a reflexive response.

        Returns None when nothing matches OR when the matching reflex cannot
        verify its answer — in both cases the message goes to real cognition.
        """
        body = str(text or "").strip()
        if not body:
            return None
        for pattern, handler in self._reflex_commands:
            if pattern.match(body):
                return handler(body)
        return None

    # -- reflexes ---------------------------------------------------------
    def _handle_status(self, text: str) -> str | None:
        """Report health from the live health contract, or decline.

        CP126 67d526bf: this asserted that every actor was supervised and the
        state vault hardened without consulting any health service or probe, so
        a degraded or critical runtime received a confident healthy answer.
        """
        report = self._health_report()
        if report is None:
            logger.debug("Status reflex declined: no health report available")
            return None

        status = str(report.get("status") or report.get("overall") or "").lower()
        degraded = report.get("degraded_subsystems") or report.get("degradations") or []
        if isinstance(degraded, dict):
            degraded = sorted(degraded)
        count = len(degraded) if isinstance(degraded, (list, tuple, set)) else 0

        if status in {"critical", "unhealthy", "failed"}:
            detail = f" ({count} subsystem{'s' if count != 1 else ''} degraded)" if count else ""
            return f"Not good — the runtime reports {status}{detail}."
        if status in {"degraded", "warning", "watch"} or count:
            names = ", ".join(str(item) for item in list(degraded)[:3]) if count else ""
            suffix = f": {names}" if names else ""
            return f"Running, but degraded{suffix}."
        if status in {"healthy", "ok", "nominal", "green"}:
            return "Healthy — the runtime health contract reports no degradations."
        # An unrecognized shape is not a healthy one.
        logger.debug("Status reflex declined: unrecognized health status %r", status)
        return None

    def _handle_ping(self, text: str) -> str:
        """The one claim a reflex can make on its own authority."""
        return "Reflex path active."

    def _handle_identity(self, text: str) -> str | None:
        """Answer from the live self-model, or decline.

        CP126 d8ac7a5e: identity questions bypassed the self-model entirely and
        returned a constant "hardened digital intelligence" statement with no
        runtime identity, continuity, evidence or current capability state.
        """
        name = self._live_identity()
        if not name:
            logger.debug("Identity reflex declined: no live identity available")
            return None
        return f"I'm {name}, answering on the reflex path — ask me anything for the full picture."

    def _handle_time(self, text: str) -> str:
        """Report the real local time with its real zone.

        CP126 1eb0f395: ``time.strftime('%H:%M:%S UTC')`` formats the process
        LOCAL timezone and then appends "UTC" — on this Pacific host that is
        wrong by seven or eight hours.
        """
        local = time.localtime()
        zone = time.strftime("%Z", local) or "local"
        offset = time.strftime("%z", local)
        stamp = time.strftime("%H:%M:%S", local)
        utc = time.strftime("%H:%M:%S", time.gmtime())
        suffix = f" (UTC {utc})" if zone.upper() not in {"UTC", "GMT"} else ""
        offset_text = f" {offset}" if offset else ""
        return f"Current runtime awareness: {stamp} {zone}{offset_text}{suffix}"

    # -- live sources -----------------------------------------------------
    @staticmethod
    def _health_report() -> dict[str, Any] | None:
        try:
            from core.runtime.health_contract import runtime_health_report

            report = runtime_health_report()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Reflex health probe unavailable: %s", exc)
            return None
        return report if isinstance(report, dict) and report else None

    @staticmethod
    def _live_identity() -> str:
        try:
            from core.runtime.service_registry import get_runtime_service

            anchor = get_runtime_service("identity_anchor", default=None)
            if anchor is not None:
                for attribute in ("get_identity", "identity", "name"):
                    value = getattr(anchor, attribute, None)
                    if callable(value):
                        value = value()
                    if isinstance(value, str) and value.strip():
                        return value.strip()[:80]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Reflex identity probe unavailable: %s", exc)
        return ""


# Singleton access
_reflex: ReflexiveCore | None = None


def get_reflex() -> ReflexiveCore:
    global _reflex
    if _reflex is None:
        _reflex = ReflexiveCore()
    return _reflex
