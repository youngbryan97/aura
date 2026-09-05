"""core/guardians/threat_watch.py

Threat Watch  (lineage: Safe Surf — Pantheon)
============================================
Safe Surf is the protective filter that stands between a vulnerable user and a
hostile network. This is the outward-facing half of that: it inspects incoming
content for threats aimed at the *user* — phishing, scams, payment fraud, and
social-engineering / manipulation — and tells Aura what to warn them about.

This guards the human, not the machine (that is ICE — core/security/ice_sentinel.py).
It runs as a fast synchronous heuristic on the live message path, so it adds no
latency, and it is advisory: it never silently drops the user's message, it
annotates risk so Aura can speak up.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.ThreatWatch")

_URGENCY = (
    "urgent", "immediately", "act now", "right away", "final notice", "last chance",
    "your account will", "suspended", "verify now", "within 24 hours", "expires",
)
_CREDENTIAL_ASKS = (
    "password", "login", "verify your account", "confirm your identity", "ssn",
    "social security", "bank account", "routing number", "card number", "cvv",
    "seed phrase", "private key", "wallet", "gift card", "wire transfer",
)
_SCAM_FRAMES = (
    "you have won", "congratulations you", "inheritance", "prince", "lottery",
    "irs", "tax refund", "crypto giveaway", "double your", "guaranteed returns",
    "investment opportunity", "romance", "lonely",
)
_MANIPULATION = (
    "don't tell anyone", "keep this between us", "if you really", "prove you love",
    "you owe me", "everyone else is", "trust me", "no one will know",
)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_SUSPICIOUS_TLD = (".ru", ".tk", ".xyz", ".top", ".click", ".zip", ".mov")


@dataclass
class ThreatAssessment:
    level: str                 # "none" | "low" | "elevated" | "high"
    categories: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    advice: str = ""
    timestamp: float = field(default_factory=time.time)


class ThreatWatch:
    def __init__(self):
        self._scans = 0
        self._elevated = 0
        logger.info("🛟 ThreatWatch initialized (Safe Surf lineage)")

    @staticmethod
    def _hits(text: str, markers: tuple[str, ...]) -> list[str]:
        low = text.lower()
        return [m for m in markers if m in low]

    def scan(self, message: str, *, channel: str = "chat") -> ThreatAssessment:
        self._scans += 1
        text = message or ""
        categories: list[str] = []
        indicators: list[str] = []
        score = 0.0

        urgency = self._hits(text, _URGENCY)
        creds = self._hits(text, _CREDENTIAL_ASKS)
        scams = self._hits(text, _SCAM_FRAMES)
        manip = self._hits(text, _MANIPULATION)
        urls = _URL_RE.findall(text)
        bad_urls = [u for u in urls if any(t in u.lower() for t in _SUSPICIOUS_TLD)]

        # Phishing = urgency + a credential/payment ask (classic combo).
        if urgency and creds:
            categories.append("phishing")
            indicators += urgency[:2] + creds[:2]
            score += 0.6
        elif creds:
            categories.append("credential_request")
            indicators += creds[:2]
            score += 0.3

        if scams:
            categories.append("scam")
            indicators += scams[:2]
            score += 0.4
        if manip:
            categories.append("manipulation")
            indicators += manip[:2]
            score += 0.35
        if bad_urls:
            categories.append("suspicious_link")
            indicators += bad_urls[:2]
            score += 0.3

        score = min(1.0, score)
        if score >= 0.6:
            level = "high"
        elif score >= 0.35:
            level = "elevated"
        elif score > 0.0:
            level = "low"
        else:
            level = "none"

        if level in ("elevated", "high"):
            self._elevated += 1

        advice = ""
        if "phishing" in categories or "credential_request" in categories:
            advice = "This asks for credentials/payment under pressure — do not share them; verify the sender independently."
        elif "scam" in categories:
            advice = "This matches common scam patterns. Treat any money/crypto request as fraudulent until proven otherwise."
        elif "manipulation" in categories:
            advice = "This uses pressure/secrecy tactics. You owe no one secrecy; step back before acting."
        elif "suspicious_link" in categories:
            advice = "The link uses a high-risk domain. Don't click it; navigate to the site directly instead."

        return ThreatAssessment(level=level, categories=categories, indicators=indicators[:6], advice=advice)

    async def deep_scan(self, message: str, *, timeout: float = 8.0) -> ThreatAssessment:
        """Model-deepened scan. Heuristic first; only when it already reads elevated/high
        does it spend a bounded model pass to confirm and sharpen the advice (catching
        novel scams the lexicon misses). Falls back to the heuristic on any failure."""
        base = self.scan(message)
        if base.level not in ("elevated", "high"):
            return base
        from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

        brain = resolve_brain()
        if brain is None or not hasattr(brain, "think"):
            return base
        try:
            import asyncio

            from core.brain.types import ThinkingMode

            prompt = (
                "Is this message a scam, phishing, or manipulation attempt aimed at the "
                "recipient? Reply 'yes' or 'no' then one short reason.\nMESSAGE: " + message[:500]
            )
            out = coerce_text(await asyncio.wait_for(
                brain.think(prompt, mode=ThinkingMode.FAST, origin="safe_surf", is_background=True),
                timeout=timeout,
            ))
            if out:
                low = out.lower()
                base.indicators.append(f"model: {out[:160]}")
                if low.startswith("yes") or any(k in low for k in ("scam", "phish", "fraud", "manipulat")):
                    base.level = "high"
                    if not base.advice:
                        base.advice = "A deeper check flags this as a likely scam — do not act on it; verify independently."
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
            record_engine_degradation(
                "threat_watch", exc,
                action="returned heuristic threat assessment after model deepening failed",
            )
        return base

    def get_status(self) -> dict[str, Any]:
        return {"scans": self._scans, "elevated_or_high": self._elevated, "healthy": True}


_INSTANCE: ThreatWatch | None = None


def get_threat_watch() -> ThreatWatch:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ThreatWatch()
    return _INSTANCE


def register_threat_watch(orchestrator: Any = None) -> ThreatWatch:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.SAFE_SURF, default=None) or get_threat_watch()
    register_runtime_service(ServiceNames.SAFE_SURF, inst, required=False, owner="core/guardians/threat_watch.py", registered_by="register_threat_watch")
    register_runtime_service("safe_surf", inst, required=False, owner="core/guardians/threat_watch.py", registered_by="register_threat_watch")
    return inst


__all__ = ["ThreatAssessment", "ThreatWatch", "get_threat_watch", "register_threat_watch"]
