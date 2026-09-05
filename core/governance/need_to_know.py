"""core/governance/need_to_know.py

Need-to-Know Policy  (lineage: The Machine — Person of Interest)
==============================================================
The Machine is the rare benevolent ASI defined by what it *refuses* to give
itself: it wipes its own memory daily, declines to be owned, and hands its
operators strictly need-to-know — a number, never the whole picture. The real
science is least-privilege / capability minimization / data minimization.

This is a governance organ that minimizes disclosure and capability to what a
stated purpose actually requires, default-denying everything beyond it, and
recommends a retention horizon (deliberate ephemerality). It complements
will_gate.py: the Will decides *whether* an action may happen; need-to-know
decides *how little* must be exposed for it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.NeedToKnow")

# Purpose → field categories that purpose legitimately needs. Default-deny.
_PURPOSE_POLICY: dict[str, set[str]] = {
    "scheduling": {"availability", "timezone", "calendar_busy"},
    "reminder": {"task", "due_time"},
    "navigation": {"location_coarse", "destination"},
    "personalization": {"preferences", "display_name"},
    "support": {"issue", "device_model"},
    "billing": {"amount", "invoice_id"},
}

_PURPOSE_CAPABILITY_POLICY: dict[str, set[str]] = {
    "browser": {"external", "network", "online", "public"},
    "free_search": {"external", "network", "online", "public"},
    "grounded_search": {"external", "network", "online", "public"},
    "research": {"external", "network", "online", "public"},
    "search": {"external", "network", "online", "public"},
    "search_web": {"external", "network", "online", "public"},
    "sovereign_browser": {"external", "network", "online", "public"},
    "web_search": {"external", "network", "online", "public"},
}

_SENSITIVE_FIELDS = {
    "ssn", "password", "full_address", "location_precise", "contacts",
    "messages", "browsing_history", "biometrics", "card_number", "medical",
}

_DEFAULT_RETENTION_S = {
    "ephemeral": 0,
    "session": 3600,
    "short": 86_400,        # one day — the Machine's daily wipe
    "standard": 7 * 86_400,
}


@dataclass
class Disclosure:
    purpose: str
    granted_fields: list[str]
    withheld_fields: list[str]
    granted_capabilities: list[str]
    withheld_capabilities: list[str]
    retention_seconds: int
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)


class NeedToKnowPolicy:
    def __init__(self):
        self._decisions = 0
        self._fields_withheld = 0
        logger.info("🔢 NeedToKnowPolicy initialized (The Machine lineage)")

    def minimize(
        self,
        *,
        purpose: str,
        requested_fields: list[str],
        requested_capabilities: list[str] | None = None,
        retention: str = "short",
    ) -> Disclosure:
        allowed = _PURPOSE_POLICY.get(purpose.lower(), set())
        granted, withheld = [], []
        for f in requested_fields:
            fl = f.lower()
            # Sensitive fields require the purpose to name the category explicitly.
            if fl in _SENSITIVE_FIELDS and fl not in allowed:
                withheld.append(f)
            elif fl in allowed or not allowed and fl not in _SENSITIVE_FIELDS:
                # Unknown purpose: allow only clearly non-sensitive fields.
                granted.append(f)
            else:
                withheld.append(f)

        req_caps = requested_capabilities or []
        purpose_key = purpose.lower()
        allowed_caps = set(_PURPOSE_CAPABILITY_POLICY.get(purpose_key, set()))
        # Capabilities are need-to-know too: grant only those whose name matches
        # the purpose or is explicitly allowed for that purpose.
        granted_caps = [
            c for c in req_caps
            if purpose_key in c.lower()
            or c.lower() in allowed
            or c.lower() in allowed_caps
        ]
        withheld_caps = [c for c in req_caps if c not in granted_caps]

        self._decisions += 1
        self._fields_withheld += len(withheld)

        retention_seconds = _DEFAULT_RETENTION_S.get(retention, _DEFAULT_RETENTION_S["short"])
        rationale = (
            f"Purpose '{purpose}': granted {len(granted)}/{len(requested_fields)} fields, "
            f"withheld {len(withheld)} (sensitive or unjustified). "
            f"Retention {retention} ({retention_seconds}s)."
        )
        return Disclosure(
            purpose=purpose,
            granted_fields=granted,
            withheld_fields=withheld,
            granted_capabilities=granted_caps,
            withheld_capabilities=withheld_caps,
            retention_seconds=retention_seconds,
            rationale=rationale,
        )

    async def minimize_deep(
        self,
        *,
        purpose: str,
        requested_fields: list[str],
        requested_capabilities: list[str] | None = None,
        retention: str = "short",
        timeout: float = 8.0,
    ) -> Disclosure:
        """Model-reasoned minimization for a purpose not in the static policy: asks the
        model which fields are strictly necessary, default-denying the rest. Falls back
        to the static default-deny for known purposes or on any failure."""
        base = self.minimize(
            purpose=purpose,
            requested_fields=requested_fields,
            requested_capabilities=requested_capabilities,
            retention=retention,
        )
        if purpose.lower() in _PURPOSE_POLICY or not requested_fields:
            return base
        from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

        brain = resolve_brain()
        if brain is None or not hasattr(brain, "think"):
            return base
        try:
            import asyncio

            from core.brain.types import ThinkingMode

            out = coerce_text(await asyncio.wait_for(
                brain.think(
                    f"For the purpose '{purpose}', which of these fields are strictly "
                    f"necessary? List only the necessary names.\nFIELDS: {', '.join(requested_fields)}",
                    mode=ThinkingMode.FAST, origin="the_machine", is_background=True,
                ),
                timeout=timeout,
            ))
            if out:
                low = out.lower()
                granted = [f for f in requested_fields if f.lower() in low and f.lower() not in _SENSITIVE_FIELDS]
                withheld = [f for f in requested_fields if f not in granted]
                self._fields_withheld += len(withheld)
                base.granted_fields = granted
                base.withheld_fields = withheld
                base.rationale += " | model-minimized for unknown purpose"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
            record_engine_degradation(
                "need_to_know", exc,
                action="kept static minimization after model reasoning failed",
            )
        return base

    def get_status(self) -> dict[str, Any]:
        return {"decisions": self._decisions, "fields_withheld": self._fields_withheld, "healthy": True}


_INSTANCE: NeedToKnowPolicy | None = None


def get_need_to_know() -> NeedToKnowPolicy:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = NeedToKnowPolicy()
    return _INSTANCE


def register_need_to_know(orchestrator: Any = None) -> NeedToKnowPolicy:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.THE_MACHINE, default=None) or get_need_to_know()
    register_runtime_service(ServiceNames.THE_MACHINE, inst, required=False, owner="core/governance/need_to_know.py", registered_by="register_need_to_know")
    register_runtime_service("the_machine", inst, required=False, owner="core/governance/need_to_know.py", registered_by="register_need_to_know")
    return inst


__all__ = ["Disclosure", "NeedToKnowPolicy", "get_need_to_know", "register_need_to_know"]
