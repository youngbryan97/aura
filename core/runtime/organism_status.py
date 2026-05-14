from __future__ import annotations

from typing import Any, Dict, Optional

from core.health.degraded_events import get_unified_failure_state
from core.runtime.service_access import (
    resolve_canonical_self,
    resolve_identity_model,
    resolve_orchestrator,
    resolve_state_repository,
)


def _canonical_self_version(current: Any) -> Optional[int]:
    try:
        return int(getattr(current, "version", 0) or 0)
    except (TypeError, ValueError):
        return None


_USER_FACING_STATUS_ORIGINS = {
    "user",
    "api",
    "chat",
    "desktop",
    "gui",
    "voice",
    "web",
    "websocket",
    "ws",
    "direct",
    "external",
}


def _is_user_facing_status_origin(origin: Any) -> bool:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    if not normalized:
        return False
    tokens = {token for token in normalized.split("_") if token}
    return normalized in _USER_FACING_STATUS_ORIGINS or bool(tokens & _USER_FACING_STATUS_ORIGINS)


def _looks_like_stale_user_prompt(text: str) -> bool:
    lowered = text.lower()
    if not text:
        return False
    if "?" in text and any(marker in lowered for marker in ("aura", "you", "your", "what", "why", "how", "can ", "could ", "please")):
        return True
    return len(text) > 120 and any(
        marker in lowered
        for marker in (
            "what is actually on your mind",
            "tell me",
            "answer like",
            "why does",
            "can you",
            "could you",
        )
    )


def _clean_current_intention_for_status(intention: Any, live_objective: Any = "", live_origin: Any = "") -> str:
    text = " ".join(str(intention or "").split())
    objective = " ".join(str(live_objective or "").split())
    origin = str(live_origin or "").strip()
    if objective and origin.lower() not in {"system", "unknown"} and not _is_user_facing_status_origin(origin):
        return objective[:260]
    lowered = text.lower()
    if not text:
        return objective[:260] if objective and not _is_user_facing_status_origin(live_origin) else ""
    if "[referential anchor]" in lowered or len(text) > 320 or _looks_like_stale_user_prompt(text):
        return objective[:260] if objective and not _is_user_facing_status_origin(live_origin) else "idle"
    return text[:260]


def get_organism_status(orchestrator: Any = None) -> Dict[str, Any]:
    orch = orchestrator or resolve_orchestrator(default=None)
    repo = resolve_state_repository(orch, default=None)
    state = getattr(repo, "_current", None) if repo is not None else None
    cognition = getattr(state, "cognition", None) if state is not None else None

    canonical_self = resolve_canonical_self(default=None)
    canonical_self_version = _canonical_self_version(canonical_self)
    identity_model = resolve_identity_model(default=None)
    failure_state = get_unified_failure_state(limit=25)
    resource_state: Dict[str, Any] = {}
    try:
        from core.container import ServiceContainer

        stakes = ServiceContainer.get("resource_stakes", default=None)
        if stakes is not None:
            state_obj = stakes.state()
            resource_state = {
                "viability": float(state_obj.viability),
                "energy": float(state_obj.energy),
                "integrity": float(state_obj.integrity),
                "degradation_events": int(state_obj.degradation_events),
                "action_envelope": stakes.action_envelope("normal").as_dict(),
            }
    except Exception:
        resource_state = {}

    current_intention = ""
    if canonical_self is not None:
        current_intention = _clean_current_intention_for_status(
            getattr(canonical_self, "current_intention", "") or "",
            str(getattr(cognition, "current_objective", "") or "") if cognition is not None else "",
            str(getattr(cognition, "current_origin", "") or "") if cognition is not None else "",
        )
    identity_name = ""
    if canonical_self is not None:
        identity_name = str(getattr(getattr(canonical_self, "identity", None), "name", "") or "")
    if not identity_name and identity_model is not None:
        identity_name = str(
            getattr(identity_model, "name", "")
            or getattr(getattr(identity_model, "state", None), "self_narrative", "")
            or ""
        )
    if not identity_name:
        identity_name = "Aura"

    return {
        "identity_surface": "canonical_self" if canonical_self is not None else "fallback_identity",
        "identity_name": identity_name,
        "canonical_self_version": canonical_self_version,
        "state_version": getattr(state, "version", None) if state is not None else None,
        "failure_state": failure_state,
        "failure_pressure": float(failure_state.get("pressure", 0.0) or 0.0),
        "resource_stakes": resource_state,
        "current_objective": str(getattr(cognition, "current_objective", "") or "") if cognition is not None else "",
        "current_intention": current_intention,
        "pending_initiatives": len(list(getattr(cognition, "pending_initiatives", []) or [])) if cognition is not None else 0,
        "active_goals": len(list(getattr(cognition, "active_goals", []) or [])) if cognition is not None else 0,
    }
