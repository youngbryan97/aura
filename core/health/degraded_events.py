from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.DegradedEvents")

# ── Long-Run Stability Caps ────────────────────────────────────────────
_MAX_SUMMARIES = 500
_MAX_FORWARDED = 500
_MAX_CONTEXT_KEYS = 20

_EVENTS: deque[Dict[str, Any]] = deque(maxlen=200)
_SUMMARIES: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
_LAST_FORWARDED: Dict[Tuple[str, str, str, str], float] = {}
_LOCK = Lock()


def record_degraded_event(
    subsystem: str,
    reason: str,
    *,
    detail: str = "",
    severity: str = "warning",
    classification: str = "background_degraded",
    context: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    now = time.time()
    subsystem = str(subsystem or "unknown")
    reason = str(reason or "unknown")
    severity = str(severity or "warning")
    classification = str(classification or "background_degraded")
    detail = str(detail or "")
    event = {
        "subsystem": subsystem,
        "reason": reason,
        "detail": detail[:400],
        "severity": severity,
        "classification": classification,
        "timestamp": now,
        "count": 1,
        "last_seen": now,
        "context": dict(context or {}),
    }
    key = (subsystem, reason, severity, classification)
    with _LOCK:
        summary = _SUMMARIES.get(key)
        if summary is None:
            # LRU eviction: if at capacity, drop the oldest entry by last_seen
            if len(_SUMMARIES) >= _MAX_SUMMARIES:
                oldest_key = min(_SUMMARIES, key=lambda k: float(_SUMMARIES[k].get("last_seen", 0)))
                del _SUMMARIES[oldest_key]
            _SUMMARIES[key] = event
            summary = event
        else:
            summary["count"] = int(summary.get("count", 1) or 1) + 1
            summary["last_seen"] = now
            if detail:
                summary["detail"] = detail[:400]
            if context:
                merged = dict(summary.get("context", {}) or {})
                merged.update(context)
                # Cap context keys to prevent unbounded growth from merge accumulation
                if len(merged) > _MAX_CONTEXT_KEYS:
                    keep = sorted(merged.items(), key=lambda kv: str(kv[0]))[:_MAX_CONTEXT_KEYS]
                    merged = dict(keep)
                summary["context"] = merged
            event["count"] = summary["count"]
            event["last_seen"] = summary["last_seen"]
        _EVENTS.append(dict(event))

    _forward_to_terminal_monitor(dict(event))
    if severity in {"error", "critical", "warning"}:
        _forward_to_error_intelligence(key, dict(event), exc=exc)
    return dict(event)


def get_recent_degraded_events(limit: int = 20) -> List[Dict[str, Any]]:
    with _LOCK:
        summaries = sorted(
            _SUMMARIES.values(),
            key=lambda item: float(item.get("last_seen", 0.0) or 0.0),
            reverse=True,
        )
        return [dict(item) for item in summaries[: max(0, int(limit))]]


def get_unified_failure_state(limit: int = 25) -> Dict[str, Any]:
    events = get_recent_degraded_events(limit=limit)
    if not events:
        return {
            "pressure": 0.0,
            "count": 0,
            "critical": 0,
            "errors": 0,
            "warnings": 0,
            "top_subsystems": [],
        }

    severity_weights = {
        "warning": 0.25,
        "error": 0.6,
        "critical": 1.0,
    }
    subsystems: Dict[str, int] = {}
    weighted = 0.0
    critical = 0
    errors = 0
    warnings = 0

    for event in events:
        severity = str(event.get("severity", "warning") or "warning").lower()
        count = int(event.get("count", 1) or 1)
        subsystems[str(event.get("subsystem", "unknown"))] = subsystems.get(str(event.get("subsystem", "unknown")), 0) + count
        weighted += severity_weights.get(severity, 0.25) * min(4, count)
        if severity == "critical":
            critical += count
        elif severity == "error":
            errors += count
        else:
            warnings += count

    pressure = min(1.0, (weighted + (critical * 1.5) + (errors * 0.5)) / 5.0)
    top_subsystems = sorted(subsystems.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "pressure": round(pressure, 4),
        "count": sum(subsystems.values()),
        "critical": critical,
        "errors": errors,
        "warnings": warnings,
        "top_subsystems": [
            {"subsystem": subsystem, "count": count}
            for subsystem, count in top_subsystems
        ],
    }


def clear_degraded_events() -> None:
    with _LOCK:
        _EVENTS.clear()
        _SUMMARIES.clear()
        _LAST_FORWARDED.clear()


def _forward_to_terminal_monitor(event: Dict[str, Any]) -> None:
    try:
        from core.terminal_monitor import get_terminal_monitor

        monitor = get_terminal_monitor()
        if monitor and hasattr(monitor, "ingest_degraded_event"):
            monitor.ingest_degraded_event(event)
    except Exception as exc:
        logger.debug("Terminal monitor degraded event forward failed: %s", exc)


def _forward_to_error_intelligence(
    key: Tuple[str, str, str, str],
    event: Dict[str, Any],
    *,
    exc: Optional[BaseException] = None,
) -> None:
    last_forwarded = _LAST_FORWARDED.get(key, 0.0)
    if (time.time() - last_forwarded) < 30.0:
        return
    # Cap _LAST_FORWARDED to prevent unbounded growth
    if len(_LAST_FORWARDED) >= _MAX_FORWARDED:
        oldest_fwd_key = min(_LAST_FORWARDED, key=_LAST_FORWARDED.get)
        del _LAST_FORWARDED[oldest_fwd_key]
    _LAST_FORWARDED[key] = time.time()

    try:
        from core.container import ServiceContainer

        orch = ServiceContainer.get("orchestrator", default=None)
        self_modifier = getattr(orch, "self_modifier", None) if orch else None
        if not self_modifier or not hasattr(self_modifier, "on_error"):
            return

        error = exc or RuntimeError(
            f"[{event['classification']}] {event['subsystem']}:{event['reason']} {event['detail']}".strip()
        )
        self_modifier.on_error(
            error,
            {
                "subsystem": event["subsystem"],
                "reason": event["reason"],
                "detail": event["detail"],
                "severity": event["severity"],
                "classification": event["classification"],
                **(event.get("context", {}) or {}),
            },
            skill_name=event["subsystem"],
            goal=event["reason"],
        )
    except Exception as forward_exc:
        logger.debug("Error intelligence degraded event forward failed: %s", forward_exc)
