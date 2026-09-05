from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.DegradedEvents")

# ── Long-Run Stability Caps ────────────────────────────────────────────
_MAX_SUMMARIES = working_history_retention_policy("AURA_DEGRADED_EVENT_SUMMARY_MAX").max_items
_MAX_FORWARDED = working_history_retention_policy("AURA_DEGRADED_EVENT_FORWARDED_MAX").max_items
_MAX_CONTEXT_KEYS = 20
_FAILURE_EVENT_HALF_LIFE_S = 150.0   # pressure halves every 2.5 minutes of no new failures
_FAILURE_EVENT_MAX_AGE_S = 300.0     # events expire after 5 minutes — prevents lockdown spiral

_EVENTS: deque[dict[str, Any]] = deque(maxlen=_MAX_SUMMARIES)
_SUMMARIES: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_LAST_FORWARDED: dict[tuple[str, str, str, str], float] = {}
_LOCK = Lock()
_DEGRADED_EVENTS_RECOVERABLE_ERRORS = (RuntimeError, AttributeError, TypeError, ValueError)


def _record_degraded_events_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "degraded_events",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.AUDIT_GAP,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


def _schedule_awaitable(awaitable: Any, *, label: str) -> None:
    if is_shutdown_requested():
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        def _runner() -> None:
            try:
                asyncio.run(awaitable)
            except _DEGRADED_EVENTS_RECOVERABLE_ERRORS as exc:
                _record_degraded_events_degradation(
                    exc,
                    action="retained degraded event locally after threaded async forward failed",
                    extra={"label": label},
                )
                logger.debug("%s async forward failed: %s", label, exc)

        threading.Thread(target=_runner, name=f"aura_{label}", daemon=True).start()
        return

    try:
        task = get_task_tracker().create_task(awaitable, name=f"degraded_events.{label}")
    except RuntimeError as exc:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        _record_degraded_events_degradation(
            exc,
            action="retained degraded event locally after async forward scheduling failed",
            extra={"label": label},
        )
        return

    def _consume_result(done: asyncio.Task) -> None:
        try:
            done.result()
        except _DEGRADED_EVENTS_RECOVERABLE_ERRORS as exc:
            _record_degraded_events_degradation(
                exc,
                action="retained degraded event locally after scheduled async forward failed",
                extra={"label": label},
            )
            logger.debug("%s async forward failed: %s", label, exc)

    task.add_done_callback(_consume_result)


def record_degraded_event(
    subsystem: str,
    reason: str,
    *,
    detail: str = "",
    severity: str = "warning",
    classification: str = "background_degraded",
    context: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    now = time.time()
    subsystem = str(subsystem or "unknown")
    reason = str(reason or "unknown")
    severity = str(severity or "warning").lower()
    classification = str(classification or "background_degraded").lower()
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

    if severity != "info" and classification != "non_critical_fallback":
        _forward_to_terminal_monitor(dict(event))
    should_forward_to_error_intelligence = (
        severity in {"error", "critical"}
        or classification == "foreground_blocking"
    )
    if should_forward_to_error_intelligence:
        _forward_to_error_intelligence(key, dict(event), exc=exc)
    return dict(event)


def get_recent_degraded_events(limit: int = 20) -> list[dict[str, Any]]:
    with _LOCK:
        summaries = sorted(
            _SUMMARIES.values(),
            key=lambda item: float(item.get("last_seen", 0.0) or 0.0),
            reverse=True,
        )
        return [dict(item) for item in summaries[: max(0, int(limit))]]


def get_unified_failure_state(limit: int = 25) -> dict[str, Any]:
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
        "info": 0.0,
        "warning": 0.25,
        "error": 0.6,
        "critical": 1.0,
    }
    classification_weights = {
        "foreground_blocking": 1.0,
        "system_crash": 1.0,
        "background_degraded": 0.25,
        "non_critical_fallback": 0.0,
    }
    now = time.time()
    subsystems: dict[str, float] = {}
    weighted = 0.0
    critical = 0.0
    errors = 0.0
    warnings = 0.0

    for event in events:
        severity = str(event.get("severity", "warning") or "warning").lower()
        classification = str(event.get("classification", "background_degraded") or "background_degraded").lower()
        if severity not in severity_weights:
            continue
        classification_weight = classification_weights.get(classification, 0.5)
        if classification_weight <= 0.0 or severity_weights.get(severity, 0.0) <= 0.0:
            continue
        count = int(event.get("count", 1) or 1)
        last_seen = float(event.get("last_seen", event.get("timestamp", now)) or now)
        age_s = max(0.0, now - last_seen)
        if age_s > _FAILURE_EVENT_MAX_AGE_S:
            continue

        recency = 0.5 ** (age_s / _FAILURE_EVENT_HALF_LIFE_S)
        active_count = min(4.0, float(count)) * recency
        subsystem = str(event.get("subsystem", "unknown"))
        subsystems[subsystem] = subsystems.get(subsystem, 0.0) + active_count
        weighted += severity_weights.get(severity, 0.25) * active_count * classification_weight
        if severity == "critical" and classification in {"foreground_blocking", "system_crash"}:
            critical += active_count
        elif severity == "error":
            errors += active_count
        elif severity == "warning":
            warnings += active_count

    # Severity-weighted sum forms the base.  Critical events add a bonus to
    # ensure genuine critical failures still trigger lockdown quickly, but the
    # normalizer (8.0) prevents a handful of recurring warnings/errors from
    # spiraling into permanent lockdown.  The old formula (divider=5.0 with
    # double-counted critical+error terms) saturated to 1.0 from routine
    # transient errors, causing the "unified_failure_lockdown_1.00" spiral.
    pressure = min(1.0, (weighted + critical * 1.5) / 8.0)
    top_subsystems = sorted(subsystems.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "pressure": round(pressure, 4),
        "count": int(round(sum(subsystems.values()))),
        "critical": int(round(critical)),
        "errors": int(round(errors)),
        "warnings": int(round(warnings)),
        "top_subsystems": [
            {"subsystem": subsystem, "count": round(count, 3)}
            for subsystem, count in top_subsystems
        ],
    }


def clear_degraded_events() -> None:
    with _LOCK:
        _EVENTS.clear()
        _SUMMARIES.clear()
        _LAST_FORWARDED.clear()


def capture_degraded_events_state() -> dict[str, Any]:
    """Capture degraded-event pressure state for isolated proof probes."""

    with _LOCK:
        return {
            "events": [dict(event) for event in _EVENTS],
            "summaries": {
                key: dict(summary)
                for key, summary in _SUMMARIES.items()
            },
            "last_forwarded": dict(_LAST_FORWARDED),
        }


def restore_degraded_events_state(snapshot: dict[str, Any]) -> None:
    """Restore degraded-event pressure state captured by capture_degraded_events_state."""

    events = snapshot.get("events", [])
    summaries = snapshot.get("summaries", {})
    last_forwarded = snapshot.get("last_forwarded", {})
    with _LOCK:
        _EVENTS.clear()
        _EVENTS.extend(dict(event) for event in events)
        _SUMMARIES.clear()
        _SUMMARIES.update(
            {
                key: dict(summary)
                for key, summary in dict(summaries).items()
            }
        )
        _LAST_FORWARDED.clear()
        _LAST_FORWARDED.update(dict(last_forwarded))


@contextmanager
def isolated_degraded_event_scope(label: str = "isolated_probe") -> Iterator[dict[str, Any]]:
    """Run an expected negative-control probe without poisoning live failure pressure.

    This is for proof/evaluation probes that intentionally drive a baseline or
    ablated lane into failure. The caller must still record the probe result in
    its artifact bundle; this only prevents expected control failures from
    causing unrelated live-runtime lockdowns.
    """

    snapshot = capture_degraded_events_state()
    scope = {
        "label": str(label or "isolated_probe"),
        "started_at": time.time(),
        "restored": False,
    }
    try:
        yield scope
    finally:
        with _LOCK:
            scope["events_observed"] = max(0, len(_EVENTS) - len(snapshot.get("events", [])))
            scope["summaries_observed"] = max(0, len(_SUMMARIES) - len(snapshot.get("summaries", {})))
        restore_degraded_events_state(snapshot)
        scope["restored"] = True


def _forward_to_terminal_monitor(event: dict[str, Any]) -> None:
    try:
        from core.terminal_monitor import get_terminal_monitor

        monitor = get_terminal_monitor()
        if monitor and hasattr(monitor, "ingest_degraded_event"):
            monitor.ingest_degraded_event(event)
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_degraded_events_degradation(
            exc,
            action="retained degraded event locally after terminal monitor forward failed",
            extra={"subsystem": event.get("subsystem"), "reason": event.get("reason")},
        )
        logger.debug("Terminal monitor degraded event forward failed: %s", exc)


def _forward_to_error_intelligence(
    key: tuple[str, str, str, str],
    event: dict[str, Any],
    *,
    exc: BaseException | None = None,
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
        from core.runtime.service_registry import get_runtime_service

        orch = get_runtime_service("orchestrator", default=None)
        self_modifier = getattr(orch, "self_modifier", None) if orch else None
        if not self_modifier or not hasattr(self_modifier, "on_error"):
            return

        error = exc or RuntimeError(
            f"[{event['classification']}] {event['subsystem']}:{event['reason']} {event['detail']}".strip()
        )
        result = self_modifier.on_error(
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
        if inspect.isawaitable(result):
            _schedule_awaitable(result, label="degraded_event_forward")
    except (ImportError, AttributeError, RuntimeError) as forward_exc:
        _record_degraded_events_degradation(
            forward_exc,
            action="retained degraded event locally after error-intelligence forward failed",
            extra={"subsystem": event.get("subsystem"), "reason": event.get("reason")},
        )
        logger.debug("Error intelligence degraded event forward failed: %s", forward_exc)
