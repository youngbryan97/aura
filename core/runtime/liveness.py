"""Small runtime-service liveness registry.

This module is intentionally lightweight. It records proof that a live service
path is making progress, and forwards that proof to the stall watchdog when it
is active. Code that needs to prove a long-running visible operation is not
wedged can depend on this module without importing the watchdog thread class.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RuntimeServiceProgress:
    source: str
    updated_at: float
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["age_s"] = max(0.0, time.time() - self.updated_at)
        return payload


_LOCK = threading.RLock()
_PROGRESS: dict[str, RuntimeServiceProgress] = {}


def _normalize_source(source: str) -> str:
    normalized = str(source or "runtime").strip()
    return normalized[:160] or "runtime"


def mark_runtime_service_progress(source: str = "runtime") -> None:
    """Record progress for a runtime service path.

    The local registry is useful for status endpoints and tests. The delegated
    stall-watchdog mark is useful for live desktop runs where the browser/UI path
    is visibly moving while the event loop is under transient generation load.
    """

    normalized = _normalize_source(source)
    now = time.time()
    with _LOCK:
        prior = _PROGRESS.get(normalized)
        _PROGRESS[normalized] = RuntimeServiceProgress(
            source=normalized,
            updated_at=now,
            count=(int(prior.count) + 1 if prior is not None else 1),
        )

    try:
        from core.resilience.stall_watchdog import mark_runtime_service_progress as _mark_watchdog_progress

        _mark_watchdog_progress(normalized)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return


def get_runtime_service_progress(source_prefix: str | None = None) -> dict[str, Any]:
    """Return the latest progress record, optionally filtered by source prefix."""

    prefix = str(source_prefix or "").strip()
    with _LOCK:
        records = [
            item
            for source, item in _PROGRESS.items()
            if not prefix or source.startswith(prefix)
        ]
    if not records:
        return {"ok": False, "source": "", "updated_at": 0.0, "age_s": None, "count": 0, "matches": 0}
    latest = max(records, key=lambda item: item.updated_at)
    payload = latest.to_dict()
    payload.update({"ok": True, "matches": len(records)})
    return payload


def clear_runtime_service_progress() -> None:
    """Test helper: clear the in-process liveness registry."""

    with _LOCK:
        _PROGRESS.clear()
