"""Process-wide foreground conversation guard.

This is deliberately lower-level than the inference gate.  A live chat turn
needs protection as soon as the user speaks, including the time spent waiting
for locks, warmup, context assembly, or recovery.  Background loops can query
this module without needing access to the chat route or orchestrator instance.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
import uuid
from typing import Any, Optional


_LOCK = threading.RLock()
_ACTIVE: dict[str, dict[str, Any]] = {}
_QUIET_UNTIL = 0.0
_LAST_USER_AT = 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def default_quiet_seconds() -> float:
    return max(0.0, _env_float("AURA_FOREGROUND_QUIET_WINDOW_S", 120.0))


def _now() -> float:
    return time.time()


def notify_user_spoke(message: str = "", *, quiet_seconds: Optional[float] = None) -> None:
    """Mark foreground social pressure before a model request begins."""
    del message  # Reserved for future per-message policies.
    quiet_s = default_quiet_seconds() if quiet_seconds is None else max(0.0, float(quiet_seconds))
    now = _now()
    with _LOCK:
        global _LAST_USER_AT, _QUIET_UNTIL
        _LAST_USER_AT = now
        _QUIET_UNTIL = max(_QUIET_UNTIL, now + quiet_s)


@dataclass
class ForegroundLease:
    token: str
    quiet_seconds: float
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        end_foreground_turn(self.token, quiet_seconds=self.quiet_seconds)

    def __enter__(self) -> "ForegroundLease":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


def begin_foreground_turn(
    *,
    owner: str = "chat_api",
    source: str = "chat_api",
    quiet_seconds: Optional[float] = None,
) -> ForegroundLease:
    """Acquire a foreground conversation lease for the current process."""
    quiet_s = default_quiet_seconds() if quiet_seconds is None else max(0.0, float(quiet_seconds))
    now = _now()
    token = uuid.uuid4().hex
    with _LOCK:
        global _LAST_USER_AT, _QUIET_UNTIL
        _ACTIVE[token] = {
            "owner": str(owner or "chat_api"),
            "source": str(source or "chat_api"),
            "started_at": now,
        }
        _LAST_USER_AT = now
        _QUIET_UNTIL = max(_QUIET_UNTIL, now + quiet_s)
    return ForegroundLease(token=token, quiet_seconds=quiet_s)


def end_foreground_turn(token: str, *, quiet_seconds: Optional[float] = None) -> None:
    quiet_s = default_quiet_seconds() if quiet_seconds is None else max(0.0, float(quiet_seconds))
    now = _now()
    with _LOCK:
        global _QUIET_UNTIL
        _ACTIVE.pop(str(token or ""), None)
        _QUIET_UNTIL = max(_QUIET_UNTIL, now + quiet_s)


def foreground_activity_reason() -> str:
    now = _now()
    with _LOCK:
        if _ACTIVE:
            return "foreground_chat_active"
        if _QUIET_UNTIL > now:
            return "foreground_quiet_window"
    return ""


def snapshot() -> dict[str, Any]:
    now = _now()
    with _LOCK:
        active_items = list(_ACTIVE.values())
        newest = max(active_items, key=lambda item: float(item.get("started_at", 0.0) or 0.0), default={})
        started_at = float(newest.get("started_at", 0.0) or 0.0)
        return {
            "active": bool(active_items),
            "active_count": len(active_items),
            "owner": newest.get("owner", ""),
            "source": newest.get("source", ""),
            "active_age_s": round(max(0.0, now - started_at), 2) if started_at > 0.0 else 0.0,
            "quiet_remaining_s": round(max(0.0, _QUIET_UNTIL - now), 2),
            "last_user_age_s": round(max(0.0, now - _LAST_USER_AT), 2) if _LAST_USER_AT > 0.0 else None,
            "reason": foreground_activity_reason(),
        }


def _reset_for_tests() -> None:
    with _LOCK:
        global _QUIET_UNTIL, _LAST_USER_AT
        _ACTIVE.clear()
        _QUIET_UNTIL = 0.0
        _LAST_USER_AT = 0.0
