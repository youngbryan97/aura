"""Shared helpers for the six fictional-AI engines.

The engines were one 2,200-line module. They share four small helpers and
nothing else — no state, no call graph — so they are six files now, and
this is what they actually had in common.
"""

from __future__ import annotations

import re
from typing import Any

from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

__all__ = [
    "WORD_TOKEN_RE",
    "engine_state_path",
    "save_engine_state",
    "as_float",
    "coerce_insight_text",
    "disk_percent_value",
    "record_fictional_degradation",
]

#: Word-boundary tokenizer for social cue matching. Substring matching made
#: single-letter and short cues fire on ordinary prose.
WORD_TOKEN_RE = re.compile(r"[a-z\']+")


def as_float(value: Any) -> float:
    """Best-effort float. NaN for anything that is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def disk_percent_value(reader: Any) -> float:
    """Call a disk-percent reader and return a float, never a surprise."""
    try:
        return float(reader())
    except (OSError, RuntimeError, TypeError, ValueError):
        return 0.0


def coerce_insight_text(result: Any) -> str:
    """Extract usable text from a brain.think() result of any supported shape."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("content", "text", "response", "answer", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    for attr in ("content", "text", "response", "answer"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def record_fictional_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "fictional_ai_synthesis",
        exc,
        severity=severity,
        action=action,
    )


def engine_state_path(explicit: Any, *default_parts: str) -> Path:
    """Resolve an engine's state file and make sure its directory exists.

    One owner for the three engines that keep a journal. Each had its own
    ``mkdir`` and its own ``atomic_write_text``, which is three effect
    call sites for one behaviour and three places for the behaviour to
    drift.
    """
    if explicit:
        path = Path(explicit)
    else:
        try:
            from core.config import config

            root = Path(config.paths.data_dir)
        except (ImportError, AttributeError):
            from core.runtime.state_ownership import state_root

            root = state_root() / "data"
        path = root.joinpath(*default_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_engine_state(path: Path, payload: dict, *, engine: str) -> bool:
    """Write one engine's journal atomically. Returns whether it landed."""
    import json

    try:
        atomic_write_text(path, json.dumps(payload, indent=2))
        return True
    except (OSError, TypeError, ValueError) as exc:
        record_fictional_degradation(
            exc,
            action=f"kept {engine} state in memory after its journal failed to save",
        )
        return False
