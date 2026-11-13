"""A copy of provider metadata that nothing else holds a reference into, for the inference gate's evidence path."""
from __future__ import annotations

from typing import Any

__all__ = ["_deep_snapshot", "_deep_snapshot_inner"]


def _deep_snapshot(value: Any) -> Any:
    """A copy nothing else holds a reference into.

    ``copy.deepcopy`` on provider metadata can meet a live client handle, a
    lock or a coroutine — objects that either refuse to copy or are actively
    harmful to duplicate. So this copies the containers (which is where the
    aliasing lives) and leaves leaves alone, replacing anything uncopyable
    with its repr rather than raising inside an evidence path.
    """
    return _deep_snapshot_inner(value, depth=0, seen=set())


def _deep_snapshot_inner(value: Any, *, depth: int, seen: set[int]) -> Any:
    if depth > 12:
        return "<snapshot depth limit>"
    if isinstance(value, dict):
        if id(value) in seen:
            return "<cycle>"
        seen = seen | {id(value)}
        # list(...) FIRST. This walks live provider metadata while other
        # threads are still writing to it, and a Python-level comprehension
        # over .items() can be interrupted between elements — "dictionary
        # changed size during iteration", raised inside an evidence path, in
        # a subsystem on the fail-closed list, so it escalated to CRITICAL and
        # held the runtime DEGRADED across health pulses (live 2026-08-03
        # 22:13, repeating). The C-level copy cannot be interrupted.
        return {
            key: _deep_snapshot_inner(item, depth=depth + 1, seen=seen)
            for key, item in list(value.items())
        }
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            return "<cycle>"
        seen = seen | {id(value)}
        return [
            _deep_snapshot_inner(item, depth=depth + 1, seen=seen)
            for item in list(value)
        ]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_deep_snapshot_inner(item, depth=depth + 1, seen=seen) for item in value),
            key=repr,
        )
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    # A client handle, a lock, a coroutine — record what it was, not the
    # object itself.
    return repr(value)
