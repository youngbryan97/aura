"""Where two snapshots of her differ, path by path, largest first.

Bryan, 21 October 2026: continuity while changing, with the chain written out;
change local to what was experienced. Two forks that lived the same turns, one
of which met something the other did not, differ exactly where that meeting
reached. Walking both snapshots and naming every number that differs, with how
far it moved against its own size, is that reach written out.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["diff_snapshots"]

#: How deep the walk goes into nested state before it stops naming paths.
_DEPTH = 8


def _walk(a: Any, b: Any, path: str, out: list[tuple[str, float]], counted: list[int], depth: int) -> None:
    if depth > _DEPTH:
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type) and type(a) is type(b):
        for item in dataclasses.fields(a):
            _walk(getattr(a, item.name, None), getattr(b, item.name, None), f"{path}.{item.name}", out, counted, depth + 1)
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b), key=str):
            _walk(a.get(key), b.get(key), f"{path}.{key}", out, counted, depth + 1)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and len(a) <= 64:
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            _walk(left, right, f"{path}[{index}]", out, counted, depth + 1)
        return
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.number):
            return
        counted[0] += 1
        gap = float(np.linalg.norm(b.astype(np.float64) - a.astype(np.float64)))
        if gap > 0.0:
            size = float(np.linalg.norm(a.astype(np.float64)))
            out.append((path, gap / size if size > 1e-12 else gap))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isfinite(a) and math.isfinite(b)):
            return
        counted[0] += 1
        if a != b:
            out.append((path, abs(b - a) / abs(a) if abs(a) > 1e-12 else abs(b - a)))


def diff_snapshots(a: Any, b: Any, *, top: int = 40) -> dict[str, Any]:
    """The numbers that differ between two snapshots, each as a change relative to its own size."""
    out: list[tuple[str, float]] = []
    counted = [0]
    _walk(a, b, "", out, counted, 0)
    out.sort(key=lambda pair: -pair[1])
    return {
        "numbers_compared": counted[0],
        "numbers_changed": len(out),
        "share_changed": round(len(out) / counted[0], 6) if counted[0] else 0.0,
        "largest": [{"path": path.lstrip("."), "relative_change": round(value, 6)} for path, value in out[:top]],
    }
