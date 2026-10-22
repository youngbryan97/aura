"""Each large array's own leading axes, fixed before a run records, for the sketch to read along.

K reads each of her large arrays through four moments and four projections.
The projections were frozen random directions: two arrays that differ only
where none of the four looks read the same (a reviewer, 21 October 2026, two
weight matrices with one sketch and next states 1.72 apart), and a change her
dynamics make along a direction the four barely see reaches K as almost
nothing. The directions an array actually moves in are its leading principal
axes over her ordinary life. Taken from a calibration pass before recording
and then held fixed, they cost no column and change no arm: every fork reads
along the same axes, because the axes belong to the apparatus, which no fork
carries.

Optional, off unless a run asks for it (`--calibrated-axes`), and named in the
preregistration before a run that reads it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.subject.sketch import SKETCH_WIDTH

__all__ = [
    "axes_for", "calibrate", "calibration_pass", "clear", "described", "fitted", "load", "save", "sketched_sources",
]

_AXES: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DESCRIBED: dict[str, dict[str, Any]] = {}


def fitted(samples: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[float]] | None:
    """The mean and the leading axes of these readings of one array, and the share each axis holds."""
    rows = [np.asarray(item, dtype=np.float64).reshape(-1) for item in samples]
    rows = [row for row in rows if row.size and np.all(np.isfinite(row))]
    if len(rows) < SKETCH_WIDTH + 1 or len({row.size for row in rows}) != 1:
        return None
    matrix = np.stack(rows)
    centre = matrix.mean(axis=0)
    centred = matrix - centre
    _u, values, vectors = np.linalg.svd(centred, full_matrices=False)
    if values.size < SKETCH_WIDTH or values[SKETCH_WIDTH - 1] <= 0.0:
        return None
    total = float(np.sum(values**2))
    shares = [float(v**2 / total) for v in values[:SKETCH_WIDTH]] if total > 0 else []
    return centre, vectors[:SKETCH_WIDTH], shares


def calibrate(readings: Mapping[str, Sequence[np.ndarray]]) -> dict[str, dict[str, Any]]:
    """Fix the axes for every array these readings cover. Returns what was fixed, per source."""
    for source, samples in readings.items():
        result = fitted(samples)
        if result is None:
            _DESCRIBED[source] = {"calibrated": False, "readings": len(samples)}
            continue
        centre, vectors, shares = result
        _AXES[source] = (centre, vectors)
        _DESCRIBED[source] = {
            "calibrated": True,
            "readings": len(samples),
            "size": int(centre.size),
            "variance_held_by_each_axis": [round(share, 6) for share in shares],
        }
    return dict(_DESCRIBED)


def axes_for(source: str) -> tuple[np.ndarray, np.ndarray] | None:
    return _AXES.get(source)


def described() -> dict[str, dict[str, Any]]:
    return dict(_DESCRIBED)


def clear() -> None:
    _AXES.clear()
    _DESCRIBED.clear()


def save(path: Any) -> int:
    """Write the fixed axes, so another process reads K along the same ones. Returns how many."""
    import io

    from core.runtime.atomic_writer import atomic_write_bytes

    sources = sorted(_AXES)
    arrays: dict[str, np.ndarray] = {"sources": np.asarray(sources, dtype=str)}
    for index, source in enumerate(sources):
        centre, vectors = _AXES[source]
        arrays[f"centre_{index}"], arrays[f"vectors_{index}"] = centre, vectors
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    atomic_write_bytes(path, buffer.getvalue())
    return len(sources)


def load(path: Any) -> int:
    """Read axes another process fixed, in place of fitting them here. Returns how many."""
    clear()
    with np.load(path, allow_pickle=False) as held:
        for index, source in enumerate(held["sources"].tolist()):
            _AXES[str(source)] = (held[f"centre_{index}"], held[f"vectors_{index}"])
            _DESCRIBED[str(source)] = {"calibrated": True, "loaded_from": str(path)}
    return len(_AXES)


def sketched_sources() -> list[str]:
    """Every organ attribute K reads through the sketch, as `organ:<organ>.<attr>`."""
    from core.subject.state import DOMAINS, schema

    out: list[str] = []
    for domain in DOMAINS:
        block = schema(domain)
        for name, source in zip(block.features, block.sources, strict=True):
            if source.startswith("organ:") and name.endswith("_p0") and source not in out:
                out.append(source)
    return out


def _resolve(organs: Any, source: str) -> Any:
    """The array a source names; a reader method is called, as K's reader calls it."""
    organ_name, _, path = source[len("organ:"):].partition(".")
    held = getattr(organs, organ_name, None)
    for part in path.split("."):
        held = getattr(held, part, None) if held is not None else None
    if callable(held):
        try:
            held = held()
        except Exception:  # noqa: BLE001 - a reader that raises gives no axes, as K reads a miss
            return None
    return held


async def calibration_pass(runtime: Any, conditions: Sequence[Any], rounds: int = 1) -> dict[str, dict[str, Any]]:
    """Live `rounds` over the conditions, read every sketched array each frame, and fix their axes.

    An array whose readings would take more than a quarter of the memory free
    at the start keeps its random directions, and the description says so.
    """
    import psutil

    budget = psutil.virtual_memory().available / 4.0
    sources = sketched_sources()
    readings: dict[str, list[np.ndarray]] = {source: [] for source in sources}
    frames_expected = max(1, rounds * len(conditions) * 16)
    skipped: set[str] = set()

    def take(_frame: Any) -> None:
        for source in sources:
            if source in skipped:
                continue
            value = _resolve(runtime.organs, source)
            if not isinstance(value, np.ndarray) or value.size == 0:
                continue
            if value.size * 8 * frames_expected > budget:
                skipped.add(source)
                continue
            readings[source].append(np.array(value, dtype=np.float64, copy=True).reshape(-1))

    for _ in range(rounds):
        for condition in conditions:
            await runtime.turn_once(condition, on_frame=take)
    out = calibrate({source: samples for source, samples in readings.items() if source not in skipped})
    for source in skipped:
        _DESCRIBED[source] = {"calibrated": False, "why": "too large to hold a pass of readings"}
        out[source] = _DESCRIBED[source]
    return out
