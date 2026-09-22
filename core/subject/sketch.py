"""An array as a fixed number of numbers.

The closure test reads everything outside the core this way, and the core reads
the two recurrent states it holds this way too, so one definition serves both:
the liquid substrate's activations and the mesh's column activations are not
things the core may describe with different numbers from the ones the test uses
to ask whether it described them fully.

Four moments and four projections. The projections are a seeded Gaussian, which
preserves distances in expectation, so two states that differ differ in the
sketch; the moments are there because a projection of a constant array is a
constant and the moments say it was one. The width is fixed, so a reservoir of
a thousand units and a vector of three cost the same, and the seed is frozen,
so two runs sketch the same directions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["SKETCH_FIELDS", "SKETCH_SEED", "SKETCH_WIDTH", "is_array", "sketch"]

#: How many projections an array is summarised into, beyond its four moments.
SKETCH_WIDTH: int = 4

#: The frozen seed the directions are drawn from.
SKETCH_SEED: int = 20250911

#: The names of what `sketch` returns, in the order a schema lists them.
SKETCH_FIELDS: tuple[str, ...] = ("mean", "sd", "min", "max", *(f"p{i}" for i in range(SKETCH_WIDTH)))

#: Directions drawn once per array length, since the draw is the same every time.
_DIRECTIONS: dict[int, np.ndarray] = {}


def is_array(value: Any) -> bool:
    """A numpy array, or something that will become one without side effects."""
    if isinstance(value, np.ndarray):
        return True
    # Torch and MLX tensors both answer to these and neither imports cleanly
    # here; going through the array protocol keeps this from knowing which.
    return hasattr(value, "shape") and hasattr(value, "dtype") and not callable(value) and not isinstance(value, type)


def _directions(size: int) -> np.ndarray:
    held = _DIRECTIONS.get(size)
    if held is None:
        rng = np.random.default_rng(SKETCH_SEED)
        held = rng.normal(size=(SKETCH_WIDTH, size)) / math.sqrt(size)
        _DIRECTIONS[size] = held
    return held


def sketch(value: Any) -> dict[str, float] | None:
    """The array's moments and projections, or None when it holds no finite number."""
    try:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
    # not a failure: the docstring above says it: None when it holds no finite number.
    except (TypeError, ValueError):
        return None
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    out = {
        "mean": float(flat.mean()),
        "sd": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }
    for index, column in enumerate(_directions(flat.size) @ flat):
        out[f"p{index}"] = float(column)
    return out
