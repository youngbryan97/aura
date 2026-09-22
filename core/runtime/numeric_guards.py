"""One correct way to accept a number from outside.

CP126 raised the same finding against seven modules independently — belief
confidence, global stability pressure, scheduler intervals, telemetry
bounds, inquiry scores, execution timeouts, fine-tune quality — all of the
form "accepts non-finite and out-of-range values". They were separate
findings because they are separate call sites, but they are one defect.

The codebase already carries a dozen near-identical private helpers
(``_safe_float``, ``_finite_float``, ``_clamp01``, ``_finite``, ``_clamp``),
each subtly different and none of them shared. That is why the same hole
kept being rediscovered: fixing one taught the others nothing.

The subtlety that makes ad-hoc clamping wrong is NaN. Every comparison with
NaN is False, so the usual idiom does not clamp it — it propagates it, and
which value survives depends on argument order::

    max(float("nan"), 0.0)   # nan
    max(0.0, float("nan"))   # 0.0
    min(max(nan, 0.0), 1.0)  # nan  → passed on as a score

``core/volition.py`` had exactly this: a NaN inquiry score reached the
priority comparison, and because every comparison against it is False it
sorted as though it were the highest priority available. A corrupt number
does not merely produce a wrong answer there, it wins.

So: reject non-finite input explicitly rather than relying on comparison,
and make the caller state a default. Nothing here guesses.
"""
from __future__ import annotations

import math
from typing import Any

__all__ = [
    "bounded_float",
    "bounded_int",
    "is_finite_number",
    "positive_float",
    "unit_float",
]


def is_finite_number(value: Any) -> bool:
    """True only for a real, finite number.

    Booleans are excluded on purpose: ``True`` is arithmetically 1 and
    almost never what a caller passing a "score" or "interval" meant, so
    silently accepting it hides a type error at the boundary.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce to a finite float inside [minimum, maximum], or use ``default``.

    ``default`` is required, and is itself clamped — a caller that supplies
    an out-of-range default gets the bound rather than a value the contract
    said was impossible.

    Non-finite input (NaN, ±inf), non-numeric input, and anything that fails
    to coerce all resolve to ``default``. They are NOT clamped, because
    clamping NaN is what produced the original defect.
    """
    if isinstance(value, bool):
        # Enforced here as well as in is_finite_number: True is
        # arithmetically 1 and almost never what a caller passing a score or
        # an interval meant. Coercing it would hide a type error at exactly
        # the boundary this function exists to guard.
        return _clamp(default, minimum, maximum)
    if not is_finite_number(value):
        try:
            candidate = float(value)  # strings like "0.5" are still accepted
        # not a failure: a value that will not become a number falls to the
        # default, which is what the docstring above describes.
        except (TypeError, ValueError):
            candidate = None
        if candidate is None or not math.isfinite(candidate):
            return _clamp(default, minimum, maximum)
        value = candidate
    return _clamp(float(value), minimum, maximum)


def unit_float(value: Any, *, default: float = 0.0) -> float:
    """A confidence, probability or score in [0, 1]."""
    return bounded_float(value, default=default, minimum=0.0, maximum=1.0)


def positive_float(
    value: Any,
    *,
    default: float,
    minimum: float = 1e-9,
    maximum: float | None = None,
) -> float:
    """A duration, interval or timeout that must be strictly positive.

    Zero and negatives are REJECTED to the default rather than clamped up to
    ``minimum``. Clamping looks safer and is not: a zero interval clamped to
    1e-9 is still a busy loop, and a zero timeout clamped to 1e-9 still means
    "already expired". Both are almost always a missing config, and the
    caller's stated default is a better answer than an epsilon.
    """
    resolved = bounded_float(value, default=default, minimum=None, maximum=maximum)
    if resolved <= 0.0:
        resolved = bounded_float(default, default=minimum, minimum=None, maximum=maximum)
    return max(minimum, resolved)


def bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce to an int inside [minimum, maximum], or use ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        resolved = default
    else:
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            resolved = default
        else:
            resolved = int(as_float) if math.isfinite(as_float) else default
    if minimum is not None:
        resolved = max(minimum, resolved)
    if maximum is not None:
        resolved = min(maximum, resolved)
    return resolved


def _clamp(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value
