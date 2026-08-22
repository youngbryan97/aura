"""Bounded arithmetic for the phenomenal substrate.

A non-finite value used to pass straight through the bounds, and not
harmlessly. ``clamp`` was ``max(lo, min(hi, x))``, and Python's ``min``
returns its first argument whenever the comparison is False — so
``min(1.0, nan)`` is ``1.0`` and ``clamp(nan)`` was ``1.0``. A NaN arriving
from anywhere upstream did not surface as a NaN; it surfaced as full
intensity, on whichever channel it landed on. ``clamp_signed(nan)`` was
``1.0`` for the same reason.

Every array path in this repository already guards for this —
``core/memory/sqlite_vector_store.py`` and ``core/learning/rl_glue.py`` both
call ``np.nan_to_num(..., nan=0.0)``. The substrate is scalar and dict-shaped
so it never adopted the same guard, which is how one convention ended up
holding on the numpy paths and not on the one that decides how something
feels.

``nan=0.0`` matches those callers: a NaN carries no information, and the
neutral element for both an intensity in [0, 1] and a signed value in
[-1, 1] is zero. It is also recorded, because a NaN reaching here means a
computation upstream produced one and somebody should see that rather than a
saturated channel.

An infinity is treated differently from a NaN, because it is different. It
carries a direction, so inside a bound it should saturate at that bound —
``clamp(inf)`` is 1.0 and ``clamp(-inf)`` is 0.0, which is the answer the
caller wanted. Only where there is no bound to saturate against, in the sums
and the blends, is an infinity as useless as a NaN and replaced by zero.
"""
from __future__ import annotations

from math import exp, isfinite, isnan
from math import tanh as _tanh

Vector = dict[str, float]


def _as_float(x: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def without_nan(x: float, default: float = 0.0, *, field: str = "") -> float:
    """``x`` unless it is a NaN, which carries nothing.

    Infinities pass through: the caller is about to bound them, and saturating
    at the bound is the answer an infinity means.
    """
    value = _as_float(x)
    if not isnan(value):
        return value
    _record_non_finite(value, field)
    return default


def finite(x: float, default: float = 0.0, *, field: str = "") -> float:
    """``x`` when it is a real number, ``default`` when it is not.

    For the unbounded operations — sums, blends, prediction errors — where
    there is no bound for an infinity to saturate against and it would poison
    every term it touches.

    Records a degradation rather than substituting quietly: the substitution
    keeps the substrate running, and the record is what lets somebody find the
    computation that produced it.
    """
    value = _as_float(x)
    if isfinite(value):
        return value
    _record_non_finite(value, field)
    return default


def _record_non_finite(value: float, field: str) -> None:
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "phenomenal_substrate.maths",
            ValueError(f"non-finite substrate value{f' for {field}' if field else ''}: {value!r}"),
            severity="warning",
            action="substituted the neutral value so the substrate kept running",
            extra={"field": field or "unknown"},
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        # The substrate must not fail because reporting failed.
        pass


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, without_nan(x, lo)))


def clamp_signed(x: float) -> float:
    return max(-1.0, min(1.0, without_nan(x, 0.0)))


def sigmoid(x: float, gain: float = 1.0, bias: float = 0.0) -> float:
    z = max(-60.0, min(60.0, finite(gain, 1.0) * (finite(x, 0.0) - finite(bias, 0.0))))
    return 1.0 / (1.0 + exp(-z))


def tanh(x: float) -> float:
    return _tanh(max(-20.0, min(20.0, finite(x, 0.0))))


def l1(v: Vector) -> float:
    return sum(abs(finite(x)) for x in v.values())


def l2(v: Vector) -> float:
    return sum(finite(x) * finite(x) for x in v.values()) ** 0.5


def add(a: Vector, b: Vector) -> Vector:
    keys = set(a) | set(b)
    return {k: finite(a.get(k, 0.0), field=k) + finite(b.get(k, 0.0), field=k) for k in keys}


def sub(a: Vector, b: Vector) -> Vector:
    keys = set(a) | set(b)
    return {k: finite(a.get(k, 0.0), field=k) - finite(b.get(k, 0.0), field=k) for k in keys}


def mul(a: Vector, scalar: float) -> Vector:
    factor = finite(scalar, 0.0)
    return {k: finite(v, field=k) * factor for k, v in a.items()}


def mix(old: Vector, new: Vector, rate: float) -> Vector:
    keys = set(old) | set(new)
    blend = clamp(rate, 0.0, 1.0)
    return {
        k: finite(old.get(k, 0.0), field=k) * (1 - blend)
        + finite(new.get(k, 0.0), field=k) * blend
        for k in keys
    }


def bound01(v: Vector) -> Vector:
    return {k: clamp(x, 0.0, 1.0) for k, x in v.items()}


def bound_signed(v: Vector) -> Vector:
    return {k: clamp_signed(x) for k, x in v.items()}


def weighted_error(predicted: Vector, observed: Vector, precision: Vector) -> Vector:
    keys = set(predicted) | set(observed) | set(precision)
    return {
        k: (finite(observed.get(k, 0.0), field=k) - finite(predicted.get(k, 0.0), field=k))
        * finite(precision.get(k, 1.0), 1.0, field=k)
        for k in keys
    }


def normalize_sum(v: Vector) -> Vector:
    cleaned = {k: max(0.0, finite(x, field=k)) for k, x in v.items()}
    total = sum(cleaned.values())
    if total <= 1e-9:
        n = len(cleaned) or 1
        return {k: 1.0 / n for k in cleaned}
    return {k: x / total for k, x in cleaned.items()}
