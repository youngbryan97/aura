"""Canonical validated-scalar primitives.

CP126 turned up the same defect in ~260 files: a caller-supplied or
model-supplied number is compared against a threshold without ever being
checked, so a NaN makes *every* comparison false and the code falls through to
whichever branch is cheapest — usually the permissive one. A closely related
shape clamps the value for display but compares the raw one.

Two rules make the class go away:

1. **Validate before you compare.** ``NaN >= 0.75`` is False and so is
   ``NaN < 0.75``; there is no threshold that catches it. Validation has to
   happen before the number reaches a comparison, not inside the formatter.
2. **Fail toward caution, and say so.** An unusable risk signal is not a low
   risk signal. Every helper here returns the repaired value *and* a fault
   string, so callers can escalate, log, and receipt the repair instead of
   silently proceeding on a number nobody produced.

These are deliberately dependency-free so `core/runtime` may import them.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "ValidatedScalar",
    "validated_scalar",
    "validated_unit",
    "validated_probability",
    "validated_positive",
    "validated_int",
    "clamp",
    "is_usable",
    "safe_ratio",
    "safe_mean",
    "all_faults",
]


class ValidatedScalar(float):
    """A float that remembers whether it had to be repaired.

    It *is* a float, so it drops into arithmetic and comparisons unchanged;
    ``.fault`` is empty when the input was already usable.
    """

    __slots__ = ("fault", "original")

    def __new__(cls, value: float, fault: str = "", original: Any = None):
        instance = super().__new__(cls, value)
        instance.fault = fault
        instance.original = original
        return instance

    @property
    def repaired(self) -> bool:
        return bool(self.fault)


def is_usable(value: Any) -> bool:
    """Whether ``value`` is a finite number safe to compare."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Plain clamp for values already known to be finite."""
    return max(low, min(high, float(value)))


def validated_scalar(
    value: Any,
    *,
    name: str = "value",
    low: float | None = None,
    high: float | None = None,
    default: float = 0.0,
    on_unusable: float | None = None,
) -> ValidatedScalar:
    """A finite number inside [low, high], with the repair recorded.

    ``on_unusable`` is the value substituted when the input is non-numeric or
    non-finite. Choose it by what is CAUTIOUS for the caller: for a risk or
    uncertainty signal that is the maximum; for a confidence or a reward it is
    the minimum. It defaults to ``default``.
    """
    fallback = default if on_unusable is None else on_unusable
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ValidatedScalar(
            fallback, f"{name} was not numeric ({value!r}); using {fallback}", value
        )
    if math.isnan(number):
        return ValidatedScalar(fallback, f"{name} was NaN; using {fallback}", value)
    if math.isinf(number):
        return ValidatedScalar(fallback, f"{name} was infinite; using {fallback}", value)
    if low is not None and number < low:
        return ValidatedScalar(low, f"{name} was below range ({number}); clamped to {low}", value)
    if high is not None and number > high:
        return ValidatedScalar(high, f"{name} was above range ({number}); clamped to {high}", value)
    return ValidatedScalar(number, "", value)


def validated_unit(value: Any, *, name: str = "value", cautious_high: bool = False) -> ValidatedScalar:
    """A [0, 1] scalar.

    ``cautious_high=True`` treats an unusable input as 1.0 — the right default
    for risk, uncertainty, novelty, cost and anything else where "unknown"
    must not read as "safe".
    """
    return validated_scalar(
        value, name=name, low=0.0, high=1.0,
        default=0.0, on_unusable=1.0 if cautious_high else 0.0,
    )


def validated_probability(value: Any, *, name: str = "probability") -> ValidatedScalar:
    """A [0, 1] probability; an unusable input becomes 0.0 (no evidence)."""
    return validated_scalar(value, name=name, low=0.0, high=1.0, default=0.0)


def validated_positive(
    value: Any, *, name: str = "value", default: float = 1.0, high: float | None = None
) -> ValidatedScalar:
    """A strictly positive number — timeouts, budgets, rates, divisors."""
    result = validated_scalar(value, name=name, low=None, high=high, default=default)
    if result.fault:
        return result
    if result <= 0:
        return ValidatedScalar(
            default, f"{name} must be positive ({float(result)}); using {default}", value
        )
    return result


def validated_int(
    value: Any,
    *,
    name: str = "value",
    low: int | None = None,
    high: int | None = None,
    default: int = 0,
) -> tuple[int, str]:
    """An int inside [low, high] plus a fault string."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return default, f"{name} was not an integer ({value!r}); using {default}"
        if not math.isfinite(as_float):
            return default, f"{name} was not finite ({value!r}); using {default}"
        number = int(as_float)
    if low is not None and number < low:
        return low, f"{name} was below range ({number}); clamped to {low}"
    if high is not None and number > high:
        return high, f"{name} was above range ({number}); clamped to {high}"
    return number, ""


def safe_ratio(
    numerator: Any, denominator: Any, *, default: float = 0.0, epsilon: float = 1e-12
) -> ValidatedScalar:
    """``numerator / denominator`` without a ZeroDivisionError or a NaN."""
    top = validated_scalar(numerator, name="numerator", default=0.0)
    bottom = validated_scalar(denominator, name="denominator", default=0.0)
    if top.fault or bottom.fault:
        return ValidatedScalar(default, top.fault or bottom.fault, (numerator, denominator))
    if abs(float(bottom)) < epsilon:
        return ValidatedScalar(
            default, f"denominator was ~zero ({float(bottom)}); using {default}", denominator
        )
    return ValidatedScalar(float(top) / float(bottom), "", None)


def safe_mean(values: Iterable[Any], *, default: float = 0.0) -> ValidatedScalar:
    """Mean over only the usable values.

    An empty or wholly-unusable input returns ``default`` WITH a fault, so a
    caller can tell "measured zero" from "measured nothing" — the distinction
    CP126 flagged repeatedly as unavailable state reported as a real reading.
    """
    usable = [float(item) for item in (values or ()) if is_usable(item)]
    total = len(list(values or ()))
    if not usable:
        return ValidatedScalar(default, f"no usable values among {total}", None)
    mean = sum(usable) / len(usable)
    fault = "" if len(usable) == total else f"{total - len(usable)} of {total} values were unusable"
    return ValidatedScalar(mean, fault, None)


def all_faults(*scalars: ValidatedScalar | tuple[Any, str]) -> tuple[str, ...]:
    """Collect the non-empty faults from a batch of validations."""
    faults: list[str] = []
    for item in scalars:
        if isinstance(item, ValidatedScalar):
            if item.fault:
                faults.append(item.fault)
        elif isinstance(item, tuple) and len(item) == 2 and item[1]:
            faults.append(str(item[1]))
    return tuple(faults)


def validated_sequence(
    values: Sequence[Any], *, name: str = "values", low: float = 0.0, high: float = 1.0
) -> tuple[list[float], tuple[str, ...]]:
    """Validate a whole vector, returning the repaired values and the faults."""
    repaired: list[float] = []
    faults: list[str] = []
    for index, value in enumerate(values or ()):
        scalar = validated_scalar(value, name=f"{name}[{index}]", low=low, high=high)
        repaired.append(float(scalar))
        if scalar.fault:
            faults.append(scalar.fault)
    return repaired, tuple(faults)
