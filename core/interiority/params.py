"""core/interiority/params.py — every number in this package carries its reason.

A model of a feeling is mostly coefficients, and a coefficient with no
stated origin is an opinion wearing a decimal point. Five of the six
outside prototypes this package was written against fail the same way:
``sanctity_prior = 10.0``, ``resolvability = 0.85``, ``agency_weight =
0.9``. Each looks like physics and is a preference, and two of them make
their mechanism a constant — Gemini's conscientious objection returns
True for every input in range, so the function has no causal dependence
on the thing it claims to weigh.

So a parameter here is a declared object with four required parts:

* ``value``, ``unit`` and bounds, so a reader knows what it is;
* ``basis``, a citation or a derivation, which may not be empty;
* ``kind``, saying what sort of claim the basis is;
* ``sensitivity``, saying what changes if the value moves.

:data:`ParamKind` is the honest part. ``CITED`` means a published
measurement says this number. ``DERIVED`` means it follows from another
number here. ``MEASURED`` means this runtime measured it. ``CALIBRATION``
means nobody has measured it and the value is a guess — and a guess is
allowed, on one condition: the faculty's *ordering* of outcomes must
survive the parameter moving across its whole plausible range.
:func:`sweep` generates that range and
``tests/interiority/test_parameter_discipline.py`` enforces the rule.

The registry is global and append-only within a process. ``make
interiority-params`` walks it and fails on an empty basis, on a value
outside its own bounds, and on a calibration parameter with no sweep
range.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterator


class ParamKind(StrEnum):
    """What sort of claim the basis makes."""

    #: A published measurement reports this value.
    CITED = "cited"
    #: Follows arithmetically from other declared parameters or from a
    #: definition, and moves when they move.
    DERIVED = "derived"
    #: This runtime measured it, and it is refreshed from live state.
    MEASURED = "measured"
    #: Nobody has measured it. The value is a guess and the faculty's
    #: ordering must be invariant to it across :attr:`Param.sweep_range`.
    CALIBRATION = "calibration"


class ParameterError(ValueError):
    """A parameter declaration that would let an unjustified number through."""


@dataclass(frozen=True)
class Param:
    """One declared number, with the reason it holds that value."""

    name: str
    value: float
    unit: str
    basis: str
    kind: ParamKind
    sensitivity: str
    lower: float = 0.0
    upper: float = 1.0
    owner: str = ""
    #: Plausible range for a calibration parameter, as (low, high).
    sweep_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ParameterError("a parameter needs a name")
        if not self.basis.strip():
            raise ParameterError(
                f"{self.name}: basis is empty. A number with no stated origin is "
                "an opinion; cite it, derive it, measure it, or declare it "
                "CALIBRATION with a sweep range"
            )
        if not self.sensitivity.strip():
            raise ParameterError(
                f"{self.name}: sensitivity is empty. Say what changes when this "
                "value moves, or nobody can tell whether it matters"
            )
        if not math.isfinite(self.value):
            raise ParameterError(f"{self.name}: value is not finite")
        if not self.lower <= self.value <= self.upper:
            raise ParameterError(
                f"{self.name}: value {self.value} is outside its own bounds "
                f"[{self.lower}, {self.upper}]"
            )
        if self.kind is ParamKind.CALIBRATION and self.sweep_range is None:
            raise ParameterError(
                f"{self.name}: a calibration parameter must declare the range it "
                "could plausibly take, so a test can show the faculty's ordering "
                "survives moving it"
            )
        if self.sweep_range is not None:
            low, high = self.sweep_range
            if not (self.lower <= low < high <= self.upper):
                raise ParameterError(
                    f"{self.name}: sweep range {self.sweep_range} does not sit "
                    f"inside the bounds [{self.lower}, {self.upper}]"
                )
            if not low <= self.value <= high:
                raise ParameterError(
                    f"{self.name}: value {self.value} is outside its own sweep "
                    "range, so the declared range is not the range in use"
                )

    def sweep(self, steps: int = 7) -> tuple[float, ...]:
        """Values across the plausible range, for an ordering-invariance test."""
        if self.sweep_range is None:
            return (self.value,)
        low, high = self.sweep_range
        if steps < 2:
            return (low, high)
        span = high - low
        return tuple(low + span * i / (steps - 1) for i in range(steps))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "basis": self.basis,
            "kind": str(self.kind),
            "sensitivity": self.sensitivity,
            "bounds": [self.lower, self.upper],
            "owner": self.owner,
            "sweep_range": list(self.sweep_range) if self.sweep_range else None,
        }


class ParamRegistry:
    """Every parameter declared anywhere in the package."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._params: dict[str, Param] = {}

    def declare(self, param: Param) -> Param:
        with self._lock:
            existing = self._params.get(param.name)
            if existing is not None and existing != param:
                raise ParameterError(
                    f"{param.name} is already declared with a different value or "
                    "basis. Parameter names are a contract; rename rather than "
                    "redefine"
                )
            self._params[param.name] = param
        return param

    def get(self, name: str) -> Param | None:
        with self._lock:
            return self._params.get(name)

    def all(self) -> tuple[Param, ...]:
        with self._lock:
            return tuple(sorted(self._params.values(), key=lambda p: p.name))

    def for_owner(self, owner: str) -> tuple[Param, ...]:
        return tuple(p for p in self.all() if p.owner == owner)

    def calibration(self) -> tuple[Param, ...]:
        return tuple(p for p in self.all() if p.kind is ParamKind.CALIBRATION)

    def __iter__(self) -> Iterator[Param]:
        return iter(self.all())

    def __len__(self) -> int:
        with self._lock:
            return len(self._params)

    def clear_for_test(self) -> None:
        with self._lock:
            self._params.clear()


_REGISTRY = ParamRegistry()


def registry() -> ParamRegistry:
    return _REGISTRY


def declare(
    name: str,
    value: float,
    *,
    unit: str,
    basis: str,
    kind: ParamKind,
    sensitivity: str,
    lower: float = 0.0,
    upper: float = 1.0,
    owner: str = "",
    sweep_range: tuple[float, float] | None = None,
) -> Param:
    """Declare a parameter and register it. Returns the parameter."""
    return _REGISTRY.declare(
        Param(
            name=name,
            value=value,
            unit=unit,
            basis=basis,
            kind=kind,
            sensitivity=sensitivity,
            lower=lower,
            upper=upper,
            owner=owner,
            sweep_range=sweep_range,
        )
    )


def cited(name: str, value: float, *, unit: str, basis: str, sensitivity: str,
          lower: float = 0.0, upper: float = 1.0, owner: str = "") -> Param:
    return declare(name, value, unit=unit, basis=basis, kind=ParamKind.CITED,
                   sensitivity=sensitivity, lower=lower, upper=upper, owner=owner)


def derived(name: str, value: float, *, unit: str, basis: str, sensitivity: str,
            lower: float = 0.0, upper: float = 1.0, owner: str = "") -> Param:
    return declare(name, value, unit=unit, basis=basis, kind=ParamKind.DERIVED,
                   sensitivity=sensitivity, lower=lower, upper=upper, owner=owner)


def calibration(name: str, value: float, *, unit: str, basis: str, sensitivity: str,
                sweep_range: tuple[float, float], lower: float = 0.0,
                upper: float = 1.0, owner: str = "") -> Param:
    return declare(name, value, unit=unit, basis=basis, kind=ParamKind.CALIBRATION,
                   sensitivity=sensitivity, lower=lower, upper=upper, owner=owner,
                   sweep_range=sweep_range)


__all__ = [
    "Param",
    "ParamKind",
    "ParamRegistry",
    "ParameterError",
    "calibration",
    "cited",
    "declare",
    "derived",
    "registry",
]
