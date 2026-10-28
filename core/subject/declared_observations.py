"""Budgeted, explicit array reads for prospective closure recordings.

Only state already declared held may be recorded here. Exact mode preserves
the complete array. Read-basis mode preserves the supplied linear reads; it
does not assert that those are every read the runtime can make. Missing values,
shape drift, nonfinite coordinates and exhausted budgets raise, never become
zeros. The ordinary recorder is unchanged unless a plan is explicitly supplied.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.observation_basis import sufficient_basis


class ObservationError(ValueError):
    """The declared measurement could not be made completely."""


@dataclass(frozen=True)
class ArrayObservation:
    unit: str
    path: tuple[str | int, ...]
    shape: tuple[int, ...]
    mode: str
    axes: np.ndarray | None
    reconstruction_error: float | None

    @property
    def width(self) -> int:
        return math.prod(self.shape) if self.axes is None else len(self.axes)

    def read(self, units: Mapping[str, Any]) -> np.ndarray:
        value = units.get(self.unit)
        if value is None:
            raise ObservationError(f"missing held unit: {self.unit}")
        foreign = {id(item) for key, item in units.items() if key != self.unit and item is not None}
        for part in self.path:
            if isinstance(value, Mapping):
                value = value.get(part)
            elif isinstance(part, int) and isinstance(value, (list, tuple)):
                value = value[part] if 0 <= part < len(value) else None
            elif isinstance(part, str) and hasattr(value, "__dict__"):
                # Stored fields only: do not execute a property or reader method.
                value = vars(value).get(part)
            else:
                value = None
            if value is None:
                raise ObservationError(f"missing array path: {self.unit}:{self.path}")
            if id(value) in foreign:
                raise ObservationError("an observation path crossed into a different unit")
        if not isinstance(value, np.ndarray) or value.shape != self.shape or value.dtype.kind not in "biuf":
            raise ObservationError(f"array shape/type changed: {self.unit}:{self.path}, expected {self.shape}")
        if ((value.dtype.kind in "iu" and (int(value.min()) < -(2**53) or int(value.max()) > 2**53))
                or (value.dtype.kind == "f" and value.dtype.itemsize > 8)):
            raise ObservationError("the float64 recording cannot preserve this array's numeric precision")
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat).all():
            raise ObservationError(f"nonfinite coordinate: {self.unit}:{self.path}")
        observed = flat.copy() if self.axes is None else self.axes @ flat
        if not np.isfinite(observed).all():
            raise ObservationError(f"nonfinite declared read: {self.unit}:{self.path}")
        return observed


class ObservationPlan:
    def __init__(self, declaration: Mapping[str, Any]):
        from dataclasses import fields

        from core.subject.carrier_reads import ORGANS_WHOLE, held_attributes_of, held_whole
        from core.subject.state import Organs

        if declaration.get("version") != 1:
            raise ObservationError("observation plan version must be 1")
        self.max_bytes = declaration.get("max_bytes")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ObservationError("max_bytes must be a positive integer")
        entries = declaration.get("arrays")
        if not isinstance(entries, list) or not entries:
            raise ObservationError("the plan must declare at least one array")
        organs = {f"organ:{field.name}" for field in fields(Organs)} if ORGANS_WHOLE else set()
        self.arrays: list[ArrayObservation] = []
        seen: set[tuple[str, tuple[str | int, ...]]] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ObservationError("an array declaration must be an object")
            unit, path, shape = entry.get("unit"), entry.get("path"), entry.get("shape")
            if (not isinstance(unit, str) or not isinstance(path, list) or not path
                    or any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in path)
                    or not isinstance(shape, list) or not shape
                    or any(type(size) is not int or size <= 0 for size in shape)):
                raise ObservationError("declare a unit, stored-field path, and positive array shape")
            if unit not in organs and not held_whole(unit) and path[0] not in held_attributes_of(unit):
                raise ObservationError(f"the observation is outside the declared carrier: {unit}:{path[0]}")
            key = (unit, tuple(path))
            if key in seen:
                raise ObservationError("duplicate array declaration")
            seen.add(key)
            mode = entry.get("mode")
            axes, error = None, None
            size = math.prod(shape)
            if mode == "read_basis":
                readers = np.asarray(entry.get("readers"), dtype=float)
                if readers.ndim != 2 or readers.shape[1] != size or not readers.shape[0]:
                    raise ObservationError("read_basis requires a nonempty read matrix matching the array size")
                basis = sufficient_basis(np.empty((0, size)), readers)
                axes = basis.axes.copy()
                axes.setflags(write=False)
                error = basis.reconstruction_error
                if not len(axes):
                    raise ObservationError("a zero read basis records no state")
            elif mode != "exact":
                raise ObservationError("mode must be exact or read_basis")
            self.arrays.append(ArrayObservation(unit, tuple(path), tuple(shape), mode, axes, error))
        self.width = sum(entry.width for entry in self.arrays)
        self.digest = hashlib.sha256(json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.rows = 0

    def check_budget(self, rows: int) -> None:
        # The accumulator grows in 512-row chunks, and materialising its final
        # sorted matrix makes additional copies. Reserve four dense matrices.
        reserved = 4 * 8 * self.width * (math.ceil(rows / 512) * 512)
        if rows < 1 or reserved > self.max_bytes:
            raise ObservationError(f"declared observations need {reserved} reserved bytes, budget is {self.max_bytes}")

    def read(self, kernel: Any, organs: Any) -> dict[str, float]:
        return self.read_units(self._runtime_units(kernel, organs))

    @staticmethod
    def _runtime_units(kernel: Any, organs: Any) -> dict[str, Any]:
        from core.subject.carrier_reads import _units, organs_of
        from core.subject.snapshot import _built_services, _module_holdings

        units = _units(kernel, organs)
        units.update({f"organ:{name}": value for name, value in organs_of(organs).items()})
        # The held object can reference an outside object. Include all known
        # owners in the alias guard, without allowing them as plan roots.
        units.update({f"other-service:{name}": value for name, value in _built_services().items()})
        units.update({f"other-module:{name}": value for name, value in _module_holdings().items()})
        units.update({f"other-phase:{index}": value for index, value in enumerate(getattr(kernel, "_phases", ()))})
        return units

    def digest_budget(self) -> int:
        """Reserve working copies of the largest exact array, not a trajectory."""
        if any(entry.mode != "exact" for entry in self.arrays):
            raise ObservationError("whole-array digest comparison requires exact-mode entries")
        reserved = max(4 * 8 * entry.width for entry in self.arrays)
        if reserved > self.max_bytes:
            raise ObservationError(f"array digests need {reserved} reserved bytes, budget is {self.max_bytes}")
        return reserved

    def digests(self, kernel: Any, organs: Any) -> dict[str, str]:
        return self.digest_units(self._runtime_units(kernel, organs))

    def digest_units(self, units: Mapping[str, Any]) -> dict[str, str]:
        """Content hashes of complete declared arrays, with an explicit byte budget.

        SHA-256 equality is collision-resistant evidence, not a mathematical
        injectivity claim. Unlike the eight-number sketch it does not discard
        arbitrary directions before comparing. Canonical float64 bytes match
        the numeric precision of the declared observation path.
        """
        self.digest_budget()
        out = {}
        for index, entry in enumerate(self.arrays):
            values = entry.read(units)
            out[f"held.array_sha256_{self.digest[:16]}.{index}"] = hashlib.sha256(values.tobytes()).hexdigest()
        return out

    def require_repeatable_digests(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
        """A missing or changing untouched array cannot disappear from comparison."""
        keys = [f"held.array_sha256_{self.digest[:16]}.{index}" for index in range(len(self.arrays))]
        missing = [key for key in keys if key not in before or key not in after]
        changed = [key for key in keys if key in before and key in after and before[key] != after[key]]
        if missing or changed:
            raise ObservationError(f"untouched declared arrays are not comparable: missing={missing}, changed={changed}")

    def read_units(self, units: Mapping[str, Any]) -> dict[str, float]:
        self.check_budget(self.rows + 1)
        out = {}
        for index, entry in enumerate(self.arrays):
            values = entry.read(units)
            for coordinate, value in enumerate(values):
                out[f"held.declared_{self.digest[:16]}.{index}.x{coordinate}"] = float(value)
        self.rows += 1
        return out

    def description(self) -> dict[str, Any]:
        return {
            "sha256": self.digest, "width": self.width, "rows": self.rows,
            "max_bytes": self.max_bytes, "diagnostic_only": True,
            "arrays": [{"unit": item.unit, "path": list(item.path), "shape": list(item.shape),
                        "mode": item.mode, "width": item.width,
                        "read_reconstruction_error": item.reconstruction_error} for item in self.arrays],
            "limits": "The byte reservation covers additional observation columns, not the existing runtime/recording. Exact mode covers each declared array; read_basis covers the supplied reads. Neither certifies a complete runtime read inventory or a complete causal boundary.",
        }
