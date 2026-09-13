"""core/runtime/parameters.py — declared, validated, observable parameters.

Clean-room adoption of the ROS 2 parameter system: declaration with
descriptors, constraint validation, atomic multi-set, and on-set
callbacks.

Aura's tunable numbers currently live in three places, none of which can
do the job on its own. Literals in source cannot be changed without a
restart. Environment flags (`core/runtime/flags.py`) are declared and
owned, which is good, but they are read once at boot and are strings.
Config files are neither validated nor observed. The result is the
familiar one: a threshold nobody can find, a magic number nobody can
justify, and a change that requires a restart of a runtime whose whole
value proposition is continuity.

ROS 2's answer, and the design here:

* **Declare or it does not exist.** Reading an undeclared parameter
  raises. This is the rule that makes the parameter list an accurate
  inventory instead of an aspiration — you cannot quietly introduce a
  knob.
* **Descriptors carry the constraint.** Type, range, allowed set,
  read-only, and a human description travel with the parameter, so
  validation happens in one place and the answer to "what is a legal
  value" is not "read the code that consumes it".
* **On-set callbacks may veto.** A parameter change is a *request*; a
  consumer that knows the change is unsafe right now rejects it with a
  reason. This is how a live runtime accepts retuning without accepting
  incoherence.
* **Atomic multi-set.** Parameters that are only meaningful together
  (a threshold and its hysteresis, a budget and its window) are validated
  as a set and applied together or not at all. Applying them one at a
  time passes through a state that is invalid, and something always
  observes that state.

Every change is recorded with its old value, new value, source, and
reason, so "why is this 0.8" has an answer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Parameters")


class ParameterType(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    STRING_ARRAY = "string_array"


class ParameterError(KeyError):
    """Raised when reading or setting a parameter that was never declared."""


@dataclass(frozen=True)
class ParameterDescriptor:
    """What a parameter is, and what values are legal for it."""

    description: str
    type: ParameterType
    owner: str
    read_only: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    allowed: tuple[Any, ...] = ()
    #: An environment variable that seeds the value at declaration time,
    #: so boot-time configuration still works and stays in one inventory.
    env_var: str = ""
    #: Persisted parameters survive a restart; transient ones reset to
    #: their declared default, which is usually what a tuning knob wants.
    persistent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "type": str(self.type),
            "owner": self.owner,
            "read_only": self.read_only,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "allowed": list(self.allowed),
            "env_var": self.env_var,
            "persistent": self.persistent,
        }


@dataclass(frozen=True)
class SetResult:
    successful: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> SetResult:
        return cls(successful=True)

    @classmethod
    def reject(cls, reason: str) -> SetResult:
        return cls(successful=False, reason=reason)


@dataclass(frozen=True)
class ParameterChange:
    name: str
    old: Any
    new: Any
    at: float
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "old": self.old,
            "new": self.new,
            "at": self.at,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass
class _Parameter:
    name: str
    value: Any
    default: Any
    descriptor: ParameterDescriptor
    declared_at: float = field(default_factory=lambda: time.time())
    changes: int = 0


ValidateFn = Callable[[str, Any, Any], SetResult | bool | None]
NotifyFn = Callable[[str, Any, Any], None]


def _coerce(value: Any, kind: ParameterType) -> Any:
    if kind is ParameterType.BOOL:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{value!r} is not a boolean")
    if kind is ParameterType.INT:
        if isinstance(value, bool):
            raise ValueError("a boolean is not an integer parameter value")
        return int(value)
    if kind is ParameterType.FLOAT:
        if isinstance(value, bool):
            raise ValueError("a boolean is not a float parameter value")
        return float(value)
    if kind is ParameterType.STRING:
        return str(value)
    if kind is ParameterType.STRING_ARRAY:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, Sequence):
            return tuple(str(v) for v in value)
        raise ValueError(f"{value!r} is not a string array")
    raise ValueError(f"unknown parameter type {kind}")


class ParameterServer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._params: dict[str, _Parameter] = {}
        self._validators: dict[str, list[ValidateFn]] = {}
        self._observers: dict[str, list[NotifyFn]] = {}
        self._global_observers: list[NotifyFn] = []
        self._history: list[ParameterChange] = []
        self.rejections = 0

    # ── declaration ───────────────────────────────────────────────────
    def declare(
        self, name: str, default: Any, descriptor: ParameterDescriptor
    ) -> Any:
        """Declare a parameter. Idempotent for an identical declaration."""
        with self._lock:
            existing = self._params.get(name)
            if existing is not None:
                if existing.descriptor != descriptor:
                    raise ValueError(
                        f"parameter {name!r} already declared by "
                        f"{existing.descriptor.owner} with a different descriptor; "
                        "a parameter has one meaning"
                    )
                return existing.value

            value = _coerce(default, descriptor.type)
            if descriptor.env_var:
                raw = os.environ.get(descriptor.env_var)
                if raw is not None and raw.strip():
                    try:
                        value = _coerce(raw, descriptor.type)
                    except ValueError as exc:
                        logger.warning(
                            "parameter %s: %s=%r is not usable (%s); keeping the default",
                            name,
                            descriptor.env_var,
                            raw,
                            exc,
                        )
            problem = self._constraint_problem(value, descriptor)
            if problem:
                raise ValueError(f"parameter {name!r} default is invalid: {problem}")
            self._params[name] = _Parameter(
                name=name, value=value, default=value, descriptor=descriptor
            )
            return value

    @staticmethod
    def _constraint_problem(value: Any, descriptor: ParameterDescriptor) -> str:
        if descriptor.allowed and value not in descriptor.allowed:
            return f"{value!r} is not one of {list(descriptor.allowed)}"
        if descriptor.type in (ParameterType.INT, ParameterType.FLOAT):
            if descriptor.minimum is not None and value < descriptor.minimum:
                return f"{value} is below the minimum {descriptor.minimum}"
            if descriptor.maximum is not None and value > descriptor.maximum:
                return f"{value} is above the maximum {descriptor.maximum}"
            if descriptor.step:
                base = descriptor.minimum if descriptor.minimum is not None else 0
                offset = (value - base) / descriptor.step
                if abs(offset - round(offset)) > 1e-9:
                    return f"{value} is not on the {descriptor.step} step grid from {base}"
        return ""

    # ── reading ───────────────────────────────────────────────────────
    def get(self, name: str) -> Any:
        with self._lock:
            param = self._params.get(name)
        if param is None:
            raise ParameterError(
                f"parameter {name!r} was never declared; declare it where it is "
                "owned so the inventory stays accurate"
            )
        return param.value

    def get_or(self, name: str, default: Any) -> Any:
        """For read paths that must tolerate an undeclared parameter."""
        try:
            return self.get(name)
        except ParameterError:
            return default

    def declared(self, name: str) -> bool:
        with self._lock:
            return name in self._params

    def describe(self, name: str) -> ParameterDescriptor | None:
        with self._lock:
            param = self._params.get(name)
            return param.descriptor if param else None

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._params)

    # ── writing ───────────────────────────────────────────────────────
    def set(
        self, name: str, value: Any, *, source: str = "runtime", reason: str = ""
    ) -> SetResult:
        return self.set_atomically({name: value}, source=source, reason=reason)

    def set_atomically(
        self,
        values: dict[str, Any],
        *,
        source: str = "runtime",
        reason: str = "",
    ) -> SetResult:
        """Validate every change, then apply all of them or none.

        Applying a coupled set one at a time passes through an invalid
        intermediate state, and something always observes it.
        """
        with self._lock:
            staged: list[tuple[_Parameter, Any, Any]] = []
            for name, raw in values.items():
                param = self._params.get(name)
                if param is None:
                    return self._reject(
                        name, f"parameter {name!r} was never declared"
                    )
                if param.descriptor.read_only:
                    return self._reject(
                        name, f"parameter {name!r} is read-only"
                    )
                try:
                    coerced = _coerce(raw, param.descriptor.type)
                except ValueError as exc:
                    return self._reject(name, str(exc))
                problem = self._constraint_problem(coerced, param.descriptor)
                if problem:
                    return self._reject(name, problem)
                staged.append((param, param.value, coerced))

            # Validators see the whole proposed set before anything applies.
            for param, old, new in staged:
                for validator in self._validators.get(param.name, ()):
                    outcome = _coerce_set_result(validator(param.name, old, new))
                    if not outcome.successful:
                        return self._reject(
                            param.name,
                            outcome.reason or "rejected by an on-set validator",
                        )

            applied: list[tuple[str, Any, Any]] = []
            for param, old, new in staged:
                if old == new:
                    continue
                param.value = new
                param.changes += 1
                applied.append((param.name, old, new))
                self._history.append(
                    ParameterChange(
                        name=param.name,
                        old=old,
                        new=new,
                        at=time.time(),
                        source=source,
                        reason=reason,
                    )
                )
            if len(self._history) > 256:
                del self._history[:-256]
            observers = {
                name: list(self._observers.get(name, ())) for name, _, _ in applied
            }
            global_observers = list(self._global_observers)

        for name, old, new in applied:
            logger.info(
                "🎛️ parameter %s: %r → %r (%s%s)",
                name,
                old,
                new,
                source,
                f": {reason}" if reason else "",
            )
            for observer in observers.get(name, ()):
                _safe_notify(observer, name, old, new)
            for observer in global_observers:
                _safe_notify(observer, name, old, new)
        return SetResult.ok()

    def _reject(self, name: str, reason: str) -> SetResult:
        self.rejections += 1
        logger.info("🎛️ parameter set rejected for %s: %s", name, reason)
        return SetResult.reject(f"{name}: {reason}")

    def reset(self, name: str, *, source: str = "runtime") -> SetResult:
        with self._lock:
            param = self._params.get(name)
            if param is None:
                return self._reject(name, "never declared")
            default = param.default
        return self.set(name, default, source=source, reason="reset to declared default")

    # ── callbacks ─────────────────────────────────────────────────────
    def add_validator(self, name: str, fn: ValidateFn) -> None:
        """A consumer that can veto a change it knows is unsafe right now."""
        with self._lock:
            self._validators.setdefault(name, []).append(fn)

    def add_observer(self, name: str, fn: NotifyFn) -> None:
        with self._lock:
            self._observers.setdefault(name, []).append(fn)

    def add_global_observer(self, fn: NotifyFn) -> None:
        with self._lock:
            self._global_observers.append(fn)

    # ── persistence ───────────────────────────────────────────────────
    def persistable(self) -> dict[str, Any]:
        with self._lock:
            return {
                name: param.value
                for name, param in self._params.items()
                if param.descriptor.persistent and param.value != param.default
            }

    def load(self, values: dict[str, Any], *, source: str = "persisted") -> dict[str, str]:
        """Apply persisted values one at a time; report what did not apply."""
        problems: dict[str, str] = {}
        for name, value in values.items():
            if not self.declared(name):
                problems[name] = "no longer declared"
                continue
            outcome = self.set(name, value, source=source, reason="restored")
            if not outcome.successful:
                problems[name] = outcome.reason
        return problems

    # ── reporting ─────────────────────────────────────────────────────
    def report(self) -> dict[str, Any]:
        with self._lock:
            params = {
                name: {
                    "value": list(param.value)
                    if isinstance(param.value, tuple)
                    else param.value,
                    "default": list(param.default)
                    if isinstance(param.default, tuple)
                    else param.default,
                    "changed": param.value != param.default,
                    "changes": param.changes,
                    "descriptor": param.descriptor.to_dict(),
                }
                for name, param in sorted(self._params.items())
            }
            history = [c.to_dict() for c in self._history[-16:]]
        return {
            "count": len(params),
            "parameters": params,
            "changed_from_default": [n for n, e in params.items() if e["changed"]],
            "rejections": self.rejections,
            "recent_changes": history,
            "owners": sorted({e["descriptor"]["owner"] for e in params.values()}),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._params.clear()
            self._validators.clear()
            self._observers.clear()
            self._global_observers.clear()
            self._history.clear()
            self.rejections = 0


def _coerce_set_result(outcome: Any) -> SetResult:
    if isinstance(outcome, SetResult):
        return outcome
    if outcome is None or outcome is True:
        return SetResult.ok()
    if outcome is False:
        return SetResult.reject("validator returned False")
    if isinstance(outcome, str):
        return SetResult.reject(outcome)
    return SetResult.ok()


def _safe_notify(fn: NotifyFn, name: str, old: Any, new: Any) -> None:
    try:
        fn(name, old, new)
    except Exception:  # noqa: BLE001 — an observer must not undo an applied change
        logger.warning("parameter observer for %s failed", name, exc_info=True)


_SERVER = ParameterServer()


def get_parameter_server() -> ParameterServer:
    return _SERVER


def declare(
    name: str,
    default: Any,
    *,
    type: ParameterType,  # noqa: A002 — mirrors the descriptor field name
    description: str,
    owner: str,
    read_only: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    allowed: tuple[Any, ...] = (),
    env_var: str = "",
    persistent: bool = False,
) -> Any:
    """Declare a parameter next to the code that owns it::

        THRESHOLD = declare(
            "eviction.soft_memory_fraction", 0.15,
            type=ParameterType.FLOAT,
            description="available-memory fraction at which reclaim begins",
            owner="core/runtime/eviction.py",
            minimum=0.01, maximum=0.9,
        )
    """
    return _SERVER.declare(
        name,
        default,
        ParameterDescriptor(
            description=description,
            type=type,
            owner=owner,
            read_only=read_only,
            minimum=minimum,
            maximum=maximum,
            step=step,
            allowed=allowed,
            env_var=env_var,
            persistent=persistent,
        ),
    )


def get(name: str) -> Any:
    return _SERVER.get(name)


def set_parameter(name: str, value: Any, *, source: str = "runtime", reason: str = "") -> SetResult:
    return _SERVER.set(name, value, source=source, reason=reason)


def parameters_report() -> dict[str, Any]:
    return _SERVER.report()


def reset_parameters_for_test() -> None:
    _SERVER.reset_for_test()


__all__ = [
    "NotifyFn",
    "ParameterChange",
    "ParameterDescriptor",
    "ParameterError",
    "ParameterServer",
    "ParameterType",
    "SetResult",
    "ValidateFn",
    "declare",
    "get",
    "get_parameter_server",
    "parameters_report",
    "reset_parameters_for_test",
    "set_parameter",
]
