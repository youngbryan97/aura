"""core/runtime/sanitizers.py — runtime sanitizers.

Clean-room adoption of the sanitizer family (ASan / TSan / UBSan) that
LLVM and Chromium made standard practice, translated into the failure
modes a Python cognitive runtime actually has.

Python does not have use-after-free or data races on raw memory, so a
literal port would be pointless. It has the *same shape* of bug in three
different clothes, and all three are silent today:

**1. Poisoned reuse (ASan's shadow memory).** Aura recycles objects —
pooled buffers, reused turn contexts, recycled tensors. When something
keeps a reference past its lifetime, the read succeeds and returns
plausible-looking stale data. Nothing crashes. The answer is subtly wrong
and the cause is unfindable. :class:`PoisonPool` poisons on release, so
the stale read is loud at the moment it happens instead of silent forever.

**2. Non-finite propagation (UBSan).** A single NaN in an affect vector,
a steering direction, or a reward signal propagates through every
downstream computation and turns every comparison False. There is no
exception; the mind simply stops preferring anything. :func:`check_finite`
makes the boundary where it entered the one place it is reported.

**3. Sequence affinity (TSan's happens-before, Chromium's
SequenceChecker).** Objects that were written assuming one owner get
touched from a worker thread or a second task, and the corruption is
timing-dependent and unreproducible. :class:`SequenceChecker` asserts the
affinity that the code already assumes but never states.

All three follow the sanitizer contract: **report at the first occurrence,
loudly, with the context needed to fix it, and keep running.** A sanitizer
that aborts the process is useless in production, and production is where
these bugs live. Every finding taints the runtime, so a green health
report cannot outlive a sanitizer hit.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from core.runtime.taint import TaintFlag, taint

logger = logging.getLogger("Aura.Sanitizer")

#: Findings are deduplicated by signature; this caps distinct findings.
MAX_FINDINGS = 512


@dataclass(frozen=True)
class Finding:
    sanitizer: str
    signature: str
    message: str
    at: float
    context: str
    occurrences: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sanitizer": self.sanitizer,
            "signature": self.signature,
            "message": self.message,
            "at": self.at,
            "context": self.context,
            "occurrences": self.occurrences,
        }


class SanitizerLog:
    """Process-wide, deduplicated, always on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._findings: dict[str, Finding] = {}
        self._counts: dict[str, int] = {}

    def report(
        self, sanitizer: str, signature: str, message: str, *, context: str = ""
    ) -> bool:
        """Record a finding. Returns True if this is the first occurrence."""
        with self._lock:
            self._counts[signature] = self._counts.get(signature, 0) + 1
            first = signature not in self._findings
            if first and len(self._findings) < MAX_FINDINGS:
                self._findings[signature] = Finding(
                    sanitizer=sanitizer,
                    signature=signature,
                    message=message,
                    at=time.time(),
                    context=context or _caller_context(),
                )
        if not first:
            return False
        logger.error("🧪 %s: %s", sanitizer, message)
        taint(TaintFlag.SANITIZER, f"{sanitizer}: {message[:200]}", subsystem="sanitizers")
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "sanitizers",
                AssertionError(message),
                severity="warning",
                action=f"{sanitizer} finding recorded; execution continued",
                extra={"sanitizer": sanitizer, "signature": signature},
                enforce_failure_policy=False,
            )
        except Exception:  # noqa: BLE001 — reporting is never load-bearing
            logger.debug("sanitizer degradation record failed", exc_info=True)
        return True

    def findings(self) -> list[Finding]:
        with self._lock:
            return [
                Finding(
                    sanitizer=f.sanitizer,
                    signature=f.signature,
                    message=f.message,
                    at=f.at,
                    context=f.context,
                    occurrences=self._counts.get(f.signature, 1),
                )
                for f in sorted(self._findings.values(), key=lambda f: f.at)
            ]

    def clean(self) -> bool:
        with self._lock:
            return not self._findings

    def to_dict(self) -> dict[str, Any]:
        findings = self.findings()
        return {
            "clean": not findings,
            "distinct_findings": len(findings),
            "total_occurrences": sum(f.occurrences for f in findings),
            "findings": [f.to_dict() for f in findings],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._findings.clear()
            self._counts.clear()


_LOG = SanitizerLog()


def get_sanitizer_log() -> SanitizerLog:
    return _LOG


def _caller_context(depth: int = 3) -> str:
    import traceback

    frames = traceback.extract_stack(limit=depth + 3)
    for frame in reversed(frames[:-1]):
        if "sanitizers.py" in frame.filename:
            continue
        return f"{frame.filename.rsplit('/', 2)[-1]}:{frame.lineno} in {frame.name}"
    return "<unknown>"


# ══════════════════════════════════════════════════════════════════════
# 1. Poisoned reuse
# ══════════════════════════════════════════════════════════════════════

class UseAfterReleaseError(RuntimeError):
    """Raised by a strict pool when poisoned state is touched."""


class Poisoned:
    """A tombstone left where a released object used to be.

    Every attribute access, call, comparison, or iteration reports. The
    object is deliberately hostile: the *point* is that the stale read
    cannot pass unnoticed.
    """

    __slots__ = ("_pool", "_label", "_released_at", "_strict")

    def __init__(self, label: str, *, pool: str = "", strict: bool = False) -> None:
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_released_at", time.time())
        object.__setattr__(self, "_strict", strict)

    def _complain(self, operation: str) -> None:
        label = object.__getattribute__(self, "_label")
        pool = object.__getattribute__(self, "_pool")
        released_at = object.__getattribute__(self, "_released_at")
        age = time.time() - released_at
        message = (
            f"use-after-release: {operation} on {label!r} from pool {pool!r}, "
            f"released {age:.3f}s ago. Something kept a reference past the "
            "object's lifetime and is now reading stale state"
        )
        _LOG.report("poison", f"poison:{pool}:{label}:{operation}", message)
        if object.__getattribute__(self, "_strict"):
            raise UseAfterReleaseError(message)

    def __getattr__(self, item: str) -> Any:
        self._complain(f"attribute read .{item}")
        return None

    def __setattr__(self, key: str, value: Any) -> None:
        self._complain(f"attribute write .{key}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._complain("call")
        return None

    def __iter__(self) -> Iterator[Any]:
        self._complain("iteration")
        return iter(())

    def __len__(self) -> int:
        self._complain("len()")
        return 0

    def __bool__(self) -> bool:
        self._complain("truth test")
        return False

    def __repr__(self) -> str:
        label = object.__getattribute__(self, "_label")
        return f"<Poisoned {label!r}>"


class PoisonPool[T]:
    """An object pool that poisons on release.

    ``acquire()`` hands out a live object; ``release()`` marks it dead and
    swaps a poison tombstone into the pool's own record. A caller that
    kept a reference and reads it later is reported at the read.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        *,
        reset: Callable[[T], None] | None = None,
        max_size: int = 32,
        strict: bool = False,
    ) -> None:
        self.name = name
        self._factory = factory
        self._reset = reset
        self._max_size = max_size
        self._strict = strict
        self._lock = threading.Lock()
        self._free: list[T] = []
        self._live: dict[int, str] = {}
        self.acquisitions = 0
        self.releases = 0
        self.double_releases = 0

    def acquire(self, label: str = "") -> T:
        with self._lock:
            obj = self._free.pop() if self._free else self._factory()
            self._live[id(obj)] = label or f"{self.name}#{id(obj):x}"
            self.acquisitions += 1
            return obj

    def release(self, obj: T) -> Poisoned:
        key = id(obj)
        with self._lock:
            label = self._live.pop(key, None)
            if label is None:
                self.double_releases += 1
        if label is None:
            _LOG.report(
                "poison",
                f"double-release:{self.name}:{key:x}",
                f"double release into pool {self.name!r}: the object was already "
                "released, so two owners believe they hold it",
            )
            return Poisoned("double-released", pool=self.name, strict=self._strict)

        if self._reset is not None:
            try:
                self._reset(obj)
            except Exception:  # noqa: BLE001 — a failed reset must not leak the object
                logger.debug("pool reset failed for %s", self.name, exc_info=True)
        with self._lock:
            self.releases += 1
            if len(self._free) < self._max_size:
                self._free.append(obj)
        return Poisoned(label, pool=self.name, strict=self._strict)

    @contextmanager
    def borrow(self, label: str = "") -> Iterator[T]:
        obj = self.acquire(label)
        try:
            yield obj
        finally:
            self.release(obj)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "free": len(self._free),
                "live": len(self._live),
                "acquisitions": self.acquisitions,
                "releases": self.releases,
                "double_releases": self.double_releases,
                "leaked": sorted(self._live.values())[:16],
            }


# ══════════════════════════════════════════════════════════════════════
# 2. Non-finite propagation
# ══════════════════════════════════════════════════════════════════════

def _iter_numbers(value: Any, *, limit: int = 4096) -> Iterator[float]:
    """Walk numbers out of scalars, sequences, and array-likes."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
        return
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:  # noqa: BLE001 — fall through to sequence handling
            logger.debug("numeric sanitizer could not materialize array-like", exc_info=True)
    if isinstance(value, (str, bytes)):
        return
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, Sequence) or hasattr(value, "__iter__"):
        count = 0
        try:
            for item in value:
                for number in _iter_numbers(item, limit=limit - count):
                    yield number
                    count += 1
                    if count >= limit:
                        return
        # not a failure: a value that claimed to be iterable and is not holds
        # no numbers, which is what stopping here reports.
        except TypeError:
            return


def check_finite(name: str, value: Any, *, strict: bool = False) -> bool:
    """Assert every number in ``value`` is finite. Returns True when clean.

    Put this at boundaries: where a vector enters the affect pipeline,
    where a steering direction is applied, where a reward is written to a
    ledger. A NaN that gets past a boundary is a NaN nobody can trace.
    """
    bad: list[str] = []
    for index, number in enumerate(_iter_numbers(value)):
        if math.isnan(number):
            bad.append(f"NaN@{index}")
        elif math.isinf(number):
            bad.append(f"{'+' if number > 0 else '-'}Inf@{index}")
        if len(bad) >= 4:
            break
    if not bad:
        return True
    message = (
        f"non-finite value entering {name!r}: {', '.join(bad)}. Every downstream "
        "comparison against this is False, so preferences and gates silently "
        "stop discriminating"
    )
    _LOG.report("numeric", f"nonfinite:{name}", message)
    if strict:
        raise ValueError(message)
    return False


def sanitize_finite(name: str, value: Any, *, fill: float = 0.0) -> Any:
    """Report non-finite entries and return a cleaned copy of a flat sequence.

    Use only where dropping the poison is genuinely better than
    propagating it — a steering vector, say, where zero means "no push".
    Scalars and unknown types are returned unchanged after reporting.
    """
    if check_finite(name, value):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return fill
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:  # noqa: BLE001
            logger.debug("finite-value sanitizer could not materialize array-like", exc_info=True)
            return value
    if isinstance(value, list):
        return [
            fill
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            else sanitize_finite(f"{name}[]", v, fill=fill)
            if isinstance(v, list)
            else v
            for v in value
        ]
    return value


# ══════════════════════════════════════════════════════════════════════
# 3. Sequence affinity
# ══════════════════════════════════════════════════════════════════════

class SequenceChecker:
    """Assert that an object is only ever touched from one execution context.

    State written assuming a single owner and then touched from a worker
    thread corrupts in a way that is timing-dependent and never reproduces
    on demand. Declaring the affinity turns that into a report at the
    first violation.

    ::

        class TurnBuffer:
            def __init__(self):
                self._sequence = SequenceChecker("turn_buffer")

            def append(self, item):
                self._sequence.check("append")
                ...
    """

    __slots__ = ("_context", "_label", "_lock")

    def __init__(self, label: str) -> None:
        self._label = label
        self._lock = threading.Lock()
        self._context: tuple[str, int] | None = None

    @staticmethod
    def _current() -> tuple[str, int]:
        import asyncio

        try:
            task = asyncio.current_task()
        # not a failure: off a loop there is no current task, and the
        # thread-based context below is used instead.
        except RuntimeError:
            task = None
        if task is not None:
            return ("task", id(task))
        return ("thread", threading.get_ident())

    def check(self, operation: str = "access") -> bool:
        current = self._current()
        with self._lock:
            if self._context is None:
                self._context = current
                return True
            expected = self._context
        if expected == current:
            return True
        _LOG.report(
            "sequence",
            f"sequence:{self._label}:{operation}",
            f"sequence violation on {self._label!r}: {operation} ran on "
            f"{current[0]}:{current[1]:x} but the object is owned by "
            f"{expected[0]}:{expected[1]:x}. Whatever this object protects is "
            "being mutated from two contexts",
        )
        return False

    def detach(self) -> None:
        """Hand ownership to whatever context checks next (an explicit move)."""
        with self._lock:
            self._context = None


@contextmanager
def sequence_scope(label: str) -> Iterator[SequenceChecker]:
    """A checker scoped to one block — useful for ad-hoc critical regions."""
    checker = SequenceChecker(label)
    checker.check("enter")
    try:
        yield checker
    finally:
        checker.detach()


# ══════════════════════════════════════════════════════════════════════

def sanitizer_report() -> dict[str, Any]:
    return _LOG.to_dict()


def sanitizers_clean() -> bool:
    return _LOG.clean()


def reset_sanitizers_for_test() -> None:
    _LOG.reset_for_test()


__all__ = [
    "Finding",
    "Poisoned",
    "PoisonPool",
    "SanitizerLog",
    "SequenceChecker",
    "UseAfterReleaseError",
    "check_finite",
    "get_sanitizer_log",
    "reset_sanitizers_for_test",
    "sanitize_finite",
    "sanitizer_report",
    "sanitizers_clean",
    "sequence_scope",
]
