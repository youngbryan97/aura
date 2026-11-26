"""core/runtime/taint.py — the runtime taint register.

Clean-room adoption of the Linux kernel's taint-flag discipline
(`/proc/sys/kernel/tainted`, `add_taint()`, the `Tainted: G W L` line on
every oops).

The rule the kernel enforces and Aura did not: **once something happened
that invalidates the assumptions the rest of the system reasons under, no
later report is allowed to look clean.** A kernel that force-loaded an
unsigned module, or that already hit a WARN, prints its taint on every
subsequent bug report forever — because the person reading report #2 needs
to know report #1 happened, and a fresh-looking oops from a corrupt machine
wastes days.

Aura's failure mode is the same shape and is documented in
KNOWN_FAILURE_MODES.md: a subsystem degrades, a fallback silently absorbs
it, and thirty minutes later the health endpoint says HEALTHY. The health
endpoint is not lying about *now*; it simply has no memory. This module is
that memory.

Semantics, deliberately matching the kernel:

* Taint is **one-way**. Nothing clears a flag except process restart. There
  is no `untaint()` and adding one would defeat the purpose. (A test-only
  reset exists, gated on the test-mode predicate, for suite hygiene.)
* Taint is **cheap and always on**. Setting a flag is a dict write under a
  lock; reading is a copy. No flag gates it.
* Taint is **carried, not judged**. This module does not decide whether the
  runtime should stop. It records that an assumption broke and lets
  health, the incident narrator, the diagnostics bundle, and the flight
  recorder carry it.

Every taint carries the first reason, the first timestamp, and an
occurrence count, so `Tainted: DL (2)` expands to a real story.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Taint")


class TaintFlag(StrEnum):
    """One letter per flag, kernel-style, so the compact form is greppable.

    Letters are chosen to be mnemonic for Aura, not to mirror the kernel's
    own letter assignments (those encode hardware concerns Aura has no
    analogue for).
    """

    #: An organ crashed and was restarted under it. State it published
    #: before the crash may be inconsistent with state published after.
    CRASHED_ORGAN = "C"
    #: Code was hot-swapped into a running process. The running image no
    #: longer matches any commit.
    HOT_SWAPPED = "H"
    #: A lock-order violation was observed (see lockdep.py). The process
    #: is one interleaving away from a deadlock.
    LOCK_ORDER = "L"
    #: An out-of-memory victim was selected and shed.
    OOM_SHED = "O"
    #: A hard assertion failed and was survived rather than aborted.
    ASSERTION = "A"
    #: A degraded fallback path served a user-visible result.
    DEGRADED_RESULT = "D"
    #: A governance or admission gate was bypassed (emergency path).
    GATE_BYPASSED = "G"
    #: Weights, adapters, or model artifacts were loaded without passing
    #: their provenance/signature check.
    UNVERIFIED_ARTIFACT = "U"
    #: Fault injection or a lesion controller deliberately broke something.
    FAULT_INJECTED = "F"
    #: The wall clock jumped backwards or forwards discontinuously; any
    #: duration measured across the jump is suspect.
    CLOCK_JUMP = "T"
    #: A sanitizer (sequence checker, poisoned-reuse detector) fired.
    SANITIZER = "S"
    #: A second runtime instance was detected on this host.
    DUPLICATE_RUNTIME = "R"
    #: Persistent state was restored from a backup or repaired in place.
    STATE_REPAIRED = "P"
    #: A schema/state migration ran and could not be fully verified.
    UNVERIFIED_MIGRATION = "M"


#: Human-readable expansion, used by the incident narrator and the bundle.
_FLAG_MEANING: dict[TaintFlag, str] = {
    TaintFlag.CRASHED_ORGAN: "an organ crashed and was restarted in-process",
    TaintFlag.HOT_SWAPPED: "code was hot-swapped; the image differs from any commit",
    TaintFlag.LOCK_ORDER: "a lock-order violation was observed",
    TaintFlag.OOM_SHED: "an organ was shed under memory pressure",
    TaintFlag.ASSERTION: "a hard assertion failed and was survived",
    TaintFlag.DEGRADED_RESULT: "a degraded fallback served a user-visible result",
    TaintFlag.GATE_BYPASSED: "a governance or admission gate was bypassed",
    TaintFlag.UNVERIFIED_ARTIFACT: "an unverified model artifact was loaded",
    TaintFlag.FAULT_INJECTED: "faults were deliberately injected",
    TaintFlag.CLOCK_JUMP: "the wall clock jumped discontinuously",
    TaintFlag.SANITIZER: "a runtime sanitizer fired",
    TaintFlag.DUPLICATE_RUNTIME: "a second runtime instance was detected",
    TaintFlag.STATE_REPAIRED: "persistent state was repaired or restored",
    TaintFlag.UNVERIFIED_MIGRATION: "a state migration ran unverified",
}

#: Flags that mean "do not trust a green health verdict without reading
#: the reason" — the health surface downgrades a HEALTHY verdict carrying
#: any of these to a caveated one.
CREDIBILITY_FLAGS: frozenset[TaintFlag] = frozenset(
    {
        TaintFlag.CRASHED_ORGAN,
        TaintFlag.LOCK_ORDER,
        TaintFlag.OOM_SHED,
        TaintFlag.ASSERTION,
        TaintFlag.SANITIZER,
        TaintFlag.DUPLICATE_RUNTIME,
        TaintFlag.UNVERIFIED_MIGRATION,
    }
)


@dataclass(frozen=True)
class TaintRecord:
    flag: TaintFlag
    first_reason: str
    first_at: float
    last_at: float
    count: int
    subsystem: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": str(self.flag),
            "meaning": _FLAG_MEANING.get(self.flag, ""),
            "first_reason": self.first_reason,
            "first_at": self.first_at,
            "last_at": self.last_at,
            "count": self.count,
            "subsystem": self.subsystem,
            "age_s": max(0.0, time.time() - self.first_at),
        }


class TaintRegister:
    """Process-wide, append-only, one-way."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[TaintFlag, TaintRecord] = {}
        self._pid = os.getpid()

    def add(self, flag: TaintFlag, reason: str, *, subsystem: str = "runtime") -> TaintRecord:
        now = time.time()
        reason = (reason or "").strip() or "(no reason given)"
        first = False
        with self._lock:
            existing = self._records.get(flag)
            if existing is None:
                first = True
                record = TaintRecord(
                    flag=flag,
                    first_reason=reason,
                    first_at=now,
                    last_at=now,
                    count=1,
                    subsystem=subsystem,
                )
            else:
                record = TaintRecord(
                    flag=flag,
                    first_reason=existing.first_reason,
                    first_at=existing.first_at,
                    last_at=now,
                    count=existing.count + 1,
                    subsystem=existing.subsystem,
                )
            self._records[flag] = record

        if first:
            # First occurrence is the interesting one; repeats are counted
            # but must not become a log flood.
            logger.warning(
                "runtime tainted [%s] %s — %s (subsystem=%s)",
                str(flag),
                _FLAG_MEANING.get(flag, ""),
                reason,
                subsystem,
            )
        return record

    def is_tainted(self, flag: TaintFlag | None = None) -> bool:
        with self._lock:
            if flag is None:
                return bool(self._records)
            return flag in self._records

    def flags(self) -> list[TaintFlag]:
        with self._lock:
            return sorted(self._records, key=lambda f: str(f))

    def compact(self) -> str:
        """The kernel's one-line form: ``GDLW`` — empty string when clean."""
        return "".join(str(f) for f in self.flags())

    def records(self) -> list[TaintRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.first_at)

    def credibility_flags(self) -> list[TaintFlag]:
        return [f for f in self.flags() if f in CREDIBILITY_FLAGS]

    def report(self) -> dict[str, Any]:
        records = self.records()
        return {
            "pid": self._pid,
            "tainted": bool(records),
            "compact": "".join(str(r.flag) for r in sorted(records, key=lambda r: str(r.flag))),
            "flags": [r.to_dict() for r in records],
            "credibility_affecting": [str(f) for f in self.credibility_flags()],
        }

    def narrative(self) -> str:
        """One sentence an operator or the incident narrator can read."""
        records = self.records()
        if not records:
            return "runtime is untainted"
        parts = []
        for r in records:
            age_min = max(0.0, time.time() - r.first_at) / 60.0
            times = "" if r.count == 1 else f" ×{r.count}"
            parts.append(
                f"{_FLAG_MEANING.get(r.flag, str(r.flag))}{times} "
                f"({age_min:.0f}m ago: {r.first_reason})"
            )
        return "runtime tainted — " + "; ".join(parts)


_REGISTER = TaintRegister()


def taint(flag: TaintFlag, reason: str, *, subsystem: str = "runtime") -> TaintRecord:
    """Record that an assumption broke. One-way, cheap, always on."""
    return _REGISTER.add(flag, reason, subsystem=subsystem)


def is_tainted(flag: TaintFlag | None = None) -> bool:
    return _REGISTER.is_tainted(flag)


def taint_flags() -> list[TaintFlag]:
    return _REGISTER.flags()


def taint_compact() -> str:
    return _REGISTER.compact()


def taint_report() -> dict[str, Any]:
    return _REGISTER.report()


def taint_narrative() -> str:
    return _REGISTER.narrative()


def credibility_caveat() -> str | None:
    """Non-None when a green verdict must be read with a caveat."""
    flags = _REGISTER.credibility_flags()
    if not flags:
        return None
    return (
        "health verdict is reported over a tainted runtime ("
        + "".join(str(f) for f in flags)
        + "); see taint report for what broke"
    )


def reset_taint_for_test() -> None:
    """Test-suite hygiene only. Never call this from runtime code —
    a taint that can be cleared is not a taint."""
    _REGISTER._records.clear()  # noqa: SLF001 — the register owns no public clear by design


__all__ = [
    "CREDIBILITY_FLAGS",
    "TaintFlag",
    "TaintRecord",
    "TaintRegister",
    "credibility_caveat",
    "is_tainted",
    "reset_taint_for_test",
    "taint",
    "taint_compact",
    "taint_flags",
    "taint_narrative",
    "taint_report",
]
