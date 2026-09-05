"""core/verify/work_ledger.py — what work actually ran, per turn.

Clean-room adoption of Springdrift's fabrication audit (AGPL; mechanism
reimplemented, no code taken). Its premise, restated for Aura:

    A persisted claim about work is not evidence the work happened. The
    tool record is.

Aura has a long history of the inverse failure — prose that describes an
analysis nobody ran. The code demo that "checked" a file it never opened.
The screen read answered with a step count instead of the reading. A
correlation reported to three decimal places by a turn that called no
analyser. In every case the sentence was fluent and the substrate was
idle, and nothing in the runtime could tell the difference afterwards
because nothing wrote down what had actually executed.

This module is the write-down. It is deliberately dumb: a bounded ring of
``(turn, unit, at, ok)`` rows and a query that answers "which units ran
during this turn". It makes no judgement — :mod:`core.verify.fabrication_audit`
does that, and keeping the two apart means the record stays trustworthy
even when the audit's patterns are wrong.

Three properties matter and are tested:

* **Recording is non-fatal.** A ledger that can raise into a tool call
  would make the audit infrastructure a new failure mode for the work it
  audits. Every entry point swallows into ``record_degradation``.
* **Absence is not proof of absence.** :func:`tools_for_turn` returns
  ``None`` for a turn the ledger never saw, distinct from an empty set for
  a turn that ran nothing. An auditor that conflates the two manufactures
  fabrication findings out of eviction, which is exactly the "absence of a
  check reported as a passed check" inversion this codebase keeps finding.
* **It is bounded.** The ring is capped and evicts oldest-first; eviction
  marks the turn unknown rather than empty.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock

__all__ = [
    "WorkRecord",
    "WorkLedger",
    "get_work_ledger",
    "record_work",
    "tools_for_turn",
]

#: How many turns keep a work record. Sized so a long live session stays
#: auditable without the ring becoming a memory surface of its own; a turn
#: older than this is reported UNKNOWN rather than empty.
_MAX_TURNS = 512

#: Per-turn cap on distinct units. A turn that ran more than this is
#: pathological on its own terms; the cap stops one runaway loop from
#: evicting every other turn's evidence.
_MAX_UNITS_PER_TURN = 256


@dataclass(frozen=True)
class WorkRecord:
    """One unit of work that actually executed."""

    turn_id: str
    unit: str
    at: float
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "unit": self.unit,
            "at": self.at,
            "ok": self.ok,
            "detail": self.detail,
        }


class WorkLedger:
    """A bounded, turn-keyed record of executed work units."""

    def __init__(self, *, max_turns: int = _MAX_TURNS) -> None:
        self._lock = checked_lock("work_ledger", rank=LockRank.LEAF)
        self._max_turns = max(1, int(max_turns))
        # turn_id -> {unit: WorkRecord}. OrderedDict gives oldest-first eviction.
        self._turns: OrderedDict[str, dict[str, WorkRecord]] = OrderedDict()
        self._evicted_turns = 0
        self._dropped_units = 0

    # ------------------------------------------------------------------ write

    def record(
        self,
        unit: str,
        *,
        turn_id: str | None = None,
        ok: bool = True,
        detail: str = "",
    ) -> WorkRecord | None:
        """Record that ``unit`` executed during ``turn_id``.

        Returns the record, or ``None`` when there is no turn to attribute
        it to — background work, tests and tool probes legitimately run
        outside a turn, and inventing a turn id for them would put rows in
        the ledger that no audit can ever match.
        """
        name = str(unit or "").strip()
        if not name:
            return None
        resolved = turn_id if turn_id is not None else _current_turn_id()
        if not resolved:
            return None
        record = WorkRecord(
            turn_id=str(resolved), unit=name, at=time.time(), ok=bool(ok), detail=str(detail or "")
        )
        with self._lock:
            units = self._turns.get(record.turn_id)
            if units is None:
                units = {}
                self._turns[record.turn_id] = units
            else:
                self._turns.move_to_end(record.turn_id)
            if name in units or len(units) < _MAX_UNITS_PER_TURN:
                units[name] = record
            else:
                self._dropped_units += 1
            while len(self._turns) > self._max_turns:
                self._turns.popitem(last=False)
                self._evicted_turns += 1
        return record

    # ------------------------------------------------------------------- read

    def tools_for_turn(self, turn_id: str) -> frozenset[str] | None:
        """Units that ran in ``turn_id``.

        ``None`` means the ledger has no record of this turn — it was
        evicted, or predates the ledger. That is NOT the same as "the turn
        ran no tools", and callers must not treat it as such.
        """
        key = str(turn_id or "")
        if not key:
            return None
        with self._lock:
            units = self._turns.get(key)
            if units is None:
                return None
            return frozenset(units)

    def successful_tools_for_turn(self, turn_id: str) -> frozenset[str] | None:
        """Units that ran AND succeeded. A failed tool did not do the work."""
        key = str(turn_id or "")
        if not key:
            return None
        with self._lock:
            units = self._turns.get(key)
            if units is None:
                return None
            return frozenset(name for name, rec in units.items() if rec.ok)

    def knows_turn(self, turn_id: str) -> bool:
        return self.tools_for_turn(turn_id) is not None

    def records_for_turn(self, turn_id: str) -> tuple[WorkRecord, ...]:
        with self._lock:
            units = self._turns.get(str(turn_id or ""))
            if not units:
                return ()
            return tuple(sorted(units.values(), key=lambda r: r.at))

    def known_turns(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._turns)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turns_tracked": len(self._turns),
                "max_turns": self._max_turns,
                "evicted_turns": self._evicted_turns,
                "dropped_units": self._dropped_units,
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._turns.clear()
            self._evicted_turns = 0
            self._dropped_units = 0


_LEDGER = WorkLedger()


def get_work_ledger() -> WorkLedger:
    return _LEDGER


def _current_turn_id() -> str:
    """The bound turn's id, or "" when nothing is bound.

    Imported lazily: ``core.runtime.turn_outcome`` is a peer, and a module
    that records evidence must not be the reason the runtime cannot import.
    """
    try:
        from core.runtime.turn_outcome import current_turn

        turn = current_turn()
        return str(getattr(turn, "turn_id", "") or "") if turn is not None else ""
    except Exception as exc:  # pragma: no cover - defensive
        record_degradation(
            "work_ledger",
            exc,
            severity="debug",
            action="attributed work to no turn",
        )
        return ""


def record_work(
    unit: str,
    *,
    turn_id: str | None = None,
    ok: bool = True,
    detail: str = "",
) -> WorkRecord | None:
    """Record executed work. Never raises into the caller.

    This sits on tool-execution paths, so a defect here must not become a
    defect in the tool. Failure to record degrades the audit (a claim
    becomes unverifiable), which is strictly better than failing the work.
    """
    try:
        record = _LEDGER.record(unit, turn_id=turn_id, ok=ok, detail=detail)
    except Exception as exc:
        record_degradation(
            "work_ledger",
            exc,
            action="work went unrecorded; claims from this turn are unverifiable",
        )
        return None
    if record is not None:
        _mirror_into_turn(record)
    return record


def _mirror_into_turn(record: WorkRecord) -> None:
    """Attach the unit to the live turn's receipts, when one is bound.

    The ring is for the auditor; the receipt is for the turn's own account
    of itself. Both, so a finalized turn carries its evidence even after
    the ring has evicted it.
    """
    try:
        from core.runtime.turn_outcome import current_turn

        turn = current_turn()
        if turn is None or turn.is_finalized:
            return
        turn.record_receipt("work_unit", {"unit": record.unit, "ok": record.ok})
    except Exception as exc:  # pragma: no cover - defensive
        record_degradation(
            "work_ledger",
            exc,
            severity="debug",
            action="turn receipt not written; ring record stands",
        )


def tools_for_turn(turn_id: str) -> frozenset[str] | None:
    return _LEDGER.tools_for_turn(turn_id)


def iter_records(turn_id: str) -> Iterator[WorkRecord]:
    yield from _LEDGER.records_for_turn(turn_id)


def status() -> Mapping[str, Any]:
    return _LEDGER.status()
