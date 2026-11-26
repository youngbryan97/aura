"""core/fsw/assertions.py — FW_ASSERT.

Clean-room adoption of F Prime's assertion mechanism.

Python's `assert` is the wrong tool for a long-running autonomous system,
for two independent reasons. It is removed under `-O`, so an assertion is
either present or silently absent depending on how the process was
launched — and "silently absent" is the mode you discover during an
incident. And when it does fire it raises `AssertionError` with a message
and nothing else: no file, no line, no argument values, no record that
survives the traceback being swallowed by an `except Exception` three
frames up. Aura has a great many of those `except` blocks, by design.

F Prime's `FW_ASSERT` is built differently, and the differences are all
the point:

* It **always runs**. There is no build mode where the check disappears.
* It **records** the file, the line, and up to a few argument values into
  a dedicated assertion log that survives the process.
* It **triggers a declared response** rather than an exception. Flight
  software usually restarts on assertion failure, because a violated
  invariant means the state is not what the code believes and continuing
  is a guess. What it never does is continue *silently*.

Aura's declared responses:

* ``RECORD`` — log it, taint the runtime, keep going. For invariants
  whose violation is informative but survivable.
* ``RAISE`` — record, then raise :class:`AssertionFailure`. For code paths
  where the caller genuinely can handle it.
* ``RESTART`` — record, then request a controlled restart. For invariants
  whose violation means the in-memory state is untrustworthy.

Every failure taints the runtime, so no later report reads clean over a
broken invariant.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Assert")

#: Distinct assertion sites retained. Failures dedupe by site.
MAX_SITES = 512


class Response(StrEnum):
    RECORD = "record"
    RAISE = "raise"
    RESTART = "restart"


class AssertionFailure(RuntimeError):
    """Raised by :func:`fw_assert` when the response is RAISE."""


@dataclass(frozen=True)
class AssertionRecord:
    condition: str
    file: str
    line: int
    function: str
    args: dict[str, Any]
    at: float
    response: Response
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "file": self.file,
            "line": self.line,
            "function": self.function,
            "args": {k: _summarize(v) for k, v in self.args.items()},
            "at": self.at,
            "response": str(self.response),
            "count": self.count,
        }

    @property
    def site(self) -> str:
        return f"{self.file}:{self.line}"


def _summarize(value: Any, *, limit: int = 200) -> Any:
    if isinstance(value, (int, float, bool, str, type(None))):
        text = str(value)
        return value if len(text) <= limit else text[:limit] + "…"
    return repr(value)[:limit]


class AssertionLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, AssertionRecord] = {}
        self._counts: dict[str, int] = {}
        self.failures = 0

    def record(self, record: AssertionRecord) -> AssertionRecord:
        with self._lock:
            self.failures += 1
            self._counts[record.site] = self._counts.get(record.site, 0) + 1
            first = record.site not in self._records
            if first and len(self._records) < MAX_SITES:
                self._records[record.site] = record
            stored = self._records.get(record.site, record)
        if first:
            logger.critical(
                "ASSERTION FAILED at %s in %s: %s | args=%s | response=%s",
                record.site,
                record.function,
                record.condition,
                {k: _summarize(v) for k, v in record.args.items()},
                record.response,
            )
            _persist(record)
        return stored

    def records(self) -> list[AssertionRecord]:
        with self._lock:
            return [
                AssertionRecord(
                    condition=r.condition,
                    file=r.file,
                    line=r.line,
                    function=r.function,
                    args=r.args,
                    at=r.at,
                    response=r.response,
                    count=self._counts.get(r.site, 1),
                )
                for r in sorted(self._records.values(), key=lambda r: r.at)
            ]

    def clean(self) -> bool:
        with self._lock:
            return not self._records

    def report(self) -> dict[str, Any]:
        records = self.records()
        return {
            "clean": not records,
            "distinct_sites": len(records),
            "total_failures": self.failures,
            "records": [r.to_dict() for r in records],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._records.clear()
            self._counts.clear()
            self.failures = 0


_LOG = AssertionLog()


def get_assertion_log() -> AssertionLog:
    return _LOG


def _persist(record: AssertionRecord) -> None:
    """A record that dies with the process is not a record."""
    try:
        from core.config import config
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        path = config.paths.data_dir / "error_logs" / "assertions.jsonl"
        line = json.dumps(record.to_dict(), separators=(",", ":")) + "\n"
        with local_internal_governed_scope("fsw.assertions"):
            get_file_write_gateway().append_text(path, line, source="fsw_assertions")
    except Exception:  # noqa: BLE001 — best effort; the log and taint already happened
        logger.debug("assertion persist failed", exc_info=True)


def _caller(depth: int = 3) -> tuple[str, int, str]:
    frames = traceback.extract_stack(limit=depth + 3)
    for frame in reversed(frames[:-1]):
        if "fsw/assertions.py" in frame.filename:
            continue
        return frame.filename.rsplit("/", 2)[-1], frame.lineno, frame.name
    return "<unknown>", 0, "<unknown>"


def fw_assert(
    condition: Any,
    message: str = "",
    *,
    response: Response = Response.RECORD,
    **args: Any,
) -> bool:
    """Assert an invariant. Always runs; records; never silently continues.

    ::

        fw_assert(
            lane.owner == self.identity,
            "model lane owner changed under us",
            response=Response.RESTART,
            expected=self.identity, observed=lane.owner,
        )

    Returns whether the condition held, so callers may branch on it.
    """
    if condition:
        return True

    file, line, function = _caller()
    record = AssertionRecord(
        condition=message or "assertion failed",
        file=file,
        line=line,
        function=function,
        args=dict(args),
        at=time.time(),
        response=response,
    )
    _LOG.record(record)

    from core.runtime.taint import TaintFlag, taint

    taint(
        TaintFlag.ASSERTION,
        f"{record.condition} at {record.site} in {function}",
        subsystem="fsw_assertions",
    )
    try:
        from core.fsw.telemetry_dictionary import EventSeverity, emit_event

        emit_event(
            "assertion_failed",
            severity=EventSeverity.FATAL if response is Response.RESTART else EventSeverity.WARNING_HI,
            condition=record.condition,
            site=record.site,
            function=function,
            response=str(response),
        )
    except Exception:  # noqa: BLE001
        logger.debug("assertion telemetry failed", exc_info=True)

    if response is Response.RESTART:
        _request_restart(record)
    elif response is Response.RAISE:
        raise AssertionFailure(
            f"{record.condition} at {record.site} in {function} | args={record.args}"
        )
    return False


def _request_restart(record: AssertionRecord) -> None:
    """A violated invariant means the in-memory state is a guess."""
    logger.critical(
        "assertion at %s requests a controlled restart: state is not what the "
        "code believes, and continuing would be guessing",
        record.site,
    )
    try:
        from core.runtime.shutdown_coordinator import request_shutdown

        request_shutdown(reason=f"assertion_failed: {record.condition} at {record.site}")
    except Exception:  # noqa: BLE001
        logger.debug("restart request from assertion failed", exc_info=True)
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "fsw_assertions",
            AssertionFailure(record.condition),
            severity="critical",
            action="requested controlled restart after invariant violation",
            extra=record.to_dict(),
        )
    except Exception:  # noqa: BLE001
        logger.debug("assertion degradation record failed", exc_info=True)


def assertions_report() -> dict[str, Any]:
    return _LOG.report()


def assertions_clean() -> bool:
    return _LOG.clean()


def reset_assertions_for_test() -> None:
    _LOG.reset_for_test()


__all__ = [
    "AssertionFailure",
    "AssertionLog",
    "AssertionRecord",
    "Response",
    "assertions_clean",
    "assertions_report",
    "fw_assert",
    "get_assertion_log",
    "reset_assertions_for_test",
]
