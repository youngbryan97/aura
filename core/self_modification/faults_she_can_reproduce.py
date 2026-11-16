"""Faults from her own running, each made again as a test that fails the way she failed (Master Todo F15).

Her degradation records keep the call a fault was raised in (``core/runtime/a_call_that_failed.py``). This hears
every record as it is made and keeps the calls it can make again, one for each fault, however often it recurs. Each
is then written as a test that makes the same call with what it was given, and run in a workspace of hers under her
test harness. A test is kept only when running it raises what she raised. That test is the defect repair works to
(``repair_by_selection.Defect``), so a fault of her own running reaches her repair with no person writing it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.a_call_that_failed import FailedCall
from core.runtime.errors import DegradationRecord, get_degradation_tracker, record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.SelfModification.Faults")

#: Where in a workspace of hers the tests of her faults are written.
FAULT_TESTS = Path("tests") / "her_faults"

#: As long as a test of hers is given to run: as long as one of her gates is (core/agency/her_goal_at_work.py).
_RUN_SECONDS = 2400.0


@dataclass(frozen=True)
class HeardFault:
    """A fault her running raised, the call it was raised in, and when it was first and last heard."""

    call: FailedCall
    subsystem: str
    first_at: float
    last_at: float
    times: int = 1


@dataclass(frozen=True)
class Reproduction:
    """A test written for a fault, and whether running it raised what she raised."""

    fault: HeardFault
    test: str
    reproduced: bool
    output: str


_LOCK = checked_lock("faults_she_can_reproduce.heard")
_HEARD: dict[tuple[str, str, str], HeardFault] = {}
_LISTENING = [False]
#: How many faults have been heard for the first time in this process, and how many of those were last kept on disk.
_FIRST_HEARD = [0]
_KEPT = [0]
#: Called when a fault is heard for the first time: her repair cycle waits on one (self_modification_engine).
_WAKERS: list[Any] = []


def _how_many_are_kept() -> int:
    """As many faults as her degradation tracker keeps records."""
    return int(getattr(get_degradation_tracker(), "_max", 500) or 500)


#: The levels at which a recorder says what it recorded is expected, not a fault (CLAUDE.md: expected backpressure is
#: logged at info). LIVE 16 November 2026 her reaper's NoSuchProcess for a child that exited mid-sweep, recorded at
#: debug, was heard as a fault every boot.
_EXPECTED = frozenset({"debug", "info"})


def hear(record: DegradationRecord) -> None:
    """A degradation listener: keep the call of each fault that can be made again (O(1), under the tracker's rule).

    A record its recorder marked as expected (``_EXPECTED``) is not a fault and is not heard."""
    if str(getattr(record, "severity", "") or "").lower() in _EXPECTED:
        return
    call = getattr(record, "failed_call", None)
    if not isinstance(call, FailedCall):
        _say_why_not_once(call, record)
        return
    with _LOCK:
        known = _HEARD.get(call.fingerprint)
        if known is not None:
            _HEARD[call.fingerprint] = HeardFault(known.call, known.subsystem, known.first_at, record.timestamp,
                                                  known.times + 1)
            return
        if len(_HEARD) >= _how_many_are_kept():
            return
        _HEARD[call.fingerprint] = HeardFault(call, record.subsystem, record.timestamp, record.timestamp)
        _FIRST_HEARD[0] += 1
    logger.info("🩹 A fault she can make again: %s.%s raised %s, from %s (%s:%d).", call.module, call.qualname,
                call.error_type.rsplit(".", 1)[-1], record.subsystem, call.file, call.line)
    with _LOCK:
        wakers = list(_WAKERS)
    for wake in wakers:
        try:
            wake()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("a fault's waker failed: %s", exc)


def when_a_fault_is_first_heard(wake: Any) -> bool:
    """Call ``wake()`` each time a fault is heard for the first time (at most as many wakers as the tracker allows
    listeners); whether it was added."""
    with _LOCK:
        if wake in _WAKERS or len(_WAKERS) >= 8:
            return False
        _WAKERS.append(wake)
        return True


#: Each reason a fault could not be kept, said once: how much of her running can be made again, and what stops the rest.
_SAID_WHY_NOT: set[str] = set()


def _say_why_not_once(why: Any, record: DegradationRecord) -> None:
    reason = str(why or "").strip()
    if not reason:
        return
    with _LOCK:
        if reason in _SAID_WHY_NOT or len(_SAID_WHY_NOT) >= _how_many_are_kept():
            return
        _SAID_WHY_NOT.add(reason)
    logger.info("🩹 Not a fault she can make again: %s (%s from %s).", reason, record.error_type, record.subsystem)


def listen() -> bool:
    """Hear every degradation this process records, after the faults kept from her last run; whether this began it."""
    with _LOCK:
        if _LISTENING[0]:
            return False
        _LISTENING[0] = True
    recall()
    return bool(get_degradation_tracker().add_listener(hear))


def where_heard_faults_are_kept() -> Path:
    """Her state, beside her goal records: a fault heard before a restart is still hers to mend after it."""
    from core.runtime.state_ownership import state_root

    return state_root() / "her_faults" / "heard.json"


def _as_row(fault: HeardFault) -> dict[str, Any]:
    return {**dataclasses.asdict(fault), "call": dataclasses.asdict(fault.call)}


def _from_row(row: Any) -> HeardFault | None:
    try:
        call = dict(row["call"])
        call["keywords"] = tuple((str(key), str(value)) for key, value in call.get("keywords") or ())
        for name in ("positional", "below"):
            call[name] = tuple(str(part) for part in call.get(name) or ())
        return HeardFault(call=FailedCall(**call), subsystem=str(row["subsystem"]), first_at=float(row["first_at"]),
                          last_at=float(row["last_at"]), times=int(row["times"]))
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("a kept fault could not be read back: %s", exc)
        return None


def recall() -> int:
    """The faults kept from her last run, heard again; how many were new to this process."""
    path = where_heard_faults_are_kept()
    try:
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError) as exc:
        record_degradation("faults_she_can_reproduce", exc, severity="warning",
                           action=f"could not read {path}; faults heard before this run are not mended")
        return 0
    added = 0
    with _LOCK:
        for fault in filter(None, (_from_row(row) for row in rows if isinstance(row, dict))):
            if fault.call.fingerprint not in _HEARD and len(_HEARD) < _how_many_are_kept():
                _HEARD[fault.call.fingerprint] = fault
                added += 1
        _KEPT[0] = _FIRST_HEARD[0]
    if added:
        logger.info("🩹 %d fault(s) she heard before this run are hers to mend in it.", added)
    return added


async def keep_what_was_heard() -> bool:
    """Write the faults heard so far to her state, when one was heard for the first time since the last write.

    LIVE 15 November 2026 she heard a fault she could make again (a UnicodeDecodeError in her subprocess gateway), and
    restarted before her loop reached it; what she heard lived only in the process that heard it.
    """
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with _LOCK:
        heard_by = _FIRST_HEARD[0]
        if heard_by == _KEPT[0]:
            return False
        rows = [_as_row(fault) for fault in _HEARD.values()]
    path = where_heard_faults_are_kept()
    gateway = get_file_write_gateway()
    with local_internal_governed_scope("faults_she_can_reproduce.keep", domain="file_write",
                                       receipt_prefix="her-heard-faults",
                                       constraints={"artifact": "aura.her_heard_faults.v1", "operation": "write"}):
        # The atomic writer makes a missing parent itself.
        await gateway.write_text_async(path, json.dumps(rows, indent=1), source="faults_she_can_reproduce")
    with _LOCK:
        _KEPT[0] = max(_KEPT[0], heard_by)
    return True


def faults_heard() -> list[HeardFault]:
    """The faults heard so far that can be made again, the most frequent first."""
    with _LOCK:
        heard = list(_HEARD.values())
    return sorted(heard, key=lambda fault: (-fault.times, fault.first_at))


def forget_for_test() -> None:
    with _LOCK:
        _HEARD.clear()
        _SAID_WHY_NOT.clear()
        _FIRST_HEARD[0] = _KEPT[0] = 0


def name_of(fault: HeardFault) -> str:
    """The test's file name: the function, and a digest of the fault, so the same fault is written once."""
    call = fault.call
    digest = hashlib.sha256(repr(call.fingerprint).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-z0-9]+", "_", call.qualname.lower()).strip("_")[:60] or "call"
    return f"test_fault_{stem}_{digest}.py"


def test_source(fault: HeardFault) -> str:
    """A test that makes the call again and fails if it raises."""
    call = fault.call
    owner, _, rest = call.qualname.partition(".")
    callee = f"{owner}.{rest}" if rest else owner
    when = time.strftime("%d %B %Y %H:%M", time.localtime(fault.first_at))
    said = call.error_message.replace('"""', "'''")
    return (
        f'"""A fault from her own running, made again: {call.module}.{call.qualname} raised {call.error_type}.\n\n'
        f"Heard {fault.times} time(s) from {fault.subsystem}, first at {when}: {said}\n"
        f"Raised at {call.file}:{call.line}. Written by core/self_modification/faults_she_can_reproduce.py.\n"
        '"""\n'
        f"from {call.module} import {owner}\n\n\n"
        f"def {Path(name_of(fault)).stem}():\n"
        f"    {call.call_source(callee)}\n"
    )


def _error_name(call: FailedCall) -> str:
    return call.error_type.rsplit(".", 1)[-1]


async def reproduce(fault: HeardFault, workspace: Path) -> Reproduction:
    """Write the fault's test in ``workspace`` and run it there; reproduced when it raises what she raised."""
    import asyncio

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    folder = Path(workspace) / FAULT_TESTS
    path = folder / name_of(fault)
    test = (FAULT_TESTS / name_of(fault)).as_posix()
    gateway = get_file_write_gateway()
    with local_internal_governed_scope("faults_she_can_reproduce.test", domain="file_write",
                                       receipt_prefix="her-fault-test",
                                       constraints={"artifact": "aura.her_fault_test.v1", "operation": "write"}):
        await gateway.ensure_directory_async(folder, source="faults_she_can_reproduce")
        await gateway.write_text_async(path, test_source(fault), source="faults_she_can_reproduce")
    passed, output = await asyncio.to_thread(_run_the_test, test, Path(workspace))
    raised = not passed and re.search(rf"\b{re.escape(_error_name(fault.call))}\b", output) is not None
    return Reproduction(fault=fault, test=test, reproduced=raised, output=output)


def _run_the_test(test: str, workspace: Path) -> tuple[bool, str]:
    """The test under her harness in her workspace, as her gates run tests (her_goal_at_work)."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    env = {**os.environ, "AURA_LOG_DIR": tempfile.mkdtemp(prefix="her-fault-logs-"),
           "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    argv = ["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test]
    try:
        with local_internal_governed_scope("faults_she_can_reproduce.run", domain="tool_execution",
                                           receipt_prefix="her-fault-run",
                                           constraints={"artifact": "aura.her_fault_run.v1", "operation": "check"}):
            done = get_subprocess_gateway().run(argv, cwd=workspace, env=env, timeout=_RUN_SECONDS,
                                                capture_output=True, source="faults_she_can_reproduce.run",
                                                accelerator_capability="none")
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation("faults_she_can_reproduce", exc, severity="warning",
                           action=f"could not run {test}; the fault is not counted as reproduced")
        return False, f"the test could not be run: {type(exc).__name__}: {exc}"
    return done.returncode == 0, ((done.stdout or "") + (done.stderr or "")).strip()[-3000:]


def as_defect(reproduction: Reproduction) -> Any:
    """The reproduced fault as repair by selection takes a defect: its test, the file it was raised in, its failure."""
    from core.self_modification.repair_by_selection import Defect

    if not reproduction.reproduced:
        raise ValueError("only a fault whose test raises what she raised is a defect")
    return Defect(tests=(reproduction.test,), files=(reproduction.fault.call.file,), seen=reproduction.output)


__all__ = [
    "FAULT_TESTS",
    "HeardFault",
    "Reproduction",
    "as_defect",
    "faults_heard",
    "hear",
    "keep_what_was_heard",
    "listen",
    "name_of",
    "reproduce",
    "recall",
    "test_source",
    "when_a_fault_is_first_heard",
    "where_heard_faults_are_kept",
]
