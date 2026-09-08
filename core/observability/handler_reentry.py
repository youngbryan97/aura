"""Stop a log record written inside a log handler from dispatching in place.

A logging handler holds its own lock for the whole of `emit`. A handler that
logs, or that calls into another handler, therefore asks for a second lock
while holding the first — and two handlers that each do it in the other's
direction deadlock the process. That is not hypothetical: on 2026-09-07 the
live runtime wedged with eleven threads blocked in `logging.Handler.acquire`,
the port still listening and every endpoint returning nothing. `TerminalMonitor`
logged a sepsis warning from inside its own `emit`, and `OmniLogHandler` called
the terminal monitor from inside its own. Each held one lock and wanted the
other.

Fixing the two callers would leave the shape in place for the next handler. The
guard is at the one funnel every record passes through instead: while a thread
is dispatching a record, a further record from that thread is deferred, and the
deferred records go out after the outermost dispatch returns — outside every
handler lock. Nothing is dropped while there is room to hold it, and what is
dropped is counted.
"""

from __future__ import annotations
from core.runtime.lockdep import checked_lock

import logging
import threading
from typing import Any

# A handler that logs once per record can defer one record per dispatch. A
# handler that logs in a loop is a defect, and the cap is what keeps that
# defect from becoming an allocation failure instead of a log line.
MAX_DEFERRED = 256

# The replay is itself dispatch, so a deferred record may defer again. The
# bound stops a handler that logs about the record it was given from looping.
MAX_REPLAY_ROUNDS = 8

_local = threading.local()
_original_call_handlers: Any = None
_dropped = 0
_deferred_total = 0
_drop_lock = checked_lock("core.observability.handler_reentry._drop_lock")


def _state() -> Any:
    if not hasattr(_local, "depth"):
        _local.depth = 0
        _local.deferred = []
    return _local


def statistics() -> dict[str, int]:
    """What the guard has had to do, for the health surface to report."""
    with _drop_lock:
        return {"deferred": _deferred_total, "dropped": _dropped}


def in_dispatch() -> bool:
    """True when this thread is inside a logging handler."""
    return bool(getattr(_local, "depth", 0))


def _guarded_call_handlers(logger: logging.Logger, record: logging.LogRecord) -> None:
    global _dropped, _deferred_total
    state = _state()
    if state.depth:
        if len(state.deferred) < MAX_DEFERRED:
            state.deferred.append((logger, record))
            with _drop_lock:
                _deferred_total += 1
        else:
            with _drop_lock:
                _dropped += 1
        return

    state.depth = 1
    rounds = 0
    try:
        _original_call_handlers(logger, record)
    finally:
        state.depth = 0
        pending, state.deferred = state.deferred, []

    while pending and rounds < MAX_REPLAY_ROUNDS:
        rounds += 1
        batch, pending = pending, []
        for deferred_logger, deferred_record in batch:
            state.depth = 1
            try:
                _original_call_handlers(deferred_logger, deferred_record)
            finally:
                state.depth = 0
                pending.extend(state.deferred)
                state.deferred = []
    if pending:
        with _drop_lock:
            _dropped += len(pending)


def install() -> bool:
    """Wrap `Logger.callHandlers` once. Returns whether this call installed it."""
    global _original_call_handlers
    if _original_call_handlers is not None:
        return False
    _original_call_handlers = logging.Logger.callHandlers
    logging.Logger.callHandlers = _guarded_call_handlers  # type: ignore[method-assign]
    return True


def is_installed() -> bool:
    return _original_call_handlers is not None


def uninstall() -> None:
    """Only for tests that need to observe the unguarded behaviour."""
    global _original_call_handlers
    if _original_call_handlers is None:
        return
    logging.Logger.callHandlers = _original_call_handlers  # type: ignore[method-assign]
    _original_call_handlers = None
