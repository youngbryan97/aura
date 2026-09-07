"""Two logging handlers that call into each other must not deadlock.

The live runtime wedged on 2026-09-07 with eleven threads blocked in
`logging.Handler.acquire`, the process alive and the port listening. The cause
was an ordinary ABBA: `TerminalMonitor.emit` logged a sepsis warning, which
reached `OmniLogHandler`; `OmniLogHandler.emit` called the terminal monitor.
Each thread held one handler lock and waited on the other's.

These tests hold the general fix — a record written inside a handler is
deferred and dispatched after the outermost handler returns — and they fail
without it by hanging, which is why every one of them is bounded by a clock.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from core.observability import handler_reentry


class _CrossCallingHandler(logging.Handler):
    """A handler that logs from inside emit, which is the shape that wedges."""

    def __init__(self, partner_logger: str, tag: str) -> None:
        super().__init__()
        self.partner_logger = partner_logger
        self.tag = tag
        self.seen: list[str] = []
        self.entered = threading.Event()
        self.hold = 0.05

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "guard_probe_reply", False):
            self.seen.append(record.getMessage())
            return
        self.entered.set()
        # Widen the window the way real work does, so the interleaving that
        # deadlocks is reached rather than hoped for.
        time.sleep(self.hold)
        logging.getLogger(self.partner_logger).warning(
            "%s reply", self.tag, extra={"guard_probe_reply": True}
        )
        self.seen.append(record.getMessage())


@pytest.fixture
def guarded_root():
    installed = handler_reentry.install()
    root = logging.getLogger()
    previous = list(root.handlers)
    previous_level = root.level
    for handler in previous:
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    yield root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in previous:
        root.addHandler(handler)
    root.setLevel(previous_level)
    if installed:
        handler_reentry.uninstall()


def test_two_handlers_that_call_each_other_do_not_deadlock(guarded_root) -> None:
    first = _CrossCallingHandler("probe.second", "first")
    second = _CrossCallingHandler("probe.first", "second")
    guarded_root.addHandler(first)
    guarded_root.addHandler(second)

    done = threading.Event()
    errors: list[BaseException] = []

    def drive(name: str) -> None:
        try:
            for index in range(20):
                logging.getLogger(name).error("%s %d", name, index)
        except BaseException as error:  # noqa: BLE001 - the thread must report
            errors.append(error)

    threads = [
        threading.Thread(target=drive, args=("probe.first",), daemon=True),
        threading.Thread(target=drive, args=("probe.second",), daemon=True),
    ]
    for thread in threads:
        thread.start()

    def wait() -> None:
        for thread in threads:
            thread.join(timeout=20)
        done.set()

    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(timeout=25)

    assert done.is_set(), "the handlers deadlocked: no record reached the sink"
    assert not errors, errors
    assert first.seen and second.seen


def test_a_record_written_inside_a_handler_still_reaches_the_sinks(guarded_root) -> None:
    """Deferring is not dropping."""
    partner = _CrossCallingHandler("probe.unused", "partner")
    partner.hold = 0.0
    guarded_root.addHandler(partner)
    before = handler_reentry.statistics()["deferred"]

    logging.getLogger("probe.origin").error("origin")

    assert "origin" in partner.seen
    assert "partner reply" in partner.seen, "the deferred record never dispatched"
    assert handler_reentry.statistics()["deferred"] > before


def test_the_guard_reports_what_it_had_to_drop(guarded_root) -> None:
    class _Runaway(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if getattr(record, "runaway", False):
                return
            for index in range(handler_reentry.MAX_DEFERRED + 40):
                logging.getLogger("probe.runaway").error(
                    "flood %d", index, extra={"runaway": True}
                )

    guarded_root.addHandler(_Runaway())
    before = handler_reentry.statistics()["dropped"]
    logging.getLogger("probe.runaway").error("start")
    assert handler_reentry.statistics()["dropped"] > before


def test_in_dispatch_is_false_outside_a_handler(guarded_root) -> None:
    assert handler_reentry.in_dispatch() is False


def test_setup_logging_installs_the_guard() -> None:
    """A guard nobody installs is a guard that does not exist."""
    source = (
        __import__("pathlib").Path("core/observability/logging_config.py").read_text()
    )
    assert "handler_reentry.install()" in source
