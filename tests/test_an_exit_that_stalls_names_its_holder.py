"""An exit that stalls after every service terminated names the thread holding it.

2026-09-15: one shutdown took 14 seconds and the next 119, with nothing in
the log for the difference. asyncio.run joins the default executor on its way
out, so one worker thread still busy is a silent two-minute exit.
"""

from __future__ import annotations

import logging
import threading
import time

from core.ops import graceful_shutdown


def test_a_busy_thread_is_named_and_an_idle_one_is_not(caplog):
    release = threading.Event()

    def busy() -> None:
        # Working: a loop that is not waiting on a queue or a condition.
        while not release.is_set():
            time.sleep(0.005)

    worker = threading.Thread(target=busy, name="still-writing-the-ledger", daemon=True)
    worker.start()
    try:
        with caplog.at_level(logging.WARNING, logger=graceful_shutdown.logger.name):
            graceful_shutdown._arm_exit_stall_dump(after_s=0.1)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and "Exit stalled" not in caplog.text:
                time.sleep(0.02)
    finally:
        release.set()
        worker.join(2.0)
    assert "Exit stalled" in caplog.text
    assert "still-writing-the-ledger" in caplog.text
