"""A starved event loop says whether its own process or the host starved it (third day run, 6 November 2026).

"Event loop starved, not blocked ... The host is not scheduling it." was said 30 times with the Mac 77% idle and a load
average under 3 across its cores.
"""
from __future__ import annotations

from core.utils import concurrency


def test_her_own_process_busy_on_a_quiet_host(monkeypatch):
    monkeypatch.setattr(concurrency.os, "getloadavg", lambda: (2.6, 2.8, 2.8))
    monkeypatch.setattr(concurrency.os, "cpu_count", lambda: 16)
    said = concurrency._who_starved_the_loop(1.3)
    assert said.startswith("Its own process held it")
    assert "1.30 cores" in said and "0.16 per core" in said


def test_a_full_run_queue_is_the_host(monkeypatch):
    monkeypatch.setattr(concurrency.os, "getloadavg", lambda: (34.0, 30.0, 28.0))
    monkeypatch.setattr(concurrency.os, "cpu_count", lambda: 18)
    assert concurrency._who_starved_the_loop(0.2).startswith("The host is not scheduling it")


def test_nothing_busy_says_so(monkeypatch):
    monkeypatch.setattr(concurrency.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    monkeypatch.setattr(concurrency.os, "cpu_count", lambda: 16)
    assert concurrency._who_starved_the_loop(0.05).startswith("Neither the host nor this process was busy")


def test_the_threads_that_ran_while_she_waited_are_named():
    """14 November 2026: 10.49 s of lag on 7% of a core while her other threads used 1.13 cores, and nothing said
    which thread. The CPU each thread spent over the window says it, where a stack taken afterwards may not."""
    import sys
    import threading
    import time

    from core.runtime.thread_cpu import every_thread_cpu, who_ran_meanwhile

    if sys.platform not in ("darwin", "linux"):
        return
    stop = time.monotonic() + 0.8

    def holds_the_interpreter():
        while time.monotonic() < stop:
            sum(range(1000))

    before = every_thread_cpu()
    began = time.monotonic()
    busy = threading.Thread(target=holds_the_interpreter, name="a busy worker")
    busy.start()
    time.sleep(0.4)
    said = who_ran_meanwhile(before, time.monotonic() - began, leave_out=threading.get_ident())
    busy.join()

    assert said.startswith("Ran meanwhile: a busy worker "), said
    assert "holds_the_interpreter" in said


def test_the_names_reach_the_starvation_line(monkeypatch):
    monkeypatch.setattr(concurrency.os, "getloadavg", lambda: (2.6, 2.8, 2.8))
    monkeypatch.setattr(concurrency.os, "cpu_count", lambda: 16)
    said = concurrency._who_starved_the_loop(1.3, "Ran meanwhile: mlx-reader 96% of a core.")
    assert said.startswith("Its own process held it") and said.endswith("Ran meanwhile: mlx-reader 96% of a core.")
