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
