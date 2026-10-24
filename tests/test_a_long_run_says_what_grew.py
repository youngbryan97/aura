"""A long run's process says what it holds and what grew, when asked to.

The carrier's workers grew by 6.5 GB in an hour of scoring cuts on 23 October,
and nothing said what grew.
"""

from __future__ import annotations

import collections
import threading

from core.subject.memory_trace import MEMORY_TRACE_ENV, describe_growth, start_memory_trace


def test_unasked_it_does_nothing(monkeypatch):
    monkeypatch.delenv(MEMORY_TRACE_ENV, raising=False)
    assert start_memory_trace(lambda line: None) is None


def test_asked_it_logs_what_it_holds(monkeypatch):
    monkeypatch.setenv(MEMORY_TRACE_ENV, "60")
    lines, said = [], threading.Event()

    def log(line: str) -> None:
        lines.append(line)
        said.set()

    thread = start_memory_trace(log)
    assert said.wait(30.0)
    thread.stop.set()
    assert lines[0].startswith("[memory]") and "GB resident" in lines[0]


def test_growth_names_the_kinds_that_grew_most():
    before = collections.Counter({"dict": 10, "list": 5, "Frame": 1})
    after = collections.Counter({"dict": 110, "list": 6, "Frame": 1, "Anchor": 3})
    assert describe_growth(before, after, top=2) == "dict +100, Anchor +3"
    assert describe_growth(after, after, top=3) == "nothing grew"


def test_kinds_are_counted_by_name():
    counted = __import__("core.subject.memory_trace", fromlist=["count_kinds"]).count_kinds()
    assert counted["dict"] > 0 and counted["function"] > 0


def test_from_a_run_the_count_runs_on_the_run_s_own_thread(monkeypatch):
    """Never on the trace's thread: its list of every object once broke a tuple the run was building."""
    import asyncio

    import core.subject.memory_trace as trace

    monkeypatch.setenv(MEMORY_TRACE_ENV, str(0.05 / 60))
    counted_on: list[int] = []
    original = trace.count_kinds

    def counting():
        counted_on.append(threading.get_ident())
        return original()

    monkeypatch.setattr(trace, "count_kinds", counting)
    lines: list[str] = []

    async def run():
        thread = start_memory_trace(lines.append)
        for _ in range(40):
            await asyncio.sleep(0.02)
            if len(lines) >= 3:
                break
        thread.stop.set()
        return threading.get_ident()

    own = asyncio.run(run())
    assert len(lines) >= 3 and "grew:" in lines[-1]
    assert counted_on and set(counted_on) == {own}


def test_with_no_run_it_reports_resident_memory_only(monkeypatch):
    import core.subject.memory_trace as trace

    monkeypatch.setenv(MEMORY_TRACE_ENV, str(0.05 / 60))
    monkeypatch.setattr(trace, "count_kinds", lambda: (_ for _ in ()).throw(AssertionError("counted off the run")))
    lines, said = [], threading.Event()

    def log(line: str) -> None:
        lines.append(line)
        if len(lines) >= 2:
            said.set()

    thread = start_memory_trace(log)
    assert said.wait(10.0)
    thread.stop.set()
    assert all("grew" not in line for line in lines)
