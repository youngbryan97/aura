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
