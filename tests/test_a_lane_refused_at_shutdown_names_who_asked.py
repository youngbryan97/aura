"""A thread lane that refuses work once shutdown has begun says what asked for it, once.

LIVE 10 November 2026: as her app shut down, the aesthetic journal and the router's history failed to save with
"runtime_shutdown" and nothing more, and no probe off her runtime refused.
"""
from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.unit


def _asks_for_the_lane():
    from core.runtime.executors import submit_blocking_io

    return submit_blocking_io(lambda: None, label="probe")


def test_the_first_refusal_on_a_lane_names_its_caller(monkeypatch, caplog):
    import core.runtime.executors as executors
    import core.runtime.shutdown_coordinator as shutdown

    monkeypatch.setattr(executors, "_REFUSED_AT_SHUTDOWN", set())
    monkeypatch.setattr(shutdown, "is_shutdown_requested", lambda: True)
    with caplog.at_level(logging.INFO, logger="Aura.Executors"):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="runtime_shutdown"):
                _asks_for_the_lane()
    said = [r.getMessage() for r in caplog.records if "refused new work" in r.getMessage()]
    assert len(said) == 1 and "_asks_for_the_lane" in said[0]
