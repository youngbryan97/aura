"""Our deadline running out is not the endpoint's failure.

A caller timeout tripped the local circuit, so every short-budget internal
call knocked the shared lane out for everybody — and this router already says
why that is wrong, a few hundred lines above the place it did it: "Hitting it
says nothing about the worker's health; it says this turn ran out of time."

LIVE 2026-08-26: her move decisions were given four seconds for a
nine-hundred-token prompt, timed out, tripped Cortex, and the next decision
found "no endpoints matched routing plan for tier 'primary'" and came back
empty. She played whole games without a thought reaching her, and the lane
the person was talking to went with it.
"""
from __future__ import annotations

import inspect
import time

from core.brain.llm_health_router import _worker_still_healthy


class _Client:
    def __init__(self, alive: bool, beat_age_s: float):
        self._alive = alive
        self._last_heartbeat = time.time() - beat_age_s if beat_age_s >= 0 else 0.0

    def is_alive(self):
        return self._alive


class _Endpoint:
    def __init__(self, client):
        self.client = client


def test_a_live_worker_that_beat_recently_is_healthy():
    assert _worker_still_healthy(_Endpoint(_Client(True, 1.0)))


def test_a_dead_worker_is_not():
    assert not _worker_still_healthy(_Endpoint(_Client(False, 1.0)))


def test_a_worker_that_stopped_beating_is_not():
    assert not _worker_still_healthy(_Endpoint(_Client(True, 600.0)))


def test_a_worker_that_never_beat_is_not():
    assert not _worker_still_healthy(_Endpoint(_Client(True, -1)))


def test_an_endpoint_that_cannot_answer_for_itself_is_not_assumed_healthy():
    """Unknowable counts as unhealthy, so a caller timeout on an endpoint with
    no client still trips the circuit."""
    assert not _worker_still_healthy(_Endpoint(None))
    assert not _worker_still_healthy(object())


def test_the_timeout_path_asks_before_tripping():
    from core.brain import llm_health_router

    source = inspect.getsource(llm_health_router)
    where = source.index("_worker_still_healthy(ep)")
    block = source[where - 200 : where + 500]
    assert "ep.is_local and _worker_still_healthy(ep)" in block
    # A worker that is genuinely wedged still trips.
    assert "elif ep.is_local:" in block
    assert "ep.trip_temporarily(last_error)" in block


def test_empty_text_from_a_healthy_worker_does_not_open_the_circuit():
    """A cancelled generation returns no text, and no text opened the circuit:
    "Circuit OPEN for Cortex on transient runtime failure. Reason:
    client_returned_no_text". The next request then found no endpoint for the
    primary tier and came back empty too, which opened it again — a loop fed
    entirely by our own deadlines.
    """
    from core.brain import llm_health_router

    source = inspect.getsource(llm_health_router)
    where = source.index('ep.trip_temporarily("client_returned_no_text")')
    block = source[where - 1600 : where + 200]
    assert "ep.is_local and _worker_still_healthy(ep)" in block
    # A worker that is not alive still trips.
    assert "elif ep.is_local:" in block
