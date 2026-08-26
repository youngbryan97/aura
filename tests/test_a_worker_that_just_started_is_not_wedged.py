"""A worker that has only just been spawned is neither wedged nor idle.

LIVE 2026-08-26: spawn, "Loading model", "Model loaded", force-killed,
respawn — five times over, while every caller that needed her writing was
told "worker_not_alive" and the runtime's own health reported the lane ready.

The guard against this exists and is well argued: killing a worker mid-load
was a doom loop once already (2026-07-15, 216s per turn, zero real cortex
answers for an hour). It reads lane bookkeeping, which is written when warmup
BEGINS — and there is a window after the process exists where none of it is
true yet. In that window a running worker reads as idle-but-running, which is
the wedged case, and gets killed.

Process creation time cannot be absent for a process that exists.
"""
from __future__ import annotations

import os
import time

from core.brain.inference_gate import InferenceGate, _worker_process_started_at


class _Client:
    """A worker with no lane bookkeeping written yet — the window."""

    def __init__(self, pid: int):
        self._process = type("P", (), {"pid": pid})()
        self._warmup_in_flight = False
        self._lane_state = ""
        self._lane_transition_at = 0.0


def test_a_process_that_exists_has_a_start_time():
    started = _worker_process_started_at(_Client(os.getpid()))
    if not started:
        # Some sandboxes refuse process introspection. The decision below is
        # what this file is about; reading a pid is the platform's business.
        return
    assert time.time() - started >= 0


def test_a_client_with_no_process_reads_zero_rather_than_raising():
    assert _worker_process_started_at(_Client(0)) == 0.0
    assert _worker_process_started_at(None) == 0.0


def test_a_freshly_spawned_worker_is_treated_as_loading(monkeypatch):
    """This is the window that was missing: running, no warmup flag, no lane
    state, no transition time."""
    from core.brain import inference_gate

    client = _Client(1234)
    assert not client._warmup_in_flight
    assert client._lane_transition_at == 0.0
    monkeypatch.setattr(
        inference_gate, "_worker_process_started_at", lambda _client: time.time()
    )
    assert InferenceGate._cortex_worker_is_legitimately_loading(client)


def test_a_worker_older_than_the_deadline_is_no_longer_protected_by_its_age(monkeypatch):
    """Age is what makes a new worker safe, not what makes any worker safe."""
    from core.brain import inference_gate

    client = _Client(1234)
    monkeypatch.setattr(
        inference_gate,
        "_worker_process_started_at",
        lambda _client: time.time() - 100000.0,
    )
    assert not InferenceGate._cortex_worker_is_legitimately_loading(client)


def test_a_worker_still_warming_within_the_deadline_is_unchanged():
    client = _Client(os.getpid())
    client._warmup_in_flight = True
    client._lane_transition_at = time.time()
    assert InferenceGate._cortex_worker_is_legitimately_loading(client)


def test_the_deadline_still_applies_to_a_worker_that_cannot_be_dated():
    """Nothing is loosened for a worker whose process cannot be read: it falls
    back to the lane bookkeeping exactly as before."""
    client = _Client(0)
    client._warmup_in_flight = True
    client._lane_transition_at = time.time() - 100000.0
    assert not InferenceGate._cortex_worker_is_legitimately_loading(client)
