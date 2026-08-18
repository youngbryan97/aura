"""A full machine tripped the circuit breaker on a healthy endpoint.

The neural stream's loudest repeat after the fault-severity bug was 128
"Circuit OPEN for Brainstem after N failures. Reason: candidate_worker_not_ready"
plus 128 "Endpoint Brainstem failed validation", on a host that was simply out
of memory.

candidate_worker_not_ready is raised by _ModelLoadAdmissionDeniedError when the
model lane controller CANCELS a spawn for want of headroom. It is the same
condition that produces background_deferred:memory_pressure, reached by a
different route — admission control working, not an endpoint misbehaving.

Counting it as an endpoint failure inverts the recovery: the breaker opens
because the machine is full, and then keeps the endpoint out after the memory
frees up. Backpressure has to be distinguishable from unreliability, or the
system punishes itself for being busy.
"""
from __future__ import annotations

import pytest

from core.brain.llm_health_router import _background_error_is_quiet


@pytest.mark.parametrize(
    "backpressure",
    [
        "candidate_worker_not_ready",
        "background_deferred:memory_pressure",
        "background_deferred:cortex_resident",
        "foreground_busy",
        "client_returned_no_text",
    ],
)
def test_backpressure_is_quiet(backpressure):
    """Being full is not being broken."""
    assert _background_error_is_quiet(backpressure)


@pytest.mark.parametrize(
    "real_fault",
    [
        "worker_died_during_generation",
        "model_returned_garbage",
        "unexpected_protocol_error",
        "",
    ],
)
def test_a_real_fault_is_still_loud(real_fault):
    """A worker that started and then died is an event, not backpressure.

    The value of quieting the one is that the other keeps meaning something.
    """
    assert not _background_error_is_quiet(real_fault)


def test_the_admission_denial_and_the_deferral_agree():
    """Two routes to one condition must not be classified differently.

    candidate_worker_not_ready and background_deferred:memory_pressure are the
    same machine state — the lane refusing to start a worker because there is
    no room. One being loud and the other quiet is how the breaker opened.
    """
    assert _background_error_is_quiet(
        "candidate_worker_not_ready"
    ) == _background_error_is_quiet("background_deferred:memory_pressure")
