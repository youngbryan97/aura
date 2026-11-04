"""Shedding memory waits off the event loop.

On her day run of 3 November the loop stalled three times, 5-6 s each, "the loop thread accrued no CPU: blocked". The
stall dump shows the metabolic coordinator's coroutine calling the resource governor's eviction, which asked her model's
worker to clear its cache and waited for the answer on the loop thread.
"""
from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


def test_the_coordinator_evicts_on_a_worker_thread():
    from core.coordinators.metabolic_coordinator import MetabolicCoordinator

    source = inspect.getsource(MetabolicCoordinator._process_cycle_inner)
    assert "await asyncio.to_thread(gov.execute_eviction, snap.eviction_tier)" in source
    assert "\\n                        gov.execute_eviction(" not in source
