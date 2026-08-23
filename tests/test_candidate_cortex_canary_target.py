from __future__ import annotations

import threading

from core.runtime.resource_observation import SimulatedResourceObserver
from tools import run_candidate_cortex_canary_target as target


def test_host_sampler_fails_closed_when_observation_is_unavailable() -> None:
    observer = SimulatedResourceObserver()
    observer.configure_memory(
        observation_available=False,
        error="simulated_blind",
    )
    state = {
        "sample_count": 0,
        "min_available_bytes": 2**63 - 1,
        "max_used_percent": 0.0,
        "max_process_rss_bytes": 0,
    }
    stop = threading.Event()
    stop.set()

    target._sample_host(stop, state, observer)

    assert state == {
        "sample_count": 1,
        "min_available_bytes": 0,
        "max_used_percent": 100.0,
        "max_process_rss_bytes": 1024**3,
    }
