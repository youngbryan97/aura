"""A second process may not load a second cortex beside the first.

core/runtime/lease.py elects a leader across processes, file-backed, with a
liveness check on the holder's pid and boot id. Its ``is_leader`` docstring
names "launching a model" as the canonical thing to gate — and ``is_leader``
had no caller anywhere but the invariant that checks it. The admission plane
that does gate model loads is the in-process control plane: it weighs
priority, fairness and declared footprint, and cannot see another Aura at
all. Single residence was observed, not enforced.

That is the incident core/runtime/oom_policy.py opens with: "a duplicate 32B
load that doubled memory and took the wedged runtime with it".
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import _another_live_runtime_blocks_worker_spawn
from core.runtime import lease


@pytest.fixture(autouse=True)
def _clean_leases(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_RUNTIME_LEASE_DIR", str(tmp_path))
    lease.reset_leases_for_test()
    yield
    lease.reset_leases_for_test()


CORTEX = "/models/Qwen3.8-27B-cortex-4bit"
SMALL = "/models/Qwen2.5-1.5B-Instruct-4bit"


def test_a_light_lane_is_never_blocked_by_the_lease():
    # The guard is about a second 20GB resident, not about every model.
    assert _another_live_runtime_blocks_worker_spawn(SMALL) is None


def test_no_election_running_does_not_block_anything():
    # Most callers of this path are tools and tests, where nobody runs an
    # election. Refusing there would be an outage manufactured from
    # bookkeeping.
    assert _another_live_runtime_blocks_worker_spawn(CORTEX) is None


def test_holding_the_lease_ourselves_does_not_block_us():
    elector = lease.get_elector(lease.RUNTIME_LEASE)
    assert elector is not None
    assert _another_live_runtime_blocks_worker_spawn(CORTEX) is None


def test_a_live_holder_in_another_process_blocks_a_heavy_spawn(monkeypatch):
    lease.get_elector(lease.RUNTIME_LEASE)
    # The provable condition, and the only one this refuses on: another
    # process, alive, holding an unexpired lease.
    monkeypatch.setattr(lease, "should_act_as_singleton", lambda _name: False)
    assert (
        _another_live_runtime_blocks_worker_spawn(CORTEX)
        == "another_live_runtime_holds_the_lease"
    )


def test_an_unreadable_lease_fails_open(monkeypatch):
    # Unlike the memory probe beside it. An unreadable lease is not evidence
    # of a second runtime.
    def _raises(_name):
        raise OSError("lease directory is gone")

    monkeypatch.setattr(lease, "should_act_as_singleton", _raises)
    assert _another_live_runtime_blocks_worker_spawn(CORTEX) is None


def test_the_spawn_gate_consults_it():
    import inspect

    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client)
    # A guard nothing calls is the state this fixes.
    assert source.count("_another_live_runtime_blocks_worker_spawn") >= 2
