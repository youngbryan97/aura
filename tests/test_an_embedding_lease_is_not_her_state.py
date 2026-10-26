"""The embedding engine and the leases on it belong to the process, not to her.

Seed 73's carrier built a bank of anchors for an hour, and all three workers
died the first time they enumerated module state after restoring one: the fork
had carried the shared embedding runtime, the coordinator's list of open leases
was written over the worker's, and a lease the worker held then raised
RuntimeError when `hasattr` asked whether it had a `__dict__` (26 October).
"""

from __future__ import annotations

import copy
import sys
import types

import pytest

from core.memory.embedding_runtime import SharedEmbeddingLease, SharedEmbeddingRuntime
from core.subject.copies import _is_process_furniture
from core.subject.snapshot import _module_holdings

pytestmark = pytest.mark.unit

PROBE = "core._embedding_lease_probe"


class _Engine:
    def __init__(self) -> None:
        self.dimension = 8

    def close(self) -> None:
        return None


def _dead_lease() -> SharedEmbeddingLease:
    runtime = SharedEmbeddingRuntime(_Engine)
    lease = runtime.acquire("probe")
    runtime._owners.clear()  # what a restore of another process's runtime did
    return lease


def test_a_live_lease_still_answers_for_its_engine() -> None:
    lease = SharedEmbeddingRuntime(_Engine).acquire("probe")
    assert lease.dimension == 8


def test_a_dead_lease_answers_questions_about_itself() -> None:
    lease = _dead_lease()
    assert not hasattr(lease, "__dict__")
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = lease.dimension


def test_a_copy_of_a_lease_is_the_lease() -> None:
    lease = SharedEmbeddingRuntime(_Engine).acquire("probe")
    assert copy.copy(lease) is lease
    assert copy.deepcopy({"lease": lease})["lease"] is lease
    dead = _dead_lease()
    assert copy.deepcopy(dead) is dead


def test_the_runtime_and_its_leases_are_furniture() -> None:
    runtime = SharedEmbeddingRuntime(_Engine)
    assert _is_process_furniture(runtime)
    assert _is_process_furniture(runtime.acquire("probe"))


def test_module_state_with_a_dead_lease_is_still_enumerated(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(PROBE)
    module._LEASE = _dead_lease()
    module._RUNTIME = SharedEmbeddingRuntime(_Engine)
    module._KEPT = {"lease": module._LEASE}
    monkeypatch.setitem(sys.modules, PROBE, module)
    holdings = _module_holdings()
    assert f"{PROBE}:_LEASE" not in holdings
    assert f"{PROBE}:_RUNTIME" not in holdings
    assert f"{PROBE}:_KEPT" not in holdings
