"""The last rung before the runtime kills itself must pull something.

2026-07-29, second Orca Demo attempt:

    🚨 [MEMWATCH] LETHAL ceiling: managed RSS 48237MB ≥ 43008MB.
       Reclaimed (killed=0). Next confirmation aborts.

Nothing was killed, and the process exited. Underneath it sat model workers and
other organs sharing the same generic ``spawn_main`` command line. Matching
that command line first missed the workers; treating every match as killable
then killed stateful non-model children. The gateway role is the authority.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.resilience import memory_watchdog as mw

#: Verbatim from the live process (pid 31863, 18.3GB) on 2026-07-29.
_REAL_WORKER_CMD = (
    "/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/"
    "Versions/3.12/Resources/Python.app/Contents/MacOS/Python -c "
    "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=7, "
    "pipe_handle=9)"
)
#: Same shape, with the checkout resolved here rather than pinned to the
#: machine the incident happened on — the marker under test is the script
#: name, and a fixture that only holds on one laptop proves nothing elsewhere.
_SENTINEL_CMD = (
    f"/opt/homebrew/.../Python {Path(__file__).resolve().parents[1]}/tools/"
    "memory_sentinel.py --pid 31812 --lethal-mb 43008.0 --interval 0.5"
)


class _FakeChild:
    def __init__(self, pid: int, rss_gb: float, cmd: str, *, role: str | None) -> None:
        self.pid = pid
        self._rss = int(rss_gb * (1024**3))
        self._cmd = cmd
        self.name = f"child-{pid}"
        self._alive = True
        self.terminated = False
        if role is not None:
            self._aura_python_process_contract = {
                "source": f"test.{role}",
                "name": self.name,
                "role": role,
                "requested_privileges": (),
                "accelerator_capability": "model" if role == "model_worker" else "none",
                "start_method": "spawn",
            }

    def cmdline(self):
        return self._cmd.split(" ")

    def memory_info(self):
        return type("MI", (), {"rss": self._rss})()

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.terminated = True
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None


@pytest.fixture
def live_tree(monkeypatch):
    children = [
        _FakeChild(31863, 18.3, _REAL_WORKER_CMD, role="model_worker"),
        _FakeChild(32308, 5.0, _REAL_WORKER_CMD, role="model_worker"),
        _FakeChild(32604, 1.6, _REAL_WORKER_CMD, role="coordinator"),
        _FakeChild(24822, 0.03, _SENTINEL_CMD, role=None),
    ]
    monkeypatch.setattr(mw.mp, "active_children", lambda: children)
    monkeypatch.setattr(
        mw,
        "_phys_footprint_mb",
        lambda pid: next(child._rss for child in children if child.pid == pid) / (1024**2),
    )
    monkeypatch.setattr(
        mw,
        "capture_identity",
        lambda process, **_kwargs: SimpleNamespace(
            bound=True,
            describe=lambda: f"pid={process.pid}",
        ),
    )
    monkeypatch.setattr(mw, "assert_owned", lambda *_args, **_kwargs: True)
    return children


def test_a_declared_model_worker_is_recognised_as_killable(live_tree) -> None:
    """The regression itself: killed=0 against a tree full of workers."""
    killed = mw.terminate_heavy_child_workers()
    assert killed >= 1, (
        "no worker was recognised — this is the 'Reclaimed (killed=0)' that "
        "preceded the abort"
    )


def test_the_out_of_process_sentinel_is_never_killed(live_tree) -> None:
    """It guards the thing being reclaimed; it must outlive the reclaim."""
    mw.terminate_heavy_child_workers()
    sentinel = next(c for c in live_tree if "memory_sentinel.py" in c._cmd)
    assert not sentinel.terminated


def test_reclaim_stops_once_the_shortfall_is_covered(live_tree) -> None:
    """Getting under the ceiling should cost one model reload, not the tree."""
    killed = mw.terminate_heavy_child_workers(
        free_at_least_bytes=int(6 * (1024**3))
    )
    assert killed == 1, f"killed {killed} workers to free 6GB"
    biggest = next(c for c in live_tree if c.pid == 31863)
    assert biggest.terminated, "largest-first: the 18.3GB worker goes first"
    assert not any(c.terminated for c in live_tree if c.pid in (32308, 32604))


def test_reclaim_sheds_burstable_worker_before_larger_guaranteed_cortex(
    live_tree, monkeypatch
) -> None:
    """The exact live incident: a 9B background lane must yield before 32B."""

    monkeypatch.setattr(
        mw,
        "_model_worker_reclaim_order",
        lambda process: (3, 0) if process.pid == 31863 else (1, 0),
    )

    killed = mw.terminate_heavy_child_workers(
        free_at_least_bytes=int(4 * (1024**3))
    )

    assert killed == 1
    cortex = next(c for c in live_tree if c.pid == 31863)
    brainstem = next(c for c in live_tree if c.pid == 32308)
    assert not cortex.terminated
    assert brainstem.terminated


def test_reclaim_policy_reads_live_mlx_client_role_and_activity(
    live_tree, monkeypatch
) -> None:
    import core.brain.llm.mlx_client as mlx_client
    from core.runtime.model_runtime_assignment import (
        ModelRuntimeAssignment,
        locator_identity,
    )

    def assignment(path: str, role: str) -> ModelRuntimeAssignment:
        return ModelRuntimeAssignment.issue(
            model_path=path,
            artifact_identity=locator_identity(path),
            artifact_identity_kind="canonical_locator_sha256",
            artifact_identity_exact=False,
            role=role,
            purpose="serve",
            authority_source="test",
        )

    cortex = live_tree[0]
    brainstem = live_tree[1]
    clients = {
        "/models/Aura-32B-fused": SimpleNamespace(
            _process=cortex,
            _active_generations=1,
            _current_gen_future=None,
            runtime_assignment=assignment("/models/Aura-32B-fused", "cortex"),
        ),
        "/models/Qwen3.5-9B-4bit": SimpleNamespace(
            _process=brainstem,
            _active_generations=0,
            _current_gen_future=None,
            runtime_assignment=assignment("/models/Qwen3.5-9B-4bit", "brainstem"),
        ),
    }
    monkeypatch.setattr(mlx_client, "_CLIENTS", clients)

    assert mw._model_worker_reclaim_order(cortex) == (3, 1)
    assert mw._model_worker_reclaim_order(brainstem) == (1, 0)


def test_a_generic_spawned_coordinator_is_never_a_model_reclaim_candidate(live_tree) -> None:
    mw.terminate_heavy_child_workers()

    coordinator = next(child for child in live_tree if child.pid == 32604)
    assert "multiprocessing.spawn" in coordinator._cmd
    assert not coordinator.terminated


def test_pid_identity_is_revalidated_at_the_moment_of_termination(
    live_tree, monkeypatch
) -> None:
    monkeypatch.setattr(mw, "assert_owned", lambda *_args, **_kwargs: False)

    assert mw.terminate_heavy_child_workers() == 0
    assert not any(child.terminated for child in live_tree)


def test_watchdog_asks_for_exactly_its_shortfall(monkeypatch) -> None:
    """The amount comes from the ceiling breach, not from a constant."""
    asked: dict[str, int] = {}

    def _terminator(*, free_at_least_bytes=None):
        asked["bytes"] = free_at_least_bytes
        return 1

    dog = mw.MemoryWatchdog(
        worker_terminator=_terminator,
        gc_collect=lambda: 0,
        ladder_shed=lambda: (0, 0),
        process_exit=lambda code: None,
    )
    sample = type("S", (), {"managed_rss_mb": 48237.0, "swap_used_gb": 0.0})()
    killed = dog._terminate_workers(sample, already_freed=0)

    expected = int((48237.0 - dog.thresholds.hard_mb) * (1024 * 1024))
    assert killed == 1
    assert asked["bytes"] == expected


def test_swap_driven_reclaim_still_sheds_workers(monkeypatch) -> None:
    """RSS under the ceiling does not mean there is nothing to reclaim.

    The hard tier also fires on swap exhaustion. Asking for a shortfall of
    zero bytes there would kill nothing — the same empty rung, wearing the
    budget as a disguise — so with no number to aim at it sheds everything
    eligible, which is what this tier always did.
    """
    asked: list[Any] = []

    def _terminator(*, free_at_least_bytes=None):
        asked.append(free_at_least_bytes)
        return 2

    dog = mw.MemoryWatchdog(
        worker_terminator=_terminator,
        gc_collect=lambda: 0,
        ladder_shed=lambda: (0, 0),
        process_exit=lambda code: None,
    )
    sample = type("S", (), {"managed_rss_mb": 1000.0, "swap_used_gb": 9.9})()

    assert dog._terminate_workers(sample) == 2, "swap pressure must still shed"
    assert asked == [None], "with no RSS breach there is no byte budget to ask for"
