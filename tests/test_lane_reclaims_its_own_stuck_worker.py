"""She must be able to recover from a state she created.

The orphan-reclamation scan before a worker spawn deliberately skips this
client's own process — it exists to kill workers left by PREVIOUS incarnations.
So a worker that loaded the model but never finished initializing kept holding
its weights, while the lane that had already given up on it refused to spawn a
replacement for want of the very memory it was holding.

Measured live 2026-07-26: available 17.4GB against a 24GB gate, with roughly
16GB wired to our own unusable worker. Killing the instance by hand dropped
wired memory 21.7GB -> 5.2GB and freed 34.6GB. On Apple Silicon those MLX
allocations do not appear in RSS, which is why the lane could not see what it
was doing to itself.

The reclaim is narrow by construction: it runs only when the spawn is about to
be refused anyway, only against our own process, only one that never became
usable, and only when it is serving no one.
"""
from __future__ import annotations

import re
from pathlib import Path

from tests.source_contract import family_text_at

SOURCE = Path("core/brain/llm/mlx_client.py")


def _reclaim_block() -> str:
    """Just the reclaim — it ends where the pre-existing reclaim *wait* begins."""
    src = family_text_at(SOURCE)
    start = src.index("# The orphan scan above only reaps workers from PREVIOUS")
    return src[start : src.index("# A worker we just killed", start)]


def test_the_reclaim_runs_before_refusing_for_headroom() -> None:
    src = family_text_at(SOURCE)
    reclaim_at = src.index("Reclaiming our own never-initialized worker")
    # CP126 5ce89b9e made the refusal a TYPED exception rather than a
    # RuntimeError carrying a magic substring, because the substring could be
    # matched by accident. The ordering contract this test guards is
    # unchanged; only the constructor's name is.
    refuse_at = src.index("error = ModelLoadAdmissionRefused(memory_block)")
    assert reclaim_at < refuse_at, (
        "reclaiming must be attempted before the spawn is refused"
    )


def test_it_only_touches_a_worker_that_never_became_usable() -> None:
    """A serving worker, or one that initialized, must never be reaped here."""
    block = _reclaim_block()
    assert "not self._init_done" in block
    assert 'int(getattr(self, "_active_generations", 0) or 0) == 0' in block


def test_it_only_touches_our_own_process() -> None:
    block = _reclaim_block()
    assert "stale = self._process" in block
    assert "is_alive" in block, "liveness must be checked before killing"


def test_headroom_is_rechecked_after_the_reclaim() -> None:
    """Freeing the memory is pointless if the refusal uses the stale reading."""
    block = _reclaim_block()
    assert block.count("_memory_pressure_blocks_worker_spawn(self.model_path)") >= 1
    assert "self._process = None" in block
    assert "self._init_done = False" in block


def test_the_deep_solver_lane_still_refuses_instantly() -> None:
    """The optional lane must not start killing workers to get itself loaded."""
    block = _reclaim_block()
    assert "_is_deep_solver_lane()" in block


def test_a_failed_reclaim_does_not_break_the_spawn_path() -> None:
    """Reclaim is opportunistic; its failure must not lose the turn."""
    block = _reclaim_block()
    assert re.search(r"except \([^)]*\) as reclaim_exc", block)
    assert "_record_mlx_degradation" in block
