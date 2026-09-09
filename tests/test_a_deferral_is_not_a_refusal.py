"""A write the governor put off is not a write the governor said no to.

Both arrive as ``approved == False``; the only thing separating them is the
reason string. The ontogeny organ is allowed to EXPLORE by deferring at the
executive admission control point, and the comment justifying that says a
deferral "only costs time" — true where somebody comes back, and false at
every call site that drops what it was holding.

`core/world_model/acg.py` found it first: 108 causal links dropped in one
sampled window, systematically rather than randomly, because the thing
deferring is a policy and will defer again under the same conditions.

LIVE, 2026-09-07: on a fresh boot every episodic write came back
"sync_approved|ontogeny:deferred", so nothing was recorded at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.memory.a_deferral_is_not_a_refusal import DeferredWrites, is_a_deferral

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "reason",
    [
        "sync_approved|ontogeny:deferred",
        "deferred_by_executive",
        "capacity_full",
        "backpressure",
        "not_now",
        "retry_after_s=30",
    ],
)
def test_not_now_is_read_as_not_now(reason: str) -> None:
    assert is_a_deferral(reason)


@pytest.mark.parametrize(
    "reason",
    ["policy_violation", "rejected", "constitutional_denial", "", "unsafe_content"],
)
def test_no_is_read_as_no(reason: str) -> None:
    assert not is_a_deferral(reason)


def test_a_held_write_lands_on_the_next_chance() -> None:
    landed: list[int] = []
    allow = False

    def retry(item: int) -> bool:
        if not allow:
            return False
        landed.append(item)
        return True

    queue = DeferredWrites("test", retry, interval_s=0.0)
    queue.hold(1, "ontogeny:deferred")
    queue.hold(2, "ontogeny:deferred")
    assert queue.replay() == 0
    assert len(queue) == 2

    allow = True
    assert queue.replay() == 2
    assert landed == [1, 2]
    assert len(queue) == 0


def test_the_queue_is_bounded_and_says_when_it_sheds() -> None:
    queue = DeferredWrites("test", lambda _item: False, limit=3, interval_s=0.0)
    for index in range(6):
        queue.hold(index, "deferred")
    assert len(queue) == 3
    assert queue.shed == 3
    assert queue.state()["shed"] == 3


def test_a_replay_that_raises_keeps_the_item() -> None:
    """A queue that can kill its caller is worse than the loss it prevents."""

    def explode(_item: int) -> bool:
        raise RuntimeError("the store is gone")

    queue = DeferredWrites("test", explode, interval_s=0.0)
    queue.hold(1, "deferred")
    assert queue.replay() == 0
    assert len(queue) == 1


def test_a_replayed_write_cannot_recursively_drain_its_own_queue() -> None:
    """The episodic writer offers replay again while replaying an episode."""

    queue = None
    landed = []

    def retry(item: int) -> bool:
        landed.append(item)
        assert queue is not None
        assert queue.replay() == 0
        return True

    queue = DeferredWrites("recursive", retry, interval_s=0.0)
    queue.hold(1, "deferred")
    queue.hold(2, "deferred")
    assert queue.replay() == 2
    assert landed == [1, 2]
    assert queue.state()["queued"] == 0


def test_the_episodic_store_holds_what_it_cannot_write_yet() -> None:
    """The live case: the module that was dropping every episode."""

    source = (ROOT / "core/memory/episodic_memory.py").read_text("utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_hold_if_deferred" in names
    assert "_write_a_held_episode" in names
    assert "deferred_state" in names
    assert "is_a_deferral" in source


def test_a_replayed_episode_carries_an_idempotency_key() -> None:
    """A replay racing the original must write one episode, not two."""

    source = (ROOT / "core/memory/episodic_memory.py").read_text("utf-8")
    start = source.index("def _hold_if_deferred")
    end = source.index("def _write_a_held_episode")
    assert "idempotency_key" in source[start:end]


def test_a_steady_stream_of_deferrals_does_not_flood_the_feed() -> None:
    """A working mechanism must not become the loudest line in the log.

    Live on 2026-09-07 the governor deferred every episodic write, so every
    hold logged and the queue's own line outnumbered everything else. The
    facts worth having — that a queue exists, how deep, how much has landed —
    survive being said periodically, and `state()` carries the exact numbers.
    """

    import logging

    from core.memory.a_deferral_is_not_a_refusal import _SAY_EVERY

    queue = DeferredWrites("test", lambda _item: False, limit=500, interval_s=99.0)
    logger = logging.getLogger("Aura.DeferredWrites")
    records: list[logging.LogRecord] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Catch()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        for index in range(_SAY_EVERY * 2):
            queue.hold(index, "ontogeny:deferred")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    said = [record for record in records if "holding deferred writes" in record.getMessage()]
    # The first, so the queue's existence is never silent, then one per
    # _SAY_EVERY.
    assert len(said) == 3, f"{len(said)} lines for {_SAY_EVERY * 2} holds"
    assert queue.state()["held_total"] == _SAY_EVERY * 2


# ── and the retry is not behind the failure it retries ───────────────────


def test_the_replay_runs_before_the_governor_is_asked():
    """LIVE, 2026-09-08: "holding deferred writes (50 queued, 50 held so far,
    0 landed)". The replay sat below the deferral branch, so it ran only on a
    write the governor had just approved — and the queue only fills when the
    governor is deferring. Nothing could ever land."""
    import inspect

    from core.memory import episodic_memory

    source = inspect.getsource(episodic_memory)
    replay_at = source.index("self._deferred_episodes.replay()")
    approve_at = source.index("approved, governance_decision = self._approve_memory_write(")
    assert replay_at < approve_at, (
        "the replay must run before the approval that decides whether this "
        "write is itself deferred"
    )
    # And only once: a second call below the branch is the old ordering back.
    assert source.count("self._deferred_episodes.replay()") == 1
