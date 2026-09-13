"""Event-loop stall prevention contracts.

July 6 2026 live forensics caught four distinct multi-second event-loop
stalls (data/error_logs/stalls/): the /health endpoint inside episodic
get_summary, think() inside goal_engine's snapshot SQLite work, the closed
loop's numpy prediction step, and a logger.info whose synchronous sinks
blocked on disk. These tests pin the structural fixes so the class cannot
quietly return.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time


# ---------------------------------------------------------------------------
# Logging transport: emit never blocks, listener owns the sinks
# ---------------------------------------------------------------------------

def test_root_logging_goes_through_queue_handler(tmp_path, monkeypatch):
    import core.observability.logging_config as lc

    # Force a fresh init in this process regardless of prior state.
    monkeypatch.setattr(lc, "_initialised", False)
    monkeypatch.setattr(lc, "_queue_listener", None)
    lc.setup_logging(log_dir=tmp_path)

    root = logging.getLogger()
    queue_handlers = [
        h for h in root.handlers if isinstance(h, lc._DropNewestOnOverflowQueueHandler)
    ]
    assert queue_handlers, "root logger must emit through the non-blocking queue handler"
    # No synchronous stream/file sinks may remain directly on the root.
    direct_sync = [
        h
        for h in root.handlers
        if type(h) is logging.StreamHandler
        or isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert not direct_sync, f"synchronous sinks must live behind the QueueListener: {direct_sync}"
    assert lc._queue_listener is not None


def test_queue_handler_overflow_drops_oldest_never_blocks():
    import core.observability.logging_config as lc

    q: queue.Queue = queue.Queue(maxsize=2)
    handler = lc._DropNewestOnOverflowQueueHandler(q)

    def rec(msg: str) -> logging.LogRecord:
        return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)

    before = lc.get_dropped_log_count()
    start = time.monotonic()
    for i in range(10):
        handler.enqueue(rec(f"m{i}"))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, "enqueue must never block, even at overflow"
    assert lc.get_dropped_log_count() > before, "drops must be counted, not silent"
    # The newest record survives; the oldest were sacrificed.
    kept = [q.get_nowait().getMessage() for _ in range(q.qsize())]
    assert "m9" in kept


def test_sqlite_log_handler_emit_is_nonblocking(tmp_path):
    from core.utils.aura_logging import SQLiteMemoryHandler

    db_path = tmp_path / "logs.db"
    handler = SQLiteMemoryHandler(db_path=str(db_path))
    assert not db_path.exists(), "constructor performed SQLite I/O on the caller thread"
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello world", None, None)

    start = time.monotonic()
    for _ in range(500):
        handler.emit(record)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "emit must be queue-put speed, not per-record fsync"

    # The writer thread persists the rows shortly after.
    deadline = time.monotonic() + 5.0
    count = 0
    while time.monotonic() < deadline:
        conn = sqlite3.connect(str(db_path))
        try:
            try:
                count = conn.execute("SELECT COUNT(*) FROM system_logs").fetchone()[0]
            except sqlite3.OperationalError:
                count = 0
        finally:
            conn.close()
        if count >= 500 - handler.dropped:
            break
        time.sleep(0.05)
    assert count > 0, "writer thread must persist queued log rows"
    handler.close()


def test_sqlite_log_handler_defaults_to_typed_aura_root(tmp_path, monkeypatch):
    from core.utils.aura_logging import SQLiteMemoryHandler

    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("AURA_ROOT", str(runtime_root))
    handler = SQLiteMemoryHandler()
    try:
        assert handler.db_path == runtime_root / "data" / "aura_memory.db"
        assert not handler.db_path.exists()
    finally:
        handler.close()


# ---------------------------------------------------------------------------
# Goal engine: conversation hot path reads a cache, never SQLite
# ---------------------------------------------------------------------------

def _fresh_goal_engine(tmp_path):
    from core.goals.goal_engine import GoalEngine

    return GoalEngine(db_path=str(tmp_path / "goals.db"))


def test_goal_context_block_uses_cached_snapshot(tmp_path, monkeypatch):
    engine = _fresh_goal_engine(tmp_path)

    calls = {"n": 0}
    real_build = engine.build_snapshot

    def counting_build(*args, **kwargs):
        calls["n"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(engine, "build_snapshot", counting_build)

    engine.get_context_block(objective="first")
    first_calls = calls["n"]
    assert first_calls == 1, "cold call primes the cache once"

    for _ in range(5):
        engine.get_context_block(objective="again")
    assert calls["n"] == first_calls, "warm calls must not touch build_snapshot inline"


def test_goal_context_block_stale_cache_refreshes_in_background(tmp_path):
    engine = _fresh_goal_engine(tmp_path)
    engine.get_context_block()  # prime
    engine._snapshot_cache_at = 0.0  # force stale

    blocking = threading.Event()
    entered = threading.Event()
    real_build = engine.build_snapshot

    def slow_build(*args, **kwargs):
        entered.set()
        blocking.wait(timeout=5)
        return real_build(*args, **kwargs)

    engine.build_snapshot = slow_build  # type: ignore[method-assign]
    try:
        start = time.monotonic()
        engine.get_context_block()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, "stale cache must serve immediately (refresh happens off-thread)"
        assert entered.wait(timeout=2), "a background refresh must have been scheduled"
    finally:
        blocking.set()
        engine.build_snapshot = real_build  # type: ignore[method-assign]
        deadline = time.monotonic() + 3.0
        while engine._snapshot_refresh_inflight and time.monotonic() < deadline:
            time.sleep(0.02)


def test_goal_mutation_expires_snapshot_cache(tmp_path):
    engine = _fresh_goal_engine(tmp_path)
    engine.get_context_block()  # prime
    assert engine._snapshot_cache_at > 0.0

    engine._upsert_goal(
        goal_id="g-stall-test",
        name="cache invalidation goal",
        objective="prove mutations expire the snapshot cache",
        status="queued",
    )
    assert engine._snapshot_cache_at == 0.0, "goal writes must expire the context cache"


# ---------------------------------------------------------------------------
# Episodic memory: telemetry summary is TTL-cached
# ---------------------------------------------------------------------------

def test_episodic_summary_cached(tmp_path, monkeypatch):
    from core.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory(db_path=str(tmp_path / "episodes.db"))
    calls = {"n": 0}
    real_summary = mem.get_summary

    def counting_summary():
        calls["n"] += 1
        return real_summary()

    monkeypatch.setattr(mem, "get_summary", counting_summary)

    first = mem.get_summary_cached()
    for _ in range(10):
        again = mem.get_summary_cached()
    assert calls["n"] == 1, "TTL window must serve the cache"
    assert first == again

    mem._summary_cache_at = 0.0  # expire
    mem.get_summary_cached()
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Closed loop: prediction step must not run inline on the event loop
# ---------------------------------------------------------------------------

def test_prediction_observe_and_update_offloaded():
    """The predictor runs in a worker thread, wherever in the class it is called.

    This used to read the source of ``_prediction_loop`` and look for the two
    words. Splitting the loop body into ``step()`` — same call, same thread,
    one function further down — failed it, which is a test reporting on where
    the code lives rather than on what it does. Ask the syntax tree instead:
    every call to the predictor anywhere in the class has to be an argument to
    ``to_thread``, and none of them may be awaited inline.
    """
    import ast
    import inspect
    import textwrap

    from core.consciousness import closed_loop

    tree = ast.parse(textwrap.dedent(inspect.getsource(closed_loop.ClosedCausalLoop)))

    def _named(node: ast.AST) -> str:
        while isinstance(node, ast.Attribute):
            node = node.value
        return node.id if isinstance(node, ast.Name) else ""

    def _reads(node: ast.AST) -> str:
        return node.attr if isinstance(node, ast.Attribute) else _named(node)

    offloaded: list[ast.AST] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if _reads(call.func) != "to_thread":
            continue
        offloaded.extend(call.args)

    handed_off = {_reads(arg) for arg in offloaded}
    assert "observe_and_update" in handed_off, (
        "the numpy prediction step must be offloaded from the event loop "
        "(observed live as a 6.0s stall); to_thread is handed "
        f"{sorted(handed_off)}"
    )

    inline = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and _reads(call.func) == "observe_and_update"
        and call not in offloaded
    ]
    assert not inline, (
        f"{len(inline)} call(s) to the predictor run on the event loop; "
        "every one of them has to go through to_thread"
    )


# ---------------------------------------------------------------------------
# Liquid substrate: telemetry reads must not block on a busy substrate
# ---------------------------------------------------------------------------

def test_substrate_telemetry_survives_a_held_lock():
    """Observed live (Jul 7): the event loop froze 5.7s inside
    _state_snapshot when a background substrate thread held sync_lock
    through a weight-cache rebuild. Telemetry readers now use the
    non-blocking snapshot and serve the last published state instead."""
    from core.consciousness.liquid_substrate import LiquidSubstrate

    substrate = LiquidSubstrate()
    substrate._state_snapshot()  # publish one snapshot

    substrate.sync_lock.acquire()  # simulate a busy substrate thread
    try:
        start = time.monotonic()
        status = substrate.get_status()
        mood = substrate.get_mood()
        summary = substrate.get_summary()
        elapsed = time.monotonic() - start
    finally:
        substrate.sync_lock.release()

    assert elapsed < 1.0, f"telemetry blocked {elapsed:.2f}s on a held substrate lock"
    assert isinstance(status, dict) and "mood" in status
    assert isinstance(mood, str) and mood
    assert "Mood" in summary


def test_substrate_nowait_prefers_fresh_lock_when_free():
    from core.consciousness.liquid_substrate import LiquidSubstrate

    substrate = LiquidSubstrate()
    snap = substrate._state_snapshot_nowait()
    assert "x" in snap and "snapshot_age_s" in snap
    # With the lock free, the published cache is refreshed.
    assert substrate._last_published_snapshot is not None
