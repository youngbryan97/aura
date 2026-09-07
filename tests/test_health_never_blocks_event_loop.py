"""The health surface must never buy its answer with the event loop.

2026-07-29, mid-demo: an escalating stall series (5.6s → 17.6s → 103.8s)
froze the live runtime. The 103.8s dump named the path exactly —

    scheduler._run_task
      → orchestrator._pulse_subsystem_audit
        → subsystem_audit.emit_pulse
          → health_contract.runtime_health_report
            → _runtime_integrity_block
              → ontogeny.service.report → experience.stats → sqlite

— an async task on the main loop, doing unbounded disk work inline. The
health report froze the runtime, the frozen runtime failed its own health
polls at 6s, the GUI fell back to "Connecting to runtime", and the immune
system opened CRITICAL incidents for lag the health report had caused.

These tests hold the four properties that make that impossible.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


# ── 1. The integrity block never collects on a running event loop ────────


def test_integrity_snapshot_never_collects_on_the_event_loop(monkeypatch):
    """On a loop, the answer is a snapshot read — never a collection."""
    from core.runtime import health_contract

    health_contract.reset_integrity_snapshot_for_test()
    calls: list[float] = []

    def slow_block() -> dict[str, object]:
        calls.append(time.monotonic())
        time.sleep(0.5)  # stands in for the sqlite scan that took 103.8s
        return {"taint": {"clean": True}}

    monkeypatch.setattr(health_contract, "_runtime_integrity_block", slow_block)

    async def on_loop() -> tuple[dict[str, object], float]:
        started = time.monotonic()
        block = health_contract.integrity_block_snapshot()
        return block, time.monotonic() - started

    block, elapsed = asyncio.run(on_loop())

    # The loop paid nothing. Not "less"; nothing on the order of a disk scan.
    assert elapsed < 0.1, f"integrity collection blocked the loop for {elapsed:.3f}s"
    assert block["snapshot"]["collected"] is False
    assert block["snapshot"]["warming"] is True


def test_integrity_snapshot_warms_in_the_background_and_then_serves(monkeypatch):
    """The refresh a loop caller requests actually lands, and is served next."""
    from core.runtime import health_contract

    health_contract.reset_integrity_snapshot_for_test()
    monkeypatch.setattr(
        health_contract, "_runtime_integrity_block", lambda: {"taint": {"clean": True}}
    )

    asyncio.run(_snapshot_on_loop())

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if health_contract._INTEGRITY_SNAPSHOT is not None:
            break
        time.sleep(0.01)

    block, elapsed = asyncio.run(_timed_snapshot_on_loop())
    assert elapsed < 0.1
    assert block["snapshot"]["collected"] is True
    assert block["taint"] == {"clean": True}


async def _snapshot_on_loop() -> dict[str, object]:
    from core.runtime import health_contract

    return health_contract.integrity_block_snapshot()


async def _timed_snapshot_on_loop() -> tuple[dict[str, object], float]:
    from core.runtime import health_contract

    started = time.monotonic()
    block = health_contract.integrity_block_snapshot()
    return block, time.monotonic() - started


def test_integrity_snapshot_still_collects_inline_off_the_loop(monkeypatch):
    """Off the loop — tests, CLI, the health worker — semantics are unchanged."""
    from core.runtime import health_contract

    health_contract.reset_integrity_snapshot_for_test()
    calls: list[int] = []

    def block() -> dict[str, object]:
        calls.append(1)
        return {"taint": {"clean": True}}

    monkeypatch.setattr(health_contract, "_runtime_integrity_block", block)

    first = health_contract.integrity_block_snapshot()
    assert first["taint"] == {"clean": True}
    assert first["snapshot"]["collected"] is True
    assert len(calls) == 1

    # A burst inside the TTL costs one collection, not one each.
    for _ in range(5):
        health_contract.integrity_block_snapshot()
    assert len(calls) == 1


def test_runtime_health_report_on_the_loop_is_bounded(monkeypatch):
    """The end-to-end path the 2026-07-29 pulse took, now bounded."""
    from core.runtime import health_contract

    health_contract.reset_integrity_snapshot_for_test()

    def slow_block() -> dict[str, object]:
        time.sleep(0.5)
        return {"taint": {"clean": True}}

    monkeypatch.setattr(health_contract, "_runtime_integrity_block", slow_block)

    async def pulse() -> float:
        started = time.monotonic()
        report = health_contract.runtime_health_report()
        elapsed = time.monotonic() - started
        assert "integrity" in report
        return elapsed

    assert asyncio.run(pulse()) < 0.25


# ── 2. The corpus scan is not repeated once per control point ────────────


def test_concurrent_health_workers_share_one_integrity_collection(monkeypatch):
    from core.runtime import health_contract

    health_contract.reset_integrity_snapshot_for_test()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def collect():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return {"measured": True}

    monkeypatch.setattr(health_contract, "_runtime_integrity_block", collect)
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(health_contract.integrity_block_snapshot)
        assert entered.wait(2)
        others = [pool.submit(health_contract.integrity_block_snapshot) for _ in range(3)]
        try:
            cold = asyncio.run(_snapshot_on_loop())
            assert cold["snapshot"]["warming"] is True
            assert calls == [1]
        finally:
            release.set()
        assert first.result(timeout=3)["measured"] is True
        assert all(job.result(timeout=3)["measured"] for job in others)
    assert calls == [1]


def test_ontogeny_stats_are_not_rescanned_within_the_ttl(tmp_path):
    """stats() is three full-table aggregates; a report asks once per CP."""
    from core.ontogeny.experience import ExperienceSpine

    spine = ExperienceSpine(tmp_path / "episodes.db", autoflush=False)
    try:
        scans: list[int] = []
        real_connect = spine._connect

        def counting_connect():
            scans.append(1)
            return real_connect()

        spine._connect = counting_connect  # type: ignore[method-assign]

        first = spine.stats()
        assert first["available"] is True
        assert len(scans) == 1

        for _ in range(10):
            spine.stats()
        assert len(scans) == 1, "corpus rescanned inside the TTL"
    finally:
        spine.close()


def test_ontogeny_stats_keep_live_counters_fresh(tmp_path):
    """The cache holds counts, never the queue depth a caller acts on."""
    from core.ontogeny.experience import ExperienceSpine

    spine = ExperienceSpine(tmp_path / "episodes.db", autoflush=False)
    try:
        spine.stats()
        before = spine.stats()["queued"]

        spine.record(_episode("cp_live"))
        after = spine.stats()["queued"]

        assert after == before + 1, "cached stats hid a queued episode"
    finally:
        spine.close()


def test_ontogeny_stats_cache_is_dropped_by_a_write(tmp_path):
    """A flush that wrote rows changes the counts, so it must invalidate them."""
    from core.ontogeny.experience import ExperienceSpine

    spine = ExperienceSpine(tmp_path / "episodes.db", autoflush=False)
    try:
        assert spine.stats()["rows"] == 0
        assert spine._stats_cache, "expected a populated cache"

        spine.record(_episode("cp_write"))
        assert spine.flush() == 1
        assert not spine._stats_cache, "a write left a stale corpus count cached"

        assert spine.stats()["rows"] == 1
    finally:
        spine.close()


def _episode(control_point: str):
    """A minimal well-formed episode the spine will actually persist."""
    from core.ontogeny.experience import Episode

    return Episode(
        control_point=control_point,
        features={"x": 1.0},
        decision="a",
        options=("a", "b"),
        decider="test",
    )


# ── 3. A permanently-unverified launch stops re-walking the tree ─────────


def test_unverified_revision_retry_backs_off(monkeypatch):
    """Running from source must not SHA256 the shell tree every 2s forever."""
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_TTL_S", 2.0)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_VERIFIED_TTL_S", 30.0)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_FAST_RETRIES", 3)

    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_STREAK", 0)
    assert system_routes._runtime_revision_unverified_ttl_s() == 2.0

    # The first few retries stay fast, so a launch that becomes verifiable is
    # picked up promptly.
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_STREAK", 3)
    assert system_routes._runtime_revision_unverified_ttl_s() == 2.0

    # After that it backs off, and never past the verified cadence.
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_STREAK", 4)
    assert system_routes._runtime_revision_unverified_ttl_s() == 4.0
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_STREAK", 40)
    assert system_routes._runtime_revision_unverified_ttl_s() == 30.0


def test_verified_revision_resets_the_backoff(monkeypatch):
    """A launch that becomes verified returns to the fast cadence."""
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE", None)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE_COLLECTED_AT", 0.0)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_INVALIDATION_PENDING", False)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_STREAK", 7)
    monkeypatch.setattr(
        system_routes,
        "_collect_runtime_revision_uncached",
        lambda: {
            **system_routes._runtime_revision_unavailable(""),
            "verified": True,
            "source_verified": True,
            "revision_token": "a" * 64,
            "issues": [],
        },
    )

    system_routes._runtime_revision_contract()
    assert system_routes._RUNTIME_REVISION_UNVERIFIED_STREAK == 0


# ── 4. Diagnosing the freeze must not deepen it ──────────────────────────


def test_repeat_stall_dumps_are_suppressed_during_one_wedge(monkeypatch, tmp_path):
    """Composing a dump walks every thread's stack — GIL the loop needs."""
    from core.resilience import stall_watchdog as sw

    watchdog = sw.StallWatchdog.__new__(sw.StallWatchdog)
    watchdog._last_stall_dump_at = 0.0
    watchdog._last_stall_dump_path = ""
    watchdog._suppressed_stall_dumps = 0
    watchdog._loop_thread_id = None

    composed: list[float] = []
    monkeypatch.setattr(
        sw.StallWatchdog,
        "_compose_dump_text",
        lambda self, elapsed: composed.append(elapsed) or "dump",
    )
    monkeypatch.setattr(
        sw.StallWatchdog, "_prune_stall_dumps", lambda self, d: None
    )
    monkeypatch.setattr(
        sw.StallWatchdog, "_notify_diagnostics", lambda self, elapsed: None
    )
    monkeypatch.setattr(sw, "_forensics_root", lambda: tmp_path)

    written: list[str] = []

    class _Gateway:
        def write_text(self, path, text, *, source=""):
            written.append(str(path))

    monkeypatch.setattr(sw, "get_file_write_gateway", lambda: _Gateway())

    # The first stall of a wedge always gets its dump — that one is evidence.
    watchdog._report_stall(5.6)
    assert composed == [5.6]

    # Its near-identical successors do not; they only cost GIL.
    watchdog._report_stall(17.6)
    watchdog._report_stall(103.8)
    assert composed == [5.6], "a repeat dump was composed during the same wedge"
    assert watchdog._suppressed_stall_dumps == 2
    assert len(written) == 1

    # Once the interval has passed, a fresh wedge is dumped again.
    watchdog._last_stall_dump_at -= sw.StallWatchdog._STALL_DUMP_MIN_INTERVAL_S + 1
    watchdog._report_stall(6.0)
    assert composed == [5.6, 6.0]


# ── 5. Vector provenance is read once per file version, not per poll ─────


def test_caa_vector_provenance_is_not_reloaded_for_an_unchanged_file(tmp_path):
    """np.load of every .npz on every health collection was a GIL burst."""
    np = pytest.importorskip("numpy")
    from core.consciousness.caa import readiness_report

    readiness_report._VECTOR_PROVENANCE_CACHE.clear()
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    np.savez(
        vectors / "valence_layer12.npz",
        v=np.zeros(8, dtype=np.float32),
        source="runtime_derived_caa",
        extracted=True,
        derived_at=1.0,
    )

    loads: list[int] = []
    real_load = readiness_report.np.load

    def counting_load(*args, **kwargs):
        loads.append(1)
        return real_load(*args, **kwargs)

    readiness_report.np.load = counting_load  # type: ignore[assignment]
    try:
        first = readiness_report.scan_vector_files(vectors)
        assert first["files"] == 1
        assert len(loads) == 1

        for _ in range(10):
            again = readiness_report.scan_vector_files(vectors)
        assert len(loads) == 1, "re-read provenance that could not have changed"
        assert again["files"] == 1
        assert again["details"][0]["source"] == "runtime_derived_caa"
    finally:
        readiness_report.np.load = real_load  # type: ignore[assignment]


def test_caa_vector_provenance_is_reread_when_the_file_changes(tmp_path):
    """The cache is keyed on the file, not on having answered once."""
    np = pytest.importorskip("numpy")
    from core.consciousness.caa import readiness_report

    readiness_report._VECTOR_PROVENANCE_CACHE.clear()
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    path = vectors / "valence_layer12.npz"
    np.savez(path, v=np.zeros(8, dtype=np.float32), source="fallback_random",
             extracted=False, derived_at=1.0)
    assert readiness_report.scan_vector_files(vectors)["fallback"] == 1

    np.savez(path, v=np.zeros(16, dtype=np.float32), source="runtime_derived_caa",
             extracted=True, derived_at=2.0)
    import os

    os.utime(path, (time.time() + 10, time.time() + 10))

    rescanned = readiness_report.scan_vector_files(vectors)
    assert rescanned["extracted"] == 1
    assert rescanned["fallback"] == 0
