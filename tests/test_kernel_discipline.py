"""Contract tests for the kernel-derived runtime disciplines.

Covers core/runtime/{taint,lockdep,pressure_stall,oom_policy}.py. These
are validators, so the tests are mostly "does it actually catch the thing"
— a lockdep that never reports is indistinguishable from no lockdep.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from core.runtime import lockdep as lockdep_mod
from core.runtime import oom_policy as oom_mod
from core.runtime import pressure_stall as psi_mod
from core.runtime import taint as taint_mod
from core.runtime.lockdep import (
    LockRank,
    assert_no_locks_held,
    checked_async_lock,
    checked_lock,
    lockdep_report,
)
from core.runtime.oom_policy import (
    OOM_SCORE_ADJ_MIN,
    OomPolicy,
)
from core.runtime.pressure_stall import PressureMonitor, Resource
from core.runtime.taint import TaintFlag, TaintRegister


@pytest.fixture(autouse=True)
def _clean_state():
    taint_mod.reset_taint_for_test()
    lockdep_mod.reset_lockdep_for_test()
    psi_mod.reset_pressure_for_test()
    oom_mod.reset_oom_policy_for_test()
    yield
    taint_mod.reset_taint_for_test()
    lockdep_mod.reset_lockdep_for_test()
    psi_mod.reset_pressure_for_test()
    oom_mod.reset_oom_policy_for_test()


# ── taint ─────────────────────────────────────────────────────────────

def test_taint_is_one_way_and_counts_repeats():
    register = TaintRegister()
    assert not register.is_tainted()

    register.add(TaintFlag.CRASHED_ORGAN, "cortex died", subsystem="cortex")
    register.add(TaintFlag.CRASHED_ORGAN, "cortex died again", subsystem="cortex")

    assert register.is_tainted(TaintFlag.CRASHED_ORGAN)
    records = register.records()
    assert len(records) == 1
    # The FIRST reason is kept — the first break is the one that explains
    # everything after it.
    assert records[0].first_reason == "cortex died"
    assert records[0].count == 2
    # There is deliberately no untaint(); the only exit is process restart.
    assert not hasattr(register, "untaint")
    assert not hasattr(register, "clear")


def test_taint_compact_form_and_credibility_caveat():
    assert taint_mod.credibility_caveat() is None
    taint_mod.taint(TaintFlag.LOCK_ORDER, "ABBA between lane and cache")
    taint_mod.taint(TaintFlag.DEGRADED_RESULT, "served from cache")

    compact = taint_mod.taint_compact()
    assert set(compact) == {"L", "D"}
    caveat = taint_mod.credibility_caveat()
    # DEGRADED_RESULT alone does not undermine a health verdict; LOCK_ORDER does.
    assert caveat is not None and "L" in caveat and "D" not in caveat.split("(")[1]
    assert "tainted" in taint_mod.taint_narrative()


def test_health_report_carries_the_taint():
    from core.runtime.health_contract import _runtime_integrity_block

    taint_mod.taint(TaintFlag.OOM_SHED, "shed research lane")
    block = _runtime_integrity_block()
    assert block["taint"]["tainted"] is True
    assert "O" in block["taint_compact"]
    assert "credibility_caveat" in block


# ── lockdep ───────────────────────────────────────────────────────────

def test_lockdep_detects_abba_without_the_deadlock_happening():
    a = checked_lock("alpha")
    b = checked_lock("beta")

    # Thread 1 order: alpha -> beta
    with a:
        with b:
            pass

    assert lockdep_report()["clean"] is True

    # Thread 2 order: beta -> alpha. The deadlock never occurs because the
    # two never run concurrently — lockdep must report anyway.
    with b:
        with a:
            pass

    report = lockdep_report()
    assert report["clean"] is False
    kinds = {s["kind"] for s in report["splats"]}
    assert "order_inversion" in kinds
    assert taint_mod.is_tainted(TaintFlag.LOCK_ORDER)


def test_lockdep_allows_consistent_order_forever():
    a = checked_lock("outer")
    b = checked_lock("inner")
    for _ in range(25):
        with a, b:
            pass
    assert lockdep_report()["clean"] is True


def test_lockdep_declared_rank_inversion_fires_on_first_offence():
    leaf = checked_lock("leaf_cache", rank=LockRank.LEAF)
    registry = checked_lock("registry", rank=LockRank.REGISTRY)

    # LEAF must be innermost; taking REGISTRY under it inverts declared order
    # on the very first occurrence, with no opposing observation needed.
    with leaf:
        with registry:
            pass

    report = lockdep_report()
    kinds = {s["kind"] for s in report["splats"]}
    assert "rank_inversion" in kinds


def test_lockdep_rejects_conflicting_rank_declaration():
    checked_lock("dual", rank=LockRank.LANE)
    with pytest.raises(ValueError, match="already declared"):
        checked_lock("dual", rank=LockRank.LEAF)


def test_lockdep_catches_self_deadlock_on_non_reentrant_lock():
    lock = checked_lock("solo")
    lock.acquire()
    try:
        # Do not actually re-acquire (that would block); the validator is
        # consulted before the blocking call, which is the point.
        lockdep_mod.get_validator().on_acquire(
            "solo", rank=LockRank.UNRANKED, is_async=False, reentrant=False
        )
        lockdep_mod.get_validator().on_release("solo", is_async=False)
    finally:
        lock.release()

    kinds = {s["kind"] for s in lockdep_report()["splats"]}
    assert "self_deadlock" in kinds


def test_lockdep_catches_sync_lock_held_across_await():
    async def scenario():
        sync = checked_lock("blocking_cache")
        gate = checked_async_lock("model_lane")
        with sync:
            async with gate:
                pass

    asyncio.run(scenario())
    report = lockdep_report()
    kinds = {s["kind"] for s in report["splats"]}
    assert "sync_held_across_await" in kinds
    splat = next(s for s in report["splats"] if s["kind"] == "sync_held_across_await")
    assert "blocking_cache" in splat["message"]


def test_lockdep_reports_loop_blocking_hold():
    validator = lockdep_mod.get_validator()
    validator.note_loop_thread()
    lock = checked_lock("slow_section")
    with lock:
        time.sleep(lockdep_mod.LOOP_BLOCKING_HOLD_S * 1.5)

    kinds = {s["kind"] for s in lockdep_report()["splats"]}
    assert "loop_blocking_hold" in kinds


def test_assert_no_locks_held_reports_once_and_can_raise():
    lock = checked_lock("held_during_io")
    with lock:
        held = assert_no_locks_held("fsync")
        assert held == ["held_during_io"]
        assert_no_locks_held("fsync")  # deduplicated
        with pytest.raises(lockdep_mod.LockHeldError):
            assert_no_locks_held("fsync", strict=True)

    splats = [s for s in lockdep_report()["splats"] if s["kind"] == "blocking_op_under_lock"]
    assert len(splats) == 1
    assert splats[0]["occurrences"] >= 3


class TestAcquireStaysCheapEnoughToLeaveOn:
    """Lockdep is on in production, so its cost is part of its contract.

    It only sees locks it wraps, which means coverage grows by wrapping
    more locks — including hot caches and registries. A per-acquire cost
    that is fine for eight locks is not automatically fine for five
    hundred, so the cost is pinned here rather than assumed.
    """

    def test_naming_the_call_site_reads_no_source(self):
        """The regression that made an acquire 97x a raw lock.

        traceback.extract_stack resolves every frame's source line through
        linecache to build a string that is thrown away unless a splat
        happens. Reading f_lineno off the frame costs nothing and says the
        same thing. linecache staying empty is the structural version of
        the timing assertion below, and it does not flake.
        """
        import linecache

        linecache.clearcache()
        with checked_lock("cheap_site"):
            pass
        assert not linecache.cache, (
            "acquiring a checked lock read source files: call-site naming is "
            "going through traceback again"
        )

    def test_overhead_stays_within_an_order_of_magnitude_or_two(self):
        import timeit

        raw = threading.Lock()
        checked = checked_lock("overhead_probe")

        def cost(lock) -> float:
            def run():
                with lock:
                    pass

            # Best of 5: the minimum is the least noisy estimator here,
            # because scheduler noise only ever adds time.
            return min(timeit.timeit(run, number=20_000) for _ in range(5))

        ratio = cost(checked) / cost(raw)
        assert ratio < 45, (
            f"checked lock is {ratio:.0f}x a raw lock; it was ~20x when this "
            "was written. Something on the acquire path got expensive, and "
            "lockdep is on in production."
        )


class TestSanctionedBlockingLocks:
    """A sanction excuses one named lock from one check — and nothing else."""

    def test_a_sanctioned_lock_is_not_an_offender(self, monkeypatch):
        monkeypatch.setitem(
            lockdep_mod.SANCTIONED_BLOCKING_LOCKS, "chain.commit", "the fsync is the commit"
        )
        with checked_lock("chain.commit"):
            assert assert_no_locks_held("fsync") == []
            assert_no_locks_held("fsync", strict=True)  # does not raise
        assert not [
            s for s in lockdep_report()["splats"] if s["kind"] == "blocking_op_under_lock"
        ]

    def test_an_unsanctioned_lock_nested_inside_one_still_reports(self, monkeypatch):
        """The sanction covers a named lock, never everything under it."""
        monkeypatch.setitem(
            lockdep_mod.SANCTIONED_BLOCKING_LOCKS, "chain.commit", "the fsync is the commit"
        )
        with checked_lock("chain.commit"), checked_lock("some.cache"):
            assert assert_no_locks_held("fsync") == ["some.cache"]

        splats = [
            s for s in lockdep_report()["splats"] if s["kind"] == "blocking_op_under_lock"
        ]
        assert len(splats) == 1
        assert splats[0]["held"] == ["some.cache"]

    def test_a_sanction_does_not_silence_the_loop_blocking_check(self, monkeypatch):
        """The reason this is safe to have at all.

        A sanction states that a blocking call belongs under a lock. It
        states nothing about how long the lock is held on the event loop
        thread, so the check that measures exactly that must survive it —
        otherwise sanctioning a lock would quietly retire the detector for
        the freeze it was written for.
        """
        monkeypatch.setitem(
            lockdep_mod.SANCTIONED_BLOCKING_LOCKS, "chain.commit", "the fsync is the commit"
        )
        lockdep_mod.get_validator().note_loop_thread()
        with checked_lock("chain.commit"):
            time.sleep(lockdep_mod.LOOP_BLOCKING_HOLD_S * 1.5)

        assert "loop_blocking_hold" in {s["kind"] for s in lockdep_report()["splats"]}

    def test_every_entry_states_a_reason(self):
        """An entry with no reason is an exemption, not a sanction."""
        for name, reason in lockdep_mod.SANCTIONED_BLOCKING_LOCKS.items():
            assert isinstance(reason, str) and len(reason.split()) >= 12, (
                f"{name} is sanctioned without a reason that names the invariant "
                "requiring the hold"
            )

    def test_the_sanctioned_locks_exist_in_the_checked_lane(self):
        """A sanction for a lock nobody constructs is a stale exemption.

        The names have to match what checked_lock() is actually called
        with. A typo would leave the entry protecting nothing while the
        real lock kept reporting — and the reason text would still read
        as though the question had been settled.
        """
        root = Path(__file__).resolve().parent.parent
        sources = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for base in ("core", "interface")
            for path in (root / base).rglob("*.py")
        )
        for name in lockdep_mod.SANCTIONED_BLOCKING_LOCKS:
            assert f'checked_lock("{name}"' in sources, (
                f"nothing constructs a checked_lock named {name!r}; the sanction "
                "protects nothing"
            )


def test_lockdep_separates_concurrent_threads():
    a = checked_lock("t_alpha")
    b = checked_lock("t_beta")
    barrier = threading.Barrier(2)

    def worker(first, second):
        barrier.wait(timeout=5)
        with first:
            with second:
                pass

    # Both threads use the SAME order; concurrency must not manufacture a splat.
    threads = [threading.Thread(target=worker, args=(a, b)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert lockdep_report()["clean"] is True


# ── PSI ───────────────────────────────────────────────────────────────

def test_psi_some_and_full_are_distinct_signals():
    monitor = PressureMonitor()
    monitor.declare_capacity(Resource.INFERENCE, 2)

    monitor.begin_stall(Resource.INFERENCE)
    report = monitor.report()
    assert report["inference"]["stalled_now"] == 1
    # One of two workers waiting is `some`, never `full`.
    assert report["inference"]["some"]["total_s"] == 0.0  # episode still open

    monitor.begin_stall(Resource.INFERENCE)
    time.sleep(0.02)
    monitor.end_stall(Resource.INFERENCE)
    monitor.end_stall(Resource.INFERENCE)

    report = monitor.report()
    assert report["inference"]["full"]["total_s"] > 0.0
    assert report["inference"]["some"]["total_s"] >= report["inference"]["full"]["total_s"]
    assert report["inference"]["peak_stalled"] == 2


def test_psi_single_capacity_makes_some_equal_full():
    monitor = PressureMonitor()
    monitor.declare_capacity(Resource.MEMORY, 1)
    monitor.begin_stall(Resource.MEMORY)
    time.sleep(0.02)
    monitor.end_stall(Resource.MEMORY)
    entry = monitor.report()["memory"]
    assert entry["full"]["total_s"] == pytest.approx(entry["some"]["total_s"], rel=0.05)


def test_psi_stall_context_manager_is_exception_safe():
    with pytest.raises(RuntimeError):
        with psi_mod.stall(Resource.IO):
            raise RuntimeError("boom")
    assert psi_mod.psi_report()["io"]["stalled_now"] == 0


def test_psi_averages_decay_over_windows():
    monitor = PressureMonitor()
    monitor.declare_capacity(Resource.CPU, 1)
    # Drive a full accounting period of stall, then advance past it.
    monitor._period_start = time.monotonic() - psi_mod.PERIOD_S * 2
    monitor.begin_stall(Resource.CPU)
    monitor.end_stall(Resource.CPU)
    entry = monitor.report()["cpu"]
    assert set(entry["some"]) >= {"avg10", "avg60", "avg300", "total_s"}
    # Shorter windows respond at least as fast as longer ones.
    assert entry["some"]["avg10"] >= entry["some"]["avg300"]


def test_psi_saturation_threshold():
    monitor = PressureMonitor()
    monitor.declare_capacity(Resource.BUS, 1)
    assert monitor.saturated() == []
    monitor._states.setdefault("bus", psi_mod._ResourceState())
    monitor._states["bus"].full_avg.values[10] = 55.0
    assert "bus" in monitor.saturated(threshold=0.2)


# ── OOM policy ────────────────────────────────────────────────────────

def _organ(policy: OomPolicy, name: str, *, adj: int, size: int, freeable: int | None = None):
    freed = size if freeable is None else freeable
    state = {"size": size}

    def shed() -> int:
        released = min(state["size"], freed)
        state["size"] -= released
        return released

    return policy.register(
        name,
        oom_score_adj=adj,
        footprint=lambda: state["size"],
        shed=shed,
        rationale=f"{name} test organ",
    )


def test_oom_badness_is_proportional_plus_adjustment():
    policy = OomPolicy()
    total = 1_000_000
    small_volunteer = _organ(policy, "background", adj=500, size=100_000)
    big_neutral = _organ(policy, "cortex", adj=0, size=400_000)

    assert policy.badness(small_volunteer, total) == 600  # 100 + 500
    assert policy.badness(big_neutral, total) == 400
    # The volunteer is chosen even though it is four times smaller — that is
    # the whole point of oom_score_adj.
    assert policy.select_victim(total).name == "background"


def test_oom_report_scores_the_same_footprint_it_displays():
    policy = OomPolicy()
    observations = []

    def footprint():
        value = 100_000 * (len(observations) + 1)
        observations.append(value)
        return value

    policy.register("changing", oom_score_adj=0, footprint=footprint,
                    rationale="one observation per scoring snapshot")
    row = policy.scoring_table(1_000_000)[0]
    assert observations == [100_000]
    assert row["footprint_bytes"] == 100_000
    assert row["badness"] == 100


def test_oom_never_selects_an_immune_organ():
    policy = OomPolicy()
    policy.register(
        "unified_will",
        oom_score_adj=OOM_SCORE_ADJ_MIN,
        footprint=lambda: 900_000,
        shed=lambda: 900_000,
        rationale="load-bearing",
    )
    assert policy.select_victim(1_000_000) is None
    table = policy.scoring_table(1_000_000)
    assert table[0]["immune"] is True
    assert table[0]["badness"] == OOM_SCORE_ADJ_MIN


def test_oom_sheds_until_target_and_stops():
    policy = OomPolicy()
    _organ(policy, "research", adj=800, size=300_000)
    _organ(policy, "vector_cache", adj=200, size=300_000)
    free = {"bytes": 50_000}

    def free_now() -> int:
        return free["bytes"]

    events = policy.shed_until(
        target_free_bytes=100_000,
        free_bytes_now=free_now,
        reason="test pressure",
    )
    # First shed frees 300k; the harness reports the new free level.
    assert events, "expected at least one shed"
    assert events[0].victim == "research"
    assert events[0].table, "the scoring table that justified the choice must be recorded"
    assert taint_mod.is_tainted(TaintFlag.OOM_SHED)


def test_oom_does_not_re_pick_a_victim_that_freed_nothing():
    policy = OomPolicy()
    policy.register(
        "stuck",
        oom_score_adj=900,
        footprint=lambda: 500_000,
        shed=lambda: 0,
        rationale="cannot actually free",
    )
    policy.register(
        "real",
        oom_score_adj=100,
        footprint=lambda: 200_000,
        shed=lambda: 200_000,
        rationale="frees for real",
    )
    events = policy.shed_until(
        target_free_bytes=10**9,
        free_bytes_now=lambda: 0,
        reason="test",
        max_victims=3,
    )
    victims = [e.victim for e in events]
    assert victims[0] == "stuck"
    assert victims.count("stuck") == 1, "a no-op shed must not be retried in the same pass"
    assert "real" in victims


def test_oom_report_names_the_next_victim():
    policy = OomPolicy()
    _organ(policy, "speculative", adj=900, size=10_000)
    report = policy.report()
    assert report["next_victim"] == "speculative"
    assert report["sheddable_organs"] == 1


def test_oom_controlled_restart_is_idempotent(monkeypatch):
    policy = OomPolicy()
    calls: list[str] = []
    import core.runtime.shutdown_coordinator as sc

    monkeypatch.setattr(sc, "request_shutdown", lambda reason="": calls.append(reason) or {})
    assert policy.request_controlled_restart("no victims left") is True
    assert policy.request_controlled_restart("again") is False
    assert len(calls) == 1
    assert taint_mod.is_tainted(TaintFlag.OOM_SHED)


def test_immune_service_list_covers_the_spine():
    from core.runtime.foundations import IMMUNE_SERVICES

    for essential in ("unified_will", "event_bus", "memory_facade", "flight_recorder"):
        assert essential in IMMUNE_SERVICES
