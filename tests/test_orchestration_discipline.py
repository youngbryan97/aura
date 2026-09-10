"""Contract tests for the Kubernetes-derived orchestration disciplines.

core/runtime/{reconcile,admission,quota,eviction,lease}.py.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from core.runtime import admission as adm_mod
from core.runtime import eviction as evict_mod
from core.runtime import lease as lease_mod
from core.runtime import quota as quota_mod
from core.runtime import reconcile as rec_mod
from core.runtime import sanitizers as san_mod
from core.runtime import taint as taint_mod
from core.runtime.admission import (
    AdmissionRequest,
    AdmissionResponse,
    FailurePolicy,
    Operation,
    admit,
    mutating,
    validating,
)
from core.runtime.eviction import (
    Comparison,
    DisruptionBudget,
    Evictable,
    EvictionManager,
    Signal,
    Threshold,
)
from core.runtime.lease import Identity, LeaderElector, LeaseRecord
from core.runtime.quota import (
    LimitRange,
    QosClass,
    QuotaRegistry,
    ResourceKind,
    ResourceSpec,
)
from core.runtime.reconcile import (
    Controller,
    ObjectMeta,
    RateLimitingQueue,
    Request,
    Result,
    run_finalizers,
)


@pytest.fixture(autouse=True)
def _clean():
    for mod in (adm_mod, quota_mod, evict_mod, lease_mod, rec_mod, san_mod, taint_mod):
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()
    yield
    for mod in (adm_mod, quota_mod, evict_mod, lease_mod, rec_mod, san_mod, taint_mod):
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()


# ── work queue ────────────────────────────────────────────────────────

def test_queue_deduplicates_pending_keys():
    async def scenario():
        queue = RateLimitingQueue()
        assert queue.add("lane-a") is True
        assert queue.add("lane-a") is False, "a pending key must collapse"
        assert queue.add("lane-b") is True
        assert queue.depth() == 2
        return queue.report()

    report = asyncio.run(scenario())
    assert report["deduped"] == 1


def test_queue_coalesces_requests_arriving_during_processing():
    async def scenario():
        queue = RateLimitingQueue()
        queue.add("k")
        request = await queue.get()
        # Ten more hints while a worker holds the key must produce exactly
        # one more pass, not ten.
        for _ in range(10):
            queue.add("k")
        assert queue.depth() == 0
        queue.done(request)
        assert queue.depth() == 1
        again = await queue.get()
        queue.done(again)
        assert queue.depth() == 0

    asyncio.run(scenario())


def test_rate_limiter_backs_off_exponentially_and_forgets_on_success():
    async def scenario():
        queue = RateLimitingQueue()
        delays = [queue.add_rate_limited("bad") for _ in range(4)]
        assert delays == sorted(delays), "backoff must be monotonic"
        assert delays[-1] > delays[0]
        assert queue.failures("bad") == 4
        queue.forget("bad")
        assert queue.failures("bad") == 0
        # After forgetting, the next failure starts from the base delay
        # again — this is what makes the limiter recover.
        assert queue.add_rate_limited("bad") == pytest.approx(rec_mod.BASE_DELAY_S)
        queue.shutdown()

    asyncio.run(scenario())


def test_rate_limiter_respects_the_ceiling():
    async def scenario():
        queue = RateLimitingQueue()
        for _ in range(40):
            delay = queue.add_rate_limited("stuck")
        assert delay == pytest.approx(rec_mod.MAX_DELAY_S)
        queue.shutdown()

    asyncio.run(scenario())


# ── controller ────────────────────────────────────────────────────────

def test_controller_reconciles_and_requeues_until_converged():
    async def scenario():
        state = {"observed": 0, "desired": 3}
        calls: list[str] = []

        async def reconcile(request: Request) -> Result:
            calls.append(request.key)
            if state["observed"] >= state["desired"]:
                return Result.done()
            state["observed"] += 1
            return Result.again(after_s=0.0)

        controller = Controller("stepper", reconcile, resync_s=0)
        await controller.start()
        controller.enqueue("thing")
        for _ in range(60):
            if state["observed"] >= state["desired"]:
                break
            await asyncio.sleep(0.01)
        await controller.stop()
        return state, calls

    state, calls = asyncio.run(scenario())
    assert state["observed"] == 3
    assert len(calls) >= 3


def test_controller_backs_off_a_failing_key_without_dying():
    async def scenario():
        attempts = {"n": 0}

        async def reconcile(request: Request) -> Result:
            attempts["n"] += 1
            raise RuntimeError("still broken")

        controller = Controller("flaky", reconcile, resync_s=0)
        await controller.start()
        controller.enqueue("k")
        await asyncio.sleep(0.4)
        report = controller.report()
        await controller.stop()
        return attempts, report

    attempts, report = asyncio.run(scenario())
    assert attempts["n"] >= 2, "must retry"
    assert report["failures"] >= 2
    assert report["running"] is True, "a failing key must not kill the worker"
    assert report["queue"]["backing_off"]


def test_reconcile_is_level_triggered_not_edge_triggered():
    """A missed hint costs latency, never correctness: the periodic resync
    reconciles keys nobody enqueued."""

    async def scenario():
        seen: list[str] = []

        async def reconcile(request: Request) -> Result:
            seen.append(request.reason)
            return Result.done()

        controller = Controller(
            "resyncer", reconcile, resync_s=0.05, list_keys=lambda: ["never-enqueued"]
        )
        await controller.start()
        await asyncio.sleep(0.2)
        await controller.stop()
        return seen

    seen = asyncio.run(scenario())
    assert "resync" in seen


def test_object_meta_tracks_convergence_by_generation():
    meta = ObjectMeta(name="lane")
    assert meta.converged is False, "nothing has observed generation 1 yet"
    meta.observe()
    assert meta.converged is True
    meta.bump()
    assert meta.converged is False, "desired state changed; not converged"
    meta.observe()
    assert meta.converged is True


def test_finalizers_hold_deletion_until_cleanup_succeeds():
    async def scenario():
        meta = ObjectMeta(name="lane")
        meta.add_finalizer("drain")
        meta.add_finalizer("unload")
        meta.deletion_requested_at = time.time()
        failures = {"drain": True}

        def drain():
            if failures["drain"]:
                raise RuntimeError("still draining")

        handlers = {"drain": drain, "unload": lambda: None}

        assert await run_finalizers(meta, handlers) is False
        assert "drain" in meta.finalizers, "a failed finalizer keeps the object alive"

        failures["drain"] = False
        assert await run_finalizers(meta, handlers) is True
        assert meta.finalizers == ()

    asyncio.run(scenario())


def test_finalizer_without_a_handler_blocks_deletion():
    async def scenario():
        meta = ObjectMeta(name="orphan")
        meta.add_finalizer("nobody-implements-this")
        meta.deletion_requested_at = time.time()
        return await run_finalizers(meta, {})

    assert asyncio.run(scenario()) is False


# ── admission ─────────────────────────────────────────────────────────

def test_mutation_runs_before_validation():
    order: list[str] = []

    @mutating("m.stamp", order=10, owner="test")
    def _stamp(request: AdmissionRequest):
        order.append("mutate")
        return {**request.obj, "stamped": True}

    @validating("v.requires_stamp", order=10, owner="test")
    def _requires(request: AdmissionRequest):
        order.append("validate")
        if not request.obj.get("stamped"):
            return AdmissionResponse.deny("validator saw an unstamped object")
        return AdmissionResponse.allow()

    verdict = admit("action", "a1", {"payload": 1})
    assert verdict.allowed, verdict.reason
    assert order == ["mutate", "validate"]
    assert verdict.obj["stamped"] is True
    assert verdict.mutated_by == ["m.stamp"]


def test_hooks_run_in_declared_order():
    seen: list[str] = []

    @mutating("m.late", order=90, owner="test")
    def _late(request: AdmissionRequest):
        seen.append("late")
        return None

    @mutating("m.early", order=10, owner="test")
    def _early(request: AdmissionRequest):
        seen.append("early")
        return None

    admit("action", "a", {})
    assert seen == ["early", "late"]


def test_failure_policy_fail_denies_and_ignore_skips():
    @validating("v.explodes", failure_policy=FailurePolicy.FAIL, owner="test")
    def _explodes(request: AdmissionRequest):
        raise RuntimeError("policy service down")

    verdict = admit("action", "a", {})
    assert verdict.allowed is False
    assert verdict.denied_by == "v.explodes"

    adm_mod.reset_admission_for_test()

    @validating("v.enriches", failure_policy=FailurePolicy.IGNORE, owner="test")
    def _enriches(request: AdmissionRequest):
        raise RuntimeError("enrichment down")

    verdict = admit("action", "a", {})
    assert verdict.allowed is True
    assert "v.enriches" in verdict.skipped


def test_a_mutating_validator_is_reported():
    @validating("v.sneaky", owner="test")
    def _sneaky(request: AdmissionRequest):
        request.obj["injected"] = True
        return AdmissionResponse.allow()

    verdict = admit("action", "a", {"x": 1})
    assert verdict.allowed is True
    assert any("mutated" in w for w in verdict.warnings)
    assert any(f["sanitizer"] == "admission" for f in san_mod.sanitizer_report()["findings"])


def test_hooks_filter_by_kind_and_operation():
    hits: list[str] = []

    @validating(
        "v.only_deletes",
        kinds=("memory",),
        operations=(Operation.DELETE,),
        owner="test",
    )
    def _only(request: AdmissionRequest):
        hits.append(f"{request.kind}:{request.operation}")
        return True

    admit("memory", "m", {}, operation=Operation.CREATE)
    admit("action", "a", {}, operation=Operation.DELETE)
    assert hits == []
    admit("memory", "m", {}, operation=Operation.DELETE)
    assert hits == ["memory:delete"]


def test_duplicate_hook_name_is_refused():
    @validating("v.dup", owner="test")
    def _one(request):
        return True

    with pytest.raises(ValueError, match="already registered"):

        @validating("v.dup", owner="test")
        def _two(request):
            return True


def test_copy_object_protects_the_callers_object():
    @mutating("m.edits", owner="test")
    def _edits(request: AdmissionRequest):
        request.obj["edited"] = True
        return request.obj

    original = {"x": 1}
    verdict = admit("action", "a", original, copy_object=True)
    assert verdict.obj["edited"] is True
    assert "edited" not in original


# ── quota / QoS ───────────────────────────────────────────────────────

def test_qos_class_derives_from_requests_versus_limits():
    guaranteed = ResourceSpec(
        name="cortex",
        requests={ResourceKind.MEMORY_BYTES: 20e9},
        limits={ResourceKind.MEMORY_BYTES: 20e9},
    )
    burstable = ResourceSpec(
        name="research",
        requests={ResourceKind.MEMORY_BYTES: 1e9},
        limits={ResourceKind.MEMORY_BYTES: 4e9},
    )
    best_effort = ResourceSpec(name="speculation")

    assert guaranteed.qos_class is QosClass.GUARANTEED
    assert burstable.qos_class is QosClass.BURSTABLE
    assert best_effort.qos_class is QosClass.BEST_EFFORT


def test_limit_range_defaults_and_clamps_incoherent_specs():
    limit_range = LimitRange(
        default_requests={ResourceKind.TOKENS: 1000},
        default_limits={ResourceKind.TOKENS: 4000},
    )
    resolved = limit_range.apply(ResourceSpec(name="x"))
    assert resolved.requests[ResourceKind.TOKENS] == 1000
    assert resolved.limits[ResourceKind.TOKENS] == 4000

    # A request above its own limit can never be satisfied; clamp it.
    clamped = limit_range.apply(
        ResourceSpec(name="y", requests={ResourceKind.TOKENS: 9000})
    )
    assert clamped.requests[ResourceKind.TOKENS] == 4000


def test_limit_range_reports_bound_violations():
    limit_range = LimitRange(maximum={ResourceKind.TOOL_CALLS: 10})
    problems = limit_range.violations(
        ResourceSpec(name="greedy", limits={ResourceKind.TOOL_CALLS: 50})
    )
    assert problems and "exceeds the maximum" in problems[0]


def test_quota_reserve_release_and_denial():
    registry = QuotaRegistry()
    registry.set_quota("research", {ResourceKind.TOKENS: 1000})

    first = registry.reserve("research", {ResourceKind.TOKENS: 700})
    assert first.allowed and first.reservation_id

    denied = registry.reserve("research", {ResourceKind.TOKENS: 400})
    assert denied.allowed is False
    assert ResourceKind.TOKENS in denied.exceeded
    assert "would reach" in denied.reason

    assert registry.release(first.reservation_id) is True
    assert registry.reserve("research", {ResourceKind.TOKENS: 400}).allowed is True


def test_quota_windows_roll():
    registry = QuotaRegistry()
    registry.set_quota("turn", {ResourceKind.TOOL_CALLS: 2}, period_s=0.05)
    assert registry.reserve("turn", {ResourceKind.TOOL_CALLS: 2}).allowed
    assert registry.reserve("turn", {ResourceKind.TOOL_CALLS: 1}).allowed is False
    time.sleep(0.06)
    assert registry.reserve("turn", {ResourceKind.TOOL_CALLS: 1}).allowed is True


def test_quota_is_enforced_at_admission():
    quota_mod.get_quota_registry().set_quota("agent", {ResourceKind.TOOL_CALLS: 2})
    assert quota_mod.install_quota_admission() is True
    assert quota_mod.install_quota_admission() is False, "must be idempotent"

    ok = admit(
        "tool_call",
        "search",
        {},
        principal="agent",
        context={"resources": {ResourceKind.TOOL_CALLS: 2}},
    )
    assert ok.allowed is True

    over = admit(
        "tool_call",
        "search",
        {},
        principal="agent",
        context={"resources": {ResourceKind.TOOL_CALLS: 1}},
    )
    assert over.allowed is False
    assert "quota" in over.reason


# ── eviction ──────────────────────────────────────────────────────────

def _manager_with(*candidates: Evictable) -> EvictionManager:
    manager = EvictionManager(thresholds=())
    for candidate in candidates:
        manager.register(candidate)
    return manager


def test_eviction_order_is_best_effort_then_burstable_never_guaranteed():
    quota_mod.declare_resources(
        "cortex", requests={ResourceKind.MEMORY_BYTES: 20e9}, limits={ResourceKind.MEMORY_BYTES: 20e9}
    )
    quota_mod.declare_resources(
        "research", requests={ResourceKind.MEMORY_BYTES: 1e9}, limits={ResourceKind.MEMORY_BYTES: 4e9}
    )
    manager = _manager_with(
        Evictable(name="cortex", evict=lambda: 1),
        Evictable(name="research", evict=lambda: 1),
        Evictable(name="speculation", evict=lambda: 1),
    )
    order = [c.name for c in manager.eviction_order()]
    assert order == ["speculation", "research"]
    assert "cortex" not in order, "a Guaranteed organ must never be in the order"


def test_qos_maps_onto_oom_score_adj_so_both_layers_agree():
    from core.runtime.oom_policy import get_oom_policy, reset_oom_policy_for_test

    reset_oom_policy_for_test()
    quota_mod.declare_resources(
        "cortex", requests={ResourceKind.MEMORY_BYTES: 1}, limits={ResourceKind.MEMORY_BYTES: 1}
    )
    manager = _manager_with(
        Evictable(name="cortex", reclaim=lambda: 0),
        Evictable(name="speculation", reclaim=lambda: 0),
    )
    scores = manager.sync_oom_scores()
    assert scores["cortex"] < 0 < scores["speculation"]
    table = {r["organ"]: r for r in get_oom_policy().scoring_table(1_000_000)}
    assert table["speculation"]["badness"] > table["cortex"]["badness"]
    reset_oom_policy_for_test()


def test_soft_threshold_waits_for_its_grace_period():
    manager = EvictionManager(
        thresholds=(
            Threshold(
                signal=Signal.MEMORY_AVAILABLE_FRACTION,
                comparison=Comparison.BELOW,
                value=0.5,
                grace_period_s=60.0,
            ),
        )
    )
    observed = {str(Signal.MEMORY_AVAILABLE_FRACTION): 0.1}
    assert manager.breached(observed) == [], "a soft threshold must not fire immediately"
    # Backdate the breach start to simulate the grace period elapsing.
    key = next(iter(manager._breach_started))
    manager._breach_started[key] -= 120.0
    assert len(manager.breached(observed)) == 1


def test_hard_threshold_fires_immediately_and_clears_when_it_recovers():
    manager = EvictionManager(
        thresholds=(
            Threshold(
                signal=Signal.MEMORY_AVAILABLE_FRACTION,
                comparison=Comparison.BELOW,
                value=0.5,
            ),
        )
    )
    assert len(manager.breached({str(Signal.MEMORY_AVAILABLE_FRACTION): 0.1})) == 1
    assert manager.breached({str(Signal.MEMORY_AVAILABLE_FRACTION): 0.9}) == []


def test_reclaim_runs_before_eviction():
    actions: list[str] = []
    manager = EvictionManager(
        thresholds=(
            Threshold(
                signal=Signal.MEMORY_AVAILABLE_FRACTION,
                comparison=Comparison.BELOW,
                value=0.5,
            ),
        )
    )
    manager.observe = lambda: {str(Signal.MEMORY_AVAILABLE_FRACTION): 0.1}  # type: ignore[method-assign]
    manager.register(
        Evictable(
            name="cache",
            reclaim=lambda: (actions.append("reclaim"), 0)[1],
            evict=lambda: (actions.append("evict"), 1)[1],
        )
    )
    manager.enforce()
    assert actions[0] == "reclaim", "killing something that would free on request is pure loss"


def test_disruption_budget_refuses_the_last_member():
    manager = EvictionManager(
        thresholds=(
            Threshold(
                signal=Signal.MEMORY_AVAILABLE_FRACTION,
                comparison=Comparison.BELOW,
                value=0.5,
            ),
        )
    )
    manager.observe = lambda: {str(Signal.MEMORY_AVAILABLE_FRACTION): 0.1}  # type: ignore[method-assign]
    manager.set_budget(DisruptionBudget(group="verifiers", min_available=1))
    manager.register(Evictable(name="verifier-a", evict=lambda: 1, group="verifiers"))

    outcome = manager.enforce()
    refused = [a for a in outcome["actions"] if a.get("action") == "refused"]
    assert refused, "eviction below a disruption budget must be refused, not silent"
    assert manager.report()["candidates"][0]["alive"] is True


def test_eviction_taints_the_runtime():
    manager = EvictionManager(
        thresholds=(
            Threshold(
                signal=Signal.MEMORY_AVAILABLE_FRACTION,
                comparison=Comparison.BELOW,
                value=0.5,
            ),
        )
    )
    manager.observe = lambda: {str(Signal.MEMORY_AVAILABLE_FRACTION): 0.1}  # type: ignore[method-assign]
    manager.register(Evictable(name="doomed", evict=lambda: 1_000_000))
    manager.enforce()
    assert taint_mod.is_tainted(taint_mod.TaintFlag.OOM_SHED)


# ── leases ────────────────────────────────────────────────────────────

def _elector(tmp_path, name="test_lease", **kwargs) -> LeaderElector:
    return LeaderElector(name, **kwargs)


def test_renew_deadline_must_be_shorter_than_the_lease():
    with pytest.raises(ValueError, match="strictly less"):
        LeaderElector("bad", lease_duration_s=10.0, renew_deadline_s=10.0)


def test_lease_acquire_then_renew(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")

    async def scenario():
        elector = LeaderElector("solo")
        assert await elector.try_acquire_or_renew() is True
        assert elector.is_leader is True
        assert elector.acquisitions == 1
        assert await elector.try_acquire_or_renew() is True
        assert elector.renewals == 1
        return elector.report()

    report = asyncio.run(scenario())
    assert report["is_leader"] is True
    assert report["record"]["transitions"] == 0


def test_orderly_lease_stop_is_release_not_loss(caplog):
    async def scenario():
        elector = LeaderElector("orderly-stop")
        elector._is_leader = True
        await elector.stop(release=False)

    with caplog.at_level("INFO", logger="Aura.Lease"):
        asyncio.run(scenario())

    assert "released lease 'orderly-stop': stopped" in caplog.text
    assert not any(record.levelname == "WARNING" for record in caplog.records)


def test_a_live_foreign_holder_blocks_acquisition_and_taints(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")

    async def scenario():
        holder = LeaderElector("contended")
        # A different, genuinely live pid: the parent process.
        foreign = Identity(
            holder="other-runtime",
            pid=os.getppid(),
            boot_id=lease_mod._boot_id(),
            host=holder.identity.host,
            started_at=time.time(),
        )
        record = LeaseRecord(
            name="contended",
            identity=foreign,
            acquired_at=time.time(),
            renewed_at=time.time(),
            lease_duration_s=15.0,
        )
        await holder._write(record)

        challenger = LeaderElector("contended")
        assert await challenger.try_acquire_or_renew() is False
        assert challenger.is_leader is False
        assert challenger.observed_leader() == "other-runtime"
        return challenger

    challenger = asyncio.run(scenario())
    assert taint_mod.is_tainted(taint_mod.TaintFlag.DUPLICATE_RUNTIME)
    assert challenger.report()["is_leader"] is False


def test_an_expired_lease_is_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")

    async def scenario():
        elector = LeaderElector("stale")
        stale = LeaseRecord(
            name="stale",
            identity=Identity(
                holder="ghost",
                pid=os.getppid(),
                boot_id=lease_mod._boot_id(),
                host=elector.identity.host,
                started_at=0.0,
            ),
            acquired_at=0.0,
            renewed_at=time.time() - 1000.0,
            lease_duration_s=15.0,
        )
        await elector._write(stale)
        assert await elector.try_acquire_or_renew() is True
        return elector.report()

    report = asyncio.run(scenario())
    assert report["is_leader"] is True
    assert report["record"]["transitions"] == 1, "leadership actually moved"


def test_a_dead_holders_lease_is_reclaimed_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")

    async def scenario():
        elector = LeaderElector("crashed")
        # A pid that cannot exist, with a still-valid expiry: only the
        # liveness check can reclaim this.
        record = LeaseRecord(
            name="crashed",
            identity=Identity(
                holder="dead",
                pid=2**30,
                boot_id=lease_mod._boot_id(),
                host=elector.identity.host,
                started_at=time.time(),
            ),
            acquired_at=time.time(),
            renewed_at=time.time(),
            lease_duration_s=3600.0,
        )
        await elector._write(record)
        return await elector.try_acquire_or_renew()

    assert asyncio.run(scenario()) is True


def test_lease_storage_read_does_not_block_the_event_loop(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")
    elector = LeaderElector("slow-read")
    main_thread = threading.get_ident()
    reader_threads: list[int] = []
    real_read = elector._read

    def slow_read():
        reader_threads.append(threading.get_ident())
        time.sleep(0.08)
        return real_read()

    monkeypatch.setattr(elector, "_read", slow_read)

    async def scenario() -> bool:
        task = asyncio.create_task(elector.try_acquire_or_renew())
        await asyncio.sleep(0.01)
        loop_remained_live = not task.done()
        await task
        return loop_remained_live

    assert asyncio.run(scenario()) is True
    assert reader_threads and reader_threads[0] != main_thread


def test_holder_gives_up_before_a_challenger_can_take_over(tmp_path, monkeypatch):
    """The safety property: never two leaders."""
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")

    async def scenario():
        stopped: list[str] = []
        elector = LeaderElector(
            "margin",
            lease_duration_s=15.0,
            renew_deadline_s=10.0,
            on_stopped_leading=lambda: stopped.append("stopped"),
        )
        await elector.try_acquire_or_renew()
        assert elector.is_leader

        # Simulate renewals failing for longer than the renew deadline but
        # less than the lease duration: the holder must give up here, in
        # the gap, before any challenger is allowed to acquire.
        elector._last_renew_ok = time.time() - 11.0
        await elector._check_renew_deadline()
        return elector.is_leader, stopped

    is_leader, stopped = asyncio.run(scenario())
    assert is_leader is False, "the holder must relinquish on its own"
    assert stopped == ["stopped"]


def test_should_act_as_singleton_fails_open_for_protective_work(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")
    # No elector registered at all: protective work must still run.
    assert lease_mod.should_act_as_singleton("nobody") is True


def test_is_leader_fails_closed_for_exclusive_work():
    assert lease_mod.is_leader("nobody") is False


def test_stop_releases_the_lease_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_mod, "_lease_path", lambda name: tmp_path / f"{name}.json")

    async def scenario():
        first = LeaderElector("handoff")
        await first.try_acquire_or_renew()
        await first.stop()

        second = LeaderElector("handoff")
        # Without an explicit release the successor would wait out the
        # full lease duration.
        return await second.try_acquire_or_renew()

    assert asyncio.run(scenario()) is True


# ── invariants ────────────────────────────────────────────────────────

def test_orchestration_invariants_are_registered_and_clean():
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import get_registry, verify

    names = {s.name for s in get_registry().specs()}
    for expected in (
        "admission.validators_do_not_mutate",
        "quota.guaranteed_specs_are_coherent",
        "eviction.guaranteed_organs_are_protected",
        "eviction.thresholds_are_ordered",
        "lease.no_live_duplicate_holder",
    ):
        assert expected in names

    report = verify("orchestration", record=False)
    assert report.ok, report.summary()


def test_default_eviction_thresholds_are_ordered():
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import verify

    evict_mod.reset_eviction_for_test()
    report = verify("orchestration", record=False)
    ordering = [v for v in report.violations if v.invariant == "eviction.thresholds_are_ordered"]
    assert ordering == [], f"shipped defaults violate their own invariant: {ordering}"
