from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.model_lane_control import (
    InProcessModelLaneLease,
    LaneClaim,
    LaneOwnerObservation,
    LaneTransactionState,
    ModelLaneControlError,
    ModelLaneController,
    ProcessIdentity,
    ProcessLiveness,
    StandaloneModelLaneLease,
    SynchronousInProcessModelLaneLease,
    acquire_in_process_model_lane,
    acquire_standalone_model_lane,
    acquire_synchronous_in_process_model_lane,
    compensate_registered_model_owner,
    discover_external_model_processes,
    estimate_model_job_footprint_gb,
    evict_managed_process_owner,
    evict_registered_model_owner,
    infer_model_process_claim,
    managed_process_group_liveness,
    register_model_lane_owner_adapter,
    unregister_model_lane_owner_adapter,
)
from core.runtime.receipts import ReceiptStore

pytestmark = pytest.mark.unit


def test_frozen_controller_training_uses_its_measured_workload_class(
    tmp_path: Path,
) -> None:
    model = tmp_path / "quantized-model"
    model.mkdir()
    weights = model / "weights.safetensors"
    with weights.open("wb") as stream:
        stream.truncate(8 * 1024**3)

    frozen_controller_gb = estimate_model_job_footprint_gb(
        str(model),
        purpose="train_frozen_controller",
    )
    broader_training_gb = estimate_model_job_footprint_gb(
        str(model),
        purpose="train",
    )

    assert frozen_controller_gb == pytest.approx(16.0)
    assert broader_training_gb == pytest.approx(18.0)
    assert frozen_controller_gb < broader_training_gb


class AliveTable:
    def __init__(self, *pids: int) -> None:
        self.alive = {os.getpid(), *pids}

    def __call__(self, identity: ProcessIdentity) -> bool:
        return identity.pid in self.alive and identity.started_at > 0.0


class MutableLivenessProbe:
    def __init__(self, result: bool | ProcessLiveness = True) -> None:
        self.result = result
        self.error: Exception | None = None

    def __call__(self, _identity: ProcessIdentity) -> bool | ProcessLiveness:
        if self.error is not None:
            raise self.error
        return self.result


def _owner(
    owner_id: str,
    model_path: str,
    declared_gb: float,
    pid: int,
    *,
    purpose: str = "serve",
    preemptible: bool = True,
) -> LaneOwnerObservation:
    return LaneOwnerObservation(
        owner_id=owner_id,
        model_path=model_path,
        declared_gb=declared_gb,
        purpose=purpose,
        process=ProcessIdentity(pid, float(pid)),
        preemptible=preemptible,
    )


def _controller(tmp_path: Path, alive: AliveTable) -> ModelLaneController:
    return ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=alive,
        process_discovery=None,
    )


@pytest.fixture(autouse=True)
def _fixed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")


@pytest.fixture(autouse=True)
def _owned_receipt_stores(monkeypatch: pytest.MonkeyPatch):
    constructor = ReceiptStore
    stores: list[ReceiptStore] = []

    def _tracked_receipt_store(*args, **kwargs) -> ReceiptStore:
        store = constructor(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(sys.modules[__name__], "ReceiptStore", _tracked_receipt_store)
    try:
        yield
    finally:
        for store in reversed(stores):
            store.close()


def test_commit_persists_fenced_process_owner_and_one_terminal_receipt(tmp_path: Path) -> None:
    alive = AliveTable(101)
    controller = _controller(tmp_path, alive)
    claim = LaneClaim(
        owner_id="mlx:root:cortex",
        model_path="/m/Aura-32B-cortex",
        request_gb=23.0,
        request_id="commit-once",
    )

    reserved = controller.reserve_sync(claim)
    assert reserved.admitted is True
    assert reserved.ready_to_spawn is True

    committed = controller.commit_sync(
        reserved,
        process=ProcessIdentity(101, 101.0),
        observed_gb=20.5,
    )
    replay = controller.commit_sync(
        committed,
        process=ProcessIdentity(101, 101.0),
        observed_gb=20.5,
    )

    assert committed.state is LaneTransactionState.COMMITTED
    assert replay.replayed is True
    assert replay.receipt_id == committed.receipt_id
    snapshot = controller.snapshot()
    assert snapshot["committed_gb"] == pytest.approx(23.0)
    assert snapshot["reserved_gb"] == 0.0
    assert snapshot["owners"][0]["fencing_token"] == reserved.fencing_token
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 1


def test_same_owner_cannot_reserve_or_replace_a_live_model_implicitly(tmp_path: Path) -> None:
    alive = AliveTable(101, 102)
    controller = _controller(tmp_path, alive)
    first = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:stable:cortex",
            model_path="/m/Aura-32B-cortex",
            request_gb=30.0,
            request_id="same-owner-first",
        )
    )

    in_flight = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:stable:cortex",
            model_path="/m/Aura-32B-cortex",
            request_gb=30.0,
            request_id="same-owner-in-flight",
        )
    )
    assert in_flight.admitted is False
    assert in_flight.reason == "owner_id_reservation_in_flight:mlx:stable:cortex"

    committed = controller.commit_sync(
        first,
        process=ProcessIdentity(101, 101.0),
    )
    replacement = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:stable:cortex",
            model_path="/m/Aura-32B-cortex-v2",
            request_gb=30.0,
            request_id="same-owner-replacement",
        )
    )

    assert committed.state is LaneTransactionState.COMMITTED
    assert replacement.admitted is False
    assert replacement.reason == "owner_id_already_committed:mlx:stable:cortex"
    snapshot = controller.snapshot()
    assert snapshot["committed_gb"] == pytest.approx(30.0)
    assert snapshot["reserved_gb"] == pytest.approx(0.0)
    assert len(snapshot["owners"]) == 1
    assert snapshot["owners"][0]["process"] == {"pid": 101, "started_at": 101.0}


def test_exclusive_claim_requires_and_preserves_an_empty_lane(tmp_path: Path) -> None:
    alive = AliveTable(101, 102)
    controller = _controller(tmp_path, alive)
    existing = controller.reserve_sync(
        LaneClaim(
            owner_id="serve:cortex",
            model_path="/m/Aura-32B-cortex",
            request_gb=20.0,
            request_id="existing-owner",
        )
    )
    controller.commit_sync(existing, process=ProcessIdentity(101, 101.0))

    refused = controller.reserve_sync(
        LaneClaim(
            owner_id="train:exclusive",
            model_path="/m/Aura-32B-cortex",
            request_gb=20.0,
            purpose="train",
            exclusive=True,
            request_id="exclusive-refused",
        )
    )

    assert refused.admitted is False
    assert refused.reason == "exclusive_lane_requires_zero_owners"


def test_exclusive_reservation_atomically_blocks_following_claims(tmp_path: Path) -> None:
    alive = AliveTable(101)
    controller = _controller(tmp_path, alive)
    exclusive = controller.reserve_sync(
        LaneClaim(
            owner_id="train:exclusive",
            model_path="/m/Aura-32B-cortex",
            request_gb=20.0,
            purpose="train",
            exclusive=True,
            request_id="exclusive-first",
        )
    )

    while_reserved = controller.reserve_sync(
        LaneClaim(
            owner_id="serve:late",
            model_path="/m/qwen-1.5b",
            request_gb=2.0,
            request_id="ordinary-while-exclusive-reserved",
        )
    )
    assert exclusive.ready_to_spawn is True
    assert while_reserved.admitted is False
    assert while_reserved.reason == "exclusive_lane_reserved:exclusive-first"

    controller.commit_sync(exclusive, process=ProcessIdentity(101, 101.0))
    while_owned = controller.reserve_sync(
        LaneClaim(
            owner_id="serve:later",
            model_path="/m/qwen-1.5b",
            request_gb=2.0,
            request_id="ordinary-while-exclusive-owned",
        )
    )
    assert while_owned.admitted is False
    assert while_owned.reason == "exclusive_lane_owned:train:exclusive"


def test_committed_replay_rejects_a_different_live_process(tmp_path: Path) -> None:
    alive = AliveTable(101, 102)
    controller = _controller(tmp_path, alive)
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:replay:cortex",
            model_path="/m/Aura-32B-cortex",
            request_gb=23.0,
            request_id="committed-replay-process",
        )
    )
    committed = controller.commit_sync(
        decision,
        process=ProcessIdentity(101, 101.0),
    )

    with pytest.raises(ModelLaneControlError, match="committed_replay_process_mismatch"):
        controller.commit_sync(
            committed,
            process=ProcessIdentity(102, 102.0),
        )


def test_expired_reservation_cannot_commit_or_issue_delegation(tmp_path: Path) -> None:
    now = [100.0]
    alive = AliveTable(101)
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=alive,
        process_discovery=None,
        clock=lambda: now[0],
    )
    commit_candidate = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:expired:commit",
            model_path="/m/qwen-7b",
            request_gb=5.0,
            reservation_ttl_s=5.0,
            request_id="expired-commit",
        )
    )
    delegation_candidate = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:expired:delegation",
            model_path="/m/qwen-7b",
            request_gb=5.0,
            reservation_ttl_s=5.0,
            request_id="expired-delegation",
        )
    )
    now[0] = 106.0

    with pytest.raises(ModelLaneControlError, match="already_terminal:expired"):
        controller.commit_sync(
            commit_candidate,
            process=ProcessIdentity(101, 101.0),
        )
    with pytest.raises(ModelLaneControlError, match="already_terminal:expired"):
        controller.issue_inherited_claim_sync(delegation_candidate)

    states = {
        record["request_id"]: record["state"]
        for record in controller.snapshot()["reservations"]
    }
    assert states == {
        "expired-commit": LaneTransactionState.EXPIRED.value,
        "expired-delegation": LaneTransactionState.EXPIRED.value,
    }


def test_request_replay_rejects_changed_capacity_or_disruption_claim(tmp_path: Path) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    original = LaneClaim(
        owner_id="mlx:replay",
        model_path="/m/qwen-7b",
        request_gb=5.0,
        request_id="exact-replay-only",
        metadata={"origin": "unit"},
    )
    controller.reserve_sync(original)

    with pytest.raises(ModelLaneControlError, match="different_claim"):
        controller.reserve_sync(
            LaneClaim(
                owner_id=original.owner_id,
                model_path=original.model_path,
                request_gb=25.0,
                request_id=original.request_id,
                metadata=original.metadata,
            )
        )
    with pytest.raises(ModelLaneControlError, match="different_claim"):
        controller.reserve_sync(
            LaneClaim(
                owner_id=original.owner_id,
                model_path=original.model_path,
                request_gb=original.request_gb,
                allow_disruptive_eviction=True,
                request_id=original.request_id,
                metadata=original.metadata,
            )
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_model_lane_resource_claims_reject_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="request_gb"):
        LaneClaim(owner_id="claim", model_path="/models/test", request_gb=invalid)
    with pytest.raises(ValueError, match="declared_gb"):
        LaneOwnerObservation(
            owner_id="owner",
            model_path="/models/test",
            declared_gb=invalid,
        )


def test_concurrent_controllers_cannot_spend_same_unreserved_capacity(tmp_path: Path) -> None:
    alive = AliveTable(201)
    first = _controller(tmp_path, alive)
    second = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=first._receipt_store,
        process_alive=alive,
        process_discovery=None,
    )
    observed = [_owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 201)]
    claims = (
        LaneClaim(
            owner_id="job:a",
            model_path="/m/qwen-7b-a",
            request_gb=20.0,
            request_id="race-a",
        ),
        LaneClaim(
            owner_id="job:b",
            model_path="/m/qwen-7b-b",
            request_gb=20.0,
            request_id="race-b",
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(
            pool.map(
                lambda pair: pair[0].reserve_sync(pair[1], observations=observed),
                ((first, claims[0]), (second, claims[1])),
            )
        )

    assert sum(decision.admitted for decision in decisions) == 1
    assert sum(not decision.admitted for decision in decisions) == 1
    snapshot = first.snapshot()
    assert snapshot["committed_gb"] == pytest.approx(20.0)
    assert snapshot["reserved_gb"] == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_required_eviction_completes_before_reservation_becomes_ready(tmp_path: Path) -> None:
    alive = AliveTable(301, 302, 303)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 301),
        _owner("mlx:brainstem", "/m/qwen-7b", 5.0, 302),
        _owner("trainer:nightly", "/m/trainer", 8.0, 303, purpose="train"),
    ]
    claim = LaneClaim(
        owner_id="mlx:solver",
        model_path="/m/Deep-72B-solver",
        request_gb=20.0,
        priority=87,
        preemptible=False,
        foreground=True,
        allow_disruptive_eviction=True,
        allow_last_warm_eviction=True,
        reservation_ttl_s=40.0,
        owner_lease_ttl_s=90.0,
        request_id="evict-first",
        metadata={"origin": "complete-reclamation-claim"},
    )
    decision = await controller.reserve(claim, observations=observations)
    assert decision.state is LaneTransactionState.EVICTING
    assert decision.evict_owner_ids == ("trainer:nightly",)

    events: list[str] = []
    reclaimed_claims: list[LaneClaim] = []

    async def evict(owner: LaneOwnerObservation, reason: str) -> bool:
        events.append(f"evict:{owner.owner_id}:{reason}")
        alive.alive.remove(owner.process.pid)
        observations[:] = [item for item in observations if item.owner_id != owner.owner_id]
        return True

    def reclaim(candidate: LaneClaim) -> bool:
        reclaimed_claims.append(candidate)
        return True

    ready = await controller.prepare(
        decision,
        evict=evict,
        observe=lambda: list(observations),
        reclaim=reclaim,
    )

    assert events and events[0].startswith("evict:trainer:nightly")
    assert ready.state is LaneTransactionState.READY
    assert ready.ready_to_spawn is True
    assert ready.evicted_owner_ids == ("trainer:nightly",)
    assert reclaimed_claims == [claim]
    snapshot = controller.snapshot()
    assert {owner["owner_id"] for owner in snapshot["owners"]} == {
        "mlx:cortex",
        "mlx:brainstem",
    }
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 1


@pytest.mark.asyncio
async def test_transient_runtime_cost_evicts_fallback_without_becoming_owner_memory(
    tmp_path: Path,
) -> None:
    alive = AliveTable(311, 312)
    controller = _controller(tmp_path, alive)
    observations = [_owner("mlx:fallback", "/m/qwen-7b", 10.0, 311)]
    claim = LaneClaim(
        owner_id="mlx:primary",
        model_path="/m/Aura-32B-cortex",
        request_gb=25.0,
        transient_runtime_gb=18.0,
        allow_last_warm_eviction=True,
        request_id="primary-over-runtime",
    )

    decision = await controller.reserve(claim, observations=observations)
    assert decision.state is LaneTransactionState.EVICTING
    assert decision.evict_owner_ids == ("mlx:fallback",)

    async def evict(owner: LaneOwnerObservation, _reason: str) -> bool:
        alive.alive.remove(owner.process.pid)
        observations.clear()
        return True

    ready = await controller.prepare(
        decision,
        evict=evict,
        observe=lambda: list(observations),
        reclaim=lambda _claim: True,
    )
    committed = controller.commit_sync(
        ready,
        process=ProcessIdentity(312, 312.0),
        observed_gb=22.0,
    )

    assert committed.state is LaneTransactionState.COMMITTED
    snapshot = controller.snapshot()
    assert snapshot["committed_gb"] == pytest.approx(25.0)
    assert snapshot["owners"][0]["declared_gb"] == pytest.approx(25.0)


def test_transient_runtime_cost_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValueError, match="transient_runtime_gb"):
        LaneClaim(
            owner_id="mlx:primary",
            model_path="/m/Aura-32B-cortex",
            request_gb=25.0,
            transient_runtime_gb=-1.0,
        )


@pytest.mark.asyncio
async def test_failed_required_eviction_cancels_candidate_and_receipts_failure(tmp_path: Path) -> None:
    alive = AliveTable(401, 402)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 401),
        _owner("trainer:nightly", "/m/trainer", 25.0, 402, purpose="train"),
    ]
    claim = LaneClaim(
        owner_id="mlx:reflex",
        model_path="/m/qwen-1.5b-reflex",
        request_gb=4.0,
        request_id="eviction-fails",
    )
    decision = await controller.reserve(claim, observations=observations)
    assert decision.state is LaneTransactionState.EVICTING

    cancelled = await controller.prepare(
        decision,
        evict=lambda _owner, _reason: False,
        observe=lambda: list(observations),
    )

    assert cancelled.admitted is False
    assert cancelled.state is LaneTransactionState.CANCELLED
    assert "required_eviction_failed" in cancelled.reason
    assert cancelled.receipt_id
    assert controller.snapshot()["reserved_gb"] == 0.0
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 2


@pytest.mark.asyncio
async def test_wedged_eviction_callback_is_bounded_and_cancels_candidate(
    tmp_path: Path,
) -> None:
    alive = AliveTable(451, 452)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 451),
        _owner("trainer:wedged", "/m/trainer", 25.0, 452, purpose="train"),
    ]
    decision = await controller.reserve(
        LaneClaim(
            owner_id="mlx:reflex",
            model_path="/m/qwen-1.5b-reflex",
            request_gb=4.0,
            request_id="eviction-wedged",
        ),
        observations=observations,
    )

    async def _never_returns(_owner: LaneOwnerObservation, _reason: str) -> bool:
        await asyncio.sleep(60.0)
        return True

    started = time.monotonic()
    cancelled = await controller.prepare(
        decision,
        evict=_never_returns,
        observe=lambda: observations,
        timeout_s=0.05,
    )

    assert time.monotonic() - started < 0.5
    assert cancelled.state is LaneTransactionState.CANCELLED
    assert "eviction_callback_timeout" in cancelled.reason
    assert cancelled.receipt_id == ""
    assert controller.snapshot()["reserved_gb"] == 0.0
    reservation = next(
        item
        for item in controller.snapshot()["reservations"]
        if item["request_id"] == decision.request_id
    )
    assert reservation["compensation_pending_owner_ids"] == ["trainer:wedged"]
    assert reservation["eviction_intents"]["trainer:wedged"]["state"] == "invoking"


@pytest.mark.asyncio
async def test_eviction_intent_is_durable_before_callback_and_compensation(
    tmp_path: Path,
) -> None:
    alive = AliveTable(455, 456)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 455),
        _owner("trainer:ambiguous", "/m/trainer", 25.0, 456, purpose="train"),
    ]
    decision = await controller.reserve(
        LaneClaim(
            owner_id="mlx:reflex",
            model_path="/m/qwen-1.5b-reflex",
            request_gb=4.0,
            request_id="eviction-intent-before-effect",
        ),
        observations=observations,
    )
    callback_observed_intent = False
    compensation_observed_pending = False

    async def ambiguous_evict(owner: LaneOwnerObservation, _reason: str) -> bool:
        nonlocal callback_observed_intent
        reservation = next(
            item
            for item in controller.snapshot()["reservations"]
            if item["request_id"] == decision.request_id
        )
        intent = reservation["eviction_intents"][owner.owner_id]
        callback_observed_intent = bool(
            intent["state"] == "invoking"
            and intent["owner"]["process"] == owner.process.to_dict()
        )
        await asyncio.sleep(60.0)
        return True

    async def compensate(owner: LaneOwnerObservation, _reason: str) -> bool:
        nonlocal compensation_observed_pending
        reservation = next(
            item
            for item in controller.snapshot()["reservations"]
            if item["request_id"] == decision.request_id
        )
        compensation_observed_pending = owner.owner_id in reservation[
            "compensation_pending_owner_ids"
        ]
        return True

    cancelled = await controller.prepare(
        decision,
        evict=ambiguous_evict,
        observe=lambda: observations,
        compensate=compensate,
        timeout_s=0.05,
    )

    assert callback_observed_intent is True
    assert compensation_observed_pending is True
    assert cancelled.receipt_id
    reservation = next(
        item
        for item in controller.snapshot()["reservations"]
        if item["request_id"] == decision.request_id
    )
    assert reservation["compensation_pending_owner_ids"] == []
    assert reservation["compensation"]["trainer:ambiguous"] is True


@pytest.mark.asyncio
async def test_replayed_ambiguous_eviction_compensates_without_reinvoking_unload(
    tmp_path: Path,
) -> None:
    alive = AliveTable(457, 458)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 457),
        _owner("trainer:crash-window", "/m/trainer", 25.0, 458, purpose="train"),
    ]
    decision = await controller.reserve(
        LaneClaim(
            owner_id="mlx:reflex",
            model_path="/m/qwen-1.5b-reflex",
            request_gb=4.0,
            request_id="eviction-intent-replay",
        ),
        observations=observations,
    )
    controller._arm_eviction_intent_sync(decision, "trainer:crash-window")
    compensation_calls: list[str] = []

    async def compensate(owner: LaneOwnerObservation, _reason: str) -> bool:
        compensation_calls.append(owner.owner_id)
        return True

    recovered = await controller.prepare(
        decision,
        evict=lambda *_args: pytest.fail("ambiguous unload must not be invoked twice"),
        observe=lambda: observations,
        compensate=compensate,
    )

    assert recovered.state is LaneTransactionState.CANCELLED
    assert recovered.reason == "ambiguous_eviction_recovered:trainer:crash-window"
    assert recovered.receipt_id
    assert compensation_calls == ["trainer:crash-window"]


@pytest.mark.asyncio
async def test_failed_ambiguous_compensation_remains_pending_and_retries(
    tmp_path: Path,
) -> None:
    alive = AliveTable(459, 460)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 459),
        _owner("trainer:retry-restore", "/m/trainer", 25.0, 460, purpose="train"),
    ]
    decision = await controller.reserve(
        LaneClaim(
            owner_id="mlx:reflex",
            model_path="/m/qwen-1.5b-reflex",
            request_gb=4.0,
            request_id="eviction-compensation-retry",
        ),
        observations=observations,
    )

    async def ambiguous_evict(_owner: LaneOwnerObservation, _reason: str) -> bool:
        await asyncio.sleep(60.0)
        return True

    cancelled = await controller.prepare(
        decision,
        evict=ambiguous_evict,
        observe=lambda: observations,
        compensate=lambda *_args: False,
        timeout_s=0.05,
    )
    assert cancelled.receipt_id == ""
    reservation = next(
        item
        for item in controller.snapshot()["reservations"]
        if item["request_id"] == decision.request_id
    )
    assert reservation["compensation_pending_owner_ids"] == ["trainer:retry-restore"]
    assert reservation["compensation"]["trainer:retry-restore"] is False

    restored = await controller.reconcile_expired_compensations(
        compensate=lambda *_args: True,
    )

    assert restored == 1
    reservation = next(
        item
        for item in controller.snapshot()["reservations"]
        if item["request_id"] == decision.request_id
    )
    assert reservation["compensation_pending_owner_ids"] == []
    assert reservation["compensation"]["trainer:retry-restore"] is True
    assert reservation["terminal_receipt_id"]


@pytest.mark.asyncio
async def test_direct_and_recovered_compensation_share_one_durable_claim(
    tmp_path: Path,
) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="candidate:serialized-restore",
            model_path="/models/candidate-1.5b",
            request_gb=0.1,
            request_id="serialized-compensation-claim",
        )
    )
    displaced = _owner(
        "owner:serialized-restore",
        "/models/displaced-1.5b",
        0.1,
        os.getpid(),
    )
    callback_started = asyncio.Event()
    finish_callback = asyncio.Event()
    calls: list[str] = []

    async def compensate(owner: LaneOwnerObservation, _reason: str) -> bool:
        calls.append(owner.owner_id)
        callback_started.set()
        await finish_callback.wait()
        return True

    direct_cancel = asyncio.create_task(
        controller.cancel(
            decision,
            reason="candidate_failed",
            compensate=compensate,
            evicted=[displaced],
        )
    )
    await callback_started.wait()

    concurrent_recovery = await controller.reconcile_expired_compensations(
        compensate=compensate,
        request_id=decision.request_id,
    )
    assert concurrent_recovery == 0
    assert calls == [displaced.owner_id]

    finish_callback.set()
    cancelled = await direct_cancel
    assert cancelled.receipt_id
    assert calls == [displaced.owner_id]
    reservation = next(
        item
        for item in controller.snapshot()["reservations"]
        if item["request_id"] == decision.request_id
    )
    assert reservation["compensation_claims"] == {}
    assert reservation["compensation_pending_owner_ids"] == []


@pytest.mark.asyncio
async def test_blocking_sync_eviction_runs_off_loop_and_drains_before_timeout_returns(
    tmp_path: Path,
) -> None:
    alive = AliveTable(461, 462)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner("mlx:cortex", "/m/Aura-32B-cortex", 20.0, 461),
        _owner("trainer:blocking", "/m/trainer", 25.0, 462, purpose="train"),
    ]
    decision = await controller.reserve(
        LaneClaim(
            owner_id="mlx:reflex",
            model_path="/m/qwen-1.5b-reflex",
            request_gb=4.0,
            request_id="eviction-blocking-sync",
        ),
        observations=observations,
    )
    callback_finished = threading.Event()
    loop_progressed = asyncio.Event()

    def blocking_evict(_owner: LaneOwnerObservation, _reason: str) -> bool:
        time.sleep(0.15)
        callback_finished.set()
        return False

    async def prove_loop_progress() -> None:
        await asyncio.sleep(0.005)
        loop_progressed.set()

    progress_task = asyncio.create_task(prove_loop_progress())
    started = time.monotonic()
    cancelled = await controller.prepare(
        decision,
        evict=blocking_evict,
        observe=lambda: observations,
        timeout_s=0.1,
    )
    elapsed = time.monotonic() - started
    await progress_task

    assert loop_progressed.is_set()
    assert callback_finished.is_set()
    assert elapsed >= 0.14
    assert elapsed < 0.5
    assert cancelled.state is LaneTransactionState.CANCELLED
    assert "eviction_callback_timeout" in cancelled.reason


def test_non_preemptible_required_owner_refuses_without_side_effect(tmp_path: Path) -> None:
    alive = AliveTable(501)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner(
            "trainer:operator",
            "/m/trainer",
            45.0,
            501,
            purpose="train",
            preemptible=False,
        )
    ]
    claim = LaneClaim(
        owner_id="mlx:cortex",
        model_path="/m/Aura-32B-cortex",
        request_gb=23.0,
        request_id="non-preemptible",
        allow_last_warm_eviction=True,
    )

    decision = controller.reserve_sync(claim, observations=observations)

    assert decision.admitted is False
    assert decision.reason == "required_eviction_not_preemptible:trainer:operator"
    assert controller.snapshot()["owners"][0]["eviction_requested_by"] == ""


def test_last_warm_lane_is_not_evicted_into_a_cold_gap(tmp_path: Path) -> None:
    alive = AliveTable(601)
    controller = _controller(tmp_path, alive)
    observations = [_owner("mlx:solver", "/m/Deep-72B-solver", 41.0, 601)]
    claim = LaneClaim(
        owner_id="mlx:cortex",
        model_path="/m/Aura-32B-cortex",
        request_gb=23.0,
        request_id="last-warm",
    )

    decision = controller.reserve_sync(claim, observations=observations)

    assert decision.admitted is False
    assert decision.reason == "disruption_budget:last_warm_lane"


def test_foreground_disruptive_swap_can_fence_idle_guaranteed_owner(tmp_path: Path) -> None:
    alive = AliveTable(611)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner(
            "mlx:cortex",
            "/m/Aura-32B-cortex",
            38.0,
            611,
            preemptible=True,
        )
    ]
    claim = LaneClaim(
        owner_id="mlx:solver",
        model_path="/m/Deep-72B-solver",
        request_gb=46.0,
        foreground=True,
        allow_disruptive_eviction=True,
        allow_last_warm_eviction=True,
        request_id="foreground-deep-swap",
    )

    decision = controller.reserve_sync(claim, observations=observations)

    assert decision.admitted is True
    assert decision.evict_owner_ids == ("mlx:cortex",)


def test_disruptive_swap_cannot_fence_busy_guaranteed_owner(tmp_path: Path) -> None:
    alive = AliveTable(612)
    controller = _controller(tmp_path, alive)
    observations = [
        _owner(
            "mlx:cortex",
            "/m/Aura-32B-cortex",
            38.0,
            612,
            preemptible=False,
        )
    ]
    claim = LaneClaim(
        owner_id="mlx:solver",
        model_path="/m/Deep-72B-solver",
        request_gb=46.0,
        foreground=True,
        allow_disruptive_eviction=True,
        allow_last_warm_eviction=True,
        request_id="busy-deep-swap",
    )

    decision = controller.reserve_sync(claim, observations=observations)

    assert decision.admitted is False
    assert decision.reason == "required_eviction_not_preemptible:mlx:cortex"


def test_stale_fence_cannot_commit_after_transaction_cancel(tmp_path: Path) -> None:
    alive = AliveTable(701)
    controller = _controller(tmp_path, alive)
    claim = LaneClaim(
        owner_id="mlx:brainstem",
        model_path="/m/qwen-7b",
        request_gb=5.0,
        request_id="old-fence",
    )
    old = controller.reserve_sync(claim)
    controller.cancel_sync(old, reason="spawn_failed")

    with pytest.raises(ModelLaneControlError, match="already_terminal"):
        controller.commit_sync(old, process=ProcessIdentity(701, 701.0))


def test_dead_reservation_owner_is_recovered_before_next_admission(tmp_path: Path) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    first = controller.reserve_sync(
        LaneClaim(
            owner_id="job:first",
            model_path="/m/qwen-7b-first",
            request_gb=30.0,
            request_id="abandoned",
        )
    )
    assert first.admitted is True

    alive.alive.remove(os.getpid())
    replacement = controller.reserve_sync(
        LaneClaim(
            owner_id="job:replacement",
            model_path="/m/qwen-7b-replacement",
            request_gb=30.0,
            request_id="replacement",
        )
    )

    assert replacement.admitted is True
    envelope = controller._load_locked()
    assert envelope["reservations"]["abandoned"]["state"] == "expired"


@pytest.mark.asyncio
async def test_in_process_lease_counts_memory_until_explicit_release(tmp_path: Path) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)

    lease = await acquire_in_process_model_lane(
        owner_id="post-training-validator",
        model_path="/models/validator-7b",
        purpose="benchmark",
        request_gb=0.1,
        owner_lease_ttl_s=15.0,
        controller=controller,
    )

    snapshot = controller.snapshot()
    assert snapshot["committed_gb"] == pytest.approx(0.1)
    assert snapshot["owners"][0]["metadata"]["lease_mode"] == "heartbeat"
    assert await lease.set_preemptible(False) is True
    non_preemptible_owner = controller.snapshot()["owners"][0]
    assert non_preemptible_owner["preemptible"] is False
    assert non_preemptible_owner["lease_expires_at"] - non_preemptible_owner[
        "heartbeat_at"
    ] == pytest.approx(15.0)
    assert await lease.set_preemptible(True) is True
    assert controller.snapshot()["owners"][0]["preemptible"] is True
    assert await lease.release(reason="validator_unloaded") is True
    assert controller.snapshot()["owners"] == []


@pytest.mark.asyncio
async def test_registered_local_evictor_is_used_before_process_signal_fallback() -> None:
    owner = _owner("in-process:test", "/models/test-7b", 5.0, os.getpid())
    calls: list[str] = []

    async def evict(_owner: LaneOwnerObservation, reason: str) -> bool:
        calls.append(reason)
        return True

    register_model_lane_owner_adapter(owner.owner_id, evict=evict)
    try:
        assert await evict_registered_model_owner(owner, "unit-preemption") is True
    finally:
        unregister_model_lane_owner_adapter(owner.owner_id)

    assert calls == ["unit-preemption"]


def test_standalone_spoofed_inheritance_falls_back_to_owned_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    monkeypatch.setenv("AURA_MODEL_LANE_INHERITED_OWNER_ID", "forged-owner")
    monkeypatch.setenv("AURA_MODEL_LANE_INHERITED_REQUEST_ID", "forged-request")
    monkeypatch.setenv("AURA_MODEL_LANE_INHERITED_MODEL_PATH", "/models/test-1b")
    monkeypatch.setenv("AURA_MODEL_LANE_INHERITED_PURPOSE", "benchmark")
    monkeypatch.setenv("AURA_MODEL_LANE_DELEGATION_TOKEN", "forged-token")

    lease = acquire_standalone_model_lane(
        owner_id="direct-tool",
        model_path="/models/test-1b",
        purpose="benchmark",
        request_gb=0.1,
        controller=controller,
    )

    assert lease.inherited is False
    assert lease.active is True
    assert controller.snapshot()["owners"][0]["owner_id"].startswith("standalone:")
    assert lease.release() is True
    assert lease.active is False
    assert controller.snapshot()["owners"] == []


def test_inherited_standalone_lease_closes_local_active_handle() -> None:
    lease = StandaloneModelLaneLease(
        controller=None,
        decision=None,
        inherited=True,
    )

    assert lease.active is True
    assert lease.release() is False
    assert lease.active is False
    assert lease.release() is False


def test_delegation_secret_is_hashed_and_wrong_token_is_rejected(tmp_path: Path) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="subprocess:delegated",
            model_path="/models/test-1b",
            request_gb=0.1,
            purpose="benchmark",
            request_id="delegated-request",
        )
    )

    token = controller.issue_inherited_claim_sync(decision)
    snapshot = controller.snapshot()
    delegation = snapshot["reservations"][0]["delegation"]

    assert token not in str(delegation)
    assert len(delegation["token_sha256"]) == 64
    assert controller.validate_inherited_claim(
        owner_id=decision.owner_id,
        request_id=decision.request_id,
        model_path=decision.model_path,
        purpose="benchmark",
        delegation_token="wrong-token",
        child_pid=os.getpid(),
        parent_pid=os.getppid(),
    ) is False


def test_inherited_child_is_bound_to_declared_model_roots_and_purposes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    fused_root = tmp_path / "fused-model"
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="subprocess:compound",
            model_path=str(tmp_path / "base-model"),
            request_gb=40.0,
            purpose="compound",
            request_id="compound-request",
            metadata={
                "allow_inherited_model_children": True,
                "allowed_inherited_model_purposes": ["train", "fuse", "benchmark"],
                "allowed_inherited_model_roots": [str(fused_root)],
            },
        )
    )
    controller.issue_inherited_claim_sync(decision)
    committed = controller.commit_sync(
        decision,
        process=ProcessIdentity.current(observer=controller.resource_observer),
        metadata={
            "managed_model_process": True,
            "start_new_session": True,
            "process_group_id": os.getpgrp(),
            "process_session_id": os.getsid(0),
        },
    )
    monkeypatch.setattr(controller, "validate_inherited_claim", lambda **_kwargs: True)
    common = {
        "owner_id": committed.owner_id,
        "request_id": committed.request_id,
        "model_path": committed.model_path,
        "purpose": "compound",
        "delegation_token": "already-consumed-by-worker",
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "requested_gb": 20.0,
    }

    for invalid in (float("nan"), float("inf"), float("-inf")):
        assert controller.validate_inherited_child_claim(
            **{**common, "requested_gb": invalid},
            child_model_path=str(fused_root / "candidate"),
            child_purpose="benchmark",
        ) is False

    assert controller.validate_inherited_child_claim(
        **common,
        child_model_path=str(fused_root / "candidate"),
        child_purpose="benchmark",
    ) is True
    assert controller.validate_inherited_child_claim(
        **common,
        child_model_path=str(tmp_path / "unrelated-model"),
        child_purpose="benchmark",
    ) is False
    assert controller.validate_inherited_child_claim(
        **common,
        child_model_path=committed.model_path,
        child_purpose="serve",
    ) is False


def test_inherited_child_subleases_are_cumulative_idempotent_and_released(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    fused_root = tmp_path / "fused-model"
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="subprocess:compound-cumulative",
            model_path=str(tmp_path / "base-model"),
            request_gb=40.0,
            purpose="compound",
            request_id="compound-cumulative-request",
            metadata={
                "allow_inherited_model_children": True,
                "allowed_inherited_model_purposes": ["train", "fuse"],
                "allowed_inherited_model_roots": [str(fused_root)],
            },
        )
    )
    controller.issue_inherited_claim_sync(decision)
    committed = controller.commit_sync(
        decision,
        process=ProcessIdentity.current(observer=controller.resource_observer),
        metadata={
            "managed_model_process": True,
            "start_new_session": True,
            "process_group_id": os.getpgrp(),
            "process_session_id": os.getsid(0),
        },
    )
    monkeypatch.setattr(controller, "validate_inherited_claim", lambda **_kwargs: True)
    common = {
        "owner_id": committed.owner_id,
        "request_id": committed.request_id,
        "model_path": committed.model_path,
        "purpose": "compound",
        "delegation_token": "consumed-by-worker",
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "child_model_path": str(fused_root / "candidate"),
    }

    for invalid_ttl in (float("nan"), float("inf"), float("-inf"), 0.0):
        assert controller.validate_inherited_child_claim(
            **common,
            requested_gb=1.0,
            child_purpose="train",
            child_request_id="invalid-ttl",
            ttl_s=invalid_ttl,
        ) is False
    assert controller.validate_inherited_child_claim(
        **common,
        requested_gb=1.0,
        child_purpose="train",
        child_request_id="invalid\nchild",
    ) is False
    assert controller.validate_inherited_child_claim(
        **common,
        requested_gb=24.0,
        child_purpose="train",
        child_request_id="child-train",
    ) is True
    assert controller.validate_inherited_child_claim(
        **common,
        requested_gb=24.0,
        child_purpose="train",
        child_request_id="child-train",
    ) is True
    assert controller.validate_inherited_child_claim(
        **common,
        requested_gb=20.0,
        child_purpose="fuse",
        child_request_id="child-fuse",
    ) is False
    subleases = controller.snapshot()["reservations"][0]["delegation"]["child_subleases"]
    assert len(subleases) == 1
    assert sum(item["requested_gb"] for item in subleases.values()) == pytest.approx(24.0)

    assert controller.release_inherited_child_claim(
        owner_id=committed.owner_id,
        request_id=committed.request_id,
        child_request_id="child-train",
        child_pid=os.getpid(),
    ) is True
    assert controller.validate_inherited_child_claim(
        **common,
        requested_gb=20.0,
        child_purpose="fuse",
        child_request_id="child-fuse",
    ) is True
    subleases = controller.snapshot()["reservations"][0]["delegation"]["child_subleases"]
    assert len(subleases) == 1
    assert next(iter(subleases.values()))["child_request_id"] == "child-fuse"


@pytest.mark.asyncio
async def test_compensator_survives_owner_unregistration_for_failed_candidate() -> None:
    owner = _owner("in-process:recoverable", "/models/recoverable-7b", 5.0, os.getpid())
    calls: list[str] = []

    async def evict(_owner: LaneOwnerObservation, _reason: str) -> bool:
        return True

    async def compensate(_owner: LaneOwnerObservation, reason: str) -> bool:
        calls.append(reason)
        return len(calls) > 1

    register_model_lane_owner_adapter(
        owner.owner_id,
        evict=evict,
        compensate=compensate,
    )
    unregister_model_lane_owner_adapter(owner.owner_id)

    assert await compensate_registered_model_owner(owner, "candidate_spawn_failed") is False
    assert await compensate_registered_model_owner(owner, "candidate_spawn_retry") is True
    assert calls == ["candidate_spawn_failed", "candidate_spawn_retry"]
    assert await compensate_registered_model_owner(owner, "post_success_replay") is False


def test_process_table_discovery_accounts_for_external_model_identity(
    monkeypatch: pytest.MonkeyPatch,
    resource_observer,
) -> None:
    from core.runtime.resource_observation import ProcessObservation

    external_pid = os.getpid() + 100_000

    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=external_pid,
                ppid=1,
                create_time=1234.5,
                status="running",
                cmdline=(
                "/usr/bin/python3",
                "-m",
                "mlx_lm.server",
                "--model",
                "/models/qwen-7b",
                ),
                name="python3",
                rss_bytes=3 * 1024**3,
            )
        ]
    )
    monkeypatch.setattr("core.runtime.model_lane_control.os.getpgid", lambda _pid: 44_001)

    observations = discover_external_model_processes([], observer=resource_observer)

    assert len(observations) == 1
    observed = observations[0]
    assert observed.model_path == "/models/qwen-7b"
    assert observed.process == ProcessIdentity(external_pid, 1234.5)
    assert observed.observed_gb == pytest.approx(3.0)
    assert observed.preemptible is False
    assert observed.metadata["externally_discovered"] is True
    assert observed.metadata["model_identity_status"] == "resolved"


def test_process_table_discovery_ignores_non_owning_model_wrappers(
    resource_observer,
) -> None:
    from core.runtime.resource_observation import ProcessObservation

    shell_pid = os.getpid() + 110_000
    caffeinate_pid = shell_pid + 1
    model = "/models/qwen-1.5b"
    child_command = (
        "/usr/bin/python3",
        "tools/train_unified_intrinsic_recurrence.py",
        "--model",
        model,
    )
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=shell_pid,
                ppid=1,
                create_time=2234.5,
                status="running",
                cmdline=(
                    "/bin/zsh",
                    "-lc",
                    "python tools/train_unified_intrinsic_recurrence.py "
                    f"--model {model}",
                ),
                name="zsh",
                rss_bytes=1024,
            ),
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=caffeinate_pid,
                ppid=shell_pid,
                create_time=2235.5,
                status="running",
                cmdline=("/usr/bin/caffeinate", "-dims", *child_command),
                name="caffeinate",
                rss_bytes=1024,
                ancestor_pids=(shell_pid,),
            ),
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=caffeinate_pid + 1,
                ppid=caffeinate_pid,
                create_time=2236.5,
                status="running",
                cmdline=child_command,
                name="python3",
                rss_bytes=2 * 1024**3,
                ancestor_pids=(caffeinate_pid, shell_pid),
            ),
        ]
    )

    observations = discover_external_model_processes([], observer=resource_observer)

    assert len(observations) == 1
    assert observations[0].process.pid == caffeinate_pid + 1
    assert observations[0].model_path == model


def test_process_table_discovery_fails_closed_on_unknown_identity_and_marks_escape(
    monkeypatch: pytest.MonkeyPatch,
    resource_observer,
) -> None:
    from core.runtime.resource_observation import ProcessObservation

    parent_pid = os.getpid() + 300_000
    child_pid = parent_pid + 1

    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=child_pid,
                ppid=parent_pid,
                create_time=4321.0,
                status="running",
                cmdline=("/usr/bin/python3", "-m", "mlx_lm.server"),
                name="python3",
                rss_bytes=2 * 1024**3,
                ancestor_pids=(parent_pid,),
            )
        ]
    )

    known_parent = LaneOwnerObservation(
        owner_id="managed-parent",
        model_path="/models/managed-7b",
        declared_gb=5.0,
        process=ProcessIdentity(parent_pid, 4000.0),
        metadata={"managed_model_process": True, "process_group_id": 41_000},
    )
    monkeypatch.setattr("core.runtime.model_lane_control.os.getpgid", lambda _pid: 42_000)

    observations = discover_external_model_processes(
        [known_parent],
        observer=resource_observer,
    )

    assert len(observations) == 1
    observed = observations[0]
    assert observed.model_path == f"unresolved:model-process:{child_pid}"
    assert observed.declared_gb == pytest.approx(46.0)
    assert observed.preemptible is False
    assert observed.metadata["model_identity_status"] == "unresolved_fail_closed"
    assert observed.metadata["process_tree_escape"] is True
    assert observed.metadata["registered_parent_owner_id"] == "managed-parent"


def test_managed_group_liveness_rejects_reused_pgid_from_another_session(
    monkeypatch: pytest.MonkeyPatch,
    resource_observer,
) -> None:
    from core.runtime.resource_observation import ProcessObservation

    root_pid = os.getpid() + 310_000
    descendant_pid = root_pid + 1
    process_group_id = root_pid
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=descendant_pid,
                ppid=1,
                create_time=5001.0,
                status="running",
                cmdline=("/usr/bin/python3", "worker.py"),
                name="python3",
                rss_bytes=1024,
                ancestor_pids=(),
            )
        ]
    )
    monkeypatch.setattr(
        "core.runtime.model_lane_control.os.getpgid",
        lambda pid: process_group_id if pid == descendant_pid else os.getpgrp(),
    )
    observed_session = [root_pid]
    monkeypatch.setattr(
        "core.runtime.model_lane_control.os.getsid",
        lambda _pid: observed_session[0],
    )

    assert managed_process_group_liveness(
        process_group_id,
        root_started_at=5000.0,
        session_id=root_pid,
        root_pid=root_pid,
        observer=resource_observer,
    ) is ProcessLiveness.ALIVE

    observed_session[0] = root_pid + 99
    assert managed_process_group_liveness(
        process_group_id,
        root_started_at=5000.0,
        session_id=root_pid,
        root_pid=root_pid,
        observer=resource_observer,
    ) is ProcessLiveness.DEAD


@pytest.mark.asyncio
async def test_managed_eviction_never_signals_reused_or_unattested_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, int]] = []
    monkeypatch.setattr(
        "core.runtime.model_lane_control._default_process_liveness",
        lambda _identity: ProcessLiveness.DEAD,
    )

    def group_liveness(process_group_id: int, **kwargs: int) -> ProcessLiveness:
        observed.append({"process_group_id": process_group_id, **kwargs})
        return ProcessLiveness.DEAD

    monkeypatch.setattr(
        "core.runtime.model_lane_control.managed_process_group_liveness",
        group_liveness,
    )
    monkeypatch.setattr(
        "core.runtime.model_lane_control.os.killpg",
        lambda *_args: pytest.fail("reused process group must never be signalled"),
    )
    owner = LaneOwnerObservation(
        owner_id="managed:finished-session",
        model_path="/models/finished-32b",
        declared_gb=23.0,
        process=ProcessIdentity(51_001, 5_001.0),
        preemptible=True,
        metadata={
            "managed_model_process": True,
            "start_new_session": True,
            "process_group_id": 51_001,
            "process_session_id": 51_001,
        },
    )

    assert await evict_managed_process_owner(owner, "test-reused-group") is True
    assert observed == [
        {
            "process_group_id": 51_001,
            "root_started_at": 5_001.0,
            "session_id": 51_001,
            "root_pid": 51_001,
        }
    ]


def test_external_discovery_drops_process_registered_during_scan(tmp_path: Path) -> None:
    alive = AliveTable(845)
    receipt_store = ReceiptStore(tmp_path / "receipts")
    state_path = tmp_path / "model_lanes.json"
    registrar = ModelLaneController(
        state_path=state_path,
        receipt_store=receipt_store,
        process_alive=alive,
        process_discovery=None,
    )
    registered = False

    def discover(_known: list[LaneOwnerObservation]) -> list[LaneOwnerObservation]:
        nonlocal registered
        if not registered:
            registered = True
            decision = registrar.reserve_sync(
                LaneClaim(
                    owner_id="mlx:registered-during-scan",
                    model_path="/models/canonical-7b",
                    request_gb=5.0,
                    request_id="registered-during-discovery",
                )
            )
            registrar.commit_sync(
                decision,
                process=ProcessIdentity(845, 845.0),
            )
        return [
            _owner(
                "external-model:845:845000000",
                "/models/canonical-7b",
                5.0,
                845,
            )
        ]

    controller = ModelLaneController(
        state_path=state_path,
        receipt_store=receipt_store,
        process_alive=alive,
        process_discovery=discover,
    )

    owners = controller.owner_observations()

    assert [owner.owner_id for owner in owners] == ["mlx:registered-during-scan"]
    assert controller.snapshot()["committed_gb"] == pytest.approx(5.0)


def test_external_nonpreemptible_process_is_included_without_explicit_observations(
    tmp_path: Path,
) -> None:
    external_pid = os.getpid() + 200_000
    alive = AliveTable(external_pid)
    external = _owner(
        "external-model:test",
        "/models/external-32b",
        44.0,
        external_pid,
        purpose="train",
        preemptible=False,
    )
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=alive,
        process_discovery=lambda _known: [external],
    )

    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="candidate",
            model_path="/models/candidate-7b",
            request_gb=5.0,
            request_id="external-owner-blocks",
        )
    )

    assert decision.admitted is False
    assert decision.reason == "required_eviction_not_preemptible:external-model:test"
    assert controller.snapshot()["owners"][0]["process"]["pid"] == external_pid


def test_offline_front_door_does_not_claim_accelerator_lane() -> None:
    claim = infer_model_process_claim(
        ["python", "tools/front_door_demo.py"],
        source="proof_tooling:front-door-offline",
        timeout_s=60.0,
    )

    assert claim is None


@pytest.mark.parametrize(
    ("program", "purpose"),
    (
        ("tools/train_unified_intrinsic_recurrence.py", "train"),
        ("tools/evaluate_unified_intrinsic_checkpoint.py", "benchmark"),
        ("tools/evaluate_unified_intrinsic_decoding.py", "benchmark"),
    ),
)
def test_unified_recurrence_commands_claim_the_resident_lane(
    program: str,
    purpose: str,
) -> None:
    claim = infer_model_process_claim(
        [sys.executable, program, "--model", "/models/Aura-32B"],
        source="unified-recurrence-test",
        timeout_s=60.0,
    )

    assert claim is not None
    assert claim.purpose == purpose
    assert claim.model_path == "/models/Aura-32B"


def test_standalone_lane_can_refuse_all_owner_eviction() -> None:
    decision = SimpleNamespace(
        admitted=True,
        evict_owner_ids=("serving:aura",),
        ready_to_spawn=False,
        reason="eviction_required",
        receipt_id="reservation-receipt",
    )
    cancelled = SimpleNamespace(
        reason="standalone_model_owner_eviction_forbidden",
        receipt_id="cancel-receipt",
    )

    class Controller:
        def validate_inherited_claim(self, **_kwargs: object) -> bool:
            return False

        def owner_observations(self) -> list[object]:
            return []

        def reserve_sync(self, _claim: object, *, observations: object) -> object:
            assert observations == []
            return decision

        def cancel_sync(self, candidate: object, *, reason: str) -> object:
            assert candidate is decision
            assert reason == "standalone_model_owner_eviction_forbidden"
            return cancelled

    with pytest.raises(ModelLaneControlError, match="owner_eviction_forbidden"):
        acquire_standalone_model_lane(
            owner_id="resident-training",
            model_path="/models/Aura-32B",
            purpose="train",
            request_gb=38.0,
            allow_owner_eviction=False,
            controller=Controller(),  # type: ignore[arg-type]
        )


def test_import_only_mlx_probe_does_not_claim_accelerator_lane() -> None:
    claim = infer_model_process_claim(
        [
            sys.executable,
            "-c",
            "import mlx.core as mx; import mlx_lm; print('mlx_runtime_ok')",
        ],
        source="runtime_probe:mlx_runtime_probe",
        timeout_s=25.0,
    )

    assert claim is None


def test_registered_integration_import_probe_does_not_claim_accelerator_lane() -> None:
    claim = infer_model_process_claim(
        [
            sys.executable,
            "-m",
            "core.runtime.integration_liveness_probe",
            "mlx_lm",
            "__aura_no_preflight__",
        ],
        source="integration_liveness.import_probe",
        timeout_s=25.0,
    )

    assert claim is None


def test_inline_mlx_model_call_still_fails_closed_without_identity() -> None:
    with pytest.raises(RuntimeError, match="missing_model_path"):
        infer_model_process_claim(
            [sys.executable, "-c", "import mlx_lm; mlx_lm.load('/models/qwen-7b')"],
            source="runtime_probe:mislabelled-model-load",
            timeout_s=25.0,
        )


@pytest.mark.asyncio
async def test_terminal_receipt_crash_replay_adopts_body_without_recompensating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="candidate:crash-replay",
            model_path="/models/candidate-1.5b",
            request_gb=0.1,
            request_id="terminal-crash-replay",
        )
    )
    displaced = _owner(
        "displaced:recoverable",
        "/models/displaced-1.5b",
        0.1,
        os.getpid(),
    )
    compensation_calls: list[str] = []

    async def compensate(_owner: LaneOwnerObservation, reason: str) -> bool:
        compensation_calls.append(reason)
        return True

    original_save = controller._save_locked
    failed_after_receipt = False

    def _fail_once_after_receipt(state: dict[str, object]) -> None:
        nonlocal failed_after_receipt
        reservations = state.get("reservations")
        records = reservations.values() if isinstance(reservations, dict) else ()
        has_receipt = any(
            bool(record.get("terminal_receipt_id"))
            for record in records
            if isinstance(record, dict)
        )
        if has_receipt and not failed_after_receipt:
            failed_after_receipt = True
            raise OSError("injected state save crash after durable receipt")
        original_save(state)

    monkeypatch.setattr(controller, "_save_locked", _fail_once_after_receipt)

    with pytest.raises(OSError, match="injected state save crash"):
        await controller.cancel(
            decision,
            reason="candidate_spawn_failed",
            compensate=compensate,
            evicted=[displaced],
        )

    replay = await controller.cancel(
        decision,
        reason="candidate_spawn_failed",
        compensate=compensate,
        evicted=[displaced],
    )

    assert replay.replayed is True
    assert replay.receipt_id
    assert compensation_calls == [
        f"compensate_failed_candidate:{decision.transaction_id}"
    ]
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 1
    reservation = controller.snapshot()["reservations"][0]
    assert reservation["terminal_receipt_id"] == replay.receipt_id


def test_eviction_receipt_replay_uses_existing_deterministic_body(tmp_path: Path) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    controller.reserve_sync(
        LaneClaim(
            owner_id="candidate:eviction-replay",
            model_path="/models/candidate-1.5b",
            request_gb=0.1,
            request_id="eviction-receipt-replay",
        )
    )
    reservation = controller.snapshot()["reservations"][0]
    owner = _owner("owner:evicted", "/models/evicted-1.5b", 0.1, os.getpid())

    first = controller._emit_eviction_receipt(
        reservation,
        owner=owner,
        outcome="evicted",
        reason="process_dead_and_owner_absent",
        completed_at=100.0,
    )
    replay = controller._emit_eviction_receipt(
        reservation,
        owner=owner,
        outcome="evicted",
        reason="different_replay_observation",
        completed_at=200.0,
    )

    assert first == replay
    assert first.startswith("resource_admission-eviction-")
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 1


def test_expired_abandoned_reservation_gets_terminal_receipt(tmp_path: Path) -> None:
    now = [100.0]
    alive = AliveTable()
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=alive,
        process_discovery=None,
        clock=lambda: now[0],
    )
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="candidate:abandoned",
            model_path="/models/candidate-1.5b",
            request_gb=0.1,
            request_id="abandoned-expiry-receipt",
            reservation_ttl_s=5.0,
        )
    )
    assert decision.ready_to_spawn is True

    now[0] = 106.0
    reservation = controller.snapshot()["reservations"][0]

    assert reservation["state"] == "expired"
    assert reservation["reason"] == "reservation_ttl_expired"
    assert reservation["terminal_receipt_id"]
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 1


@pytest.mark.asyncio
async def test_expired_partial_eviction_compensates_before_terminal_receipt(
    tmp_path: Path,
) -> None:
    now = [100.0]
    alive = AliveTable(811)
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=alive,
        process_discovery=None,
        clock=lambda: now[0],
    )
    displaced = _owner(
        "mlx:solver",
        "/models/Deep-72B-solver",
        40.0,
        811,
    )
    decision = controller.reserve_sync(
        LaneClaim(
            owner_id="mlx:cortex",
            model_path="/models/Aura-32B-cortex",
            request_gb=23.0,
            allow_last_warm_eviction=True,
            reservation_ttl_s=5.0,
            request_id="expired-after-eviction",
        ),
        observations=[displaced],
    )
    assert decision.evict_owner_ids == (displaced.owner_id,)

    state = controller._load_locked()
    record = state["reservations"][decision.request_id]
    state["owners"].pop(displaced.owner_id)
    record["evicted_owner_ids"] = [displaced.owner_id]
    record["evicted_owners"] = {
        displaced.owner_id: controller._observation_payload(displaced)
    }
    controller._save_locked(state)

    now[0] = 106.0
    expired = controller.snapshot()["reservations"][0]
    assert expired["state"] == "expired"
    assert expired["compensation_pending_owner_ids"] == [displaced.owner_id]
    assert expired["terminal_receipt_id"] == ""

    with pytest.raises(ModelLaneControlError, match="compensation_pending"):
        controller.reserve_sync(
            LaneClaim(
                owner_id="job:must-wait",
                model_path="/models/qwen-7b",
                request_gb=5.0,
                request_id="blocked-by-compensation",
            )
        )

    calls: list[str] = []

    async def _restore(owner: LaneOwnerObservation, reason: str) -> bool:
        assert owner.owner_id == displaced.owner_id
        calls.append(reason)
        return True

    assert await controller.reconcile_expired_compensations(compensate=_restore) == 1
    assert await controller.reconcile_expired_compensations(compensate=_restore) == 0

    recovered = controller.snapshot()["reservations"][0]
    assert recovered["compensation"] == {displaced.owner_id: True}
    assert recovered["compensation_pending_owner_ids"] == []
    assert recovered["terminal_receipt_id"]
    assert calls == [f"compensate_expired_candidate:{decision.transaction_id}"]


def test_synchronous_in_process_lease_counts_until_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    claims: list[LaneClaim] = []
    reserve_sync = controller.reserve_sync

    def _capture_claim(claim: LaneClaim):
        claims.append(claim)
        return reserve_sync(claim)

    monkeypatch.setattr(controller, "reserve_sync", _capture_claim)

    lease = acquire_synchronous_in_process_model_lane(
        owner_id="embedding-model",
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        purpose="serve",
        request_gb=0.25,
        transient_runtime_gb=0.75,
        controller=controller,
    )

    assert claims[0].transient_runtime_gb == pytest.approx(0.75)
    owner = controller.snapshot()["owners"][0]
    assert owner["declared_gb"] == pytest.approx(0.25)
    assert owner["metadata"]["synchronous_loader"] is True
    assert lease.set_preemptible(False) is True
    assert controller.snapshot()["owners"][0]["preemptible"] is False
    assert lease.set_preemptible(True) is True
    assert controller.snapshot()["owners"][0]["preemptible"] is True
    assert lease.release(reason="embedding_unloaded") is True
    assert controller.snapshot()["owners"] == []


@pytest.mark.asyncio
async def test_async_in_process_heartbeat_exception_requests_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_reasons: list[str] = []

    class BrokenController:
        async def heartbeat_owner(self, _owner_id: str, *, fencing_token: int) -> bool:
            assert fencing_token == 73
            raise ModelLaneControlError("durable heartbeat store unavailable")

    from core.runtime import shutdown_coordinator

    monkeypatch.setattr(shutdown_coordinator, "request_shutdown", shutdown_reasons.append)
    lease = object.__new__(InProcessModelLaneLease)
    lease.controller = BrokenController()
    lease.decision = SimpleNamespace(owner_id="in-process:test", fencing_token=73)
    lease._heartbeat_interval_s = 0.0
    lease._heartbeat_task = None
    lease._released = False

    await asyncio.wait_for(lease._heartbeat_loop(), timeout=0.1)

    assert shutdown_reasons == ["in_process_model_lane_fence_lost:in-process:test"]


@pytest.mark.asyncio
async def test_async_in_process_release_retries_after_durable_refusal() -> None:
    release_results = iter((False, True))
    release_calls: list[str] = []

    class Controller:
        async def heartbeat_owner(self, _owner_id: str, *, fencing_token: int) -> bool:
            assert fencing_token == 74
            return True

        async def release_owner(
            self,
            _owner_id: str,
            *,
            fencing_token: int,
            reason: str,
        ) -> bool:
            assert fencing_token == 74
            release_calls.append(reason)
            return next(release_results)

    lease = InProcessModelLaneLease(
        controller=Controller(),
        decision=SimpleNamespace(owner_id="in-process:retry", fencing_token=74),
        heartbeat_interval_s=60.0,
    )

    assert await lease.release(reason="first-attempt") is False
    assert lease._released is False
    assert lease._heartbeat_task is not None
    assert await lease.release(reason="second-attempt") is True
    assert lease._released is True
    assert release_calls == ["first-attempt", "second-attempt"]


def test_sync_in_process_release_retries_after_durable_refusal() -> None:
    release_results = iter((False, True))
    release_calls: list[str] = []

    class Controller:
        def heartbeat_owner_sync(self, _owner_id: str, *, fencing_token: int) -> bool:
            assert fencing_token == 75
            return True

        def release_owner_sync(
            self,
            _owner_id: str,
            *,
            fencing_token: int,
            reason: str,
        ) -> bool:
            assert fencing_token == 75
            release_calls.append(reason)
            return next(release_results)

    lease = SynchronousInProcessModelLaneLease(
        controller=Controller(),
        decision=SimpleNamespace(owner_id="in-process-sync:retry", fencing_token=75),
        heartbeat_interval_s=60.0,
    )

    assert lease.release(reason="first-attempt") is False
    assert lease._released is False
    assert lease._heartbeat_thread.is_alive()
    assert lease.release(reason="second-attempt") is True
    assert lease._released is True
    assert release_calls == ["first-attempt", "second-attempt"]


def test_live_in_process_owner_heartbeat_expiry_fails_closed_until_recovery(
    tmp_path: Path,
) -> None:
    now = [100.0]
    alive = AliveTable()
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=alive,
        process_discovery=None,
        clock=lambda: now[0],
    )
    lease = acquire_synchronous_in_process_model_lane(
        owner_id="stale-live-embedding",
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        purpose="serve",
        request_gb=0.25,
        owner_lease_ttl_s=15.0,
        controller=controller,
    )

    now[0] = 116.0
    stale = controller.snapshot()["owners"][0]

    assert stale["lease_ttl_s"] == pytest.approx(15.0)
    assert stale["metadata"]["heartbeat_lease_stale"] is True
    assert stale["preemptible"] is False
    assert lease.set_preemptible(True) is False

    now[0] = 117.0
    assert controller.heartbeat_owner_sync(
        lease.decision.owner_id,
        fencing_token=lease.decision.fencing_token,
    ) is True
    recovered = controller.snapshot()["owners"][0]
    assert "heartbeat_lease_stale" not in recovered["metadata"]
    assert recovered["preemptible"] is True
    assert recovered["lease_expires_at"] == pytest.approx(132.0)

    assert lease.release(reason="stale-heartbeat-test-complete") is True


def test_process_observation_failure_retains_and_fences_live_owner_until_recovery(
    tmp_path: Path,
) -> None:
    probe = MutableLivenessProbe()
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=probe,
        process_discovery=None,
    )
    claim = LaneClaim(
        owner_id="trainer:unknown-liveness",
        model_path="/m/Aura-32B-cortex",
        request_gb=23.0,
        request_id="unknown-liveness-owner",
    )
    decision = controller.reserve_sync(claim)
    controller.commit_sync(decision, process=ProcessIdentity(1901, 1901.0))

    probe.error = RuntimeError("process observer temporarily unavailable")
    retained = controller.snapshot()["owners"][0]

    assert retained["owner_id"] == claim.owner_id
    assert retained["preemptible"] is False
    assert retained["metadata"]["process_liveness_unknown"] is True
    assert retained["metadata"]["preemptible_before_liveness_unknown"] is True

    probe.error = None
    probe.result = ProcessLiveness.ALIVE
    recovered = controller.snapshot()["owners"][0]

    assert recovered["preemptible"] is True
    assert "process_liveness_unknown" not in recovered["metadata"]
    assert "preemptible_before_liveness_unknown" not in recovered["metadata"]


def test_unknown_candidate_liveness_cannot_commit_or_consume_reservation(
    tmp_path: Path,
) -> None:
    probe = MutableLivenessProbe()
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=probe,
        process_discovery=None,
    )
    claim = LaneClaim(
        owner_id="trainer:commit-unknown",
        model_path="/m/Aura-32B-cortex",
        request_gb=23.0,
        request_id="unknown-liveness-commit",
    )
    decision = controller.reserve_sync(claim)
    probe.result = ProcessLiveness.UNKNOWN

    with pytest.raises(ModelLaneControlError, match="candidate_process_identity_unobservable"):
        controller.commit_sync(decision, process=ProcessIdentity(1902, 1902.0))

    reservation = next(
        item
        for item in controller.snapshot()["reservations"]
        if item["request_id"] == claim.request_id
    )
    assert reservation["state"] == LaneTransactionState.READY.value


@pytest.mark.asyncio
async def test_in_process_eviction_verifies_unload_without_root_process_death(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resource_observer,
) -> None:
    resource_observer.configure_memory(
        available_bytes=64 * 1024**3,
        total_bytes=64 * 1024**3,
        percent=0.0,
    )
    alive = AliveTable()
    controller = _controller(tmp_path, alive)
    lease_holder: dict[str, object] = {}

    async def _evict(_owner: LaneOwnerObservation, _reason: str) -> bool:
        lease = lease_holder["lease"]
        assert hasattr(lease, "release")
        return bool(lease.release(reason="unit_in_process_eviction"))

    lease = acquire_synchronous_in_process_model_lane(
        owner_id="large-optional-model",
        model_path="/models/optional-32b",
        purpose="train",
        request_gb=44.0,
        preemptible=True,
        evict=_evict,
        controller=controller,
    )
    lease_holder["lease"] = lease
    candidate = await controller.reserve(
        LaneClaim(
            owner_id="foreground-candidate",
            model_path="/models/foreground-7b",
            request_gb=5.0,
            request_id="in-process-eviction-candidate",
        )
    )
    assert candidate.state is LaneTransactionState.EVICTING

    ready = await controller.prepare(
        candidate,
        evict=evict_registered_model_owner,
        observe=lambda: controller.owner_observations(),
        reclaim=lambda _claim: True,
    )

    assert ready.ready_to_spawn is True
    assert ready.evicted_owner_ids == (lease.decision.owner_id,)
    assert controller.snapshot()["owners"] == []
    controller.cancel_sync(ready, reason="unit_cleanup")
