"""Estimated duration cannot release a model another task still owns."""

import asyncio

import pytest

from core.resilience.resource_arbitrator import ResourceArbitrator
from core.runtime.control_plane import (
    AdmissionRequest,
    PressureSnapshot,
    ResourceAdmissionController,
    WorkClass,
)


def controller():
    return ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
        poll_interval_s=0.01,
    )


def request(owner):
    return AdmissionRequest(
        owner=owner, work_class=WorkClass.INFERENCE, lane="cortex",
        timeout_s=0, lease_ttl_s=0.01,
    )


@pytest.mark.asyncio
async def test_active_owner_keeps_capacity_after_its_forecast_expires():
    admission = controller()
    held = await admission.acquire(request("owner"), holder_task=asyncio.current_task())
    await asyncio.sleep(0.02)
    assert admission.active_lease_count(WorkClass.INFERENCE) == 1
    assert not (await admission.acquire(request("competitor"))).admitted
    status = admission.status()["active_leases"][0]
    assert status["lifetime"] == "holder_task"
    assert status["holder_task_active"] is True
    await admission.release(held.lease_id)
    replacement = await admission.acquire(request("replacement"))
    assert replacement.admitted
    await admission.release(replacement.lease_id)


@pytest.mark.asyncio
async def test_an_owner_that_exits_without_release_cannot_leave_a_phantom():
    admission = controller()

    async def forget_to_release():
        return await admission.acquire(request("owner"), holder_task=asyncio.current_task())

    held = await asyncio.create_task(forget_to_release())
    assert held.admitted
    assert admission.active_lease_count(WorkClass.INFERENCE) == 0
    replacement = await admission.acquire(request("replacement"))
    assert replacement.admitted
    await admission.release(replacement.lease_id)


@pytest.mark.asyncio
async def test_an_acquirer_cannot_name_an_unrelated_holder():
    admission = controller()
    other = asyncio.create_task(asyncio.sleep(60))
    try:
        with pytest.raises(ValueError, match="acquiring task"):
            await admission.acquire(request("wrong-owner"), holder_task=other)
        assert admission.active_lease_count() == 0
    finally:
        other.cancel()
        with pytest.raises(asyncio.CancelledError):
            await other


@pytest.mark.asyncio
async def test_inference_context_holds_until_cancel_cleanup_has_finished():
    admission = controller()
    arbitrator = ResourceArbitrator(admission=admission)
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def generating():
        async with arbitrator.inference_context(worker="cortex", timeout=0):
            # Move beyond the forecast without waiting thirty wall-clock seconds.
            lease = next(iter(admission._leases.values()))
            lease.expires_at = 0
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await finish_cleanup.wait()

    task = asyncio.create_task(generating())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert not (await admission.acquire(request("before-cancel"))).admitted
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 1)
        assert not (await admission.acquire(request("during-cleanup"))).admitted
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admission.active_lease_count() == 0
        assert arbitrator.get_status()["inference_lanes"] == {}
    finally:
        finish_cleanup.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_model_load_holds_its_lease_for_as_long_as_the_load_runs():
    """LIVE 2026-09-21: "Model-load admission lease expired before release".

    The cortex load asked for a lease with a TTL derived from the handshake
    budget and nothing else. On a contended host the read of a 20GB model ran
    past it, so admission counted the memory as free while it was still being
    taken, and the release at the end found nothing to release. The whole
    load runs on the acquiring task, which is the lifetime the controller
    already knows how to hold.
    """
    import asyncio as _asyncio

    from core.brain.llm import mlx_client as mlx

    seen: dict[str, object] = {}

    class _Recorder:
        async def acquire(self, request, *, on_preempt=None, holder_task=None):
            seen["request"] = request
            seen["holder_task"] = holder_task
            raise _Stop

    class _Stop(Exception):
        pass

    class _Plane:
        admission = _Recorder()

    import hashlib

    from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

    class _Client:
        model_path = "/models/Aura-Qwen3.8-27B-persona-crsm"
        runtime_assignment = ModelRuntimeAssignment.issue(
            model_path=model_path,
            artifact_identity=hashlib.sha256(model_path.encode("utf-8")).hexdigest(),
            artifact_identity_kind="canonical_locator_sha256",
            artifact_identity_exact=False,
            role="cortex",
            purpose="serve",
            authority_source="test_model_registry",
        )

        def _warmup_timeout(self):
            return 180.0

        def _handshake_timeout(self):
            return 300.0

    import core.runtime.control_plane as cp

    original_plane = cp.get_runtime_control_plane
    original_footprint = mlx._declared_mlx_worker_footprint_gb
    cp.get_runtime_control_plane = lambda: _Plane()
    mlx._declared_mlx_worker_footprint_gb = lambda _path: 20.0
    try:
        with pytest.raises(_Stop):
            async with mlx._model_load_admission_context(
                _Client(), foreground_request=True
            ):
                pass
    finally:
        cp.get_runtime_control_plane = original_plane
        mlx._declared_mlx_worker_footprint_gb = original_footprint

    assert seen.get("holder_task") is _asyncio.current_task(), (
        "the model load must hold its lease for the life of the loading task"
    )
