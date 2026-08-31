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
