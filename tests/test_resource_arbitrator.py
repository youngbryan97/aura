import asyncio
import uuid

import pytest

from core.resilience.resource_arbitrator import (
    ResourceArbitrator,
    get_resource_arbitrator,
    reset_resource_arbitrator,
)
from core.runtime.control_plane import (
    PressureSnapshot,
    ResourceAdmissionController,
    get_runtime_control_plane,
    reset_runtime_control_plane,
)
from core.runtime.model_runtime_assignment import (
    ModelRuntimeAssignment,
    locator_identity,
)


def _arbitrator(*, lock_path=None):
    return ResourceArbitrator(
        admission=ResourceAdmissionController(
            pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
            poll_interval_s=0.01,
        ),
        lock_path=lock_path,
    )


def test_default_facade_resolves_the_control_plane_admission_singleton():
    reset_resource_arbitrator()
    reset_runtime_control_plane()
    try:
        plane = get_runtime_control_plane()
        assert get_resource_arbitrator().admission is plane.admission
    finally:
        reset_resource_arbitrator()
        reset_runtime_control_plane()


def test_runtime_worker_labels_normalize_to_model_load_lanes():
    assert ResourceArbitrator._worker_lane("MLX-Cortex") == "cortex"
    assert ResourceArbitrator._worker_lane("mlx_brainstem") == "brainstem"
    assert ResourceArbitrator._worker_lane("custom-provider") == "custom-provider"


def test_suggestive_model_path_cannot_claim_a_lane_after_load():
    path = "/models/qwen-70b-solver"
    assert ResourceArbitrator._worker_lane(path) == path


def test_runtime_assignment_is_the_authority_for_model_lane():
    path = "/models/qwen-70b-solver"
    assignment = ModelRuntimeAssignment.issue(
        model_path=path,
        artifact_identity=locator_identity(path),
        artifact_identity_kind="canonical_locator_sha256",
        artifact_identity_exact=False,
        role="auxiliary",
        purpose="serve",
        authority_source="test",
    )
    assert ResourceArbitrator._worker_lane(assignment) == "auxiliary"


@pytest.mark.asyncio
async def test_worker_token_timeout_does_not_leak_permit():
    arbitrator = _arbitrator()
    worker = f"MLX-Cortex-test-{uuid.uuid4().hex}"

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)
    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05) is False

    await arbitrator.release_inference(worker=worker)
    await asyncio.sleep(0.05)

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)
    await arbitrator.release_inference(worker=worker)


@pytest.mark.asyncio
async def test_inference_context_raises_when_worker_token_times_out():
    arbitrator = _arbitrator()
    worker = f"MLX-Cortex-timeout-{uuid.uuid4().hex}"

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)

    with pytest.raises(asyncio.TimeoutError):
        async with arbitrator.inference_context(worker=worker, timeout=0.05):
            pass

    await arbitrator.release_inference(worker=worker)


@pytest.mark.asyncio
async def test_priority_worker_timeout_respects_caller_budget():
    arbitrator = _arbitrator()
    worker = f"MLX-Cortex-priority-{uuid.uuid4().hex}"

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await arbitrator.acquire_inference(priority=True, worker=worker, timeout=0.05) is False
    elapsed = loop.time() - started

    await arbitrator.release_inference(worker=worker)

    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_releasing_one_worker_does_not_hide_other_active_inference():
    arbitrator = _arbitrator()

    assert await arbitrator.acquire_inference(worker="cortex", timeout=0.05)
    assert await arbitrator.acquire_inference(worker="brainstem", timeout=0.05)
    assert arbitrator.is_inference_busy() is True

    await arbitrator.release_inference(worker="cortex")
    assert arbitrator.is_inference_busy() is True
    assert "brainstem" in arbitrator.get_status()["inference_lanes"]

    await arbitrator.release_inference(worker="brainstem")
    assert arbitrator.is_inference_busy() is False


@pytest.mark.asyncio
async def test_evolution_is_exclusive_with_inference_and_releases_both_locks(tmp_path):
    arbitrator = _arbitrator(lock_path=tmp_path / "vram.lock")

    assert await arbitrator.acquire_inference(worker="cortex", timeout=0.05)
    assert await arbitrator.acquire_evolution(timeout=0.03) is False
    await arbitrator.release_inference(worker="cortex")

    assert await arbitrator.acquire_evolution(timeout=0.05) is True
    assert arbitrator.get_status()["evolution_active"] is True
    await arbitrator.release_evolution()
    assert arbitrator.get_status()["evolution_active"] is False

    assert await arbitrator.acquire_evolution(timeout=0.05) is True
    await arbitrator.release_evolution()
