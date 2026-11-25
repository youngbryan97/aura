from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from core.runtime.model_lane_control import ModelLaneController
from core.runtime.receipts import ReceiptStore
from core.skills.image_gen import ImageGenInput, ImageGenSkill


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


@pytest.mark.asyncio
async def test_diffusion_pipeline_has_one_durable_lane_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.runtime import model_lane_control

    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
    controller = ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_discovery=None,
    )
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    skill = ImageGenSkill()
    loads = 0

    def _load(img2img: bool = False) -> bool:
        nonlocal loads
        loads += 1
        skill._img2img_pipeline = object() if img2img else None
        skill._pipeline = None if img2img else object()
        return True

    monkeypatch.setattr(skill, "_load_pipeline", _load)

    assert await skill._ensure_pipeline(img2img=False) is True
    assert await skill._ensure_pipeline(img2img=False) is True

    snapshot = controller.snapshot()
    assert loads == 1
    assert len(snapshot["owners"]) == 1
    assert snapshot["owners"][0]["declared_gb"] == pytest.approx(10.0)
    assert snapshot["owners"][0]["metadata"]["single_pipeline_residency"] is True
    assert snapshot["owners"][0]["preemptible"] is True

    await skill.on_stop_async()

    assert controller.snapshot()["owners"] == []
    assert skill._lane_lease is None


@pytest.mark.asyncio
async def test_image_model_preemption_refuses_during_active_generation() -> None:
    skill = ImageGenSkill()
    await skill._generation_lock.acquire()
    try:
        accepted = await skill._evict_for_model_lane(object(), "foreground_chat")
    finally:
        skill._generation_lock.release()

    assert accepted is False


@pytest.mark.asyncio
async def test_cancelled_image_request_retains_generation_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class _Pipeline:
        def __call__(self, **_kwargs):
            started.set()
            assert release.wait(2.0)
            return SimpleNamespace(images=[object()])

    torch = ModuleType("torch")
    torch.Generator = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    skill = ImageGenSkill()
    skill._pipeline = _Pipeline()
    skill._model_loaded = True

    request = asyncio.create_task(
        skill.execute(
            ImageGenInput(prompt="test image", steps=10, width=256, height=256),
            {},
        )
    )
    assert await asyncio.to_thread(started.wait, 1.0)
    request.cancel()
    await asyncio.sleep(0.02)

    assert request.done() is False
    assert skill._generation_lock.locked() is True
    assert await skill._evict_for_model_lane(object(), "foreground") is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert skill._generation_lock.locked() is False
    await skill.on_stop_async()


@pytest.mark.asyncio
async def test_parallel_pipeline_requests_do_not_double_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import model_lane_control

    class _Lease:
        def __init__(self) -> None:
            self.released = False

        async def set_preemptible(self, preemptible: bool) -> bool:
            return preemptible

        async def release(self, *, reason: str) -> bool:
            self.released = True
            return True

    lease = _Lease()
    acquire_calls = 0

    async def _acquire(**_kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        return lease

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    skill = ImageGenSkill()
    load_calls = 0

    def _load(_img2img: bool = False) -> bool:
        nonlocal load_calls
        load_calls += 1
        skill._pipeline = object()
        return True

    monkeypatch.setattr(skill, "_load_pipeline", _load)

    results = await asyncio.gather(
        skill._ensure_pipeline(img2img=False),
        skill._ensure_pipeline(img2img=False),
    )

    assert results == [True, True]
    assert acquire_calls == 1
    assert load_calls == 1
    await skill.on_stop_async()
    assert lease.released is True
