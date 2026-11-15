from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_organ_load_records_structured_fallback(monkeypatch):
    import core.kernel.organs as organs

    records = []
    organ = organs.OrganStub("voice", SimpleNamespace())

    async def failing_resolver():
        message = "voice engine unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(organ, "_resolve", failing_resolver)
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await organ.load()

    assert organ.ready.is_set()
    assert organ.fallback_used is True
    assert organ.resolved_kind == "FallbackVoice"
    assert "voice engine unavailable" in organ.failure_reason
    assert records
    assert records[0][0][0] == "organs"
    assert records[0][1]["extra"]["organ"] == "voice"
    assert "bounded fallback" in records[0][1]["action"]
    assert organ.status()["fallback_used"] is True


@pytest.mark.asyncio
async def test_organ_load_marks_fallback_returned_by_resolver(monkeypatch):
    import core.kernel.organs as organs

    records = []
    organ = organs.OrganStub("neural", SimpleNamespace())

    async def fallback_resolver():
        return organs.FallbackNeural()

    monkeypatch.setattr(organ, "_resolve", fallback_resolver)
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await organ.load()

    assert organ.fallback_used is True
    assert organ.resolved_kind == "FallbackNeural"
    assert "resolved to FallbackNeural" in organ.failure_reason
    assert records == []


@pytest.mark.asyncio
async def test_organ_shutdown_records_hook_failure(monkeypatch):
    import core.kernel.organs as organs

    records = []

    class ServiceWithBrokenStop:
        def stop(self):
            message = "cleanup failed"
            raise RuntimeError(message)

    organ = organs.OrganStub("voice", SimpleNamespace())
    organ.instance = ServiceWithBrokenStop()
    organ.resolved_kind = "ServiceWithBrokenStop"
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await organ.shutdown()

    assert records
    assert "shutdown hook failed" in records[0][1]["action"]
    assert records[0][1]["extra"]["organ"] == "voice"


@pytest.mark.asyncio
async def test_kernel_publishes_live_ice_instance_to_authority_container(monkeypatch):
    from core.kernel.aura_kernel import AuraKernel

    live_ice = SimpleNamespace()
    organ = SimpleNamespace(
        name="ice_layer",
        instance=None,
        fallback_used=False,
    )

    async def load():
        organ.instance = live_ice

    organ.load = load
    registrations = []
    monkeypatch.setattr(
        "core.kernel.aura_kernel.ServiceContainer.register_instance",
        lambda *args, **kwargs: registrations.append((args, kwargs)),
    )
    kernel = AuraKernel.__new__(AuraKernel)
    kernel._gui_queue = asyncio.Queue()

    await kernel._supervise_organ_load(organ)

    assert registrations == [
        (
            ("ice_layer", live_ice),
            {
                "required": True,
                "owner": "aura_kernel",
                "registered_by": "AuraKernel._supervise_organ_load",
                "required_for": "authority_containment",
                "failure_policy": "fail_closed",
            },
        )
    ]
    assert await kernel._gui_queue.get() == {
        "type": "ORGAN_READY",
        "name": "ice_layer",
    }


@pytest.mark.asyncio
async def test_an_organ_still_resolving_past_the_stall_bound_is_the_real_one(monkeypatch):
    """Five seconds in all cut her NeuralBridge while it was still loading, and the fallback stood for the run."""
    import core.kernel.organs as organs

    monkeypatch.setattr(organs, "_ORGAN_STALL_S", 0.2)
    organ = organs.OrganStub("neural", SimpleNamespace())
    real = object()

    async def a_resolution_that_keeps_moving():
        for _step in range(12):  # 0.6 s in all, three times the stall bound
            await asyncio.sleep(0.05)
        return real

    monkeypatch.setattr(organ, "_resolve", a_resolution_that_keeps_moving)
    await organ.load()

    assert organ.instance is real and organ.fallback_used is False


@pytest.mark.asyncio
async def test_an_organ_stuck_on_one_await_gets_its_fallback(monkeypatch):
    import core.kernel.organs as organs

    monkeypatch.setattr(organs, "_ORGAN_STALL_S", 0.2)
    monkeypatch.setattr(organs, "record_degradation", lambda *args, **kwargs: None)
    organ = organs.OrganStub("neural", SimpleNamespace())

    async def a_resolution_that_never_answers():
        await asyncio.Event().wait()

    monkeypatch.setattr(organ, "_resolve", a_resolution_that_never_answers)
    await asyncio.wait_for(organ.load(), timeout=5.0)

    assert organ.fallback_used is True and organ.resolved_kind == "FallbackNeural"
