"""Stopped execution owners cannot admit replacement model work."""
import asyncio

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.runtime.what_stops_it import AnExecutionContext, under


def test_stopped_owner_is_rejected_before_any_client_or_worker_mutation():
    async def exercise():
        owner = AnExecutionContext(doing="desktop turn")
        owner.stopping.stop("user_requested_stop")
        # No client internals exist: admission must stop before touching them.
        client = object.__new__(MLXLocalClient)
        with under(owner):
            with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
                await client.generate("fallback must not run")
            with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
                await client.generate("retry must not run")

    asyncio.run(exercise())


def test_router_stop_cancels_the_endpoint_without_trying_a_replacement(monkeypatch):
    from core.brain.llm_health_router import HealthAwareLLMRouter

    async def exercise():
        owner = AnExecutionContext(doing="desktop task")
        router = object.__new__(HealthAwareLLMRouter)
        entered, cleaned = asyncio.Event(), asyncio.Event()
        calls = []

        async def endpoint(*args, **kwargs):
            calls.append("original")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(router, "_generate_with_metadata_gated", endpoint)
        with under(owner):
            task = asyncio.create_task(router.generate_with_metadata("owned work", _gate_already_held=True))
        await asyncio.wait_for(entered.wait(), 1.)
        owner.stopping.stop("user_requested_stop")
        with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
            await asyncio.wait_for(task, 1.)
        with under(owner):
            with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
                await router.generate_with_metadata("retry must not run")
        assert cleaned.is_set() and calls == ["original"]

    asyncio.run(exercise())


def test_stop_cancels_an_inflight_gate_and_preserves_request_metadata_cleanup(monkeypatch):
    from core.brain.inference_gate import _GENERATE_ENTERED_AT, InferenceGate

    async def exercise():
        owner = AnExecutionContext(doing="desktop task")
        gate = object.__new__(InferenceGate)
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def endpoint(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(gate, "_generate_with_metadata_sink", endpoint)
        before = _GENERATE_ENTERED_AT.get()
        with under(owner):
            task = asyncio.create_task(gate.generate("owned work", context={"origin": "desktop_ui"}))
        await entered.wait()
        owner.stopping.stop("user_requested_stop")
        with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
            await asyncio.wait_for(task, 1.)
        assert cleaned.is_set() and _GENERATE_ENTERED_AT.get() == before

    asyncio.run(exercise())


def test_stop_reaches_the_local_client_while_it_waits_for_generation_admission(monkeypatch):
    """Use the real public client endpoint with its first admission wait held."""
    from types import SimpleNamespace

    from core.brain.llm import mlx_client

    async def exercise():
        owner = AnExecutionContext(doing="desktop task")
        entered = asyncio.Event()
        client = object.__new__(MLXLocalClient)
        client.model_path = "/owned-model"
        client.max_tokens = 100
        monkeypatch.setattr(client, "_set_task_surface_control_receipt", lambda value: None)
        monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: SimpleNamespace(
            should_gc=False, refuse_heavy_local_generation=False))
        monkeypatch.setattr(mlx_client, "_apply_memory_pressure_generation_controls", lambda kwargs, *args, **kw: kwargs)

        async def acquire(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(client, "_acquire_request_lock", acquire)
        with under(owner):
            task = asyncio.create_task(client.generate("owned work"))
        await asyncio.wait_for(entered.wait(), 1.)
        owner.stopping.stop("user_requested_stop")
        with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
            await asyncio.wait_for(task, 1.)

    asyncio.run(exercise())
