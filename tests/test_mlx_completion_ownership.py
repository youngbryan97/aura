"""Foreground work follows progress; bounded probes retain their budgets."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.brain.inference_gate import InferenceGate
from core.brain.llm import mlx_client
from core.brain.llm.mlx_client import MLXLocalClient
from core.utils.deadlines import get_deadline


class CompletionClient(MLXLocalClient):
    def __init__(self, delay=0.03):
        self.delay = delay
        self.cancelled = False
        self.seen = {}

    async def generate_text_async(self, prompt, **kwargs):
        self.seen = kwargs
        try:
            await asyncio.sleep(self.delay)
            return "Complete answer."
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_gate_delegates_resident_completion_instead_of_cancelling_it(monkeypatch):
    async def run():
        monkeypatch.setattr(mlx_client, "USER_FACING_COMPLETION_DEADLINE_MAX_S", 0.2)
        client = CompletionClient()
        gate = InferenceGate.__new__(InferenceGate)
        metadata = []
        gate._record_client_generation_metadata = lambda *args, **kw: metadata.append(kw)
        result = await gate._generate_with_client(
            client, "Compare two choices", "", [], get_deadline(0.01), "Cortex",
            foreground_request=True, origin="desktop_ui",
        )
        assert result == "Complete answer."
        assert not client.cancelled
        assert client.seen["deadline"].is_expired
        assert metadata[-1]["success"] is True

    asyncio.run(run())


@pytest.mark.parametrize("flag", [
    "is_background", "benchmark_request", "proof_evaluation_contract",
    "strict_answer_contract", "internal_inference_call", "health_probe",
])
def test_non_conversational_calls_keep_their_declared_deadline(monkeypatch, flag):
    async def run():
        monkeypatch.setattr(mlx_client, "USER_FACING_COMPLETION_DEADLINE_MAX_S", 0.2)
        client = CompletionClient()
        with pytest.raises(TimeoutError):
            await client.generate_text_to_completion(
                "probe", deadline=get_deadline(0.01), foreground_request=True, **{flag: True}
            )
        assert client.cancelled
        assert client.seen["_progress_owned_completion"] is False

    asyncio.run(run())


def test_completion_does_not_cancel_healthy_work_at_absolute_ceiling(monkeypatch):
    async def run():
        monkeypatch.setattr(mlx_client, "USER_FACING_COMPLETION_DEADLINE_MAX_S", 0.01)
        client = CompletionClient(delay=0.03)
        deadline = get_deadline(0.02)
        assert await client.generate_text_to_completion(
            "answer", deadline=deadline, foreground_request=True
        ) == "Complete answer."
        assert not client.cancelled
        assert client.seen["deadline"] is deadline

    asyncio.run(run())


def test_user_cancellation_propagates_to_completion():
    async def run():
        client = CompletionClient(delay=10)
        task = asyncio.create_task(client.generate_text_to_completion(
            "answer", deadline=get_deadline(1), foreground_request=True
        ))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.cancelled

    asyncio.run(run())


@pytest.mark.parametrize("active", [True, False])
def test_expired_wait_uses_current_request_progress_without_busy_polling(monkeypatch, active):
    async def run():
        client = MLXLocalClient.__new__(MLXLocalClient)
        client._last_token_progress_at = time.time() if active else 0.0
        client._stale_after = lambda **kw: 40.0
        client._first_token_sla = lambda **kw: 40.0
        client._token_stall_after = lambda **kw: 40.0
        slices = []

        async def resolve(future, *, timeout_s):
            slices.append(timeout_s)
            return {"text": "done"}

        monkeypatch.setattr(mlx_client, "_await_shared_future", resolve)
        call = client._wait_for_generation_result(
            "request", object(), get_deadline(-1), foreground_request=True,
        )
        if active:
            assert await call == {"text": "done"}
            assert slices == [2.0]
        else:
            with pytest.raises(TimeoutError):
                await call
            assert not slices

    asyncio.run(run())


def test_progress_owned_wait_has_no_total_duration_kill(monkeypatch):
    async def run():
        client = MLXLocalClient.__new__(MLXLocalClient)
        client._stale_after = lambda **kw: 40.0
        client._first_token_sla = lambda **kw: 40.0
        client._token_stall_after = lambda **kw: 40.0
        monkeypatch.setattr(mlx_client, "_generation_wait_hard_cap_s", lambda *a, **kw: -1)

        async def resolve(future, *, timeout_s):
            assert timeout_s == 2.0
            return {"text": "complete"}

        monkeypatch.setattr(mlx_client, "_await_shared_future", resolve)
        assert await client._wait_for_generation_result(
            "request", object(), get_deadline(-1), foreground_request=True,
            progress_owned_completion=True,
        ) == {"text": "complete"}

    asyncio.run(run())


@pytest.mark.parametrize("dead", [True, False])
def test_progress_owner_distinguishes_long_prefill_from_dead_worker(monkeypatch, dead):
    async def run():
        client = MLXLocalClient(model_path="/models/test-small")
        client._process = SimpleNamespace(is_alive=lambda: not dead)
        client._current_request_id = "request"
        client._current_request_started_at = 1.0
        client._last_heartbeat = 1000.0
        client._last_progress_at = 1000.0
        client._last_ready_at = 1000.0
        client._current_prefill_tokens_total = 2000
        client._current_prefill_tokens_processed = 1000
        client._prefill_observed_at = 999.0
        client._cold_lane_first_token_allowance = lambda: 0.0
        client._prefill_floor_seconds = lambda count: 0.0
        client._first_token_hard_ceiling = lambda **kw: 20.0
        client._record_degraded_event = lambda *a, **kw: None
        monkeypatch.setattr(mlx_client.time, "time", lambda: 1000.0)
        monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: SimpleNamespace(
            should_gc=False, refuse_heavy_local_generation=False
        ))
        future = asyncio.get_running_loop().create_future()
        client._pending_generations["request"] = future
        calls = 0

        async def resolve(future, *, timeout_s):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError
            return {"text": "complete"}

        monkeypatch.setattr(mlx_client, "_await_shared_future", resolve)
        result = await client._wait_for_generation_result(
            "request", future, get_deadline(-1), foreground_request=True,
            progress_owned_completion=True,
        )
        if dead:
            assert result is None
            assert future.cancelled()
            assert client._deferred_reboot_reason == "worker_died_during_generation"
        else:
            assert result == {"text": "complete"}
            assert not future.cancelled()

    asyncio.run(run())
