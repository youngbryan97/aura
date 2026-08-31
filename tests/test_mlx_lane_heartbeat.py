from __future__ import annotations

import asyncio
import contextlib
import queue
from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.runtime.shutdown_coordinator import clear_shutdown_request


def test_all_decode_passes_refresh_worker_activity_only_on_yield():
    from core.brain.llm.mlx_worker import _generation_stream_with_activity

    activity = []
    watchdog = SimpleNamespace(activity=lambda: activity.append("token"))

    def generate(model, tokenizer, *, prompt, max_tokens):
        assert model == "model" and tokenizer == "tokenizer"
        yield prompt
        yield max_tokens

    for prompt in ("initial", "answer-continuation"):
        before = len(activity)
        stream = _generation_stream_with_activity(
            generate, "model", "tokenizer", prompt=prompt,
            generation_kwargs={"max_tokens": 2}, watchdog=watchdog,
        )
        assert len(activity) == before
        assert next(stream) == prompt
        assert len(activity) == before + 1
        assert list(stream) == [2]
        assert len(activity) == before + 2


@pytest.mark.parametrize("fails", [False, True])
def test_decode_activity_stream_releases_generator_and_tap(fails):
    from core.brain.llm.mlx_worker import _generation_stream_with_activity

    events = []

    @contextlib.contextmanager
    def tap():
        events.append("entered")
        try:
            yield
        finally:
            events.append("exited")

    def generate(*args, **kwargs):
        try:
            yield "first"
            if fails:
                raise ValueError("decode failed")
            yield "second"
        finally:
            events.append("closed")

    stream = _generation_stream_with_activity(
        generate, None, None, prompt="prompt", generation_kwargs={},
        watchdog=SimpleNamespace(activity=lambda: events.append("activity")), tap=tap(),
    )
    assert next(stream) == "first"
    if fails:
        with pytest.raises(ValueError, match="decode failed"):
            next(stream)
    else:
        stream.close()
    assert events == ["entered", "activity", "closed", "exited"]


class _ProcessProbe:
    def __init__(self) -> None:
        self.alive = True
        self.killed = False

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        return None


@pytest.fixture(autouse=True)
def _clear_shutdown() -> None:
    clear_shutdown_request()
    yield
    clear_shutdown_request()


@pytest.mark.asyncio
async def test_worker_heartbeat_renews_durable_lane_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.container import ServiceContainer
    from core.runtime import model_lane_control

    renewed = asyncio.Event()
    calls: list[tuple[str, int]] = []
    released: list[tuple[str, int, str]] = []

    class _Controller:
        async def heartbeat_owner(self, owner_id: str, *, fencing_token: int) -> bool:
            calls.append((owner_id, fencing_token))
            renewed.set()
            return True

        def release_owner_sync(
            self,
            owner_id: str,
            *,
            fencing_token: int,
            reason: str,
        ) -> bool:
            released.append((owner_id, fencing_token, reason))
            return True

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: _Controller())
    monkeypatch.setattr(ServiceContainer, "get", lambda *_args, **_kwargs: None)
    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._res_q = queue.Queue()
    client._model_lane_owner_id = "mlx:test:heartbeat"
    client._model_lane_fencing_token = 77
    listener = asyncio.create_task(client._response_listener_loop())
    try:
        client._res_q.put({"status": "heartbeat"})
        await asyncio.wait_for(renewed.wait(), timeout=2.0)
    finally:
        listener.cancel()
        await listener
        client.close()

    assert calls == [("mlx:test:heartbeat", 77)]
    assert released == [("mlx:test:heartbeat", 77, "client_close")]


@pytest.mark.asyncio
async def test_slow_lane_renewal_does_not_block_terminal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.container import ServiceContainer
    from core.runtime import model_lane_control

    renewal_started = asyncio.Event()
    release_renewal = asyncio.Event()

    class _Controller:
        async def heartbeat_owner(self, _owner_id: str, *, fencing_token: int) -> bool:
            assert fencing_token == 79
            renewal_started.set()
            await release_renewal.wait()
            return True

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: _Controller())
    monkeypatch.setattr(ServiceContainer, "get", lambda *_args, **_kwargs: None)
    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._res_q = queue.Queue()
    client._model_lane_owner_id = "mlx:test:nonblocking-heartbeat"
    client._model_lane_fencing_token = 79
    client._current_request_id = "foreground-request"
    client._current_gen_future = asyncio.get_running_loop().create_future()
    listener = asyncio.create_task(client._response_listener_loop())
    try:
        client._res_q.put({"status": "heartbeat"})
        await asyncio.wait_for(renewal_started.wait(), timeout=2.0)
        client._res_q.put(
            {
                "status": "ok",
                "action": "generate",
                "id": "foreground-request",
                "text": "answer arrived while lease storage was slow",
            }
        )
        result = await asyncio.wait_for(
            asyncio.shield(client._current_gen_future),
            timeout=0.5,
        )
    finally:
        release_renewal.set()
        renewal = client._lane_renewal_task
        if renewal is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await renewal
        listener.cancel()
        await listener
        client._model_lane_fencing_token = 0

    assert result["text"] == "answer arrived while lease storage was slow"


@pytest.mark.asyncio
async def test_stale_lane_renewal_cannot_kill_replacement_owner() -> None:
    renewal_started = asyncio.Event()
    release_renewal = asyncio.Event()

    class _Controller:
        async def heartbeat_owner(self, _owner_id: str, *, fencing_token: int) -> bool:
            assert fencing_token == 81
            renewal_started.set()
            await release_renewal.wait()
            return False

    client = MLXLocalClient(model_path="/models/test-1.5b")
    process = _ProcessProbe()
    client._process = process
    client._model_lane_owner_id = "mlx:test:old-owner"
    client._model_lane_fencing_token = 81
    client._set_lane_state("ready")
    renewal = asyncio.create_task(
        client._renew_durable_lane_lease_in_background(
            _Controller(),
            "mlx:test:old-owner",
            81,
            2,
        )
    )
    await asyncio.wait_for(renewal_started.wait(), timeout=2.0)

    client._model_lane_owner_id = "mlx:test:replacement-owner"
    client._model_lane_fencing_token = 82
    release_renewal.set()
    await asyncio.wait_for(renewal, timeout=2.0)

    assert process.killed is False
    assert client._process is process
    assert client._model_lane_fencing_token == 82
    assert client._lane_state == "ready"
    client._model_lane_fencing_token = 0
    client._process = None


@pytest.mark.asyncio
async def test_lost_heartbeat_fence_stops_stale_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.container import ServiceContainer
    from core.runtime import model_lane_control

    class _Controller:
        async def heartbeat_owner(self, _owner_id: str, *, fencing_token: int) -> bool:
            assert fencing_token == 88
            return False

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: _Controller())
    monkeypatch.setattr(ServiceContainer, "get", lambda *_args, **_kwargs: None)
    client = MLXLocalClient(model_path="/models/test-1.5b")
    process = _ProcessProbe()
    client._process = process
    client._res_q = queue.Queue()
    client._model_lane_owner_id = "mlx:test:stale"
    client._model_lane_fencing_token = 88
    client._res_q.put({"status": "heartbeat"})

    await asyncio.wait_for(client._response_listener_loop(), timeout=2.0)

    assert process.killed is True
    assert client._process is None
    assert client._model_lane_fencing_token == 0
    assert client._lane_state == "cold"
    assert client._deferred_reboot_reason == "model_lane_fence_lost"


def test_forced_abort_releases_exact_durable_lane_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import model_lane_control

    released: list[tuple[str, int, str]] = []
    unregistered: list[str] = []

    class _Controller:
        def release_owner_sync(
            self,
            owner_id: str,
            *,
            fencing_token: int,
            reason: str,
        ) -> bool:
            released.append((owner_id, fencing_token, reason))
            return True

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: _Controller())
    monkeypatch.setattr(
        model_lane_control,
        "unregister_model_lane_owner_adapter",
        lambda owner_id: unregistered.append(owner_id),
    )
    client = MLXLocalClient(model_path="/models/test-1.5b")
    process = _ProcessProbe()
    client._process = process
    client._active_generations = 1
    client._model_lane_owner_id = "mlx:test:forced-abort"
    client._model_lane_fencing_token = 101
    client._model_lane_terminal_receipt_id = "receipt-101"
    monkeypatch.setattr(client, "_replace_ipc_queues", lambda: None)

    assert client.force_abort_active_generation("foreground_deadline") is True

    assert process.killed is True
    assert released == [("mlx:test:forced-abort", 101, "foreground_deadline")]
    assert unregistered == ["mlx:test:forced-abort"]
    assert client._model_lane_fencing_token == 0
    assert client._model_lane_terminal_receipt_id == ""


@pytest.mark.asyncio
async def test_dead_worker_releases_stale_fence_before_respawn_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.llm import mlx_client
    from core.runtime import model_lane_control

    events: list[str] = []

    class _Controller:
        def release_owner_sync(
            self,
            owner_id: str,
            *,
            fencing_token: int,
            reason: str,
        ) -> bool:
            events.append(f"release:{owner_id}:{fencing_token}:{reason}")
            return True

    @contextlib.asynccontextmanager
    async def _admitted(client, *, foreground_request):
        assert foreground_request is True
        assert client._model_lane_fencing_token == 0
        events.append("admitted")
        yield SimpleNamespace()

    async def _spawned(**_kwargs: object) -> bool:
        events.append("spawned")
        return True

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: _Controller())
    monkeypatch.setattr(model_lane_control, "unregister_model_lane_owner_adapter", lambda _owner: None)
    monkeypatch.setattr(mlx_client, "_model_load_admission_context", _admitted)
    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._model_lane_owner_id = "mlx:test:dead-worker"
    client._model_lane_fencing_token = 102
    client._model_lane_terminal_receipt_id = "receipt-102"
    monkeypatch.setattr(client, "_ensure_worker_alive_inner", _spawned)

    assert await client._ensure_worker_alive(foreground_request=True) is True
    assert events == [
        "release:mlx:test:dead-worker:102:dead_worker_before_respawn",
        "admitted",
        "spawned",
    ]


@pytest.mark.asyncio
async def test_generation_boundary_updates_durable_owner_preemptibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import model_lane_control

    calls: list[tuple[str, int, bool]] = []

    class _Controller:
        async def update_owner_preemptibility(
            self,
            owner_id: str,
            *,
            fencing_token: int,
            preemptible: bool,
        ) -> bool:
            calls.append((owner_id, fencing_token, preemptible))
            return True

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: _Controller())
    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._model_lane_owner_id = "mlx:test:generation-boundary"
    client._model_lane_fencing_token = 91
    future = asyncio.get_running_loop().create_future()
    client._pending_generations["request-1"] = future
    client._current_gen_future = future
    client._active_generations = 1

    assert await client._set_durable_lane_preemptible(False) is True
    await client._finish_generation_ownership("request-1", future, None)

    assert calls == [
        ("mlx:test:generation-boundary", 91, False),
        ("mlx:test:generation-boundary", 91, True),
    ]
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert client._current_gen_future is None
    client._model_lane_fencing_token = 0
    client.close()


@pytest.mark.asyncio
async def test_mlx_lane_preemption_refuses_active_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._model_lane_owner_id = "mlx:test:active"
    client._active_generations = 1
    rebooted: list[str] = []

    async def _reboot_worker(*, reason: str, mark_failed: bool) -> None:
        rebooted.append(f"{reason}:{mark_failed}")

    monkeypatch.setattr(client, "reboot_worker", _reboot_worker)
    monkeypatch.setattr(mlx_client, "_CLIENTS", {client.model_path: client})

    accepted = await mlx_client._evict_model_lane_owner(
        SimpleNamespace(owner_id="mlx:test:active"),
        "foreground_candidate",
    )

    assert accepted is False
    assert rebooted == []
    client._active_generations = 0
    client.close()


@pytest.mark.asyncio
async def test_generation_enqueue_failure_releases_busy_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._req_q = SimpleNamespace(put=lambda *_args, **_kwargs: None)

    async def _worker_ready(**_kwargs: object) -> bool:
        return True

    async def _broken_enqueue(*_args: object, **_kwargs: object) -> object:
        raise BrokenPipeError("injected queue failure")

    monkeypatch.setattr(client, "_ensure_worker_alive", _worker_ready)
    monkeypatch.setattr(mlx_client, "run_io_bound", _broken_enqueue)

    result = await client._generate_inner(
        "hello",
        _retry=False,
        foreground_request=False,
        strict_answer_contract=True,
    )

    assert result is None
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert client._current_gen_future is None
    assert client._foreground_generation_watchdog is None
    client._req_q = None
    client.close()
