import asyncio
import pytest
import httpx
import subprocess
import threading
from collections import deque
from unittest.mock import AsyncMock, MagicMock

from core.brain.llm import local_server_client
from core.brain.llm.local_server_client import LocalServerClient, _SERVER_CLIENTS, _thread_lock_context
from core.brain.memory_guard import ContextPruner


def _register(client: LocalServerClient) -> LocalServerClient:
    _SERVER_CLIENTS[client.model_path] = client
    return client


@pytest.fixture(autouse=True)
def _clear_runtime_clients():
    _SERVER_CLIENTS.clear()
    yield
    _SERVER_CLIENTS.clear()


@pytest.mark.asyncio
async def test_thread_lock_context_cancellation_does_not_park_executor_waiter():
    lock = threading.Lock()
    lock.acquire()

    async def _wait_for_lock():
        async with _thread_lock_context(lock, timeout=1.0, label="test_lock"):
            return True

    task = asyncio.create_task(_wait_for_lock())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    lock.release()
    await asyncio.sleep(0.1)
    assert lock.acquire(False) is True
    lock.release()


@pytest.mark.asyncio
async def test_background_lane_will_not_evict_ready_cortex():
    cortex = _register(LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf"))
    cortex._lane_state = "ready"

    brainstem = _register(LocalServerClient("/tmp/qwen2.5-7b-instruct-q4_k_m.gguf"))

    original = local_server_client._parallel_lane_runtime_allowed
    local_server_client._parallel_lane_runtime_allowed = lambda: False
    try:
        allowed = await brainstem._yield_runtime_slot(foreground_request=False)
    finally:
        local_server_client._parallel_lane_runtime_allowed = original

    assert allowed is False
    assert brainstem.get_lane_status()["last_error"] == "background_deferred:foreground_reserved"
    assert cortex.get_lane_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_background_brainstem_can_coexist_with_ready_cortex_when_parallel_runtime_allowed():
    cortex = _register(LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf"))
    cortex._lane_state = "ready"

    brainstem = _register(LocalServerClient("/tmp/qwen2.5-7b-instruct-q4_k_m.gguf"))

    original = local_server_client._parallel_lane_runtime_allowed
    local_server_client._parallel_lane_runtime_allowed = lambda: True
    try:
        allowed = await brainstem._yield_runtime_slot(foreground_request=False)
    finally:
        local_server_client._parallel_lane_runtime_allowed = original

    assert allowed is True
    assert brainstem.get_lane_status()["last_error"] == ""
    assert cortex.get_lane_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_foreground_solver_evicts_existing_cortex_before_start():
    cortex = _register(LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf"))
    cortex._lane_state = "ready"

    calls = []

    async def _fake_reboot_worker(reason: str = "manual_reboot", mark_failed: bool = False):
        calls.append((reason, mark_failed))
        cortex._lane_state = "cold"

    cortex.reboot_worker = _fake_reboot_worker

    solver = _register(LocalServerClient("/tmp/qwen2.5-72b-instruct-q4_k_m.gguf"))

    allowed = await solver._yield_runtime_slot(foreground_request=True)

    assert allowed is True
    assert calls == [("yield_to:Solver", False)]
    assert cortex.get_lane_status()["state"] == "cold"


@pytest.mark.asyncio
async def test_message_payload_sanitization_drops_non_json_metadata():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    client._lane_state = "ready"

    captured = {}

    class _FakeHttpClient:
        async def post(self, _url, json=None, timeout=None):
            captured["json"] = json
            captured["timeout"] = timeout
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "Hello from Cortex."}}
                    ]
                },
            )

    async def _fake_client():
        return _FakeHttpClient()

    client._client = _fake_client
    client._ensure_runtime_ready = AsyncMock(return_value=True)

    result = await client.generate_text_async(
        "hello",
        messages=[
            {
                "role": "user",
                "content": {"text": "hello"},
                "metadata": {"type": "skill_result", "opaque": object()},
                "timestamp": object(),
            }
        ],
        foreground_request=False,
    )

    assert result == "Hello from Cortex."
    assert captured["json"]["messages"] == [
        {"role": "user", "content": '{"text": "hello"}'}
    ]


def test_is_alive_repairs_adopted_runtime_without_owned_process():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    client._lane_state = "recovering"
    client._runtime_identity_ok = True
    client._http_health_check = lambda: True

    assert client.is_alive() is True
    assert client.get_lane_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_response_parser_handles_part_arrays_and_reasoning_fallback():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    client._lane_state = "ready"
    client._ensure_runtime_ready = AsyncMock(return_value=True)

    responses = deque([
        httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]}}
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "", "reasoning_content": "I can still answer here."}}
                ]
            },
        ),
    ])

    class _FakeHttpClient:
        async def post(self, _url, json=None, timeout=None):
            return responses.popleft()

    async def _fake_client():
        return _FakeHttpClient()

    client._client = _fake_client

    first = await client.generate_text_async("hello", foreground_request=False)
    second = await client.generate_text_async("hello", foreground_request=False)

    assert first == "Hello world"
    assert second == "I can still answer here."


@pytest.mark.asyncio
async def test_warmup_treats_empty_generation_as_benign_runtime_readiness():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    client._ensure_runtime_ready = AsyncMock(return_value=True)

    class _FakeHttpClient:
        async def post(self, _url, json=None, timeout=None):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": ""}}
                    ]
                },
            )

    async def _fake_client():
        return _FakeHttpClient()

    client._client = _fake_client

    await client.warmup()

    assert client.get_lane_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_solver_background_warmup_preserves_background_runtime_intent():
    client = LocalServerClient("/tmp/qwen2.5-72b-instruct-q4_k_m.gguf")
    client._ensure_runtime_ready = AsyncMock(return_value=False)

    await client.warmup(foreground_request=False)

    assert client._ensure_runtime_ready.await_args.kwargs["foreground_request"] is False


@pytest.mark.asyncio
async def test_single_empty_generation_does_not_immediately_crash_foreground_lane():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    client._lane_state = "ready"
    client._ensure_runtime_ready = AsyncMock(return_value=True)

    class _FakeHttpClient:
        async def post(self, _url, json=None, timeout=None):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": ""}}
                    ]
                },
            )

    async def _fake_client():
        return _FakeHttpClient()

    client._client = _fake_client

    result = await client.generate_text_async("hello", foreground_request=True)

    assert result is None
    assert client.get_lane_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_restart_server_wait_is_offloaded_from_event_loop(monkeypatch):
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    wait_calls = []
    offload_calls = []

    class _Proc:
        def poll(self):
            return None

        def kill(self):
            return None

        def wait(self, timeout=None):
            wait_calls.append(timeout)
            return 0

    async def _fake_to_thread(func, *args, **kwargs):
        offload_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    client._process = _Proc()
    client.warmup = AsyncMock(return_value=None)
    monkeypatch.setattr(local_server_client.asyncio, "to_thread", _fake_to_thread)

    await client._restart_server()

    assert wait_calls == [5.0]
    assert offload_calls and offload_calls[0][0].__name__ == "wait"
    client.warmup.assert_awaited_once()


@pytest.mark.asyncio
async def test_server_health_detects_wrong_model_on_reserved_lane_port():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")

    class _FakeHttpClient:
        async def get(self, url, timeout=None):
            if url.endswith("/health"):
                return httpx.Response(200, json={"status": "ok"})
            if url.endswith("/v1/models"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "qwen2.5-72b-instruct-q3_k_m-00001-of-00009.gguf"}
                        ]
                    },
                )
            raise AssertionError(f"unexpected url: {url}")

    async def _fake_client():
        return _FakeHttpClient()

    client._client = _fake_client

    healthy, mismatch = await client._server_healthy()

    assert healthy is False
    assert mismatch is True
    assert client.get_lane_status()["runtime_identity_ok"] is False
    assert client.get_lane_status()["detected_models"] == [
        "qwen2.5-72b-instruct-q3_k_m-00001-of-00009.gguf"
    ]
    assert client.get_lane_status()["last_error"].startswith("runtime_model_mismatch:")


def test_spawn_server_uses_single_slot_and_disables_prompt_cache_by_default(tmp_path, monkeypatch):
    model_path = tmp_path / "qwen2.5-32b-instruct-q5_k_m.gguf"
    model_path.write_text("stub", encoding="utf-8")

    client = LocalServerClient(str(model_path))
    client._resolve_llama_server_bin = lambda: "/opt/homebrew/bin/llama-server"
    client._log_path = lambda: tmp_path / "runtime.log"

    monkeypatch.delenv("AURA_LOCAL_PROMPT_CACHE", raising=False)
    monkeypatch.delenv("AURA_LOCAL_CACHE_RAM_MIB", raising=False)
    monkeypatch.delenv("AURA_LOCAL_PARALLEL_SLOTS", raising=False)

    captured = {}

    def _fake_popen(cmd, stdout=None, stderr=None, cwd=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        return MagicMock(spec=subprocess.Popen)

    original = subprocess.Popen
    subprocess.Popen = _fake_popen
    try:
        client._spawn_server_blocking()
    finally:
        subprocess.Popen = original

    assert "--parallel" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--parallel") + 1] == "1"
    assert "--cache-ram" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--cache-ram") + 1] == "256"
    assert "--no-cache-prompt" in captured["cmd"]


def test_fit_messages_to_context_trims_oversized_background_payload():
    client = LocalServerClient("/tmp/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    messages = [
        {"role": "system", "content": "S" * 6000},
        {"role": "user", "content": "U" * 12000},
        {"role": "assistant", "content": "A" * 8000},
        {"role": "user", "content": "latest prompt"},
    ]

    fitted = client._fit_messages_to_context(messages, max_tokens=256)

    assert fitted
    assert fitted[0]["role"] == "system"
    assert fitted[-1]["content"] == "latest prompt"
    assert sum(client._estimate_message_tokens(msg) for msg in fitted) < sum(
        client._estimate_message_tokens(msg) for msg in messages
    )


def test_memory_guard_handles_deque_history_without_index_pop_failure():
    pruner = ContextPruner()
    history = deque(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "u" * 12000},
            {"role": "assistant", "content": "a" * 12000},
            {"role": "user", "content": "final prompt"},
        ]
    )

    pruned = pruner.prune_context(history, tier="reflex")

    assert isinstance(pruned, list)
    assert pruned[0]["role"] == "system"
    assert any(msg.get("content") == "final prompt" for msg in pruned)


def test_local_server_supervision_status_and_fragmentation_recycle():
    client = LocalServerClient("/tmp/qwen2.5-32b-instruct-q5_k_m.gguf")
    proc = MagicMock()
    proc.poll.return_value = None
    client._process = proc
    client._lane_state = "ready"
    client._process_started_at = 100.0
    client._last_generation_completed_at = 600.0

    original_time = local_server_client.time.time
    local_server_client.time.time = lambda: 2000.0
    try:
        status = client.get_supervision_status()
        recyclable = client.should_recycle_for_fragmentation(
            max_uptime_s=900.0,
            min_idle_s=300.0,
        )
    finally:
        local_server_client.time.time = original_time

    assert status["alive"] is True
    assert status["process_uptime_s"] == pytest.approx(1900.0)
    assert status["idle_for_s"] == pytest.approx(1400.0)
    assert recyclable is True
