import asyncio
import contextlib
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.mlx_worker import IPCWriterThread, _should_emit_generation_progress
from core.utils.deadlines import get_deadline


class TestMLXClientResilience(unittest.IsolatedAsyncioTestCase):
    async def test_heavy_model_hotswap_reboots_other_heavy_client_before_spawn(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"

        primary = MLXLocalClient(model_path=primary_path)
        solver = MLXLocalClient(model_path=deep_path)

        primary_proc = MagicMock()
        primary_proc.is_alive.return_value = True
        primary._process = primary_proc
        primary._init_done = True
        primary.reboot_worker = AsyncMock()

        solver_proc = MagicMock()
        solver_proc.is_alive.return_value = True

        async def _spawn_solver():
            solver._init_future.set_result({"status": "ok", "action": "init"})
            return solver_proc

        old_clients = dict(mlx_module._CLIENTS)
        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._CLIENTS = {
            primary_path: primary,
            deep_path: solver,
        }
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = ""
        mlx_module._GLOBAL_LAST_SWAP_TIME = 0.0

        try:
            with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 patch("core.brain.llm.model_registry.DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit"), \
                 patch("core.brain.llm.model_registry.get_model_path", side_effect=lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 patch("core.brain.llm.mlx_client.os.path.realpath", side_effect=lambda path: path), \
                 patch.object(solver, "_spawn_worker", side_effect=_spawn_solver):
                await solver._ensure_worker_alive()
        finally:
            mlx_module._CLIENTS = old_clients
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        primary.reboot_worker.assert_awaited_once()
        self.assertTrue(solver._init_done)

    async def test_ensure_worker_sets_init_future_before_spawn(self):
        client = MLXLocalClient(model_path="/tmp/test-model")

        async def spawn_side_effect():
            self.assertIsNotNone(client._init_future)
            self.assertFalse(client._init_future.done())
            client._init_future.set_result({"status": "ok", "action": "init"})
            proc = MagicMock()
            proc.is_alive.return_value = True
            return proc

        with patch.object(client, "_spawn_worker", side_effect=spawn_side_effect):
            await client._ensure_worker_alive()

        self.assertTrue(client._init_done)
        self.assertTrue(client.is_alive())

    async def test_ensure_worker_reuses_existing_handshake_future(self):
        client = MLXLocalClient(model_path="/tmp/test-model")

        live_process = MagicMock()
        live_process.is_alive.return_value = True
        client._process = live_process
        client._init_done = False
        fut = AsyncMock()
        client._init_future = AsyncMock()
        # Replace the async mock with a real Future to match runtime behavior.
        import asyncio
        real_future = asyncio.get_running_loop().create_future()
        real_future.set_result({"status": "ok", "action": "init"})
        client._init_future = real_future

        with patch.object(client, "_spawn_worker", new=AsyncMock()) as spawn_mock:
            await client._ensure_worker_alive()

        spawn_mock.assert_not_awaited()
        live_process.kill.assert_not_called()
        self.assertTrue(client._init_done)

    async def test_ensure_worker_reuses_cross_loop_handshake_future(self):
        client = MLXLocalClient(model_path="/tmp/test-model")

        live_process = MagicMock()
        live_process.is_alive.return_value = True
        client._process = live_process
        client._init_done = False

        holder = {}
        ready = threading.Event()

        def _loop_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            future = loop.create_future()
            holder["future"] = future
            ready.set()

            async def _complete():
                await asyncio.sleep(0.05)
                future.set_result({"status": "ok", "action": "init"})
                await asyncio.sleep(0.05)

            loop.run_until_complete(_complete())
            loop.close()

        thread = threading.Thread(target=_loop_thread, name="mlx-cross-loop-init", daemon=True)
        thread.start()
        ready.wait(timeout=1.0)
        client._init_future = holder["future"]

        try:
            with patch.object(client, "_spawn_worker", new=AsyncMock()) as spawn_mock:
                await client._ensure_worker_alive()
        finally:
            thread.join(timeout=1.0)

        spawn_mock.assert_not_awaited()
        self.assertTrue(client._init_done)
        live_process.kill.assert_not_called()

    async def test_cancelled_generation_preserves_healthy_worker(self):
        client = MLXLocalClient(model_path="/tmp/test-model")
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        client._last_heartbeat = client._last_progress_at = client._last_ready_at = 10_000.0

        async def _cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        with patch.object(client, "_ensure_worker_alive", new=AsyncMock(return_value=True)):
            with patch.object(client, "_wait_for_generation_result", side_effect=_cancelled):
                with patch.object(client, "reboot_worker", new=AsyncMock()) as reboot_mock:
                    with patch("time.time", return_value=10_001.0):
                        with self.assertRaises(asyncio.CancelledError):
                            await client._generate_inner("hello", foreground_request=True)

        reboot_mock.assert_not_awaited()

    async def test_generate_times_out_waiting_for_foreground_owner(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:cortex"
        mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = time.time()

        try:
            with patch.object(client, "_acquire_request_lock", new=AsyncMock(return_value=True)):
                with patch.object(client, "_generate_inner", new=AsyncMock()) as inner:
                    with patch("core.brain.llm.mlx_client._foreground_owner_wait_budget", return_value=0.0):
                        result = await client.generate(
                            "hello",
                            foreground_request=True,
                            owner_label="test",
                            deadline=get_deadline(30.0),
                        )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = old_owned_at

        self.assertIsNone(result)
        inner.assert_not_awaited()

    async def test_generate_clears_stale_foreground_owner_and_continues(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:cortex"
        mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = time.time() - 120.0

        try:
            with patch.object(client, "_acquire_request_lock", new=AsyncMock(return_value=True)):
                with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")) as inner:
                    result = await client.generate(
                        "hello",
                        foreground_request=True,
                        owner_label="test",
                        deadline=get_deadline(30.0),
                    )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = old_owned_at

        self.assertEqual(result, "ok")
        inner.assert_awaited_once()


    async def test_primary_lane_generate_requires_explicit_foreground_request(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")

        with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")) as inner:
            result = await client.generate("hello")

        self.assertEqual(result, "ok")
        self.assertFalse(inner.await_args.kwargs["foreground_request"])
        self.assertFalse(inner.await_args.kwargs["request_is_background"])

    async def test_generate_suppresses_stale_unlock_in_finally(self):
        client = MLXLocalClient(model_path="/tmp/test-model")
        fake_lock = MagicMock()
        fake_lock.acquire.return_value = True
        fake_lock.release.side_effect = RuntimeError("release unlocked lock")
        client._request_lock = fake_lock

        with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")):
            result = await client.generate("hello")

        self.assertEqual(result, "ok")
        fake_lock.release.assert_called()

    async def test_generate_soft_times_out_init_budget_without_killing_worker(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = False
        client._set_lane_state("handshaking")
        client._init_future = asyncio.get_running_loop().create_future()

        result = await client._generate_inner(
            "hello",
            foreground_request=True,
            owner_label="test",
            deadline=get_deadline(0.5),
        )

        self.assertIsNone(result)
        proc.kill.assert_not_called()
        self.assertIs(client._process, proc)
        self.assertFalse(client._init_future.done())
        self.assertEqual(client._lane_state, "recovering")

    async def test_listener_routes_init_error_without_action_to_init_future(self):
        client = MLXLocalClient(model_path="/tmp/test-model")
        client._init_future = asyncio.get_running_loop().create_future()

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put({"status": "error", "message": "Init failed: boom"})
            result = await asyncio.wait_for(asyncio.shield(client._init_future), timeout=2.0)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(result["action"], "init")
        self.assertEqual(result["message"], "Init failed: boom")

    async def test_generation_waiter_flags_first_token_sla_breach(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-1"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 0
        deadline = get_deadline(None)

        with patch("core.brain.llm.mlx_client.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch(
                "core.brain.llm.mlx_client.time.time",
                return_value=100.0 + client._first_token_sla(foreground_request=True) + 1.0,
            ):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    deadline,
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "first_token_sla_exceeded")

    async def test_long_prompt_extends_first_token_sla_for_heavy_lane(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        cold_sla = client._first_token_sla(foreground_request=True)

        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 24_740

        warm_long_prompt_sla = client._first_token_sla(foreground_request=True)

        self.assertGreater(warm_long_prompt_sla, 22.0)
        self.assertGreater(warm_long_prompt_sla, cold_sla)

    async def test_generation_waiter_flags_token_progress_stall(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-2"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._current_first_token_at = 105.0
        client._last_token_progress_at = 105.0

        with patch("core.brain.llm.mlx_client.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch("core.brain.llm.mlx_client.time.time", return_value=105.0 + client._token_stall_after() + 1.0):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(30.0),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "token_progress_stalled")

    async def test_warmup_precompile_accepts_empty_text_as_successful_compile(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        client._warmup_in_flight = True

        with patch.object(client, "_generate_inner", new=AsyncMock(return_value="")):
            await client._run_warmup_precompile(
                request_is_background=False,
                foreground_request=True,
                owner_name="warmup:test",
                warmup_timeout=1.0,
            )

        self.assertEqual(client.get_lane_status()["state"], "ready")

    async def test_foreground_empty_generation_marks_recoverable_reboot(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        client._process = MagicMock()
        client._process.is_alive.return_value = True
        client._init_done = True
        client._set_lane_state("ready")

        with patch.object(client, "_ensure_worker_alive", new=AsyncMock(return_value=True)):
            with patch.object(
                client,
                "_wait_for_generation_result",
                new=AsyncMock(return_value={"status": "ok", "text": ""}),
            ):
                result = await client._generate_inner(
                    "hello",
                    _retry=False,
                    foreground_request=True,
                    owner_label="test",
                    deadline=get_deadline(30.0),
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_empty_generation")

    async def test_generate_reboots_recoverable_empty_generation_without_failed_lane(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")

        async def _empty_then_request_reboot(*args, **kwargs):
            client._deferred_reboot_reason = "recoverable_empty_generation"
            return None

        with patch.object(client, "_generate_inner", new=AsyncMock(side_effect=_empty_then_request_reboot)):
            with patch.object(client, "reboot_worker", new=AsyncMock()) as reboot_mock:
                result = await client.generate("hello", foreground_request=True, owner_label="test")

        self.assertIsNone(result)
        reboot_mock.assert_awaited_once_with(reason="empty_generation", mark_failed=False)

    async def test_supervision_status_reports_recycle_candidate(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._process_started_at = 100.0
        client._last_generation_completed_at = 600.0

        with patch("core.brain.llm.mlx_client.time.time", return_value=2000.0):
            status = client.get_supervision_status()
            recyclable = client.should_recycle_for_fragmentation(
                max_uptime_s=900.0,
                min_idle_s=300.0,
            )

        self.assertTrue(status["alive"])
        self.assertAlmostEqual(status["process_uptime_s"], 1900.0, places=3)
        self.assertAlmostEqual(status["idle_for_s"], 1400.0, places=3)
        self.assertTrue(recyclable)

    async def test_heavy_model_swap_respects_cooldown_window(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"
        solver = MLXLocalClient(model_path=deep_path)

        solver_proc = MagicMock()
        solver_proc.is_alive.return_value = True

        async def _spawn_solver():
            solver._init_future.set_result({"status": "ok", "action": "init"})
            return solver_proc

        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = primary_path
        mlx_module._GLOBAL_LAST_SWAP_TIME = 100.0

        try:
            with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 patch("core.brain.llm.model_registry.DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit"), \
                 patch("core.brain.llm.model_registry.get_model_path", side_effect=lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 patch("core.brain.llm.mlx_client.os.path.realpath", side_effect=lambda path: path), \
                 patch("core.brain.llm.mlx_client.time.time", return_value=105.0), \
                 patch("core.brain.llm.mlx_client.asyncio.sleep", new_callable=AsyncMock) as sleep_mock, \
                 patch.object(solver, "_spawn_worker", side_effect=_spawn_solver):
                await solver._ensure_worker_alive()
        finally:
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        sleep_mock.assert_any_await(7.0)
        self.assertTrue(solver._init_done)


class TestIPCWriterThread(unittest.TestCase):
    def test_essential_messages_bypass_full_buffer(self):
        mp_queue = MagicMock()
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "heartbeat"})

        item = {"status": "ok", "action": "generate", "text": "hello"}
        writer.put(item)

        mp_queue.put.assert_called_once_with(item, block=True, timeout=5.0)

    def test_heartbeat_is_dropped_when_buffer_full(self):
        mp_queue = MagicMock()
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "heartbeat"})

        writer.put({"status": "heartbeat", "timestamp": 1.0})

        mp_queue.put.assert_not_called()


class TestMLXWorkerProgress(unittest.TestCase):
    def test_generation_progress_emits_on_first_token(self):
        self.assertTrue(
            _should_emit_generation_progress(
                1,
                last_emit_at=100.0,
                now=100.2,
            )
        )

    def test_generation_progress_emits_on_time_gap_before_token_modulus(self):
        self.assertTrue(
            _should_emit_generation_progress(
                3,
                last_emit_at=100.0,
                now=101.7,
            )
        )

    def test_generation_progress_stays_quiet_when_recent_and_off_cycle(self):
        self.assertFalse(
            _should_emit_generation_progress(
                3,
                last_emit_at=100.0,
                now=100.4,
            )
        )


class TestMLXRuntimeProbeFailure(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_probe_failure_marks_lane_failed_without_spawn_loop(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")

        with patch.object(
            client,
            "_spawn_worker",
            side_effect=RuntimeError("mlx_runtime_probe_failed:metal_device_enumeration_crash"),
        ) as spawn_mock:
            alive = await client._ensure_worker_alive()

        self.assertFalse(alive)
        spawn_mock.assert_awaited_once()
        self.assertEqual(client.get_lane_status()["state"], "failed")
        self.assertEqual(
            client.get_lane_status()["last_error"],
            "mlx_runtime_unavailable:metal_device_enumeration_crash",
        )

    async def test_runtime_probe_recovery_clears_failed_lane_and_backoff(self):
        client = MLXLocalClient(model_path="/tmp/Qwen2.5-32B-Instruct-8bit")
        client.note_lane_failed("mlx_runtime_unavailable:metal_device_enumeration_crash")
        client._spawn_backoff_until = time.time() + 120.0
        client._consecutive_spawn_failures = 3

        proc = MagicMock()
        proc.is_alive.return_value = True

        async def _spawn():
            client._init_future.set_result({"status": "ok", "action": "init"})
            return proc

        with patch("core.brain.llm.mlx_client._probe_mlx_runtime", return_value=(True, "mlx_runtime_ok")):
            with patch.object(client, "_spawn_worker", side_effect=_spawn) as spawn_mock:
                alive = await client._ensure_worker_alive()

        self.assertTrue(alive)
        spawn_mock.assert_awaited_once()
        self.assertEqual(client.get_lane_status()["state"], "ready")
        self.assertEqual(client.get_lane_status()["last_error"], "")
        self.assertEqual(client._consecutive_spawn_failures, 0)
        self.assertEqual(client._spawn_backoff_until, 0.0)


def test_probe_reuses_fresh_positive_disk_cache(monkeypatch):
    import core.brain.llm.mlx_client as mlx_module

    monkeypatch.setattr(mlx_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(mlx_module, "_load_probe_cache_from_disk", lambda: (True, "mlx_runtime_ok", 950.0))
    monkeypatch.setattr(
        mlx_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    monkeypatch.setattr(
        mlx_module,
        "_MLX_RUNTIME_PROBE",
        {"ok": None, "detail": "", "checked_at": 0.0},
    )

    ok, detail = mlx_module._probe_mlx_runtime(force=False)

    assert ok is True
    assert detail == "mlx_runtime_ok"


def test_probe_does_not_trust_stale_negative_disk_cache(monkeypatch):
    import core.brain.llm.mlx_client as mlx_module

    class _Completed:
        returncode = 0
        stdout = "mlx_runtime_ok\n"
        stderr = ""

    calls = []

    monkeypatch.setattr(mlx_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        mlx_module,
        "_load_probe_cache_from_disk",
        lambda: (False, "metal_device_enumeration_crash", 900.0),
    )
    monkeypatch.setattr(
        mlx_module.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _Completed(),
    )
    monkeypatch.setattr(
        mlx_module,
        "_MLX_RUNTIME_PROBE",
        {"ok": None, "detail": "", "checked_at": 0.0},
    )
    monkeypatch.setattr(mlx_module, "_store_probe_cache_to_disk", lambda ok, detail: None)

    ok, detail = mlx_module._probe_mlx_runtime(force=False)

    assert ok is True
    assert detail == "mlx_runtime_ok"
    assert calls
