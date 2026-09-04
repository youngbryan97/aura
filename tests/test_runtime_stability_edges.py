import asyncio
import importlib
import inspect
import json
import os
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

from core.brain.llm.mlx_client import MLXLocalClient
from core.container import ServiceContainer
from core.senses.sensory_client import SensoryLocalClient

TMP_ROOT = Path(tempfile.gettempdir())
TEST_MODEL = str(TMP_ROOT / "test-model")
QWEN32_TEST_MODEL = str(TMP_ROOT / "Qwen2.5-32B-Instruct-8bit")


_MISSING = object()


class _RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs


class _CallRecorder:
    def __init__(
        self,
        *args,
        return_value=_MISSING,
        side_effect=None,
        wraps=None,
        spec=None,
        **attrs,
    ):
        self.return_value = None if return_value is _MISSING else return_value
        self.side_effect = side_effect
        self.wraps = wraps
        self.calls = []
        self.call_args = None
        for key, value in attrs.items():
            setattr(self, key, value)

    @property
    def called(self):
        return bool(self.calls)

    @property
    def call_count(self):
        return len(self.calls)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = _CallRecorder()
        setattr(self, name, child)
        return child

    def _next_effect(self):
        effect = self.side_effect
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, list):
            if not effect:
                raise StopIteration
            value = effect.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        return _MISSING

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        effect_value = self._next_effect()
        if effect_value is not _MISSING:
            return effect_value
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        if self.wraps is not None:
            return self.wraps(*args, **kwargs)
        return self.return_value

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args, **kwargs):
        self.assert_called_once()
        call = self.calls[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_any_call(self, *args, **kwargs):
        assert any(call.args == args and call.kwargs == kwargs for call in self.calls)

    def assert_not_called(self):
        assert not self.calls


class _AsyncCallRecorder:
    def __init__(self, result=None, *, return_value=None, side_effect=None):
        self.return_value = result if return_value is None else return_value
        self.side_effect = side_effect
        self.await_args_list = []
        self.await_args = None

    @property
    def await_count(self):
        return len(self.await_args_list)

    @property
    def called(self):
        return bool(self.await_args_list)

    def _next_effect(self):
        effect = self.side_effect
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, list):
            if not effect:
                raise StopIteration
            value = effect.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        return _MISSING

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.await_args_list.append(call)
        self.await_args = call

        async def _complete():
            effect_value = self._next_effect()
            if effect_value is not _MISSING:
                return effect_value
            if callable(self.side_effect):
                value = self.side_effect(*args, **kwargs)
            else:
                value = self.return_value
            if inspect.isawaitable(value):
                return await value
            return value

        return _complete()

    def assert_awaited_once(self):
        assert len(self.await_args_list) == 1

    def assert_awaited_once_with(self, *args, **kwargs):
        self.assert_awaited_once()
        call = self.await_args_list[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_any_await(self, *args, **kwargs):
        assert any(call.args == args and call.kwargs == kwargs for call in self.await_args_list)

    def assert_not_awaited(self):
        assert not self.await_args_list


def _resolve_dotted_path(target):
    parts = target.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        owner = module
        for part in parts[index:-1]:
            owner = getattr(owner, part)
        return owner, parts[-1]
    raise ImportError(f"Cannot resolve dotted target: {target}")


class _SwapContext:
    def __init__(self, owner, name, replacement, *, create=False):
        self.owner = owner
        self.name = name
        self.replacement = replacement
        self.create = create
        self.had_original = False
        self.original = None

    def __enter__(self):
        self.had_original = hasattr(self.owner, self.name)
        if self.had_original:
            self.original = getattr(self.owner, self.name)
        elif not self.create:
            raise AttributeError(self.name)
        setattr(self.owner, self.name, self.replacement)
        return self.replacement

    def __exit__(self, exc_type, exc, tb):
        if self.had_original:
            setattr(self.owner, self.name, self.original)
        else:
            delattr(self.owner, self.name)
        return False


class _SwapDictContext:
    def __init__(self, mapping, values, *, clear=False):
        if isinstance(mapping, str):
            owner, name = _resolve_dotted_path(mapping)
            mapping = getattr(owner, name)
        self.mapping = mapping
        self.values = dict(values)
        self.clear = clear
        self.original = None

    def __enter__(self):
        self.original = dict(self.mapping)
        if self.clear:
            self.mapping.clear()
        self.mapping.update(self.values)
        return self.mapping

    def __exit__(self, exc_type, exc, tb):
        self.mapping.clear()
        self.mapping.update(self.original)
        return False


def _replacement(new=_MISSING, *, return_value=_MISSING, side_effect=None, wraps=None):
    if new is not _MISSING:
        return new
    kwargs = {"side_effect": side_effect}
    if wraps is not None:
        return _CallRecorder(wraps=wraps, **kwargs)
    if return_value is not _MISSING:
        return _CallRecorder(return_value=return_value, **kwargs)
    return _CallRecorder(**kwargs)


class _Swap:
    def __call__(
        self,
        target,
        new=_MISSING,
        *,
        return_value=_MISSING,
        side_effect=None,
        create=False,
        wraps=None,
    ):
        owner, name = _resolve_dotted_path(target)
        return _SwapContext(
            owner,
            name,
            _replacement(new, return_value=return_value, side_effect=side_effect, wraps=wraps),
            create=create,
        )

    def object(
        self,
        owner,
        name,
        new=_MISSING,
        *,
        return_value=_MISSING,
        side_effect=None,
        create=False,
        wraps=None,
    ):
        return _SwapContext(
            owner,
            name,
            _replacement(new, return_value=return_value, side_effect=side_effect, wraps=wraps),
            create=create,
        )

    def dict(self, mapping, values, *, clear=False):
        return _SwapDictContext(mapping, values, clear=clear)


swap = _Swap()


class TestMLXCompatibility(unittest.IsolatedAsyncioTestCase):
    def _ready_client(self, model_path: str) -> MLXLocalClient:
        client = MLXLocalClient(model_path=model_path)
        now = time.time()
        client._init_done = True
        client._lane_state = "ready"
        client._lane_error = ""
        client._last_heartbeat = now
        client._last_progress_at = now
        client._last_token_progress_at = now
        client._last_ready_at = now
        client._last_generation_completed_at = now
        # A primary/deep lane is only "conversation ready" once it has proven a
        # user-facing turn can complete (visible_conversation_probe). Stamp the
        # visible-readiness anchors so this helper models a TRULY ready client.
        client._last_visible_readiness_at = now
        client._last_user_facing_completed_at = now
        return client

    async def test_warm_up_alias_delegates_to_warmup(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        client.warmup = _AsyncCallRecorder(return_value="ok")

        result = await client.warm_up()

        self.assertEqual(result, "ok")
        client.warmup.assert_awaited_once()

    async def test_32b_lane_status_rejects_missing_required_recurrent_depth(self):
        client = self._ready_client(QWEN32_TEST_MODEL)
        client._recurrent_depth_status = {
            "active": False,
            "config": None,
            "expected_loops": 2,
            "required": True,
        }

        with swap.object(client, "is_alive", new=lambda: True):
            status = client.get_lane_status()

        self.assertFalse(status["conversation_ready"])
        self.assertEqual(status["state"], "recovering")
        self.assertIn("recurrent_depth_inactive", status["readiness_blockers"])
        self.assertTrue(status["recurrent_depth"]["required"])
        self.assertEqual(status["recurrent_depth"]["expected_loops"], 2)

    async def test_32b_lane_status_accepts_active_recurrent_depth(self):
        client = self._ready_client(QWEN32_TEST_MODEL)
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 2},
            "expected_loops": 2,
            "required": True,
        }

        with swap.object(client, "is_alive", new=lambda: True):
            status = client.get_lane_status()

        self.assertTrue(status["conversation_ready"])
        self.assertNotIn("recurrent_depth_inactive", status["readiness_blockers"])
        self.assertNotIn("recurrent_depth_loop_mismatch", status["readiness_blockers"])

    async def test_health_probe_completion_does_not_prove_visible_conversation(self):
        client = MLXLocalClient(model_path=QWEN32_TEST_MODEL)

        client._mark_generation_completed(user_facing=False)

        self.assertGreater(client._last_generation_completed_at, 0.0)
        self.assertEqual(client._last_user_facing_completed_at, 0.0)
        self.assertEqual(client._last_visible_readiness_at, 0.0)

    async def test_user_facing_completion_proves_visible_conversation(self):
        client = MLXLocalClient(model_path=QWEN32_TEST_MODEL)

        client._mark_generation_completed(user_facing=True)

        self.assertGreater(client._last_user_facing_completed_at, 0.0)
        self.assertEqual(
            client._last_visible_readiness_at,
            client._last_user_facing_completed_at,
        )

    async def test_32b_lane_status_rejects_recurrent_depth_loop_mismatch(self):
        client = self._ready_client(QWEN32_TEST_MODEL)
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 1},
            "expected_loops": 2,
            "required": True,
        }

        with swap.object(client, "is_alive", new=lambda: True):
            status = client.get_lane_status()

        self.assertFalse(status["conversation_ready"])
        self.assertEqual(status["state"], "recovering")
        self.assertIn("recurrent_depth_loop_mismatch", status["readiness_blockers"])

    async def test_32b_generation_admission_rejects_inactive_recurrent_depth(self):
        client = self._ready_client(QWEN32_TEST_MODEL)
        client._process = SimpleNamespace(is_alive=lambda: True)
        client._init_done = True
        client._recurrent_depth_status = {
            "active": False,
            "config": None,
            "expected_loops": 2,
            "required": True,
        }

        alive = await client._ensure_worker_alive(foreground_request=True)

        self.assertFalse(alive)
        self.assertEqual(client._lane_state, "recovering")
        self.assertEqual(client._lane_error, "recurrent_depth_inactive")

    async def test_operator_disabled_recurrent_depth_does_not_block_32b_readiness(self):
        client = self._ready_client(QWEN32_TEST_MODEL)
        client._recurrent_depth_status = {
            "active": False,
            "config": None,
            "reason": "standard_or_operator_disabled",
        }

        with swap.dict(os.environ, {"AURA_RECURRENT_LOOPS": "0"}, clear=False), \
             swap.object(client, "is_alive", new=lambda: True):
            status = client.get_lane_status()

        self.assertTrue(status["conversation_ready"])
        self.assertFalse(status["recurrent_depth"]["required"])
        self.assertEqual(status["recurrent_depth"]["expected_loops"], 0)


class TestSensoryClientRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_start_uses_spawn_on_darwin_and_pings_worker(self):
        client = SensoryLocalClient()

        process = _CallRecorder()
        process.pid = 4321
        process.is_alive.return_value = True
        ctx = _CallRecorder()
        ctx.Process.return_value = process
        ctx.get_start_method.return_value = "spawn"

        # start() drives _request_blocking, which returns a STATUS DICT.
        # The double here stubbed _send_command with booleans — a method the
        # start path no longer calls — so this test stopped exercising the
        # ping/init sequence it names and simply failed.
        commands: list[str] = []

        def _request_blocking(command, payload=None, timeout=None):
            commands.append(command)
            return {"status": "ok"}

        with swap("core.senses.sensory_client.sys.platform", "darwin"), \
             swap("core.senses.sensory_client.mp.get_context", return_value=ctx) as get_context, \
             swap.object(client, "_request_blocking", new=_request_blocking):
            started = await client.start()

        self.assertTrue(started)
        get_context.assert_called_once_with("spawn")
        self.assertEqual(commands, ["ping", "init_vision", "init_audio"])

    async def test_send_command_restarts_dead_worker(self):
        client = SensoryLocalClient()

        async def restart_worker():
            client._req_q = _CallRecorder()
            client._res_q = _CallRecorder()
            client._process = _CallRecorder()
            client._process.is_alive.return_value = True
            return True

        client.start = _AsyncCallRecorder(side_effect=restart_worker)
        task_registry = _CallRecorder()
        task_registry.register_task.return_value = "task-1"

        with swap("core.supervisor.registry.get_task_registry", return_value=task_registry), \
             swap("core.senses.sensory_client.asyncio.to_thread", new=_AsyncCallRecorder(return_value={"status": "ok"})):
            ok = await client._send_command("ping")

        self.assertTrue(ok)
        client.start.assert_awaited_once()


class TestResponsePhaseErrorBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_user_facing_response_phase_failure_clears_stale_reply(self):
        from core.resilience.error_boundary import CircuitRegistry, wrap_phase

        CircuitRegistry._instance = None
        state = SimpleNamespace(
            cognition=SimpleNamespace(current_origin="api", last_response="old answer from a previous turn"),
            response_modifiers={},
            world=SimpleNamespace(recent_percepts=[]),
        )

        async def fail_phase(_state, objective=None, **_kwargs):
            await asyncio.sleep(0)
            raise TimeoutError("only unsafe drafts")

        result = await wrap_phase(
            "UnitaryResponsePhase",
            fail_phase,
            state,
            objective="new user turn",
        )

        self.assertIs(result, state)
        self.assertEqual(state.cognition.last_response, "")
        self.assertTrue(state.response_modifiers["suppress_stale_response_reuse"])
        self.assertEqual(state.response_modifiers["response_phase_failed"], "UnitaryResponsePhase")


class TestStallWatchdogRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_stall_recovery_is_diagnostic_only_by_default(self):
        from core.resilience.stall_watchdog import StallWatchdog

        async def long_lived():
            await asyncio.Event().wait()

        loop = asyncio.get_running_loop()
        task = asyncio.create_task(long_lived(), name="ordinary_background_task")
        dog = StallWatchdog(loop, threshold=0.1)
        dog._task_birth[id(task)] = time.time() - 300

        try:
            with swap.dict(os.environ, {}, clear=True):
                await dog._recover_on_loop(91.0)

            self.assertFalse(task.cancelled())
            self.assertFalse(task.done())
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class TestSelfHealingLoopSafety(unittest.IsolatedAsyncioTestCase):
    async def test_stop_drains_loop_and_deep_repair_ownership(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        cancelled: list[str] = []

        async def owned_task(name: str) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(name)

        loop_task = asyncio.create_task(owned_task("loop"))
        repair_task = asyncio.create_task(owned_task("repair"))
        healer._task = loop_task
        healer._running = True
        healer._deep_repairs["core/example.py"] = repair_task
        await asyncio.sleep(0)

        await healer.stop()

        self.assertCountEqual(cancelled, ["loop", "repair"])
        self.assertTrue(loop_task.done())
        self.assertTrue(repair_task.done())
        self.assertEqual(healer.get_status()["deep_repairs_active"], 0)
        self.assertGreaterEqual(healer.shutdown_timeout_s, 5.0)

    async def test_ledger_write_timeout_does_not_block_tick(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        healer._ledger_write_timeout_s = 0.01
        healer.watch("slow_watch", expected_interval_s=0.01)
        watch = healer._watches["slow_watch"]
        watch.last_heartbeat_at = time.time() - 10

        def slow_append(_record):
            import threading

            threading.Event().wait(0.25)

        started = time.perf_counter()
        with swap.object(healer, "_foreground_runtime_busy", return_value=True), \
             swap.object(healer, "_append_record", side_effect=slow_append):
            await healer._tick()

        self.assertLess(time.perf_counter() - started, 0.12)
        self.assertGreater(watch.last_heartbeat_at, time.time() - 1.0)

    async def test_restart_timeout_is_contained_and_reported(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        healer._restart_timeout_s = 0.01
        release = asyncio.Event()

        async def stalled_restart():
            await release.wait()

        healer.watch(
            "stalled_watch",
            expected_interval_s=0.01,
            restart_async=stalled_restart,
        )
        watch = healer._watches["stalled_watch"]
        append_record = _AsyncCallRecorder()

        with swap.object(healer, "_healing_defer_reason", return_value=""), \
             swap.object(healer, "_append_record_async", new=append_record):
            await asyncio.wait_for(healer._heal(watch, 10.0), timeout=0.25)

        record = append_record.await_args.args[0]
        self.assertEqual(record["result"], "restart_timeout_after_0.01s")
        self.assertEqual(watch.restart_failures, 1)
        self.assertIn("TimeoutError", watch.last_restart_error)
        self.assertGreater(watch.last_heartbeat_at, time.time() - 1.0)

    async def test_restart_failure_does_not_terminate_following_heal(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        calls = 0

        async def flaky_restart():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("first attempt failed")

        healer.watch("flaky_watch", expected_interval_s=0.01, restart_async=flaky_restart)
        watch = healer._watches["flaky_watch"]
        append_record = _AsyncCallRecorder()

        with swap.object(healer, "_healing_defer_reason", return_value=""), \
             swap.object(healer, "_append_record_async", new=append_record):
            await healer._heal(watch, 10.0)
            await healer._heal(watch, 10.0)

        self.assertEqual(calls, 2)
        self.assertEqual(append_record.await_args_list[-1].args[0]["result"], "restarted")
        self.assertEqual(watch.restart_failures, 0)
        self.assertEqual(watch.restarts, 1)

    async def test_inference_gate_foreground_dict_defers_healing_restart(self):
        from core.runtime.self_healing import SelfHealing

        ServiceContainer.clear()
        ServiceContainer.register_instance(
            "inference_gate",
            SimpleNamespace(
                get_conversation_status=lambda: {
                    "foreground_owned": True,
                    "active_generations": 1,
                    "state": "ready",
                }
            ),
        )
        healer = SelfHealing()
        restart = _AsyncCallRecorder()
        healer.watch("foreground_watch", expected_interval_s=0.01, restart_async=restart)
        watch = healer._watches["foreground_watch"]
        watch.last_heartbeat_at = time.time() - 10

        append_record = _AsyncCallRecorder()
        try:
            with swap.object(healer, "_append_record_async", new=append_record):
                await healer._tick()
        finally:
            ServiceContainer.clear()

        self.assertEqual(restart.await_count, 0)
        self.assertEqual(append_record.await_count, 1)
        self.assertEqual(append_record.await_args.args[0]["result"], "deferred_foreground_busy")
        self.assertGreater(watch.last_heartbeat_at, time.time() - 1.0)

    async def test_headless_proof_run_defers_healing_restart(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        restart = _AsyncCallRecorder()
        healer.watch("proof_watch", expected_interval_s=0.01, restart_async=restart)
        watch = healer._watches["proof_watch"]
        watch.last_heartbeat_at = time.time() - 10

        append_record = _AsyncCallRecorder()
        with swap.dict(os.environ, {"AURA_PROOF_RUN": "1"}, clear=False), \
             swap.object(healer, "_append_record_async", new=append_record):
            await healer._tick()

        self.assertEqual(restart.await_count, 0)
        self.assertEqual(append_record.await_count, 1)
        self.assertEqual(append_record.await_args.args[0]["result"], "deferred_proof_run_active")
        self.assertGreater(watch.last_heartbeat_at, time.time() - 1.0)

    async def test_heal_rechecks_defer_reason_before_restart(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        restart = _AsyncCallRecorder()
        healer.watch("raced_watch", expected_interval_s=0.01, restart_async=restart)
        watch = healer._watches["raced_watch"]
        append_record = _AsyncCallRecorder()

        with swap.object(healer, "_healing_defer_reason", return_value="shutdown_requested"), \
             swap.object(healer, "_append_record_async", new=append_record):
            await healer._heal(watch, 10.0)

        self.assertEqual(restart.await_count, 0)
        self.assertEqual(append_record.await_count, 1)
        self.assertEqual(append_record.await_args.args[0]["result"], "deferred_shutdown_requested")
        self.assertGreater(watch.last_heartbeat_at, time.time() - 1.0)

    async def test_module_path_resolution_is_offloaded_for_deep_repair(self):
        from core.runtime.self_healing import SelfHealing

        healer = SelfHealing()
        healer.watch("orchestrator", expected_interval_s=0.01, container_key="orchestrator")
        watch = healer._watches["orchestrator"]
        watch.restarts = 3
        watch.last_heartbeat_at = time.time() - 10

        to_thread = _AsyncCallRecorder(return_value="core/orchestrator/main.py")
        with swap.dict(os.environ, {"AURA_ENABLE_DEEP_REPAIR": "1"}), \
             swap("core.runtime.self_healing.asyncio.to_thread", new=to_thread), \
             swap.object(healer, "schedule_deep_repair", return_value={"result": "deep_repair_scheduled"}), \
             swap.object(healer, "_append_record_async", new=_AsyncCallRecorder()):
            await healer._heal(watch, 10.0)

        self.assertEqual(to_thread.await_count, 1)


class TestRSSFallback(unittest.TestCase):
    def test_stdlib_rss_parser_returns_feedparser_like_shape(self):
        from core.utils.rss_feed import parse_feed_bytes

        feed = parse_feed_bytes(
            b"""<?xml version="1.0"?>
            <rss version="2.0"><channel>
              <title>World</title>
              <item><title>Signal found</title><link>https://example.test/a</link><description>Useful context</description></item>
            </channel></rss>"""
        )

        self.assertEqual(feed.feed["title"], "World")
        self.assertEqual(feed.entries[0].title, "Signal found")
        self.assertEqual(feed.entries[0].summary, "Useful context")


class TestShutdownCoordination(unittest.TestCase):
    def test_supervisor_does_not_schedule_actor_restart_during_shutdown(self):
        from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown
        from core.supervisor.tree import ActorSpec, SupervisionTree

        tree = SupervisionTree()
        tree.add_actor(ActorSpec(name="worker", entry_point=lambda _pipe=None: None))
        actor = tree._actors["worker"]
        actor.process = _CallRecorder()
        actor.process.is_alive.return_value = False
        actor.process.exitcode = -15
        actor.pipe = object()

        request_shutdown("test")
        try:
            tree._poll_health()
            self.assertEqual(actor.next_restart_time, 0.0)
            self.assertIsNone(tree.start_actor("worker"))
        finally:
            clear_shutdown_request()

    def test_proof_runners_use_normal_exit_after_runtime_shutdown(self):
        root = Path(__file__).resolve().parents[1]
        proof_runners = [
            root / "tools" / "run_aletheia_live_proof.py",
            root / "tools" / "certify_boot.py",
        ]
        for runner in proof_runners:
            source = runner.read_text(encoding="utf-8")
            self.assertNotIn("os._exit", source, f"{runner} bypasses multiprocessing cleanup")

        # The DNU battery is allowed a *guarded* fast exit: after runtime
        # shutdown it must explicitly reap child processes and flush stdio
        # before os._exit, so multiprocessing cleanup happens by hand instead
        # of hanging in Py_FinalizeEx on non-daemon helper/native threads.
        dnu_source = (root / "tools" / "agi" / "run_dnu_agi_proof_battery.py").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            dnu_source,
            r"await shutdown_proof_runtime\(orch\)\n\s+return (?:1|fail_run_status\()",
        )
        self.assertRegex(
            dnu_source,
            r"_reap_proof_child_processes_sync\(\"process_exit\"\)\n"
            r"\s+sys\.stdout\.flush\(\)\n"
            r"\s+sys\.stderr\.flush\(\)\n"
            r"(?:\s+#[^\n]*\n)*"
            r"\s+os\._exit\(int\(code\)\)",
            "os._exit in the DNU battery must stay guarded by explicit child "
            "reaping and stdio flushes",
        )

    def test_agency_shutdown_prefers_final_client_close_before_reboot_fallback(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "tools" / "agency" / "run_agency_emergence_battery.py").read_text(
            encoding="utf-8"
        )
        shutdown_block = source[
            source.index("async def shutdown_agency_runtime"):
            source.index("async def main", source.index("async def shutdown_agency_runtime"))
        ]
        close_index = shutdown_block.index('for close_name in ("aclose", "close", "cleanup", "on_stop")')
        reboot_index = shutdown_block.index("reboot_worker = getattr")
        self.assertLess(close_index, reboot_index)


class TestStateTransportRuntimeEdges(unittest.IsolatedAsyncioTestCase):
    async def test_pipe_handler_failure_returns_failed_response_without_killing_dispatcher(self):
        from core.bus.local_pipe_bus import LocalPipeBus

        bus = LocalPipeBus(start_reader=False)
        bus._write_raw_message = _AsyncCallRecorder()
        handler_calls = {"count": 0}

        def bad_handler(_payload, _trace_id):
            handler_calls["count"] += 1
            raise ValueError("bad payload")

        try:
            await bus._handle_message(
                bad_handler,
                {
                    "type": "commit",
                    "payload": {"state": {}},
                    "trace_id": "trace-1",
                    "request_id": "request-1",
                    "is_request": True,
                },
            )
        finally:
            await bus.stop()

        bus._write_raw_message.assert_awaited_once()
        self.assertEqual(handler_calls["count"], 1)
        raw_response = bus._write_raw_message.await_args.args[0]
        self.assertIn('"failed": true', raw_response)
        self.assertIn('"response_to": "request-1"', raw_response)

    async def test_proxy_commit_timeout_defers_under_live_fail_closed_state_repository(self):
        from core.state.aura_state import AuraState
        from core.state.state_repository import StateRepository

        class SaturatedTransport:
            def __init__(self):
                self.calls = 0

            async def request(self, *_args, **_kwargs):
                self.calls += 1
                raise TimeoutError("commit pipe saturated")

        transport = SaturatedTransport()
        repo = StateRepository(
            db_path=str(TMP_ROOT / f"aura_state_proxy_timeout_{time.time_ns()}.db"),
            is_vault_owner=False,
        )
        repo._current = AuraState()
        repo._transport = transport

        try:
            with swap.dict(os.environ, {"AURA_MODE": "live"}, clear=False):
                ServiceContainer.clear()
                ServiceContainer.register_instance(
                    "state_repository",
                    repo,
                    required=True,
                    failure_policy="fail-closed",
                )

                next_state = repo._current.derive("foreground_commit", origin="api")
                await repo.commit(next_state, "foreground_commit", trace_id="trace-live")

                status = repo.get_runtime_status()
                self.assertTrue(status["pending_proxy_commit"])
                self.assertEqual(status["pending_proxy_commit_count"], 1)
                self.assertIn("TimeoutError", status["last_proxy_commit_error"])
                self.assertEqual(transport.calls, 2)
        finally:
            await repo.close()
            ServiceContainer.clear()


class TestShutdownCancellationHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_self_healing_loop_honors_process_shutdown_cancellation(self):
        from core.runtime.self_healing import SelfHealing
        from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

        healer = SelfHealing()
        healer._tick = _AsyncCallRecorder(return_value=None)

        try:
            await healer.start(interval=60.0)
            self.assertIsNotNone(healer._task)
            await asyncio.sleep(0)
            request_shutdown("test")
            healer._task.cancel()
            await asyncio.wait_for(healer._task, timeout=0.25)
            self.assertTrue(healer._task.done())
            self.assertFalse(healer._task.cancelled())
        finally:
            healer._running = False
            clear_shutdown_request()


class TestAffectBroadcastBackpressure(unittest.IsolatedAsyncioTestCase):
    async def test_affect_broadcast_caps_background_tasks(self):
        emit_release = asyncio.Event()

        class _Bus:
            async def emit(self, *_args, **_kwargs):
                await emit_release.wait()

        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        engine._max_background_tasks = 2

        with swap("core.container.ServiceContainer.get", return_value=_Bus()):
            await engine._broadcast_event("affect_pulse")
            await engine._broadcast_event("affect_pulse")
            await engine._broadcast_event("affect_pulse")

        self.assertEqual(len(engine._background_tasks), 2)

        emit_release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_affect_appraisal_skips_llm_when_foreground_lane_is_protected(self):
        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        guarded_gate = _CallRecorder(
            _should_quiet_background_for_cortex_startup=_CallRecorder(return_value=True)
        )
        guarded_gate.get_conversation_status.return_value = {
            "conversation_ready": False,
            "state": "warming",
            "warmup_in_flight": True,
        }

        with swap("core.container.ServiceContainer.get", return_value=guarded_gate):
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        self.assertEqual(engine._llm_failure_count, 0)

    async def test_affect_appraisal_skips_llm_when_foreground_turn_is_active(self):
        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        guarded_gate = _CallRecorder(
            _foreground_user_turn_active=_CallRecorder(return_value=True)
        )
        guarded_gate.get_conversation_status.return_value = {
            "conversation_ready": True,
            "state": "ready",
            "warmup_in_flight": False,
            "foreground_owned": False,
            "active_generations": 0,
            "request_age_s": 0.0,
        }
        guarded_gate.generate = _AsyncCallRecorder(side_effect=AssertionError("LLM appraisal should have been deferred"))

        with swap("core.container.ServiceContainer.get", return_value=guarded_gate), \
             swap("core.brain.llm.mlx_client._foreground_owner_active", return_value=False):
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        self.assertEqual(engine._llm_failure_count, 0)

    async def test_affect_appraisal_defaults_to_heuristic_for_live_background(self):
        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        guarded_gate = _CallRecorder()
        guarded_gate.get_conversation_status.return_value = {
            "conversation_ready": True,
            "state": "ready",
            "warmup_in_flight": False,
            "foreground_owned": False,
            "active_generations": 0,
            "request_age_s": 0.0,
        }
        guarded_gate.generate = _AsyncCallRecorder(side_effect=AssertionError("background affect appraisal must not call the LLM by default"))
        engine._background_llm_should_defer = _CallRecorder(return_value=False)

        with swap("core.container.ServiceContainer.get", return_value=guarded_gate), \
             swap("core.brain.llm.mlx_client._foreground_owner_active", return_value=False):
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        guarded_gate.generate.assert_not_awaited()
        self.assertEqual(engine._llm_failure_count, 0)

    async def test_affect_opt_in_parse_failure_uses_heuristic_appraisal(self):
        """A model that answers in prose must not cost the appraisal.

        a7bc6b48f moved the fallback OUT of _appraise_with_llm and into its
        caller, which is the better place for it: the inner call now reports
        that the response could not be parsed instead of returning a number it
        did not measure. The contract this test exists for — an unparseable
        response still yields a heuristic appraisal — is asserted where it
        now lives.
        """
        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        gate = _CallRecorder()
        gate.generate = _AsyncCallRecorder(return_value="I feel present.")
        gate.get_conversation_status.return_value = {
            "conversation_ready": True,
            "state": "ready",
        }

        with swap("core.container.ServiceContainer.get", return_value=gate):
            with self.assertRaises(ValueError) as caught:
                await engine._appraise_with_llm("error and frustration", {"intensity": 0.6})
        self.assertEqual(str(caught.exception), "parse_failure")

        # And the caller still produces a real appraisal from that failure.
        engine.react = _AsyncCallRecorder(side_effect=ValueError("parse_failure"))
        result = await engine.apply_stimulus("error and frustration", 6.0)
        self.assertEqual(result["status"], "heuristic_fallback")
        appraisal = result["appraisal"]
        # The contract is that a real appraisal is produced from the failure,
        # not that the words "error and frustration" score negative. That
        # second assertion was the word scan core.interiority replaced: with
        # nothing at stake, a sentence containing the word error is not a bad
        # event, and requiring it to be one is requiring the classifier back.
        self.assertIn("v", appraisal)
        self.assertIn("a", appraisal)
        self.assertGreater(appraisal["a"], 0.0)
        for axis in ("v", "a", "e"):
            if axis in appraisal:
                self.assertIsInstance(appraisal[axis], float)

    async def test_apply_stimulus_suppresses_empty_response_failures(self):
        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        engine.react = _AsyncCallRecorder(side_effect=ValueError("empty_response"))

        result = await engine.apply_stimulus("resource_strain", 9.0)

        self.assertEqual(result["status"], "heuristic_fallback")
        self.assertIn("appraisal", result)

    async def test_affect_background_timeout_falls_back_without_runtime_degradation(self):
        with swap("core.affect.damasio_v2.PhysicalActuator", return_value=_CallRecorder()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        engine._background_llm_should_defer = _CallRecorder(return_value=False)
        engine._appraise_with_llm = _AsyncCallRecorder(side_effect=TimeoutError())
        engine.iot_bridge.broadcast_affect_state = _AsyncCallRecorder(return_value=None)

        with swap("core.brain.llm.mlx_client._foreground_owner_active", return_value=False), \
             swap("core.affect.damasio_v2.record_degradation") as record_degradation:
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        self.assertEqual(engine._llm_failure_count, 0)
        record_degradation.assert_not_called()


class TestEternalMemoryCaching(unittest.IsolatedAsyncioTestCase):
    async def test_eternal_memory_reuses_recent_summary_cache(self):
        from core.kernel.upgrades_10x import EternalMemoryPhase

        phase = EternalMemoryPhase(_CallRecorder())
        phase._summary_cache = [{"role": "system", "content": "[ETERNAL MEMORY]\nsteady"}]
        import time
        phase._last_summary_refresh_at = time.time()
        phase._summary_refresh_interval_s = 120.0
        phase._load_eternal_slice = _CallRecorder(side_effect=AssertionError("should not load recent history"))

        with swap.object(phase, "_background_llm_should_defer", return_value=False):
            summary = await phase._get_cached_or_refresh_summary()

        self.assertEqual(summary, phase._summary_cache)

    async def test_eternal_memory_refresh_is_nonblocking_and_singleflight(self):
        from core.kernel.upgrades_10x import EternalMemoryPhase

        phase = EternalMemoryPhase(_CallRecorder())
        phase._background_llm_should_defer = lambda: False
        phase._load_eternal_slice = lambda limit: [{"summary": "retained"}]
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def slow_summary(history):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return [{"role": "system", "content": "[ETERNAL MEMORY]\nretained"}]

        phase._generate_eternal_summary = slow_summary

        started = time.perf_counter()
        initial = await phase._get_cached_or_refresh_summary()
        elapsed = time.perf_counter() - started

        self.assertEqual(initial, [])
        self.assertLess(elapsed, 0.05)
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        first_task = phase._summary_refresh_task
        self.assertIsNotNone(first_task)

        duplicate = await phase._get_cached_or_refresh_summary()
        self.assertEqual(duplicate, [])
        self.assertIs(phase._summary_refresh_task, first_task)
        self.assertEqual(calls, 1)

        release.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        self.assertIsNone(phase._summary_refresh_task)
        self.assertEqual(
            await phase._get_cached_or_refresh_summary(),
            [{"role": "system", "content": "[ETERNAL MEMORY]\nretained"}],
        )


class TestNativeMultimodalBridgeGuards(unittest.IsolatedAsyncioTestCase):
    async def test_native_multimodal_bridge_skips_disabled_vision_without_name_error(self):
        from core.kernel.upgrades_10x import NativeMultimodalBridge
        from core.state.aura_state import AuraState

        kernel = _CallRecorder()
        kernel.organs = {}
        phase = NativeMultimodalBridge(kernel)
        state = AuraState()
        state.cognition.current_objective = "Please inspect the screen."

        with swap.dict("os.environ", {"AURA_ENABLE_NATIVE_VISION_ACTIONS": "0"}, clear=False):
            new_state = await phase.execute(state, objective=state.cognition.current_objective)

        self.assertIs(new_state, state)


class TestJsonRepairGuards(unittest.IsolatedAsyncioTestCase):
    async def test_json_repair_handles_none_without_crashing(self):
        from core.utils.json_utils import SelfHealingJSON

        repairer = SelfHealingJSON(brain=_CallRecorder())

        parsed = await repairer.parse(None)

        self.assertEqual(parsed, {})

    async def test_json_repair_handles_python_style_dict_payloads(self):
        from core.utils.json_utils import SelfHealingJSON

        repairer = SelfHealingJSON(brain=_CallRecorder())

        parsed = await repairer.parse(
            "{'signature_phrase': 'I am steady.', 'stable_traits': ['curious'], "
            "'learned_preferences': ['space'], 'growth_edges': ['patience'],}"
        )

        self.assertEqual(parsed["signature_phrase"], "I am steady.")
        self.assertEqual(parsed["stable_traits"], ["curious"])


class TestExperienceConsolidatorGuards(unittest.IsolatedAsyncioTestCase):
    async def test_consolidator_defers_without_marking_last_run_when_foreground_is_busy(self):
        from core.consciousness.experience_consolidator import ExperienceConsolidator

        consolidator = ExperienceConsolidator(cognitive_engine=None)
        before = consolidator._last_run
        consolidator._background_should_defer = _CallRecorder(return_value=True)
        consolidator._gather_material = _CallRecorder(side_effect=AssertionError("foreground defer should skip work"))

        result = await consolidator.run_now()

        self.assertIsNone(result)
        self.assertEqual(consolidator._last_run, before)


class TestSubstrateStimulusGuards(unittest.IsolatedAsyncioTestCase):
    async def test_psych_stabilization_runs_off_event_loop(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=8))

        with swap(
            "core.consciousness.liquid_substrate.asyncio.to_thread",
            new=_AsyncCallRecorder(return_value=None),
        ) as to_thread:
            await substrate._stabilize_psych_state(0.05)

        to_thread.assert_awaited_once_with(substrate._stabilize_psych_state_sync, 0.05)

    async def test_recurrent_self_model_runs_off_event_loop(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=4))

        with swap(
            "core.consciousness.liquid_substrate.asyncio.to_thread",
            new=_AsyncCallRecorder(return_value=None),
        ) as to_thread:
            await substrate._recurrent_self_model(0.05)

        to_thread.assert_awaited_once_with(substrate._recurrent_self_model_sync, 0.05)

    async def test_plasticity_runs_off_event_loop(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=4))

        with swap(
            "core.consciousness.liquid_substrate.asyncio.to_thread",
            new=_AsyncCallRecorder(return_value=None),
        ) as to_thread:
            await substrate._apply_plasticity()

        to_thread.assert_awaited_once_with(substrate._apply_plasticity_sync)

    async def test_liquid_substrate_scales_constrained_stimulus_weight(self):
        import numpy as np

        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
        from core.consciousness.substrate_authority import AuthorizationDecision

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=4))
        substrate.x[:] = 0.0

        class _Authority:
            def authorize(self, *args, **kwargs):
                return SimpleNamespace(
                    decision=AuthorizationDecision.CONSTRAIN,
                    constraints=["neurochemical_gaba_collapse: internal_state_mutation_constrained"],
                )

        with swap(
            "core.container.ServiceContainer.get",
            staticmethod(lambda name, default=None: _Authority() if name == "substrate_authority" else default),
        ):
            await substrate.inject_stimulus([1.0, 1.0, 1.0, 1.0], weight=1.0)

        np.testing.assert_allclose(substrate.x, np.full(4, 0.02), atol=1e-6)


class TestSovereignPrunerGuards(unittest.IsolatedAsyncioTestCase):
    async def test_pruner_caps_consolidation_batch_and_preserves_deferred_memories(self):
        from core.memory.sovereign_pruner import MemoryRecord, SovereignPruner

        pruner = SovereignPruner(target_retention=0.0)
        pruner._min_prune_interval_s = 0.0
        pruner._max_consolidations_per_pass = 2
        pruner._background_should_defer = _CallRecorder(return_value=False)
        pruner._consolidate = _AsyncCallRecorder(side_effect=["insight-a", "insight-b"])

        memories = [
            MemoryRecord(
                id=f"mem-{idx}",
                content=f"memory {idx}",
                timestamp=time.time(),
                source="test",
                emotional_weight=0.3,
                identity_relevance=0.3,
            )
            for idx in range(4)
        ]

        surviving, log = await pruner.prune(memories, {})

        self.assertEqual(pruner._consolidate.await_count, 2)
        self.assertEqual(len(surviving), 4)
        self.assertTrue(any("Deferred consolidation" in entry for entry in log))

    async def test_pruner_honors_cooldown_without_spawning_more_work(self):
        from core.memory.sovereign_pruner import MemoryRecord, SovereignPruner

        pruner = SovereignPruner(target_retention=0.0)
        pruner._min_prune_interval_s = 999.0
        pruner._last_prune_at = time.time()
        pruner._background_should_defer = _CallRecorder(return_value=False)
        pruner._consolidate = _AsyncCallRecorder()

        memories = [
            MemoryRecord(
                id="mem-1",
                content="memory",
                timestamp=time.time(),
                source="test",
                emotional_weight=0.3,
                identity_relevance=0.3,
            )
        ]

        surviving, log = await pruner.prune(memories, {})

        self.assertEqual(surviving, memories)
        self.assertEqual(pruner._consolidate.await_count, 0)
        self.assertTrue(any("cooldown" in entry.lower() for entry in log))


class TestLocalVisionPermissionGuards(unittest.IsolatedAsyncioTestCase):
    async def test_capture_screen_skips_screenshot_when_permission_not_granted(self):
        from core.senses.screen_vision import LocalVision

        guard = _CallRecorder()
        guard.check_permission = _AsyncCallRecorder(
            return_value={"granted": False, "status": "deferred", "guidance": "nope"}
        )

        with swap("core.container.ServiceContainer.get", return_value=guard), swap(
            "core.senses.screen_vision._screen_capture_preflight",
            return_value=False,
        ):
            image = await LocalVision().capture_screen()

        self.assertIsNone(image)


class TestNeuralBridgeBootSafety(unittest.IsolatedAsyncioTestCase):
    async def test_neural_bridge_loads_lightweight_without_heavy_boot_dependencies(self):
        from core.senses.neural_bridge import NeuralBridge

        bridge = NeuralBridge(lightweight_mode=True)

        with swap.object(bridge, "start") as start:
            await bridge.load()

        self.assertTrue(bridge.is_trained)
        self.assertTrue(bridge.get_status()["lightweight_mode"])
        start.assert_called_once()


class TestNeuralOrganBootSafety(unittest.IsolatedAsyncioTestCase):
    async def test_neural_organ_uses_lightweight_mode_during_safe_desktop_boot(self):
        from core.kernel.organs import OrganStub

        bridge = _CallRecorder()
        bridge.load = _AsyncCallRecorder()

        with swap.dict("os.environ", {"AURA_SAFE_BOOT_DESKTOP": "1"}, clear=False):
            with swap("core.senses.neural_bridge.NeuralBridge", return_value=bridge) as bridge_cls:
                organ = OrganStub("neural", _CallRecorder())
                await organ.load()

        bridge_cls.assert_called_once_with(lightweight_mode=True)
        bridge.load.assert_awaited_once()
        self.assertIs(organ.instance, bridge)


class TestBackgroundPolicyGuards(unittest.TestCase):
    def test_background_policy_requires_user_anchor_before_unsolicited_work(self):
        from core.health.degraded_events import clear_degraded_events
        from core.runtime.background_policy import background_activity_reason
        from core.runtime.foreground_guard import _reset_for_tests as reset_foreground_guard

        clear_degraded_events()
        reset_foreground_guard()

        orch = _CallRecorder()
        orch.is_busy = False
        orch._suppress_unsolicited_proactivity_until = 0.0
        orch._foreground_user_quiet_until = 0.0
        orch._last_user_interaction_time = 0.0
        orch.status = _CallRecorder(last_user_interaction_time=0.0)

        reason = background_activity_reason(orch, require_conversation_ready=False)

        self.assertEqual(reason, "no_user_anchor")


class TestSupervisorShutdownGuards(unittest.TestCase):
    def test_shutdown_failure_path_never_schedules_actor_restart(self):
        from core.supervisor.tree import ActorSpec, SupervisionTree

        tree = SupervisionTree()
        tree.add_actor(ActorSpec(name="sensory", entry_point=lambda *_args: None))
        tree._is_running = False
        tree._shutting_down = True

        tree._handle_failure("sensory")

        actor = tree._actors["sensory"]
        self.assertEqual(actor.next_restart_time, 0.0)
        self.assertFalse(actor.is_circuit_broken)


class TestSovereignNetworkBackgroundGuards(unittest.IsolatedAsyncioTestCase):
    async def test_autonomous_network_scan_defers_during_foreground_quiet_window(self):
        from core.skills.sovereign_network import NetworkInput, SovereignNetworkSkill

        skill = SovereignNetworkSkill()
        with swap(
            "core.runtime.background_policy.background_activity_reason",
            return_value="foreground_quiet_window",
        ):
            result = await skill.execute(
                NetworkInput(mode="discovery", target="192.168.1.0/30", ports="8000"),
                {"origin": "system", "orchestrator": _CallRecorder()},
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["reason"], "foreground_quiet_window")


class TestCognitiveContextSanitization(unittest.TestCase):
    def test_runtime_status_hides_stale_referential_anchor_intention(self):
        from core.runtime.organism_status import _clean_current_intention_for_status

        stale = (
            "What do you think makes a friendship feel alive over time?\n\n"
            "[REFERENTIAL ANCHOR]\nThe user is referring to an older prompt."
        )

        self.assertEqual(_clean_current_intention_for_status(stale, ""), "idle")
        self.assertEqual(
            _clean_current_intention_for_status(stale, "current live turn"),
            "current live turn",
        )

    def test_trim_working_memory_clears_stale_speculative_autonomy_state(self):
        from core.state.aura_state import CognitiveContext

        ctx = CognitiveContext(
            current_objective="Researching The Unix Philosophy and the Art of Minimalist Tooling",
            current_origin="system",
            attention_focus="Researching The Unix Philosophy and the Art of Minimalist Tooling",
            last_response="I'm still processing that thought. Give me a moment....",
            active_goals=[
                {"goal": "Researching The Unix Philosophy and the Art of Minimalist Tooling", "origin": "system"},
                {"goal": "Run a diagnostic on shadow_ast_healer.py", "origin": "user"},
            ],
            pending_initiatives=[
                {"description": "Seek novel stimulation or internal simulation.", "origin": "system"},
                {"description": "Run a diagnostic on shadow_ast_healer.py", "origin": "user"},
            ],
        )

        ctx.trim_working_memory()

        self.assertIsNone(ctx.current_objective)
        self.assertIsNone(ctx.attention_focus)
        self.assertIsNone(ctx.last_response)
        self.assertEqual(ctx.active_goals, [{"goal": "Run a diagnostic on shadow_ast_healer.py", "origin": "user"}])
        self.assertEqual(ctx.pending_initiatives, [{"description": "Run a diagnostic on shadow_ast_healer.py", "origin": "user"}])


class TestMetabolicCoordinatorGuards(unittest.IsolatedAsyncioTestCase):
    async def test_metabolic_coordinator_preserves_live_singleton_locks(self):
        from core.coordinators import metabolic_coordinator as metabolic_module
        from core.coordinators.metabolic_coordinator import MetabolicCoordinator

        with self.subTest("live_lock_preserved_and_dead_lock_removed"):
            with swap.object(
                metabolic_module,
                "config",
                SimpleNamespace(paths=SimpleNamespace(home_dir=self._make_tmp_home())),
            ):
                lock_dir = metabolic_module.config.paths.home_dir / "locks"
                lock_dir.mkdir(parents=True, exist_ok=True)
                live_lock = lock_dir / "orchestrator.lock"
                stale_lock = lock_dir / "stale.lock"
                live_lock.write_text(str(metabolic_module.os.getpid()), encoding="utf-8")
                stale_lock.write_text("999999", encoding="utf-8")

                MetabolicCoordinator()

                self.assertTrue(live_lock.exists())
                self.assertFalse(stale_lock.exists())

    async def test_metabolic_coordinator_drains_neural_event_deque_without_pop_index_error(self):
        from core.coordinators.metabolic_coordinator import MetabolicCoordinator

        now = time.time()
        status = _CallRecorder(
            cycle_count=5,
            last_user_interaction_time=now,
            state="ready",
            acceleration_factor=1.0,
            singularity_threshold=False,
            is_processing=False,
            volition_level=0,
        )
        hooks = _CallRecorder(trigger=_AsyncCallRecorder())
        world = _CallRecorder(recent_percepts=[])
        message_queue = _CallRecorder()
        message_queue._queue = []
        message_queue.full.return_value = False
        message_queue.qsize.return_value = 0
        message_queue.empty.return_value = True

        orch = _CallRecorder(
            status=status,
            hooks=hooks,
            drive_controller=None,
            drives=None,
            is_busy=False,
            latent_core=None,
            predictive_model=None,
            kernel=_CallRecorder(organs={}),
            message_queue=message_queue,
            world=world,
            conversation_history=[],
            memory_manager=None,
            liquid_state=_CallRecorder(current=_CallRecorder(curiosity=0.5, frustration=0.0, energy=0.5)),
            lnn=None,
            homeostasis=None,
            mortality=None,
            singularity_monitor=None,
            swarm=None,
            _active_metabolic_tasks=set(),
            _last_thought_time=now,
            _last_boredom_impulse=0.0,
            _last_reflection_impulse=0.0,
            _last_pulse=now,
            _recovery_attempts=0,
        )
        orch._acquire_next_message = _AsyncCallRecorder(return_value=None)
        orch._dispatch_message = _CallRecorder()
        orch._publish_telemetry = _CallRecorder()
        orch._emit_thought_stream = _CallRecorder()

        coord = MetabolicCoordinator(orch=orch)
        coord._event_bus = object()
        coord._consume_energy = _CallRecorder(return_value=False)
        coord.update_liquid_pacing = _CallRecorder()
        coord.manage_memory_hygiene = _CallRecorder()
        coord.process_world_decay = _AsyncCallRecorder()
        coord.trigger_autonomous_thought = _AsyncCallRecorder()
        coord.run_terminal_self_heal = _AsyncCallRecorder()
        coord._neural_events.append({"command": "test", "confidence": 0.9})

        result = await coord._process_metabolic_tasks()

        self.assertFalse(result)
        self.assertEqual(world.recent_percepts[0]["command"], "test")

    def _make_tmp_home(self):
        import shutil
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp(prefix="aura-metabolic-locks-"))
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path


class TestSelfModificationBackgroundSafety(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_background_coro_swallows_cancelled_error(self):
        from core.self_modification.self_modification_engine import _schedule_background_coro

        loop = asyncio.get_running_loop()
        captured = []
        previous_handler = loop.get_exception_handler()

        def _handler(_loop, context):
            captured.append(context)

        loop.set_exception_handler(_handler)
        try:
            cancelled_attempts = []

            async def _cancelled():
                cancelled_attempts.append("attempted")
                raise asyncio.CancelledError()

            _schedule_background_coro(_cancelled(), label="test_cancelled_background")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertEqual(captured, [])
        self.assertEqual(cancelled_attempts, ["attempted"])

    async def test_error_logger_ignores_cancelled_append(self):
        from core.self_modification.error_intelligence import StructuredErrorLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            logger_system = StructuredErrorLogger(log_dir=temp_dir)
            with swap(
                "core.self_modification.error_intelligence.asyncio.to_thread",
                new=_AsyncCallRecorder(side_effect=asyncio.CancelledError),
            ):
                await logger_system._append_to_log(Path(temp_dir) / "error_events.jsonl", {"ok": True})

    async def test_error_logger_append_is_governed_under_strict_runtime(self):
        from core.governance_context import get_violations
        from core.self_modification.error_intelligence import StructuredErrorLogger

        with tempfile.TemporaryDirectory() as temp_dir, swap.dict(
            os.environ,
            {"AURA_GOVERNANCE_MODE": "strict"},
        ):
            target = Path(temp_dir) / "execution_log.jsonl"
            logger_system = StructuredErrorLogger(log_dir=temp_dir)
            before = len(get_violations(200))

            await logger_system._append_to_log(target, {"ok": True, "source": "test"})

            after = len(get_violations(200))
            lines = target.read_text(encoding="utf-8").splitlines()

        self.assertEqual(after, before)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["source"], "test")

    async def test_self_modification_diagnosis_uses_static_path_by_default(self):
        from core.self_modification.error_intelligence import (
            AutomatedDiagnosisEngine,
            ErrorEvent,
            ErrorPattern,
        )

        brain = SimpleNamespace(think=_AsyncCallRecorder(return_value=SimpleNamespace(content="{}")))
        engine = AutomatedDiagnosisEngine(brain)
        event = ErrorEvent(
            timestamp=time.time(),
            error_type="RuntimeWarning",
            error_message="coroutine 'MemoryCoordinator.prune_low_salience' was never awaited",
            stack_trace="trace",
            context={},
            file_path="core/coordinators/metabolic_coordinator.py",
            line_number=841,
        )
        pattern = ErrorPattern(
            fingerprint="await-warning",
            occurrences=2,
            first_seen=time.time() - 5,
            last_seen=time.time(),
            events=[event],
            severity="medium",
        )

        with swap.dict(os.environ, {"AURA_SELFMOD_LLM_DIAGNOSIS": "0"}):
            diagnosis = await engine.diagnose_pattern(pattern)

        brain.think.assert_not_awaited()
        self.assertTrue(diagnosis["ok"])
        self.assertEqual(diagnosis["diagnosis_source"], "deterministic_static")
        self.assertIn("coroutine", diagnosis["hypotheses"][0]["root_cause"])

    async def test_self_modification_skips_unlocated_error_patterns_as_unfixable(self):
        from core.self_modification.error_intelligence import (
            ErrorEvent,
            ErrorPattern,
            ErrorPatternAnalyzer,
        )

        analyzer = ErrorPatternAnalyzer(SimpleNamespace())
        event = ErrorEvent(
            timestamp=time.time(),
            error_type="RuntimeError",
            error_message="opaque runtime failure",
            stack_trace="trace",
            context={},
            file_path=None,
            line_number=None,
        )
        pattern = ErrorPattern(
            fingerprint="unlocated",
            occurrences=2,
            first_seen=time.time() - 5,
            last_seen=time.time(),
            events=[event],
            severity="medium",
        )

        self.assertFalse(analyzer.should_trigger_fix(pattern))

    async def test_kernel_refiner_skips_llm_deep_audit_by_default(self):
        from core.self_modification.kernel_refiner import KernelRefiner

        brain = SimpleNamespace(think=_AsyncCallRecorder(return_value=SimpleNamespace(content='{"found": true}')))
        refiner = KernelRefiner(brain, code_base_path=".")

        with swap.dict(os.environ, {"AURA_KERNEL_REFINER_LLM_AUDIT": "0"}):
            result = await refiner._perform_deep_brain_audit("def evaluate(self):\n    return None\n")

        self.assertEqual(result, [])
        brain.think.assert_not_awaited()


class TestLifecycleDeduplication(unittest.IsolatedAsyncioTestCase):
    async def test_reliability_engine_start_is_idempotent_while_tasks_are_alive(self):
        with swap.object(ServiceContainer, "_registration_locked", False):
            from core.reliability_engine import ReliabilityEngine

        engine = ReliabilityEngine()
        engine._started = True
        engine._tasks = [SimpleNamespace(done=lambda: False)]

        with swap(
            "core.reliability_engine.get_task_tracker",
            side_effect=AssertionError("duplicate tasks should not be created"),
        ):
            await engine.start()

        self.assertEqual(len(engine._tasks), 1)

    def test_session_guardian_start_reuses_existing_monitor_task(self):
        from core.session.session_guardian import SessionGuardian

        guardian = SessionGuardian()
        existing_task = SimpleNamespace(done=lambda: False)
        guardian._running = True
        guardian._monitor_task = existing_task

        with swap(
            "core.session.session_guardian.get_task_tracker",
            side_effect=AssertionError("duplicate monitor task should not be created"),
        ):
            result = guardian.start()

        self.assertIs(result, guardian)
        self.assertIs(guardian._monitor_task, existing_task)

    async def test_fictional_background_loops_noop_when_already_running(self):
        from core.fictional_ai_synthesis import (
            DistributedResilienceCore,
            ProactiveAnticipationEngine,
            TemporalDilationScheduler,
        )

        jarvis = ProactiveAnticipationEngine()
        jarvis._running = True
        await jarvis.start(interval_seconds=0.01)

        skynet = DistributedResilienceCore()
        skynet._running = True
        await skynet.start_monitoring()

        mist = TemporalDilationScheduler()
        mist._is_running = True
        await mist.run_idle_loop()


class TestLiveRuntimeFailureIsolation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        # Boot grace keys on the current process incarnation; these isolation
        # tests model a mature live runtime, not a fresh boot.
        import core.runtime.background_policy as background_policy

        original_started_at = background_policy._PROCESS_STARTED_AT
        background_policy._PROCESS_STARTED_AT = time.time() - 1000.0
        self.addCleanup(
            setattr, background_policy, "_PROCESS_STARTED_AT", original_started_at
        )

    async def test_event_bus_same_loop_delivery_avoids_threadsafe_self_wakeup(self):
        from core.event_bus import AuraEventBus

        bus = AuraEventBus()
        queue = await bus.subscribe("same-loop")
        loop = asyncio.get_running_loop()

        with swap.object(
            loop,
            "call_soon_threadsafe",
            side_effect=AssertionError("same-loop publish should not wake the selector pipe"),
        ), swap.object(loop, "call_soon", wraps=loop.call_soon) as call_soon:
            await bus.publish("same-loop", {"ok": True})
            await asyncio.sleep(0)

        _priority, _sequence, payload = await asyncio.wait_for(queue.get(), timeout=1.0)
        self.assertEqual(payload["topic"], "same-loop")
        self.assertTrue(payload["data"]["ok"])
        self.assertGreaterEqual(call_soon.call_count, 1)

    async def test_event_bus_publish_captures_redis_client_across_shutdown_race(self):
        from core.event_bus import AuraEventBus

        class _Redis:
            def __init__(self):
                self.published = []

            async def publish(self, topic, payload):
                self.published.append((topic, payload))

        redis_client = _Redis()
        bus = AuraEventBus()
        bus._use_redis = True
        bus._redis = redis_client

        async def _racy_to_thread(func, *args, **kwargs):
            bus._redis = None
            return func(*args, **kwargs)

        with swap("core.event_bus.asyncio.to_thread", side_effect=_racy_to_thread):
            await bus.publish("shutdown-race", {"ok": True})

        self.assertEqual(len(redis_client.published), 1)
        self.assertEqual(redis_client.published[0][0], "aura/events/shutdown-race")

    async def test_event_bus_loop_mismatch_drops_closed_loop_redis_without_false_error(self):
        from core.event_bus import AuraEventBus

        class _ClosedLoop:
            def is_running(self):
                return False

        class _Redis:
            closed = False

            async def aclose(self):
                self.closed = True

        bus = AuraEventBus()
        bus._use_redis = True
        bus._loop = asyncio.get_running_loop()
        bus._redis_loop = _ClosedLoop()
        redis_client = _Redis()
        bus._redis = redis_client
        bus._record_remote_error = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("closed-loop Redis disposal is not a remote transport failure")
        )

        bus._check_loop_mismatch()

        self.assertIsNone(bus._redis)
        self.assertIsNone(bus._redis_loop)
        self.assertFalse(redis_client.closed)

    def _terminal_monitor_without_handler(self):
        from collections import deque

        from core.terminal_monitor import TerminalMonitor

        monitor = TerminalMonitor.__new__(TerminalMonitor)
        monitor._error_buffer = deque(maxlen=100)
        monitor._seen = {}
        monitor._fix_attempts = {}
        monitor._failures = {}
        monitor._fix_window = []
        monitor._sepsis_mode = False
        monitor._sepsis_start = 0.0
        monitor._circuit_breaker_open = False
        monitor._ignore_patterns = []
        monitor._actionable_patterns = {}
        monitor._blacklist = set()
        return monitor

    def test_background_degraded_noise_does_not_trip_sepsis(self):
        from core.terminal_monitor import ErrorEntry

        monitor = self._terminal_monitor_without_handler()
        for idx in range(25):
            monitor._ingest_error(
                ErrorEntry(
                    message=f"background warning {idx}",
                    level="WARNING",
                    source=f"degraded.background_{idx}",
                    metadata={"classification": "background_degraded", "severity": "warning"},
                )
            )

        self.assertFalse(monitor._sepsis_mode)

    def test_foreground_failures_can_still_trip_sepsis(self):
        from core.terminal_monitor import ErrorEntry

        monitor = self._terminal_monitor_without_handler()
        now = time.time()
        with swap("core.terminal_monitor.time.time", return_value=now):
            for idx in range(12):
                monitor._ingest_error(
                    ErrorEntry(
                        message=f"foreground failure {idx}",
                        level="WARNING",
                        source=f"degraded.foreground_{idx}",
                        metadata={"classification": "foreground_blocking", "severity": "warning"},
                        timestamp=now,
                    )
                )

        self.assertTrue(monitor._sepsis_mode)

    def test_runtime_status_hides_stale_user_prompt_as_current_intention(self):
        from core.tools.runtime_tools import _clean_current_intention_for_status

        prompt = "Aura, what is actually on your mind right now?"

        self.assertEqual(
            _clean_current_intention_for_status(prompt, prompt, "user"),
            "idle",
        )
        self.assertEqual(
            _clean_current_intention_for_status(
                "Checking autonomous action pathways for one blocked capability to rewire.",
                "Auditing one live-runtime bottleneck and proposing a concrete repair.",
                "motivation_engine",
            ),
            "Auditing one live-runtime bottleneck and proposing a concrete repair.",
        )

    def test_background_failure_pressure_stays_low_for_repeated_warnings(self):
        from core.health.degraded_events import (
            clear_degraded_events,
            get_unified_failure_state,
            record_degraded_event,
        )

        clear_degraded_events()
        try:
            for _ in range(30):
                record_degraded_event(
                    "service_container",
                    "SUBSYSTEM_ABSENT",
                    detail="optional_neurochemical_regulator",
                    severity="warning",
                    classification="background_degraded",
                )

            pressure = get_unified_failure_state()["pressure"]
            self.assertLess(pressure, 0.10)
        finally:
            clear_degraded_events()

    def test_optional_service_absence_stays_out_of_neural_error_stream(self):
        from core.container import ServiceContainer
        from core.health.degraded_events import (
            clear_degraded_events,
            get_recent_degraded_events,
            get_unified_failure_state,
        )

        forwarded = []
        clear_degraded_events()
        try:
            with swap("core.health.degraded_events._forward_to_terminal_monitor", side_effect=forwarded.append):
                ServiceContainer._emit_absent_event("voice_pipeline")

            events = get_recent_degraded_events(limit=5)
            self.assertEqual(events, [])
            self.assertEqual(forwarded, [])
            self.assertEqual(get_unified_failure_state()["pressure"], 0.0)
        finally:
            clear_degraded_events()

    def test_optional_service_absence_is_recorded_once_under_polling(self):
        from core.container import ServiceContainer
        from core.health.degraded_events import (
            clear_degraded_events,
            get_recent_degraded_events,
        )

        clear_degraded_events()
        ServiceContainer._optional_absent_breadcrumbs.discard("poll_only_service")
        try:
            for _ in range(100):
                self.assertIsNone(
                    ServiceContainer.get("poll_only_service", default=None)
                )

            events = [
                event
                for event in get_recent_degraded_events(limit=20)
                if event.get("detail") == "poll_only_service"
            ]
            self.assertEqual(events, [])
        finally:
            ServiceContainer._optional_absent_breadcrumbs.discard("poll_only_service")
            clear_degraded_events()

    async def test_private_phenomenology_uses_local_reflection_by_default(self):
        from core.agency.private_phenomenology import PrivatePhenomenology
        from core.runtime.foreground_guard import _reset_for_tests as reset_foreground_guard

        reset_foreground_guard()
        engine = _CallRecorder()
        engine.think = _AsyncCallRecorder(return_value=SimpleNamespace(content="quiet inner reflection"))

        def fake_get(name, default=None):
            if name == "cognitive_engine":
                return engine
            if name == "orchestrator":
                return SimpleNamespace(is_busy=False, _last_user_interaction_time=time.time() - 300)
            return default

        with tempfile.TemporaryDirectory() as temp_dir, swap(
            "core.agency.private_phenomenology.ServiceContainer.get",
            side_effect=fake_get,
        ), swap(
            "core.runtime.background_policy.background_activity_reason",
            return_value="",
        ), swap.dict(os.environ, {"AURA_PHENOMENOLOGY_USE_LLM": "0"}):
            phenomenology = PrivatePhenomenology(storage_path=str(Path(temp_dir) / "monologue.jsonl"))
            reflection = await phenomenology.reflect({"P": 0.1, "A": 0.2, "D": 0.3}, [{"event": "test"}])

        engine.think.assert_not_awaited()
        self.assertIn("recent pattern", reflection)

    async def test_private_phenomenology_llm_mode_marks_internal_reflection_as_background(self):
        from core.agency.private_phenomenology import PrivatePhenomenology
        from core.runtime.foreground_guard import _reset_for_tests as reset_foreground_guard

        reset_foreground_guard()
        engine = _CallRecorder()
        engine.think = _AsyncCallRecorder(return_value=SimpleNamespace(content="quiet inner reflection"))

        def fake_get(name, default=None):
            if name == "cognitive_engine":
                return engine
            if name == "orchestrator":
                return SimpleNamespace(is_busy=False, _last_user_interaction_time=time.time() - 300)
            return default

        with tempfile.TemporaryDirectory() as temp_dir, swap(
            "core.agency.private_phenomenology.ServiceContainer.get",
            side_effect=fake_get,
        ), swap(
            "core.runtime.background_policy.background_activity_reason",
            return_value="",
        ), swap.dict(os.environ, {"AURA_PHENOMENOLOGY_USE_LLM": "1"}):
            phenomenology = PrivatePhenomenology(storage_path=str(Path(temp_dir) / "monologue.jsonl"))
            await phenomenology.reflect({"P": 0.1, "A": 0.2, "D": 0.3}, [{"event": "test"}])

        kwargs = engine.think.await_args.kwargs
        self.assertEqual(kwargs["origin"], "phenomenological_reflection")
        self.assertTrue(kwargs["is_background"])

    def test_email_and_reddit_adapters_remain_routable_for_autonomy(self):
        from core.capability_engine import CapabilityEngine

        engine = CapabilityEngine(orchestrator=None)

        self.assertIn("email_adapter", engine.skills)
        self.assertIn("reddit_adapter", engine.skills)
        self.assertIn("email_adapter", engine.active_skills)
        self.assertIn("reddit_adapter", engine.active_skills)

    async def test_reddit_inbox_login_unavailable_is_incomplete(self):
        from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput

        skill = RedditAdapterSkill()
        skill._ensure_logged_in = _AsyncCallRecorder(return_value=False)

        result = await skill._handle_check_inbox(_CallRecorder(), RedditInput(mode="check_inbox"))

        self.assertFalse(result["ok"])
        self.assertFalse(result["completed"])
        self.assertEqual(result["status"], "login_unavailable")


class TestStateManagerSnapshotRecovery(unittest.TestCase):
    def _write_checked_snapshot(self, path: Path, data: dict):
        snapshot = {"meta": {"iso_time": "test", "reason": "shutdown"}, "data": data}
        data_bytes = json.dumps(snapshot).encode("utf-8")
        path.write_bytes((zlib.crc32(data_bytes) & 0xFFFFFFFF).to_bytes(4, "big") + data_bytes)

    def test_unreadable_legacy_snapshot_is_quarantined_without_boot_failure(self):
        from core.resilience.state_manager import StateManager

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            manager = StateManager.__new__(StateManager)
            manager.snapshot_dir = snapshot_dir
            latest = snapshot_dir / "latest_snapshot.json"
            latest.write_bytes(b"{\xff not utf8 json")

            self.assertIsNone(manager.load_last_snapshot())
            self.assertFalse(latest.exists())
            self.assertTrue(list((snapshot_dir / "autopsy").glob("corrupted_state_*_latest_snapshot.json")))

    def test_corrupt_latest_recovers_from_newest_valid_history_snapshot(self):
        from core.resilience.state_manager import StateManager

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            manager = StateManager.__new__(StateManager)
            manager.snapshot_dir = snapshot_dir
            latest = snapshot_dir / "latest_snapshot.json"
            history = snapshot_dir / "snapshot_200_shutdown.json"
            latest.write_bytes(b"{\xff not utf8 json")
            self._write_checked_snapshot(history, {"cycle_count": 42, "boredom": 0.2})

            recovered = manager.load_last_snapshot()

            self.assertEqual(recovered["cycle_count"], 42)
            self.assertTrue(latest.exists())
            self.assertEqual(latest.read_bytes(), history.read_bytes())
            self.assertTrue(list((snapshot_dir / "autopsy").glob("corrupted_state_*_latest_snapshot.json")))

    def test_history_recovery_skips_corrupt_newest_candidate(self):
        from core.resilience.state_manager import StateManager

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            manager = StateManager.__new__(StateManager)
            manager.snapshot_dir = snapshot_dir
            latest = snapshot_dir / "latest_snapshot.json"
            corrupt_newest = snapshot_dir / "snapshot_300_shutdown.json"
            valid_older = snapshot_dir / "snapshot_200_shutdown.json"

            latest.write_bytes(b"\x00\x00\x00\x00bad latest")
            self._write_checked_snapshot(valid_older, {"cycle_count": 7, "boredom": 0.1})
            corrupt_newest.write_bytes(b"\x00\x00\x00\x00bad history")
            os.utime(valid_older, (time.time() - 10, time.time() - 10))
            os.utime(corrupt_newest, None)

            recovered = manager.load_last_snapshot()

            self.assertEqual(recovered["cycle_count"], 7)
            self.assertFalse(corrupt_newest.exists())
            self.assertEqual(latest.read_bytes(), valid_older.read_bytes())
            self.assertTrue(list((snapshot_dir / "autopsy").glob("corrupted_state_*_snapshot_300_shutdown.json")))


class TestLiveRuntimeContentionGuards(unittest.IsolatedAsyncioTestCase):
    async def test_sovereign_pruner_uses_heuristic_consolidation_without_llm_by_default(self):
        from core.memory.sovereign_pruner import MemoryRecord, SovereignPruner

        class Brain:
            called = False

            async def think(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("background pruner should not call LLM by default")

        pruner = SovereignPruner()
        pruner._brain = Brain()
        mem = MemoryRecord(
            id="m1",
            content="Bryan and Aura fixed a live desktop crash. Aura learned to reserve foreground cognition.",
            timestamp=time.time(),
            source="chat",
            emotional_weight=0.7,
            identity_relevance=0.9,
        )

        result = await pruner._consolidate(mem)

        self.assertIn("foreground cognition", result)
        self.assertFalse(pruner._brain.called)

    async def test_dialectical_crucible_lightweight_default_does_not_spawn_llm_roles(self):
        from core.adaptation.dialectics import DialecticalCrucible
        from core.container import ServiceContainer

        class Engine:
            def __init__(self):
                self.called = False

            async def think(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("background crucible should not call LLM by default")

        class Beliefs:
            def __init__(self):
                self.claims = []

            async def process_new_claim(self, **kwargs):
                self.claims.append(kwargs)
                return True

        engine = Engine()
        beliefs = Beliefs()
        old_get = ServiceContainer.get

        def fake_get(name, default=None):
            if name == "cognitive_engine":
                return engine
            if name == "belief_revision_engine":
                return beliefs
            return default

        ServiceContainer.get = staticmethod(fake_get)
        try:
            crucible = DialecticalCrucible()
            # The claim under test is that the DEFAULT path is lightweight —
            # not that background work is currently admissible. The upstream
            # background-policy gate reads a process-global health score that
            # any earlier test exercising a degradation path drives down (seen
            # live: "failure_lockdown_0.55"), so the crucible deferred before
            # reaching the branch this test is about. Establish the premise.
            crucible._background_deferral_reason = lambda _concept: ""
            result = await crucible.run_crucible("Aura should keep foreground chat responsive.", context="test")
        finally:
            ServiceContainer.get = old_get

        self.assertTrue(result["ok"])
        self.assertEqual(result["reason"], "lightweight_crucible")
        self.assertFalse(engine.called)
        self.assertEqual(len(beliefs.claims), 1)

    async def test_optional_threaded_status_times_out_as_stale_not_api_blocker(self):
        from interface.routes.system import _optional_threaded_status

        def slow_status():
            time.sleep(0.25)
            return {"ok": True}

        result = await _optional_threaded_status("slow", slow_status, timeout_s=0.01)

        self.assertTrue(result["_stale"])
        self.assertEqual(result["reason"], "status_timeout")


class TestHealingSwarmForegroundPolicy(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_env = {
            key: os.environ.get(key)
            for key in ("AURA_ENABLE_BACKGROUND_COGNITION", "AURA_FOREGROUND_ONLY")
        }
        os.environ["AURA_ENABLE_BACKGROUND_COGNITION"] = "0"
        os.environ.pop("AURA_FOREGROUND_ONLY", None)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_healing_swarm_start_deferred_when_background_cognition_disabled(self):
        from core.resilience.healing_swarm import HealingSwarmService

        service = HealingSwarmService(SimpleNamespace())

        self.assertFalse(service.start())
        self.assertFalse(service.is_running)
        self.assertIsNone(service._monitor_task)

    async def test_healing_swarm_repair_deferred_when_background_cognition_disabled(self):
        from core.resilience.healing_swarm import HealingSwarmService

        class _Swarm:
            def __init__(self):
                self.called = False

            async def spawn_shard(self, *_args, **_kwargs):
                self.called = True
                return True

        swarm = _Swarm()
        service = HealingSwarmService(SimpleNamespace(sovereign_swarm=swarm))

        await service.attempt_repair("personality_engine", {"status": "STALE"})

        self.assertFalse(swarm.called)
        self.assertNotIn("personality_engine", service._repair_history)
