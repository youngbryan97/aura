import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from core.brain.llm.mlx_client import MLXLocalClient
from core.senses.sensory_client import SensoryLocalClient


class TestMLXCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_warm_up_alias_delegates_to_warmup(self):
        client = MLXLocalClient(model_path="/tmp/test-model")
        client.warmup = AsyncMock(return_value="ok")

        result = await client.warm_up()

        self.assertEqual(result, "ok")
        client.warmup.assert_awaited_once()


class TestSensoryClientRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_start_uses_spawn_on_darwin_and_pings_worker(self):
        client = SensoryLocalClient()

        process = MagicMock()
        process.pid = 4321
        process.is_alive.return_value = True
        ctx = MagicMock()
        ctx.Process.return_value = process

        with patch("core.senses.sensory_client.sys.platform", "darwin"), \
             patch("core.senses.sensory_client.mp.get_context", return_value=ctx) as get_context, \
             patch.object(client, "_send_command", new=AsyncMock(side_effect=[True, True, True])) as send_command:
            started = await client.start()

        self.assertTrue(started)
        get_context.assert_called_once_with("spawn")
        send_command.assert_any_await("ping", timeout=2.0, auto_restart=False)
        self.assertEqual(send_command.await_count, 3)

    async def test_send_command_restarts_dead_worker(self):
        client = SensoryLocalClient()
        client.start = AsyncMock(return_value=True)
        task_registry = MagicMock()
        task_registry.register_task.return_value = "task-1"

        with patch("core.supervisor.registry.get_task_registry", return_value=task_registry), \
             patch("core.senses.sensory_client.asyncio.to_thread", new=AsyncMock(return_value={"status": "ok"})):
            ok = await client._send_command("ping")

        self.assertTrue(ok)
        client.start.assert_awaited_once()


class TestAffectBroadcastBackpressure(unittest.IsolatedAsyncioTestCase):
    async def test_affect_broadcast_caps_background_tasks(self):
        emit_release = asyncio.Event()

        class _Bus:
            async def emit(self, *_args, **_kwargs):
                await emit_release.wait()

        with patch("core.affect.damasio_v2.PhysicalActuator", return_value=MagicMock()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        engine._max_background_tasks = 2

        with patch("core.container.ServiceContainer.get", return_value=_Bus()):
            await engine._broadcast_event("affect_pulse")
            await engine._broadcast_event("affect_pulse")
            await engine._broadcast_event("affect_pulse")

        self.assertEqual(len(engine._background_tasks), 2)

        emit_release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_affect_appraisal_skips_llm_when_foreground_lane_is_protected(self):
        with patch("core.affect.damasio_v2.PhysicalActuator", return_value=MagicMock()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        guarded_gate = MagicMock()
        guarded_gate._should_quiet_background_for_cortex_startup.return_value = True
        guarded_gate.get_conversation_status.return_value = {
            "conversation_ready": False,
            "state": "warming",
            "warmup_in_flight": True,
        }

        with patch("core.container.ServiceContainer.get", return_value=guarded_gate):
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        self.assertEqual(engine._llm_failure_count, 0)


class TestEternalMemoryCaching(unittest.IsolatedAsyncioTestCase):
    async def test_eternal_memory_reuses_recent_summary_cache(self):
        from core.kernel.upgrades_10x import EternalMemoryPhase

        phase = EternalMemoryPhase(MagicMock())
        phase._summary_cache = [{"role": "system", "content": "[ETERNAL MEMORY]\nsteady"}]
        import time
        phase._last_summary_refresh_at = time.time()
        phase._summary_refresh_interval_s = 120.0
        phase._load_eternal_slice = MagicMock(side_effect=AssertionError("should not load recent history"))

        with patch.object(phase, "_background_llm_should_defer", return_value=False):
            summary = await phase._get_cached_or_refresh_summary()

        self.assertEqual(summary, phase._summary_cache)


class TestNativeMultimodalBridgeGuards(unittest.IsolatedAsyncioTestCase):
    async def test_native_multimodal_bridge_skips_disabled_vision_without_name_error(self):
        from core.kernel.upgrades_10x import NativeMultimodalBridge
        from core.state.aura_state import AuraState

        kernel = MagicMock()
        kernel.organs = {}
        phase = NativeMultimodalBridge(kernel)
        state = AuraState()
        state.cognition.current_objective = "Please inspect the screen."

        with patch.dict("os.environ", {"AURA_ENABLE_NATIVE_VISION_ACTIONS": "0"}, clear=False):
            new_state = await phase.execute(state, objective=state.cognition.current_objective)

        self.assertIs(new_state, state)


class TestJsonRepairGuards(unittest.IsolatedAsyncioTestCase):
    async def test_json_repair_handles_none_without_crashing(self):
        from core.utils.json_utils import SelfHealingJSON

        repairer = SelfHealingJSON(brain=MagicMock())

        parsed = await repairer.parse(None)

        self.assertEqual(parsed, {})


class TestSovereignPrunerGuards(unittest.IsolatedAsyncioTestCase):
    async def test_pruner_caps_consolidation_batch_and_preserves_deferred_memories(self):
        from core.memory.sovereign_pruner import MemoryRecord, SovereignPruner

        pruner = SovereignPruner(target_retention=0.0)
        pruner._min_prune_interval_s = 0.0
        pruner._max_consolidations_per_pass = 2
        pruner._background_should_defer = MagicMock(return_value=False)
        pruner._consolidate = AsyncMock(side_effect=["insight-a", "insight-b"])

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
        pruner._background_should_defer = MagicMock(return_value=False)
        pruner._consolidate = AsyncMock()

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

        guard = MagicMock()
        guard.check_permission = AsyncMock(
            return_value={"granted": False, "status": "deferred", "guidance": "nope"}
        )

        with patch("core.container.ServiceContainer.get", return_value=guard), patch(
            "core.senses.screen_vision._screen_capture_preflight",
            return_value=False,
        ):
            image = await LocalVision().capture_screen()

        self.assertIsNone(image)


class TestNeuralBridgeBootSafety(unittest.IsolatedAsyncioTestCase):
    async def test_neural_bridge_loads_lightweight_without_heavy_boot_dependencies(self):
        from core.senses.neural_bridge import NeuralBridge

        bridge = NeuralBridge(lightweight_mode=True)

        with patch.object(bridge, "start") as start:
            await bridge.load()

        self.assertTrue(bridge.is_trained)
        self.assertTrue(bridge.get_status()["lightweight_mode"])
        start.assert_called_once()


class TestNeuralOrganBootSafety(unittest.IsolatedAsyncioTestCase):
    async def test_neural_organ_uses_lightweight_mode_during_safe_desktop_boot(self):
        from core.kernel.organs import OrganStub

        bridge = MagicMock()
        bridge.load = AsyncMock()

        with patch.dict("os.environ", {"AURA_SAFE_BOOT_DESKTOP": "1"}, clear=False):
            with patch("core.senses.neural_bridge.NeuralBridge", return_value=bridge) as bridge_cls:
                organ = OrganStub("neural", MagicMock())
                await organ.load()

        bridge_cls.assert_called_once_with(lightweight_mode=True)
        bridge.load.assert_awaited_once()
        self.assertIs(organ.instance, bridge)


class TestBackgroundPolicyGuards(unittest.TestCase):
    def test_background_policy_requires_user_anchor_before_unsolicited_work(self):
        from core.runtime.background_policy import background_activity_reason
        from core.health.degraded_events import clear_degraded_events
        clear_degraded_events()

        orch = MagicMock()
        orch.is_busy = False
        orch._suppress_unsolicited_proactivity_until = 0.0
        orch._foreground_user_quiet_until = 0.0
        orch._last_user_interaction_time = 0.0
        orch.status = MagicMock(last_user_interaction_time=0.0)

        reason = background_activity_reason(orch, require_conversation_ready=False)

        self.assertEqual(reason, "no_user_anchor")


class TestCognitiveContextSanitization(unittest.TestCase):
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
            with patch.object(
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
        status = MagicMock(
            cycle_count=5,
            last_user_interaction_time=now,
            state="ready",
            acceleration_factor=1.0,
            singularity_threshold=False,
            is_processing=False,
            volition_level=0,
        )
        hooks = MagicMock(trigger=AsyncMock())
        world = MagicMock(recent_percepts=[])
        message_queue = MagicMock()
        message_queue._queue = []
        message_queue.full.return_value = False
        message_queue.qsize.return_value = 0
        message_queue.empty.return_value = True

        orch = MagicMock(
            status=status,
            hooks=hooks,
            drive_controller=None,
            drives=None,
            is_busy=False,
            latent_core=None,
            predictive_model=None,
            kernel=MagicMock(organs={}),
            message_queue=message_queue,
            world=world,
            conversation_history=[],
            memory_manager=None,
            liquid_state=MagicMock(current=MagicMock(curiosity=0.5, frustration=0.0, energy=0.5)),
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
        orch._acquire_next_message = AsyncMock(return_value=None)
        orch._dispatch_message = MagicMock()
        orch._publish_telemetry = MagicMock()
        orch._emit_thought_stream = MagicMock()

        coord = MetabolicCoordinator(orch=orch)
        coord._event_bus = object()
        coord._consume_energy = MagicMock(return_value=False)
        coord.update_liquid_pacing = MagicMock()
        coord.manage_memory_hygiene = MagicMock()
        coord.process_world_decay = AsyncMock()
        coord.trigger_autonomous_thought = AsyncMock()
        coord.run_terminal_self_heal = AsyncMock()
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
            async def _cancelled():
                raise asyncio.CancelledError()

            _schedule_background_coro(_cancelled(), label="test_cancelled_background")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertEqual(captured, [])

    async def test_error_logger_ignores_cancelled_append(self):
        from core.self_modification.error_intelligence import StructuredErrorLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            logger_system = StructuredErrorLogger(log_dir=temp_dir)
            with patch(
                "core.self_modification.error_intelligence.asyncio.to_thread",
                new=AsyncMock(side_effect=asyncio.CancelledError),
            ):
                await logger_system._append_to_log(Path(temp_dir) / "error_events.jsonl", {"ok": True})
