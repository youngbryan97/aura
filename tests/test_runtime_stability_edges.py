import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import os
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

    async def test_affect_appraisal_skips_llm_when_foreground_turn_is_active(self):
        with patch("core.affect.damasio_v2.PhysicalActuator", return_value=MagicMock()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        guarded_gate = MagicMock()
        guarded_gate._foreground_user_turn_active.return_value = True
        guarded_gate.get_conversation_status.return_value = {
            "conversation_ready": True,
            "state": "ready",
            "warmup_in_flight": False,
            "foreground_owned": False,
            "active_generations": 0,
            "request_age_s": 0.0,
        }
        guarded_gate.generate = AsyncMock(side_effect=AssertionError("LLM appraisal should have been deferred"))

        with patch("core.container.ServiceContainer.get", return_value=guarded_gate), \
             patch("core.brain.llm.mlx_client._foreground_owner_active", return_value=False):
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        self.assertEqual(engine._llm_failure_count, 0)

    async def test_affect_background_timeout_falls_back_without_runtime_degradation(self):
        with patch("core.affect.damasio_v2.PhysicalActuator", return_value=MagicMock()):
            from core.affect.damasio_v2 import AffectEngineV2

            engine = AffectEngineV2()

        engine._background_llm_should_defer = MagicMock(return_value=False)
        engine._appraise_with_llm = AsyncMock(side_effect=asyncio.TimeoutError())
        engine.iot_bridge.broadcast_affect_state = AsyncMock(return_value=None)

        with patch("core.brain.llm.mlx_client._foreground_owner_active", return_value=False), \
             patch("core.affect.damasio_v2.record_degradation") as record_degradation:
            result = await engine.react("I feel frustrated and need to reflect on recent interactions.")

        self.assertIsNotNone(result)
        self.assertEqual(engine._llm_failure_count, 0)
        record_degradation.assert_not_called()


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

    async def test_json_repair_handles_python_style_dict_payloads(self):
        from core.utils.json_utils import SelfHealingJSON

        repairer = SelfHealingJSON(brain=MagicMock())

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
        consolidator._background_should_defer = MagicMock(return_value=True)
        consolidator._gather_material = MagicMock(side_effect=AssertionError("foreground defer should skip work"))

        result = await consolidator.run_now()

        self.assertIsNone(result)
        self.assertEqual(consolidator._last_run, before)


class TestSubstrateStimulusGuards(unittest.IsolatedAsyncioTestCase):
    async def test_recurrent_self_model_runs_off_event_loop(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=4))

        with patch(
            "core.consciousness.liquid_substrate.asyncio.to_thread",
            new=AsyncMock(return_value=None),
        ) as to_thread:
            await substrate._recurrent_self_model(0.05)

        to_thread.assert_awaited_once_with(substrate._recurrent_self_model_sync, 0.05)

    async def test_plasticity_runs_off_event_loop(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=4))

        with patch(
            "core.consciousness.liquid_substrate.asyncio.to_thread",
            new=AsyncMock(return_value=None),
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

        with patch(
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
        with patch(
            "core.runtime.background_policy.background_activity_reason",
            return_value="foreground_quiet_window",
        ):
            result = await skill.execute(
                NetworkInput(mode="discovery", target="192.168.1.0/30", ports="8000"),
                {"origin": "system", "orchestrator": MagicMock()},
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

    async def test_self_modification_diagnosis_uses_static_path_by_default(self):
        from core.self_modification.error_intelligence import (
            AutomatedDiagnosisEngine,
            ErrorEvent,
            ErrorPattern,
        )

        brain = SimpleNamespace(think=AsyncMock(return_value=SimpleNamespace(content="{}")))
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

        with patch.dict(os.environ, {"AURA_SELFMOD_LLM_DIAGNOSIS": "0"}):
            diagnosis = await engine.diagnose_pattern(pattern)

        brain.think.assert_not_awaited()
        self.assertTrue(diagnosis["ok"])
        self.assertEqual(diagnosis["diagnosis_source"], "deterministic_static")
        self.assertIn("coroutine", diagnosis["hypotheses"][0]["root_cause"])

    async def test_self_modification_skips_unlocated_error_patterns_as_unfixable(self):
        from core.self_modification.error_intelligence import (
            ErrorPatternAnalyzer,
            ErrorEvent,
            ErrorPattern,
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

        brain = SimpleNamespace(think=AsyncMock(return_value=SimpleNamespace(content='{"found": true}')))
        refiner = KernelRefiner(brain, code_base_path=".")

        with patch.dict(os.environ, {"AURA_KERNEL_REFINER_LLM_AUDIT": "0"}):
            result = await refiner._perform_deep_brain_audit("def evaluate(self):\n    return None\n")

        self.assertEqual(result, [])
        brain.think.assert_not_awaited()


class TestLifecycleDeduplication(unittest.IsolatedAsyncioTestCase):
    async def test_reliability_engine_start_is_idempotent_while_tasks_are_alive(self):
        from core.reliability_engine import ReliabilityEngine

        engine = ReliabilityEngine()
        engine._started = True
        engine._tasks = [SimpleNamespace(done=lambda: False)]

        with patch(
            "core.reliability_engine.get_task_tracker",
            side_effect=AssertionError("duplicate tasks should not be created"),
        ):
            await engine.start()

        self.assertEqual(len(engine._tasks), 1)

    def test_session_guardian_start_reuses_existing_monitor_task(self):
        from core.session_guardian import SessionGuardian

        guardian = SessionGuardian()
        existing_task = SimpleNamespace(done=lambda: False)
        guardian._running = True
        guardian._monitor_task = existing_task

        with patch(
            "core.session_guardian.get_task_tracker",
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
    async def test_event_bus_same_loop_delivery_avoids_threadsafe_self_wakeup(self):
        from core.event_bus import AuraEventBus

        bus = AuraEventBus()
        queue = await bus.subscribe("same-loop")
        loop = asyncio.get_running_loop()

        with patch.object(
            loop,
            "call_soon_threadsafe",
            side_effect=AssertionError("same-loop publish should not wake the selector pipe"),
        ), patch.object(loop, "call_soon", wraps=loop.call_soon) as call_soon:
            await bus.publish("same-loop", {"ok": True})
            await asyncio.sleep(0)

        _priority, _sequence, payload = await asyncio.wait_for(queue.get(), timeout=1.0)
        self.assertEqual(payload["topic"], "same-loop")
        self.assertTrue(payload["data"]["ok"])
        self.assertGreaterEqual(call_soon.call_count, 1)

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
        with patch("core.terminal_monitor.time.time", return_value=now):
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
        from core.runtime_tools import _clean_current_intention_for_status

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
        from core.health.degraded_events import clear_degraded_events, get_unified_failure_state, record_degraded_event

        clear_degraded_events()
        try:
            for idx in range(30):
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
        from core.health.degraded_events import clear_degraded_events, get_recent_degraded_events, get_unified_failure_state

        forwarded = []
        clear_degraded_events()
        try:
            with patch("core.health.degraded_events._forward_to_terminal_monitor", side_effect=forwarded.append):
                ServiceContainer._emit_absent_event("voice_pipeline")

            events = get_recent_degraded_events(limit=5)
            self.assertEqual(events[0]["severity"], "info")
            self.assertEqual(events[0]["classification"], "non_critical_fallback")
            self.assertEqual(forwarded, [])
            self.assertEqual(get_unified_failure_state()["pressure"], 0.0)
        finally:
            clear_degraded_events()

    async def test_private_phenomenology_uses_local_reflection_by_default(self):
        from core.agency.private_phenomenology import PrivatePhenomenology

        engine = MagicMock()
        engine.think = AsyncMock(return_value=SimpleNamespace(content="quiet inner reflection"))

        def fake_get(name, default=None):
            if name == "cognitive_engine":
                return engine
            if name == "orchestrator":
                return SimpleNamespace(is_busy=False, _last_user_interaction_time=time.time() - 300)
            return default

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "core.agency.private_phenomenology.ServiceContainer.get",
            side_effect=fake_get,
        ), patch.dict(os.environ, {"AURA_PHENOMENOLOGY_USE_LLM": "0"}):
            phenomenology = PrivatePhenomenology(storage_path=str(Path(temp_dir) / "monologue.jsonl"))
            reflection = await phenomenology.reflect({"P": 0.1, "A": 0.2, "D": 0.3}, [{"event": "test"}])

        engine.think.assert_not_awaited()
        self.assertIn("recent pattern", reflection)

    async def test_private_phenomenology_llm_mode_marks_internal_reflection_as_background(self):
        from core.agency.private_phenomenology import PrivatePhenomenology

        engine = MagicMock()
        engine.think = AsyncMock(return_value=SimpleNamespace(content="quiet inner reflection"))

        def fake_get(name, default=None):
            if name == "cognitive_engine":
                return engine
            if name == "orchestrator":
                return SimpleNamespace(is_busy=False, _last_user_interaction_time=time.time() - 300)
            return default

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "core.agency.private_phenomenology.ServiceContainer.get",
            side_effect=fake_get,
        ), patch.dict(os.environ, {"AURA_PHENOMENOLOGY_USE_LLM": "1"}):
            phenomenology = PrivatePhenomenology(storage_path=str(Path(temp_dir) / "monologue.jsonl"))
            await phenomenology.reflect({"P": 0.1, "A": 0.2, "D": 0.3}, [{"event": "test"}])

        kwargs = engine.think.await_args.kwargs
        self.assertEqual(kwargs["origin"], "phenomenological_reflection")
        self.assertTrue(kwargs["is_background"])

    async def test_gemini_auth_failure_disables_adapter_without_runtime_degradation(self):
        import httpx
        from core.brain.llm.gemini_adapter import GeminiAdapter, GeminiProviderUnavailable

        adapter = GeminiAdapter(api_key="test", model="gemini-2.0-flash")
        response = httpx.Response(
            403,
            content=b'{"error":{"status":"PERMISSION_DENIED","message":"API key was reported as leaked"}}',
        )

        with self.assertRaises(GeminiProviderUnavailable):
            await adapter._handle_error(response)

        self.assertFalse(adapter.is_available())
        self.assertIn("provider_auth_failed", adapter.availability_reason())

    def test_email_and_reddit_adapters_remain_routable_for_autonomy(self):
        from core.capability_engine import CapabilityEngine

        engine = CapabilityEngine(orchestrator=None)

        self.assertIn("email_adapter", engine.skills)
        self.assertIn("reddit_adapter", engine.skills)
        self.assertIn("email_adapter", engine.active_skills)
        self.assertIn("reddit_adapter", engine.active_skills)

    async def test_reddit_inbox_login_unavailable_is_quiet_success(self):
        from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput

        skill = RedditAdapterSkill()
        skill._ensure_logged_in = AsyncMock(return_value=False)

        result = await skill._handle_check_inbox(MagicMock(), RedditInput(mode="check_inbox"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "login_unavailable")
