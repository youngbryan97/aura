import json
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest


BANNED_LIVE_FALLBACKS = re.compile(
    r"(say that again|try (?:again|me again|that again)|ask me again|"
    r"give me a moment|i'?m with you|could you repeat|repeat your question|"
    r"send your message again|lost my (?:thread|train of thought)|"
    r"hit a bump|one moment|having trouble formulating|could you try rephrasing)",
    re.IGNORECASE,
)


def assert_no_live_reset_boilerplate(text: str) -> None:
    assert not BANNED_LIVE_FALLBACKS.search(str(text or ""))


def test_consciousness_bridge_adapts_64_projection_to_512_substrate():
    from core.consciousness.consciousness_bridge import ConsciousnessBridge

    projection = np.arange(64, dtype=np.float32)
    fitted = ConsciousnessBridge._fit_vector(projection, 512, mode="tile", dtype=np.float64)

    assert fitted.shape == (512,)
    assert fitted.dtype == np.float64
    np.testing.assert_array_equal(fitted[:64], projection.astype(np.float64))
    np.testing.assert_array_equal(fitted[64:128], projection.astype(np.float64))


def test_closed_loop_predictor_resizes_to_512_substrate():
    from core.consciousness.closed_loop import ClosedCausalLoop

    loop = ClosedCausalLoop()
    loop._ensure_vector_dimensions(512)

    current = np.zeros(512, dtype=np.float32)
    predicted = loop._predictor.predict(current)
    cycle = loop._predictor.observe_and_update(np.ones(512, dtype=np.float32) * 0.05)

    assert predicted.shape == (512,)
    assert cycle is not None
    assert cycle.actual_state.shape == (512,)


def test_omni_tracer_downgrades_optional_dependency_logs():
    from core.resilience.omni_tracer import _classify_forwarded_log

    severity, classification = _classify_forwarded_log(
        "Aura.VoiceEngine",
        "pyttsx3 not installed — TTS unavailable",
        "critical",
    )

    assert severity == "warning"
    assert classification == "background_degraded"


def test_background_policy_defers_work_during_boot_grace(monkeypatch):
    from core.runtime.background_policy import background_activity_reason

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
    orch = SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 42))

    assert background_activity_reason(orch) == "boot_grace_42s"


def test_sensory_motor_idle_volition_respects_background_boot_grace(monkeypatch):
    from core.sensory_motor_cortex import SensoryMotorCortex

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
    orch = SimpleNamespace(
        status=SimpleNamespace(start_time=time.time() - 42, is_processing=False),
        _last_user_interaction_time=0.0,
    )
    cortex = SensoryMotorCortex(orchestrator=orch, config={"boredom_threshold": 1})
    cortex.last_interaction_time = time.time() - 999

    assert cortex._should_trigger_volition(now=time.time()) is False


def test_stability_guardian_uses_warmup_lag_budget(monkeypatch):
    from core.resilience.stability_guardian import StabilityGuardian

    monkeypatch.setenv("AURA_EVENT_LOOP_LAG_BOOT_GRACE_S", "300")
    orch = SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 60))
    guardian = StabilityGuardian(orch)

    assert guardian._event_loop_lag_threshold_ms() >= 6000.0


def test_viability_boot_grace_does_not_starve_from_no_interaction(monkeypatch):
    from core.organism.viability import ViabilityEngine, ViabilitySample, ViabilityState

    monkeypatch.setenv("AURA_VIABILITY_BOOT_GRACE_S", "300")
    sample = ViabilitySample(
        cpu_pct=0.0,
        ram_pct=50.0,
        disk_pct=10.0,
        error_rate_per_min=0.0,
        failed_tool_loops=0,
        unresolved_goals=0,
        successful_goals_last_hour=0,
        user_interactions_last_hour=0,
        incoherent_beliefs=0,
        broken_subsystems=1,
        runtime_uptime_s=42.0,
    )

    assert ViabilityEngine._classify(sample) == ViabilityState.HEALTHY


def test_background_enqueue_defers_stale_autonomy_during_boot(monkeypatch):
    from core.orchestrator.mixins.message_handling import MessageHandlingMixin

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")

    class FakeOrchestrator(MessageHandlingMixin):
        status = SimpleNamespace(start_time=time.time() - 60)
        _flow_controller = None

        def _is_user_facing_origin(self, origin):
            return False

    fake = FakeOrchestrator()

    assert fake.enqueue_message(
        {"content": "stale autonomous objective"},
        priority=20,
        origin="agency_core_environmental_explorer",
    ) is False


def test_continuity_generic_reentry_goal_is_not_restored_as_work(monkeypatch):
    from core.continuity import ContinuityEngine, ContinuityRecord

    monkeypatch.delenv("AURA_ENABLE_CONTINUITY_REENTRY_INITIATIVE", raising=False)
    engine = ContinuityEngine()
    engine._record = ContinuityRecord(
        last_shutdown=time.time() - 60,
        last_shutdown_reason="keyboard_interrupt",
        total_uptime_seconds=60.0,
        session_count=7,
        last_conversation_summary="",
        identity_hash="",
        current_objective="Reconcile continuity gap and re-establish the interrupted thread",
        pending_initiative_details=["Reconcile continuity gap and re-establish the interrupted thread"],
        active_goal_details=["Reconcile continuity gap and re-establish the interrupted thread"],
        active_commitments=["Reconcile continuity gap and re-establish the interrupted thread"],
    )
    engine._gap_seconds = 60.0

    cognition = SimpleNamespace(
        current_objective="",
        rolling_summary="",
        contradiction_count=0,
        pending_initiatives=[],
        active_goals=[],
        modifiers={},
        trim_working_memory=lambda: None,
    )
    state = SimpleNamespace(cognition=cognition)

    engine.apply_to_state(state)

    assert cognition.current_objective == ""
    assert cognition.pending_initiatives == []
    assert cognition.active_goals == []
    obligations = cognition.modifiers["continuity_obligations"]
    assert obligations["current_objective"] == ""
    assert obligations["pending_initiatives"] == []
    assert obligations["active_goals"] == []


@pytest.mark.asyncio
async def test_initiative_arbiter_quarantines_generic_continuity_reentry_goal():
    from core.agency.initiative_arbiter import InitiativeArbiter

    cognition = SimpleNamespace(
        pending_initiatives=[
            {"goal": "Reconcile continuity gap and re-establish the interrupted thread"},
        ],
        working_memory=[],
    )
    state = SimpleNamespace(cognition=cognition, identity=SimpleNamespace(core_values=[]))

    assert await InitiativeArbiter().arbitrate(state) is None
    assert cognition.pending_initiatives == []


@pytest.mark.asyncio
async def test_executive_authority_does_not_fallback_promote_quarantined_reentry_goal():
    from core.consciousness.executive_authority import ExecutiveAuthority

    cognition = SimpleNamespace(
        current_objective="",
        pending_initiatives=[
            {"goal": "Reconcile continuity gap and re-establish the interrupted thread"},
        ],
        working_memory=[],
    )
    state = SimpleNamespace(cognition=cognition, identity=SimpleNamespace(core_values=[]))

    new_state, initiative, decision = await ExecutiveAuthority().promote_next_initiative(state)

    assert new_state is state
    assert initiative is None
    assert decision["reason"] == "no_selectable_initiatives"
    assert cognition.pending_initiatives == []


def test_proactive_presence_respects_boot_grace(monkeypatch):
    from core.proactive_presence import ProactivePresence

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
    orch = SimpleNamespace(
        status=SimpleNamespace(start_time=time.time() - 42, is_processing=False),
        _last_user_interaction_time=0.0,
        _last_thought_time=0.0,
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
    )
    presence = ProactivePresence(orchestrator=orch)

    assert presence._should_speak_now() is False


@pytest.mark.asyncio
async def test_motivation_engine_does_not_emit_boot_grace_intention(monkeypatch):
    from core.motivation.engine import MotivationEngine

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
    orch = SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 42))
    engine = MotivationEngine(orchestrator=orch)
    engine.affect = SimpleNamespace(get_resonance_string=lambda: "Aura (Core) 100%")
    for budget in engine.budgets.values():
        budget.level = 0.0

    assert await engine._assess_needs() is None


@pytest.mark.asyncio
async def test_metabolic_terminal_self_heal_defers_during_boot_grace(monkeypatch):
    from core.coordinators.metabolic_coordinator import MetabolicCoordinator

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
    called = False

    def fail_if_called():
        nonlocal called
        called = True
        raise AssertionError("terminal monitor should not be polled during boot grace")

    monkeypatch.setattr("core.terminal_monitor.get_terminal_monitor", fail_if_called)
    orch = SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 42))

    await MetabolicCoordinator(orch=orch).run_terminal_self_heal()

    assert called is False


def test_stability_guardian_suppresses_tick_rate_during_shutdown():
    from core.resilience.stability_guardian import StabilityGuardian
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

    request_shutdown("unit_test")
    try:
        guardian = StabilityGuardian(SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 500)))
        result = guardian._check_tick_rate()
    finally:
        clear_shutdown_request()

    assert result.healthy is True
    assert "shutdown" in result.message.lower()


def test_stability_guardian_treats_boot_loop_lag_as_grace(monkeypatch):
    from core.resilience.stability_guardian import StabilityGuardian

    monkeypatch.delenv("AURA_MAX_EVENT_LOOP_LAG_BOOT_MS", raising=False)
    guardian = StabilityGuardian(SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 20)))
    guardian._loop_lag_samples.append((time.time(), 8705.0))

    result = guardian._check_tick_rate()

    assert result.healthy is True
    assert "lag" not in result.message.lower() or "not sustained" in result.message.lower()


def test_stall_watchdog_suppresses_boot_grace_stalls(monkeypatch):
    from core.resilience.stall_watchdog import StallWatchdog

    monkeypatch.setenv("AURA_WATCHDOG_BOOT_GRACE_S", "120")
    watchdog = StallWatchdog(SimpleNamespace(is_closed=lambda: False), threshold=1.0)
    watchdog._started_at = time.time()

    assert watchdog._should_suppress_stall(15.0) is True


def test_stall_watchdog_suppresses_foreground_inference_grace_stalls(monkeypatch):
    from core.container import ServiceContainer
    from core.resilience.stall_watchdog import StallWatchdog

    class FakeGate:
        def get_conversation_status(self):
            return {
                "state": "handshaking",
                "foreground_owned": True,
                "foreground_owner": "chat",
                "active_generations": 1,
                "warmup_in_flight": False,
            }

    monkeypatch.setenv("AURA_WATCHDOG_BOOT_GRACE_S", "0")
    monkeypatch.setenv("AURA_WATCHDOG_FOREGROUND_GRACE_S", "75")
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: FakeGate() if name == "inference_gate" else default))

    watchdog = StallWatchdog(SimpleNamespace(is_closed=lambda: False), threshold=1.0)
    watchdog._started_at = time.time() - 500

    assert watchdog._should_suppress_stall(15.0) is True


def test_terminal_monitor_ignores_background_phase_timeouts():
    from core.terminal_monitor import ErrorEntry, TerminalMonitor

    monitor = TerminalMonitor.__new__(TerminalMonitor)
    entry = ErrorEntry(
        message="⏰ Phase 'EternalMemoryPhase' timed out after 10s — skipping",
        level="ERROR",
        source="Aura.Core.Kernel",
    )

    assert monitor._classify_error(entry) is None


def test_chat_live_proof_classifier_catches_plain_snake_and_glass_prompts():
    from interface.routes.chat import _build_glass_arithmetic_reply, _classify_live_runtime_proof

    assert _classify_live_runtime_proof(
        "Create a simple game of Snake and save it as artifacts/live_runtime/generated/live_snake.html"
    ) == "snake"
    assert _classify_live_runtime_proof(
        "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
    ) == "novel_topic"
    assert "14'" in _build_glass_arithmetic_reply(
        "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
    )


@pytest.mark.asyncio
async def test_api_chat_live_proof_receipt_survives_quality_repair(monkeypatch, tmp_path):
    from interface.routes import chat as chat_module

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "file_operation"
            target = tmp_path / params["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(params["content"])
            return {"ok": True, "path": params["path"], "context": context}

    async def fail_if_repaired(*_args, **_kwargs):
        raise AssertionError("verified live proof replies must not be replaced by quality repair")

    async def no_op_async(*_args, **_kwargs):
        return None

    monkeypatch.chdir(tmp_path)
    chat_module._locks.pop("fg", None)
    monkeypatch.setattr(chat_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_module, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_module, "_collect_conversation_lane_status", lambda: {"state": "ready"})
    monkeypatch.setattr(chat_module, "_emit_chat_output_receipt", no_op_async)
    monkeypatch.setattr(chat_module, "_log_exchange", no_op_async)
    monkeypatch.setattr(chat_module, "_repair_final_degraded_reply", fail_if_repaired)
    monkeypatch.setattr(
        chat_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default),
    )

    response = await chat_module.api_chat(
        chat_module.ChatRequest(
            message="Create a simple game of Snake and save it as artifacts/live_runtime/generated/live_snake.html"
        ),
        SimpleNamespace(headers={}, client=None, cookies={}),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "live_proof_snake"
    assert payload["response_confidence"] == "high"
    assert "Snake game" in payload["response"]
    assert "artifacts/live_runtime/generated/live_snake.html" in payload["response"]
    assert_no_live_reset_boilerplate(payload["response"])
    assert (tmp_path / "artifacts/live_runtime/generated/live_snake.html").exists()


def test_neural_bridge_reports_continuous_band_profile():
    from core.senses import neural_bridge as nb

    bridge = nb.NeuralBridge(lightweight_mode=True)
    bridge._calibrate()
    sample = bridge._generate_synthetic_eeg(4)
    profile = bridge._band_profile(sample)

    assert set(profile) == {"delta", "theta", "alpha", "beta", "gamma"}
    assert abs(sum(profile.values()) - 1.0) < 0.01


def test_conversation_lane_degraded_messages_do_not_ask_user_to_repeat():
    from interface.routes.chat import _conversation_lane_user_message

    samples = [
        _conversation_lane_user_message({"state": "warming"}, status_override="warming_timeout"),
        _conversation_lane_user_message({"state": "warming"}, status_override="warming_failed"),
        _conversation_lane_user_message({"state": "ready"}, timed_out=True),
        _conversation_lane_user_message({"state": "recovering"}),
        _conversation_lane_user_message({"state": "failed"}),
        _conversation_lane_user_message({"state": "cold"}),
    ]

    for sample in samples:
        assert_no_live_reset_boilerplate(sample)


def test_output_guardrail_degraded_messages_do_not_ask_user_to_repeat():
    from core.security.output_guardrails import OutputGuardrails

    guard = OutputGuardrails()
    empty, empty_report = guard.check_response("")
    incomplete, incomplete_report = guard.check_response("...")

    assert empty_report["ok"] is False
    assert incomplete_report["ok"] is False
    assert_no_live_reset_boilerplate(empty)
    assert_no_live_reset_boilerplate(incomplete)


@pytest.mark.asyncio
async def test_intent_router_route_execution_drives_capability_engine():
    from core.cognitive.router import IntentRouter

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context))
            return {"ok": True, "skill": skill_name, "params": params, "context": context}

    result = await IntentRouter().route_execution(
        "file_operation",
        {"action": "exists", "path": "README.md"},
        FakeCapabilityEngine(),
    )

    assert result["ok"] is True
    assert calls == [
        (
            "file_operation",
            {"action": "exists", "path": "README.md"},
            {"origin": "api", "route": "intent_router.route_execution"},
        )
    ]


@pytest.mark.asyncio
async def test_state_machine_live_coding_artifact_writes_runnable_snake_html():
    from core.cognitive.state_machine import StateMachine

    calls = []

    class FakeOrchestrator:
        capability_engine = SimpleNamespace(
            skills={"coding_skill": object(), "file_operation": object()},
            active_skills={"coding_skill", "file_operation"},
        )

        async def execute_tool(self, tool_name, args, **kwargs):
            calls.append((tool_name, args, kwargs))
            if tool_name == "coding_skill":
                return {"ok": False, "error": "model unavailable in unit test"}
            if tool_name == "file_operation":
                assert args["action"] == "write"
                assert args["path"].endswith(".html")
                assert "<canvas" in args["content"].lower()
                assert "function tick" in args["content"]
                return {"ok": True, "summary": "written"}
            raise AssertionError(tool_name)

        def _publish_telemetry(self, _payload):
            pass

    machine = StateMachine(orchestrator=FakeOrchestrator())
    result = await machine._maybe_execute_live_coding_artifact(
        "Create a simple game of Snake and save it as artifacts/live_runtime/generated/test_snake.html",
        {},
        priority=1.0,
        origin="api",
    )

    assert result is not None
    reply, used_skills = result
    assert "artifacts/live_runtime/generated/test_snake.html" in reply
    assert used_skills == ["coding_skill", "file_operation"]
    assert [call[0] for call in calls] == ["coding_skill", "file_operation"]
    assert_no_live_reset_boilerplate(reply)


@pytest.mark.asyncio
async def test_file_operation_write_creates_nested_live_runtime_directory(tmp_path):
    from core.skills.file_operation import FileOperationSkill

    skill = FileOperationSkill()
    skill.root_dir = str(tmp_path.resolve())

    result = await skill.execute(
        {
            "action": "write",
            "path": "artifacts/live_runtime/generated/nested.txt",
            "content": "live proof",
        },
        context={"origin": "unit_test"},
    )

    assert result["ok"] is True
    assert (tmp_path / "artifacts/live_runtime/generated/nested.txt").read_text() == "live proof"


@pytest.mark.asyncio
async def test_computer_use_clock_returns_limited_payload_when_permissions_block(monkeypatch):
    from core.skills.computer_use import ComputerUseSkill

    skill = ComputerUseSkill()

    async def blocked(*_args, **_kwargs):
        return {"ok": False, "status": "denied", "error": "permission blocked"}

    monkeypatch.setattr(skill, "_require_permissions", blocked)
    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["clock_text"]
