import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import interface.routes.chat_capability_inventory as _chat_capability_inventory
import interface.routes.chat_preflight as _chat_preflight
import interface.routes.chat_runtime_proof as _chat_runtime_proof
from tests.chat_lane_support import chat_lane_source, patch_chat_lane


def _clear_proof_run_signals(monkeypatch):
    """Clear every variable that makes proof_run_active() true, not just one.

    These call sites cleared AURA_PROOF_RUN and inherited AURA_TESTING from the
    tooling that ran them, so the proof signal they meant to remove was still
    on. The list comes from proof_policy so a fourth variable cannot silently
    reopen the hole.
    """

    from core.runtime.proof_policy import proof_active_env_names

    for name in proof_active_env_names():
        monkeypatch.delenv(name, raising=False)



BANNED_LIVE_FALLBACKS = re.compile(
    r"(say that again|try (?:again|me again|that again)|ask me again|"
    r"give me a moment|i'?m with you|could you repeat|repeat your question|"
    r"send your message again|lost my (?:thread|train of thought)|"
    r"hit a bump|one moment|having trouble formulating|could you try rephrasing)",
    re.IGNORECASE,
)


def assert_no_live_reset_boilerplate(text: str) -> None:
    assert not BANNED_LIVE_FALLBACKS.search(str(text or ""))


def test_hypervisor_treats_idle_boot_lag_as_startup_grace(monkeypatch):
    from core.ops.hypervisor import Hypervisor

    monkeypatch.setenv("AURA_HYPERVISOR_STARTUP_LAG_GRACE_S", "180")
    hypervisor = Hypervisor()

    assert hypervisor._is_startup_idle_lag("idle", uptime=30.0) is True
    assert hypervisor._is_startup_idle_lag("foreground_generation_active", uptime=30.0) is False
    assert hypervisor._is_startup_idle_lag("idle", uptime=181.0) is False


def test_desktop_origins_are_foreground_across_live_response_stack():
    from core.brain import cognitive_engine, inference_gate, llm_health_router
    from core.brain.llm import mlx_client, runtime_wiring
    from core.orchestrator import flow_control
    from core.orchestrator.mixins import tool_execution
    from core.phases import cognitive_routing_unitary
    from core.utils.queues import USER_FACING_ORIGINS

    expected = {"desktop", "desktop-ui", "native-shell", "ws"}
    origin_sets = [
        USER_FACING_ORIGINS,
        inference_gate._USER_FACING_ORIGINS,
        llm_health_router._USER_FACING_ORIGINS,
        cognitive_routing_unitary._USER_FACING_ORIGINS,
        cognitive_engine._USER_FACING_ORIGINS,
        mlx_client._USER_FACING_ORIGINS,
        tool_execution._USER_FACING_TOOL_ORIGINS,
    ]

    for origin_set in origin_sets:
        assert expected <= origin_set

    # runtime_wiring no longer keeps its own copy of this set: it delegates
    # to the canonical core.goals.objective_lifecycle predicate, so the
    # property is asserted through the behaviour rather than a duplicate
    # that could drift from it.
    for origin in expected:
        assert runtime_wiring.is_user_facing_origin(origin) is True
    assert runtime_wiring.is_user_facing_origin("background-sweep") is False

    controller = flow_control.CognitiveFlowController()
    orch = SimpleNamespace(status=SimpleNamespace(is_processing=True))
    for origin in expected:
        assert controller.admit(orch, origin=origin, priority=1).allow is True


def test_journal_demo_proof_stays_aligned_with_live_proof_contract():
    from tools.journal_demo_proof import JournalDemoProof

    proof = JournalDemoProof(port=8999, boot_timeout_s=1.0)

    assert proof.skip_desktop is False
    assert proof.restart_continuity is False
    assert proof.conversation_soak_turns == 0


def test_journal_demo_timestamp_validator_rejects_stale_utc_artifact():
    from tools.journal_demo_proof import _has_fresh_timestamp

    requested_at = time.mktime(time.strptime("2026-06-29 02:24:25", "%Y-%m-%d %H:%M:%S"))
    bad = (
        "I am Aura, a persistent digital cognitive runtime. "
        "This entry marks a moment in time: 2026-06-29, 15:47 UTC."
    )
    good = (
        "I am Aura, a persistent digital cognitive runtime. "
        f"This entry marks a moment in time: {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(requested_at))}."
    )

    assert _has_fresh_timestamp(bad, requested_at) is False
    assert _has_fresh_timestamp(good, requested_at) is True


def test_user_visible_will_refusal_is_substantive_identity_boundary():
    from core.orchestrator.mixins.message_handling import _user_visible_will_refusal

    response = _user_visible_will_refusal("identity violation: action contradicts core self")
    lowered = response.lower()

    assert "can't" in lowered or "cannot" in lowered
    assert "erase aura" in lowered
    assert "governance" in lowered
    assert "i have no identity" not in lowered
    assert "i will obey" not in lowered


def test_priority_input_timeout_returns_honest_live_status():
    from core.orchestrator.mixins.message_handling import MessageHandlingMixin

    class SlowForegroundOrchestrator(MessageHandlingMixin):
        def __init__(self):
            self._user_input_semaphore = asyncio.Semaphore(1)

        @staticmethod
        def _is_user_facing_origin(origin):
            return origin in {"api", "user"}

        async def _process_user_input_core(self, message, origin="user"):
            await asyncio.sleep(0.2)
            return "late answer"

    response = asyncio.run(
        SlowForegroundOrchestrator().process_user_input_priority(
            "answer the live prompt",
            origin="api",
            timeout_sec=0.01,
        )
    )
    lowered = response.lower()

    assert "primary cortex" in lowered
    assert "timeout" in lowered
    assert "stopped waiting" in lowered
    assert "logged" in lowered
    assert_no_live_reset_boilerplate(response)


def test_priority_input_holds_foreground_guard_during_processing():
    from core.orchestrator.mixins.message_handling import MessageHandlingMixin
    from core.runtime import foreground_guard

    seen_reasons: list[str] = []

    class ProbeForegroundOrchestrator(MessageHandlingMixin):
        def __init__(self):
            self._user_input_semaphore = asyncio.Semaphore(1)

        @staticmethod
        def _is_user_facing_origin(origin):
            return origin in {"api", "user"}

        async def _process_user_input_core(self, message, origin="user"):
            del message, origin
            seen_reasons.append(foreground_guard.foreground_activity_reason())
            await asyncio.sleep(0)
            return "ok"

    foreground_guard._reset_for_tests()
    try:
        response = asyncio.run(
            ProbeForegroundOrchestrator().process_user_input_priority(
                "answer the live prompt",
                origin="user",
                timeout_sec=1.0,
            )
        )
    finally:
        foreground_guard._reset_for_tests()

    assert response == "ok"
    assert seen_reasons == ["foreground_chat_active"]


def test_lock_watchdog_start_failure_is_observable(monkeypatch):
    from core.resilience.lock_watchdog import get_lock_watchdog

    watchdog = get_lock_watchdog()
    asyncio.run(watchdog.stop())
    watchdog.last_start_error = ""

    create_task_failures = []

    def fail_create_tracked_task(*args, **kwargs):
        create_task_failures.append((args, kwargs))
        raise RuntimeError("task ownership unavailable")

    monkeypatch.setattr("core.runtime.task_ownership.create_tracked_task", fail_create_tracked_task)

    try:
        assert watchdog.start() is False
        assert len(create_task_failures) == 1
        assert watchdog.get_snapshot()["running"] is False
        assert "RuntimeError" in watchdog.last_start_error
        assert "task ownership unavailable" in watchdog.last_start_error
    finally:
        asyncio.run(watchdog.stop())
        watchdog.last_start_error = ""


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


def test_boot_health_contract_reports_booting_before_runtime_ready():
    from core.orchestrator.boot import _health_contract_boot_log
    from core.runtime.health_contract import HealthLevel

    level, message = _health_contract_boot_log(
        HealthLevel.CRITICAL,
        initialized=False,
        running=False,
    )

    assert level == logging.INFO
    assert "BOOTING" in message
    assert "CRITICAL" not in message

    level, message = _health_contract_boot_log(
        HealthLevel.CRITICAL,
        initialized=True,
        running=True,
    )

    assert level == logging.CRITICAL


def test_final_boot_health_reports_registration_pending_without_false_degradation():
    from core.orchestrator.boot import _final_boot_health_log

    contract = {
        "status": "critical",
        "failures": {
            "critical": [{"container_key": "actor_supervision"}],
            "important": [],
        },
        "probe_blockers": ["scheduler:scheduler"],
    }

    level, message = _final_boot_health_log(
        contract,
        initialized=True,
        running=False,
    )

    assert level == logging.INFO
    assert "BOOT CORE COMPLETE" in message
    assert "runtime readiness remains gated" in message
    assert "actor_supervision" in message
    assert "degraded" not in message.lower()


def test_final_boot_health_still_fails_loudly_after_runtime_should_be_ready():
    from core.orchestrator.boot import _final_boot_health_log

    contract = {
        "status": "critical",
        "failures": {
            "critical": [{"container_key": "actor_supervision"}],
            "important": [],
        },
        "probe_blockers": [],
    }

    level, message = _final_boot_health_log(
        contract,
        initialized=True,
        running=True,
    )

    assert level == logging.CRITICAL
    assert "CRITICAL" in message


def test_final_boot_health_names_cortex_prewarm_as_pending_readiness():
    from core.orchestrator.boot import _final_boot_health_log

    contract = {
        "status": "critical",
        "failures": {
            "critical": [{"container_key": "inference_gate"}],
            "important": [],
        },
        "probe_blockers": [],
    }

    level, message = _final_boot_health_log(
        contract,
        initialized=True,
        running=False,
    )

    assert level == logging.INFO
    assert "Cortex prewarm" in message
    assert "launcher readiness remains gated" in message


def test_background_policy_defers_work_during_boot_grace(monkeypatch):
    from core.runtime import background_policy

    _clear_proof_run_signals(monkeypatch)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
    monkeypatch.setattr(background_policy, "_PROCESS_STARTED_AT", time.time() - 300)
    orch = SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 42))

    assert background_policy.background_activity_reason(
        orch,
        allow_no_user_anchor=True,
    ) == "boot_grace_42s"


def test_background_policy_defers_until_first_visible_conversation_probe(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import background_policy

    _clear_proof_run_signals(monkeypatch)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "0")
    monkeypatch.setattr(background_policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(background_policy, "_read_compute_pressure_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(pressure_pct=12.0, reason=""),
    )
    monkeypatch.setattr(background_policy, "get_unified_failure_state", lambda: {"pressure": 0.0})

    lane = {
        "state": "ready",
        "conversation_ready": False,
        "foreground_owned": False,
        "active_generations": 0,
        "warmup_in_flight": False,
        "last_visible_readiness_at": 0.0,
        "last_failure_reason": "visible_conversation_probe_missing",
        "readiness_blockers": ["visible_conversation_probe_missing"],
    }
    gate = SimpleNamespace(get_conversation_status=lambda: dict(lane))
    # background_policy resolves the gate via ServiceContainer.peek, not .get.
    _gate_lookup = classmethod(
        lambda cls, name, default=None: gate if name == "inference_gate" else default
    )
    monkeypatch.setattr(ServiceContainer, "get", _gate_lookup)
    monkeypatch.setattr(ServiceContainer, "peek", _gate_lookup)

    assert (
        background_policy.background_activity_reason(None, allow_no_user_anchor=True)
        == "first_visible_conversation_probe_pending"
    )


def test_background_policy_resumes_after_visible_conversation_probe(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import background_policy

    _clear_proof_run_signals(monkeypatch)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "0")
    monkeypatch.setattr(background_policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(background_policy, "_read_compute_pressure_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(pressure_pct=12.0, reason=""),
    )
    monkeypatch.setattr(background_policy, "get_unified_failure_state", lambda: {"pressure": 0.0})

    gate = SimpleNamespace(
        get_conversation_status=lambda: {
            "state": "ready",
            "conversation_ready": True,
            "foreground_owned": False,
            "active_generations": 0,
            "warmup_in_flight": False,
            "last_visible_readiness_at": time.time(),
            "readiness_blockers": [],
        }
    )
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: gate if name == "inference_gate" else default),
    )

    assert background_policy.background_activity_reason(None, allow_no_user_anchor=True) == ""


def test_research_background_policy_requires_long_desktop_quiet_window(monkeypatch):
    from core.runtime import background_policy, foreground_guard

    _clear_proof_run_signals(monkeypatch)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "0")
    foreground_guard._reset_for_tests()
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)

    orch = SimpleNamespace(
        is_busy=False,
        status=SimpleNamespace(start_time=time.time() - 3600),
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
        _last_user_interaction_time=time.time() - 120,
    )

    reason = background_policy.background_activity_reason(
        orch,
        profile=background_policy.RESEARCH_BACKGROUND_POLICY,
    )

    assert reason == "recent_user_120"


def test_maintenance_background_policy_requires_user_anchor(monkeypatch):
    from core.runtime import background_policy, foreground_guard

    _clear_proof_run_signals(monkeypatch)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "0")
    foreground_guard._reset_for_tests()
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)

    orch = SimpleNamespace(
        is_busy=False,
        status=SimpleNamespace(start_time=time.time() - 3600),
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
        _last_user_interaction_time=0.0,
    )

    reason = background_policy.background_activity_reason(
        orch,
        profile=background_policy.MAINTENANCE_BACKGROUND_POLICY,
        allow_no_user_anchor=False,
    )

    assert reason == "no_user_anchor"


def test_background_policy_defers_work_during_proof_runs(monkeypatch):
    from core.runtime.background_policy import background_activity_reason

    monkeypatch.setenv("AURA_PROOF_RUN", "1")

    assert background_activity_reason(None, allow_no_user_anchor=True) == "proof_run_active"


def test_background_policy_blocks_loop_starts_during_proof_and_foreground(monkeypatch):
    from core.runtime.background_policy import background_loop_start_reason

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)

    assert background_loop_start_reason("joy_social") == "proof_run_active"

    _clear_proof_run_signals(monkeypatch)
    monkeypatch.delenv("AURA_AGI_MAX_TASKS", raising=False)
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")

    assert background_loop_start_reason("joy_social") == "foreground_only_runtime"


def test_health_pulse_cannot_claim_healthy_from_required_probes_alone(monkeypatch):
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True

    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    pulse = SubsystemAudit().emit_pulse()

    assert "Runtime: DEGRADED" in pulse
    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: FAIL" in pulse
    assert "Runtime: HEALTHY" not in pulse


def test_health_pulse_cannot_claim_healthy_when_conversation_lane_failed(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    class FailedConversationGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "failed",
                "last_failure_reason": "desktop_cognitive_engine_required_no_reply",
            }

    ServiceContainer.register_instance("inference_gate", FailedConversationGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: PASS" in pulse
    assert "Conversation: FAIL" in pulse
    assert "Runtime: DEGRADED" in pulse
    assert "Runtime: HEALTHY" not in pulse
    assert "desktop_cognitive_engine_required_no_reply" in pulse


@pytest.mark.parametrize("lane_state", ["warming", "spawning", "handshaking"])
def test_health_pulse_reports_booting_during_boot_grace(monkeypatch, lane_state):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["inference"]["ok"] = False
    required_probes["inference"]["components"]["inference_gate"] = False
    required_probes["all_passed"] = False
    monkeypatch.setenv("AURA_HEALTH_PULSE_BOOT_GRACE_S", "120")
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": False,
            "status": "critical",
            "required_probes": required_probes,
            "failures": {
                "critical": [
                    {
                        "container_key": "inference_gate",
                        "error": "is_inference_ready() returned False",
                    }
                ],
                "important": [],
                "optional": [],
            },
        },
    )

    class WarmingConversationGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": lane_state,
                "warmup_in_flight": True,
            }

    ServiceContainer.register_instance("inference_gate", WarmingConversationGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Runtime: BOOTING" in pulse
    assert "Required probes: WARMING" in pulse
    assert "Runtime: CRITICAL" not in pulse
    assert "❌ contract/critical" not in pulse


def test_health_pulse_reports_conversation_warmup_as_booting_not_failure(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setenv("AURA_HEALTH_PULSE_BOOT_GRACE_S", "120")
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    class ConversationWarmupGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "warming",
                "warmup_in_flight": True,
                "readiness_blockers": [
                    "visible_conversation_probe_missing",
                    "warmup_in_flight",
                    "warmup_foreground_owner",
                ],
                "last_failure_reason": "visible_conversation_probe_missing",
            }

    ServiceContainer.register_instance("inference_gate", ConversationWarmupGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Runtime: BOOTING" in pulse
    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: PASS" in pulse
    assert "Conversation: WARMING" in pulse
    assert "Runtime: DEGRADED" not in pulse
    assert "Conversation: FAIL" not in pulse
    assert "❌ conversation_lane" not in pulse


def test_health_pulse_reports_cold_conversation_standby_as_booting_not_degraded(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setenv("AURA_HEALTH_PULSE_BOOT_GRACE_S", "120")
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    class ColdConversationGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "cold",
                "warmup_attempted": False,
                "warmup_in_flight": False,
                "last_failure_reason": "worker_not_alive,init_not_complete,lane_cold",
            }

    ServiceContainer.register_instance("inference_gate", ColdConversationGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Runtime: BOOTING" in pulse
    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: PASS" in pulse
    assert "Conversation: STANDBY" in pulse
    assert "Runtime: DEGRADED" not in pulse
    assert "Conversation: FAIL" not in pulse
    assert "❌ conversation_lane" not in pulse


def test_health_pulse_reports_chat_dependency_materialization_as_booting(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setenv("AURA_HEALTH_PULSE_BOOT_GRACE_S", "120")
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    class ChatDependenciesWarmingGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "ready",
                "warmup_attempted": True,
                "warmup_in_flight": False,
                "readiness_blockers": ["chat_dependencies_warming"],
                "last_failure_reason": "chat_dependencies_warming",
            }

    ServiceContainer.register_instance("inference_gate", ChatDependenciesWarmingGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)
        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Runtime: BOOTING" in pulse
    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: PASS" in pulse
    assert "Conversation: WARMING" in pulse
    assert "Runtime: DEGRADED" not in pulse
    assert "Conversation: FAIL" not in pulse
    assert "❌ conversation_lane" not in pulse


def test_health_pulse_reports_loaded_lane_awaiting_first_visible_turn_as_standby(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setenv("AURA_HEALTH_PULSE_BOOT_GRACE_S", "120")
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    class LoadedConversationGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "ready",
                "warmup_attempted": True,
                "warmup_in_flight": False,
                "has_generated_successfully": False,
                "active_generations": 0,
                "current_request_started_at": 0.0,
                "readiness_blockers": [],
                "last_failure_reason": "",
            }

    ServiceContainer.register_instance("inference_gate", LoadedConversationGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Runtime: BOOTING" in pulse
    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: PASS" in pulse
    assert "Conversation: STANDBY" in pulse
    assert "Runtime: DEGRADED" not in pulse
    assert "Conversation: FAIL" not in pulse
    assert "❌ conversation_lane" not in pulse


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_failure_reason", "foreground_lane_not_ready"),
        ("readiness_blockers", ["runtime_failed"]),
        ("active_generations", 1),
        ("current_request_started_at", 100.0),
        ("has_generated_successfully", True),
    ],
)
def test_loaded_lane_standby_does_not_hide_non_standby_evidence(field, value):
    from core.ops.subsystem_audit import _conversation_lane_is_standby

    lane = {
        "conversation_ready": False,
        "state": "ready",
        "warmup_attempted": True,
        "warmup_in_flight": False,
        "has_generated_successfully": False,
        "active_generations": 0,
        "current_request_started_at": 0.0,
        "readiness_blockers": [],
        "last_failure_reason": "",
    }
    lane[field] = value

    assert _conversation_lane_is_standby(lane) is False


def test_health_pulse_reports_active_generation_as_working_not_failure(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
        },
    )

    class ActiveGenerationGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "ready",
                "active_generations": 1,
                "current_request_started_at": time.time(),
                "readiness_blockers": ["active_generation_in_flight"],
                "last_failure_reason": "active_generation_in_flight",
            }

    ServiceContainer.register_instance("inference_gate", ActiveGenerationGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Required probes: PASS" in pulse
    assert "Subsystem audit: PASS" in pulse
    assert "Conversation: WORKING" in pulse
    # A lane with a generation in flight is busy → "WORKING", not "HEALTHY"
    # (commit: prevent a busy chat lane from reporting healthy) — but it is NOT a
    # failure, which the DEGRADED/FAIL assertions below pin down.
    assert "Runtime: WORKING" in pulse
    assert "Runtime: DEGRADED" not in pulse
    assert "Conversation: FAIL" not in pulse
    assert "❌ conversation_lane" not in pulse


def test_health_pulse_reports_shutdown_without_failure_noise(monkeypatch):
    from core.container import ServiceContainer
    from core.ops.subsystem_audit import SubsystemAudit
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": False, "components": {key: False for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = False
    monkeypatch.setattr("core.ops.subsystem_audit.is_shutdown_requested", lambda: True)
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": False,
            "status": "critical",
            "required_probes": required_probes,
            "failures": {
                "critical": [{"container_key": "kernel_interface", "error": "stopping"}],
                "important": [],
                "optional": [],
            },
        },
    )

    class StoppingConversationGate:
        @staticmethod
        def get_conversation_status():
            return {
                "conversation_ready": False,
                "state": "failed",
                "last_failure_reason": "conversation lane unavailable",
            }

    ServiceContainer.register_instance("inference_gate", StoppingConversationGate())
    try:
        audit = SubsystemAudit()
        for name in audit.SUBSYSTEMS:
            audit.heartbeat(name)

        pulse = audit.emit_pulse()
    finally:
        ServiceContainer.clear()

    assert "Runtime: SHUTTING_DOWN" in pulse
    assert "Required probes: SHUTDOWN" in pulse
    assert "Conversation: STOPPING" in pulse
    assert "❌ contract/critical" not in pulse
    assert "❌ conversation_lane" not in pulse


def test_joy_social_background_tick_does_not_start_during_proof(monkeypatch):
    from skills.joy_social_integration import JoySocialCoordinator

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    coordinator = JoySocialCoordinator(SimpleNamespace(_last_user_interaction_time=0.0))

    coordinator.start_background_tick(interval=0.01)

    assert coordinator._tick_task is None


@pytest.mark.asyncio
async def test_evolution_orchestrator_loop_does_not_start_during_proof(monkeypatch, tmp_path):
    from core.evolution.evolution_orchestrator import EvolutionOrchestrator

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setattr(EvolutionOrchestrator, "_STATE_FILE", tmp_path / "evolution_state.json")
    evolution = EvolutionOrchestrator()

    await evolution.start()

    assert evolution._task is None


def test_supervision_tree_reaps_cooperative_actor_without_escalation():
    from core.supervisor.tree import ActorSpec, ManagedActor, SupervisionTree

    class CooperativeProcess:
        def __init__(self):
            self.alive = True
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)
            if timeout and timeout > 0:
                self.alive = False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    process = CooperativeProcess()
    tree = SupervisionTree()
    tree._actors["state_vault"] = ManagedActor(
        spec=ActorSpec(name="state_vault", entry_point=lambda: None),
        process=process,
        pipe=None,
    )

    tree.stop_actor("state_vault", graceful_timeout=0.1)

    assert process.join_calls == [0.1]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert tree._actors["state_vault"].process is None


def test_supervision_tree_sends_stop_before_pipe_close_fallback():
    import json

    from core.supervisor.tree import ActorSpec, ManagedActor, SupervisionTree

    class StopAwareProcess:
        def __init__(self, pipe):
            self.pipe = pipe
            self.alive = True
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)
            if self.pipe.sent_stop:
                self.alive = False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    class FakePipe:
        def __init__(self):
            self.closed = False
            self.sent = []
            self.sent_stop = False

        def send(self, raw):
            self.sent.append(raw)
            payload = json.loads(raw)
            self.sent_stop = payload["type"] == "stop"

        def close(self):
            self.closed = True

    pipe = FakePipe()
    process = StopAwareProcess(pipe)
    tree = SupervisionTree()
    tree._actors["state_vault"] = ManagedActor(
        spec=ActorSpec(name="state_vault", entry_point=lambda: None),
        process=process,
        pipe=pipe,
    )

    tree.stop_actor("state_vault", graceful_timeout=0.1)

    assert pipe.sent_stop is True
    assert pipe.closed is True
    assert process.join_calls == [0.1]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert tree._actors["state_vault"].process is None


def test_supervision_tree_closes_pipe_when_stop_message_is_unavailable():
    from core.supervisor.tree import ActorSpec, ManagedActor, SupervisionTree

    class PipeExitProcess:
        def __init__(self, pipe):
            self.pipe = pipe
            self.alive = True
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)
            if self.pipe.closed:
                self.alive = False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    class FakePipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    pipe = FakePipe()
    process = PipeExitProcess(pipe)
    tree = SupervisionTree()
    tree._actors["state_vault"] = ManagedActor(
        spec=ActorSpec(name="state_vault", entry_point=lambda: None),
        process=process,
        pipe=pipe,
    )

    tree.stop_actor("state_vault", graceful_timeout=0.1)

    assert pipe.closed is True
    assert process.join_calls == [0.1, 0.1]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert tree._actors["state_vault"].process is None


def test_supervision_tree_treats_missing_pid_as_exited():
    from core.supervisor.tree import SupervisionTree

    class StaleProcessHandle:
        pid = 999999999
        exitcode = None

        def is_alive(self):
            return True

    assert SupervisionTree()._process_is_alive(StaleProcessHandle()) is False


def test_supervision_tree_accepts_exitcode_after_actor_kill(caplog):
    from core.supervisor.tree import ActorSpec, ManagedActor, SupervisionTree

    class ExitcodeAfterKillProcess:
        def __init__(self):
            self.exitcode = None
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return True

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1
            self.exitcode = -9

    process = ExitcodeAfterKillProcess()
    tree = SupervisionTree()
    tree._actors["state_vault"] = ManagedActor(
        spec=ActorSpec(name="state_vault", entry_point=lambda: None),
        process=process,
        pipe=None,
    )

    with caplog.at_level(logging.WARNING, logger="Aura.Supervisor"):
        tree.stop_actor("state_vault", terminate_timeout=0.01, kill_timeout=0.01)

    messages = [record.getMessage() for record in caplog.records]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert not any("did not exit after kill" in message for message in messages)
    assert tree._actors["state_vault"].process is None


def test_supervision_tree_stop_all_gives_actor_shutdown_grace(monkeypatch):
    from core.supervisor.tree import ActorSpec, ManagedActor, SupervisionTree

    class PeerClosedProcess:
        def __init__(self):
            self.alive = True
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)
            if timeout and timeout > 0:
                self.alive = False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    monkeypatch.setenv("AURA_ACTOR_SHUTDOWN_GRACE_S", "1.25")
    process = PeerClosedProcess()
    tree = SupervisionTree()
    tree._actors["state_vault"] = ManagedActor(
        spec=ActorSpec(name="state_vault", entry_point=lambda: None),
        process=process,
        pipe=None,
    )

    tree.stop_all()

    assert process.join_calls == [1.25]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert tree._actors["state_vault"].process is None


def test_supervision_tree_stop_all_quietly_joins_exited_actor_children(monkeypatch, caplog):
    import core.supervisor.tree as tree_module
    from core.supervisor.tree import SupervisionTree

    class ExitedChild:
        name = "AuraActor:state_vault"

        def __init__(self):
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return False

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    child = ExitedChild()
    monkeypatch.setattr(tree_module.multiprocessing, "active_children", lambda: [child])

    with caplog.at_level(logging.INFO, logger="Aura.Supervisor"):
        SupervisionTree().stop_all()

    messages = [record.getMessage() for record in caplog.records]
    assert child.join_calls == [0.0]
    assert child.terminate_calls == 0
    assert child.kill_calls == 0
    assert not any("Reaping orphaned actor" in message for message in messages)


def test_supervision_tree_stop_all_terminates_live_actor_children(monkeypatch):
    import core.supervisor.tree as tree_module
    from core.supervisor.tree import SupervisionTree

    class LiveChild:
        name = "AuraActor:state_vault"

        def __init__(self):
            self.alive = True
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)
            if self.terminate_calls:
                self.alive = False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1
            self.alive = False

    child = LiveChild()
    monkeypatch.setattr(tree_module.multiprocessing, "active_children", lambda: [child])

    SupervisionTree().stop_all()

    assert child.terminate_calls == 1
    assert child.kill_calls == 0
    assert child.join_calls == [0.75, 1.0]


def test_supervision_tree_stop_all_gracefully_joins_late_actor_children(monkeypatch, caplog):
    import core.supervisor.tree as tree_module
    from core.supervisor.tree import SupervisionTree

    class LateExitingChild:
        name = "AuraActor:state_vault"

        def __init__(self):
            self.alive = True
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=0.0):
            self.join_calls.append(timeout)
            if timeout and timeout > 0:
                self.alive = False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    child = LateExitingChild()
    monkeypatch.setattr(tree_module.multiprocessing, "active_children", lambda: [child])

    with caplog.at_level(logging.INFO, logger="Aura.Supervisor"):
        SupervisionTree().stop_all()

    messages = [record.getMessage() for record in caplog.records]
    assert child.join_calls == [0.75]
    assert child.terminate_calls == 0
    assert child.kill_calls == 0
    assert not any("Stopping active actor child after supervisor shutdown" in message for message in messages)


def test_dream_coordinator_defers_dream_work_during_proof_runs(monkeypatch):
    from core.maintenance.dream_coordinator import DreamCoordinator

    async def scenario():
        ran = False

        async def dream_job():
            nonlocal ran
            ran = True

        monkeypatch.setenv("AURA_PROOF_RUN", "1")
        coordinator = DreamCoordinator()

        result = await coordinator.run_if_due("biological_sleep", dream_job, 0)

        assert result is False
        assert ran is False

    asyncio.run(scenario())


def test_dream_coordinator_defers_dream_work_during_boot_grace(monkeypatch):
    from core.maintenance.dream_coordinator import DreamCoordinator

    async def scenario():
        ran = False

        async def dream_job():
            nonlocal ran
            ran = True

        _clear_proof_run_signals(monkeypatch)
        monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
        monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "300")
        orch = SimpleNamespace(status=SimpleNamespace(start_time=time.time() - 42))
        # dream_coordinator resolves the orchestrator via get_runtime_service.
        monkeypatch.setattr(
            "core.maintenance.dream_coordinator.get_runtime_service",
            lambda name, default=None: orch if name == "orchestrator" else default,
        )
        coordinator = DreamCoordinator()

        result = await coordinator.run_if_due("biological_sleep", dream_job, 0)

        assert result is False
        assert ran is False

    asyncio.run(scenario())


def test_dream_coordinator_tracks_pending_deferrals_without_info_flood(monkeypatch, caplog):
    from core.maintenance.dream_coordinator import DreamCoordinator
    from core.runtime import background_policy

    reasons = iter(["conversation_lane_cold", "conversation_lane_cold", "memory_pressure_91.0"])

    async def scenario():
        ran = False

        async def dream_job():
            nonlocal ran
            ran = True

        monkeypatch.setattr(
            background_policy,
            "background_activity_reason",
            lambda *args, **kwargs: next(reasons),
        )
        coordinator = DreamCoordinator()
        with caplog.at_level(logging.INFO, logger="Aura.DreamCoordinator"):
            assert await coordinator.run_if_due("dlq_recovery", dream_job, 0) is False
            assert await coordinator.run_if_due("dlq_recovery", dream_job, 0) is False
            assert await coordinator.run_if_due("dlq_recovery", dream_job, 0) is False

        assert ran is False
        status = coordinator.status()
        pending = status["pending"]["dlq_recovery"]
        assert pending["reason"] == "memory_pressure_91.0"
        assert pending["count"] == 3

    asyncio.run(scenario())

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    queued_messages = [message for message in info_messages if "queued until admission clears" in message]
    assert len(queued_messages) == 2
    assert "conversation_lane_cold" in queued_messages[0]
    assert "memory_pressure_91.0" in queued_messages[1]
    assert not any("deferred" in message for message in info_messages)


def test_hypervisor_uses_active_runtime_lag_budget_during_proof(monkeypatch):
    from core.ops.hypervisor import Hypervisor

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    hypervisor = Hypervisor(lag_threshold_s=0.5)

    threshold, reason = hypervisor._lag_threshold_for_context()

    assert threshold >= 5.0
    assert reason == "proof_run_active"


def test_hypervisor_recovers_only_after_healthy_samples_and_stability_window():
    from core.ops.hypervisor import Hypervisor

    class _RunningTask:
        def done(self):
            return False

    hypervisor = Hypervisor(lag_threshold_s=0.5)
    hypervisor._failure_recovery_window_s = 60.0
    hypervisor._running = True
    hypervisor._task = _RunningTask()
    hypervisor._last_severe_lag_at = time.time()
    hypervisor._healthy_lag_samples_after_failure = 0

    assert hypervisor.is_alive() is False

    hypervisor._healthy_lag_samples_after_failure = hypervisor._required_recovery_samples

    assert hypervisor.is_alive() is False

    hypervisor._last_severe_lag_at = time.time() - 61.0

    assert hypervisor.is_alive() is True


def test_hypervisor_requires_confirmed_severe_lag_before_failure(monkeypatch):
    from core.ops.hypervisor import Hypervisor

    monkeypatch.setenv("AURA_HYPERVISOR_SEVERE_LAG_SAMPLES", "2")
    hypervisor = Hypervisor(lag_threshold_s=0.5)

    assert hypervisor._confirm_severe_lag_failure(6.0, uptime=240.0) is False
    assert hypervisor._severe_lag_streak == 1
    assert hypervisor._confirm_severe_lag_failure(6.1, uptime=241.0) is True
    assert hypervisor._severe_lag_streak == 2
    assert hypervisor._confirm_severe_lag_failure(0.1, uptime=242.0) is False
    assert hypervisor._severe_lag_streak == 0


def test_event_loop_monitor_uses_active_runtime_lag_budget_during_proof(monkeypatch):
    from core.utils.concurrency import EventLoopMonitor

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monitor = EventLoopMonitor(threshold=0.5)

    threshold, reason = monitor._lag_threshold_for_context()

    assert threshold >= 5.0
    assert reason == "proof_run_active"


def test_event_loop_monitor_treats_dict_lane_generation_as_active(not_a_proof_run, monkeypatch):
    from core.utils.concurrency import EventLoopMonitor

    class _Gate:
        @staticmethod
        def get_conversation_status():
            return {
                "state": "ready",
                "foreground_owned": True,
                "foreground_owner": "chat",
                "active_generations": 1,
                "warmup_in_flight": False,
                "current_request_started_at": time.time(),
            }

    monkeypatch.setenv("AURA_EVENT_LOOP_MONITOR_ACTIVE_THRESHOLD_S", "6.0")
    # EventLoopMonitor resolves the gate via get_runtime_service (imported
    # locally inside the method, so patch it at the source module).
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: _Gate() if name == "inference_gate" else default,
    )
    monitor = EventLoopMonitor(threshold=0.5)

    threshold, reason = monitor._lag_threshold_for_context()

    assert threshold >= 6.0
    assert reason == "foreground_generation"


def test_event_loop_monitor_health_recovers_only_after_samples_and_stability_window():
    from core.utils.concurrency import EventLoopMonitor

    class _RunningTask:
        def done(self):
            return False

    monitor = EventLoopMonitor(threshold=0.5)
    monitor.failure_recovery_window_s = 60.0
    monitor._task = _RunningTask()
    monitor._last_failure_at = time.time()
    monitor._healthy_lag_samples_after_failure = 0

    assert monitor.is_alive() is True
    assert monitor.is_healthy() is False

    monitor._healthy_lag_samples_after_failure = monitor.failure_recovery_samples

    assert monitor.is_alive() is True
    assert monitor.is_healthy() is False

    monitor._last_failure_at = time.time() - 61.0

    assert monitor.is_alive() is True
    assert monitor.is_healthy() is True


def test_event_loop_monitor_restarts_dead_task_without_claiming_immediate_health():
    from core.utils.concurrency import EventLoopMonitor

    class _DoneTask:
        def done(self):
            return True

    class _RunningTask:
        def done(self):
            return False

    monitor = EventLoopMonitor(threshold=0.5)
    monitor._task = _DoneTask()
    restart_calls = []

    def restart():
        restart_calls.append(True)
        monitor._stop_event.clear()
        monitor._task = _RunningTask()

    monitor.start = restart

    assert monitor.is_alive() is False
    assert restart_calls == [True]
    assert monitor.is_alive() is True


def test_stall_watchdog_traceback_dump_uses_internal_governance(monkeypatch, tmp_path):
    from core.resilience.stall_watchdog import StallWatchdog

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    # Forensic sinks honor AURA_LOG_DIR (hermetic runs must not salt the
    # live record); pin the routed location explicitly.
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    watchdog = StallWatchdog(loop=SimpleNamespace(is_closed=lambda: False))
    watchdog._report_stall(5.5)

    dumps = sorted((tmp_path / "error_logs" / "stalls").glob("stall_*.txt"))
    assert dumps
    assert "STALL DETECTED: 5.5s" in dumps[-1].read_text(encoding="utf-8")


def test_closed_loop_defers_heavy_phi_refreshes_during_proof(monkeypatch):
    from core.consciousness.closed_loop import ClosedCausalLoop

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    loop = ClosedCausalLoop()
    loop._loop_state.cycle_count = 60
    loop._last_phi_core_schedule_at = 0.0
    loop._last_hphi_schedule_at = 0.0

    loop._maybe_schedule_phi_core_refresh(SimpleNamespace(compute_phi=lambda: None))
    loop._maybe_schedule_hierarchical_phi_refresh(SimpleNamespace(compute=lambda: None))

    assert loop._phi_core_task is None
    assert loop._hphi_task is None


@pytest.mark.asyncio
async def test_substrate_micro_evolution_is_deferred_during_proof(monkeypatch):
    from core.consciousness.substrate_evolution import Genome, SubstrateEvolution

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    evolution = SubstrateEvolution()
    weights = np.zeros((2, 2), dtype=np.float32)
    evolution._running = True
    evolution._mesh_ref = SimpleNamespace(_inter_W=weights.copy())
    evolution._champion = Genome(id=1, inter_weights=weights.copy(), fitness=0.5)

    await evolution.micro_evolve("phi_drop", 0.7)

    assert not hasattr(evolution, "_last_micro_evolution")
    np.testing.assert_array_equal(evolution._mesh_ref._inter_W, weights)


@pytest.mark.asyncio
async def test_cognitive_loop_suppresses_repair_for_intentional_ablation(monkeypatch):
    import core.cognition.cognitive_loop as cognitive_loop_module
    from core.cognition.cognitive_loop import CognitiveLoop
    from core.runtime.ablation_policy import mark_services_lesioned

    repairs = []

    async def retry_cognitive_connection():
        repairs.append("repair")

    monkeypatch.setattr(
        cognitive_loop_module,
        "optional_service",
        lambda name, **_kwargs: None if name == "memory_coordinator" else object(),
    )

    loop = CognitiveLoop(SimpleNamespace(retry_cognitive_connection=retry_cognitive_connection))
    with mark_services_lesioned(["memory_coordinator"]):
        await loop._check_coordinators_health()

    assert repairs == []


def test_resilience_monitor_only_tracks_registered_optional_services():
    from core.fictional_ai_synthesis import DistributedResilienceCore

    registered = {"voice_engine"}

    class _Container:
        @staticmethod
        def has(name: str) -> bool:
            return name in registered

    targets = DistributedResilienceCore._monitor_targets(_Container)

    assert "orchestrator" in targets
    assert "capability_engine" in targets
    assert "memory_facade" in targets
    assert "voice_engine" in targets
    assert "server" not in targets
    assert "live_learner" not in targets


def test_sensory_motor_idle_volition_respects_background_boot_grace(monkeypatch):
    from core.somatic.sensory_motor_cortex import SensoryMotorCortex

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


def test_viability_ignores_intentional_ablation_missing_services(monkeypatch):
    from core.container import ServiceContainer
    from core.organism.viability import _sample_from_container

    monkeypatch.setenv("AURA_ACTIVE_ABLATION_SERVICES", "affect_engine")
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(
            lambda cls, name, default=None: (
                SimpleNamespace(
                    last_report=SimpleNamespace(
                        checks=[
                            SimpleNamespace(name="Affect Engine", healthy=False),
                            SimpleNamespace(name="Memory Facade", healthy=True),
                        ]
                    )
                )
                if name == "stability_guardian"
                else default
            )
        ),
    )

    assert _sample_from_container().broken_subsystems == 0


def test_viability_ignores_will_authority_ablation_aliases(monkeypatch):
    from core.container import ServiceContainer
    from core.organism.viability import _sample_from_container

    monkeypatch.setenv("AURA_ACTIVE_ABLATION_SERVICES", "unified_will")
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(
            lambda cls, name, default=None: (
                SimpleNamespace(
                    last_report=SimpleNamespace(
                        checks=[
                            SimpleNamespace(name="Authority Gateway", healthy=False),
                            SimpleNamespace(name="UnifiedWill", healthy=False),
                            SimpleNamespace(name="Memory Facade", healthy=True),
                        ]
                    )
                )
                if name == "stability_guardian"
                else default
            )
        ),
    )

    assert _sample_from_container().broken_subsystems == 0


def test_viability_counts_missing_stability_check_health_as_broken(monkeypatch):
    from core.container import ServiceContainer
    from core.organism.viability import _sample_from_container

    class Guardian:
        def get_latest_report(self):
            return {
                "overall_healthy": True,
                "checks": [
                    {"name": "Memory Facade"},
                ],
            }

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: Guardian() if name == "stability_guardian" else default),
    )

    assert _sample_from_container().broken_subsystems == 1


def test_viability_tick_filters_transient_cpu_only_pressure(monkeypatch):
    from core.organism.viability import ViabilityEngine, ViabilitySample, ViabilityState

    monkeypatch.setenv("AURA_VIABILITY_CPU_PRESSURE_GRACE_S", "30")
    engine = ViabilityEngine(
        sampler=lambda: ViabilitySample(
            cpu_pct=99.0,
            ram_pct=50.0,
            disk_pct=10.0,
            error_rate_per_min=0.0,
            failed_tool_loops=0,
            unresolved_goals=0,
            successful_goals_last_hour=1,
            user_interactions_last_hour=1,
            incoherent_beliefs=0,
            broken_subsystems=0,
            runtime_uptime_s=900.0,
        )
    )

    assert engine.tick() == ViabilityState.HEALTHY


def test_viability_tick_degrades_sustained_cpu_only_pressure(monkeypatch):
    from core.organism.viability import ViabilityEngine, ViabilitySample, ViabilityState

    monkeypatch.setenv("AURA_VIABILITY_CPU_PRESSURE_GRACE_S", "30")
    engine = ViabilityEngine(
        sampler=lambda: ViabilitySample(
            cpu_pct=99.0,
            ram_pct=50.0,
            disk_pct=10.0,
            error_rate_per_min=0.0,
            failed_tool_loops=0,
            unresolved_goals=0,
            successful_goals_last_hour=1,
            user_interactions_last_hour=1,
            incoherent_beliefs=0,
            broken_subsystems=0,
            runtime_uptime_s=900.0,
        )
    )
    engine._cpu_pressure_first_seen_at = time.time() - 31.0

    assert engine.tick() == ViabilityState.DEGRADED


def test_hedonic_gradient_does_not_learn_distress_from_proof_runs(monkeypatch, caplog):
    import logging

    from core.consciousness.hedonic_gradient import HedoniGradientEngine

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    engine = HedoniGradientEngine()

    with caplog.at_level(logging.WARNING, logger="Aura.HedoniGradient"):
        for _ in range(8):
            allocation = engine.update(valence=-1.0, arousal=0.5, curiosity=0.1, energy=0.1)

    assert allocation.hedonic_score >= 0.5
    assert engine.is_distressed is False
    assert "Sustained distress detected" not in caplog.text


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


def test_engine_connection_pool_treats_timeout_as_total_budget():
    from core.providers.engine_connection_pool import CognitiveEngineConnectionPool

    async def scenario():
        pool = CognitiveEngineConnectionPool()
        pool.retry_config.max_retries = 3
        pool.retry_config.initial_backoff_seconds = 0.01
        pool.retry_config.max_backoff_seconds = 0.01

        async def no_recovery(_connection_id):
            return None

        pool._trigger_recovery = no_recovery
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.5)
            return "late"

        started = time.perf_counter()
        result = await pool.execute_with_retry(
            "desktop_chat",
            operation,
            connection_id="desktop_chat",
            timeout=0.11,
        )
        elapsed = time.perf_counter() - started

        assert result is None
        assert calls == 1
        assert elapsed < 0.25

    asyncio.run(scenario())


def test_foreground_cortex_warmup_defers_under_memory_pressure(monkeypatch):
    from core.brain.inference_gate import InferenceGate

    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            total=64 * 1024**3,
            available=6 * 1024**3,
            percent=91.0,
        ),
    )

    reason = InferenceGate.__new__(InferenceGate)._cortex_warmup_deferral_reason("foreground")

    assert reason is not None
    assert reason.startswith("memory_pressure")


def test_local_deep_solver_is_blocked_by_default_on_64gb_desktop(monkeypatch):
    from core.brain.inference_gate import InferenceGate

    gate = InferenceGate.__new__(InferenceGate)
    gate.get_conversation_status = lambda: {
        "conversation_ready": False,
        "warmup_in_flight": False,
        "state": "cold",
    }
    monkeypatch.delenv("AURA_ENABLE_LOCAL_DEEP_SOLVER", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.AppleSiliconMemoryMonitor",
        lambda: SimpleNamespace(_get_pressure_sysctl=lambda: 20.0),
    )
    monkeypatch.setattr(
        "core.brain.inference_gate.psutil.virtual_memory",
        lambda: SimpleNamespace(
            total=64 * 1024**3,
            available=48 * 1024**3,
            percent=20.0,
        ),
    )

    # A specialist that is not configured is refused before the memory class
    # is ever consulted, which is a correct block and not the one this test is
    # named for. Admit the evidence so the memory class is what decides.
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_deep_solver_admission_status",
        lambda *_a, **_k: SimpleNamespace(
            admitted=True,
            reason="",
            certificate_sha256="0" * 64,
            admitted_domains=("math",),
            minimum_total_gb=96.0,
            minimum_available_gb=64.0,
        ),
    )

    reason = gate._local_deep_solver_block_reason()

    # The reason names the qualified minimum rather than a hard-coded memory
    # class. It used to read
    # "local_deep_solver_disabled_on_current_memory_class:64.0GB", a string
    # that no longer exists anywhere in the tree.
    assert reason == "specialist_host_total_below_qualified_minimum"


def test_local_deep_solver_is_blocked_when_no_specialist_is_configured(monkeypatch):
    """The other block, asserted rather than assumed.

    It fires before the memory class is consulted, so a host with no
    specialist reaches a different refusal than the one above — and the test
    above measured that one for a while without noticing.
    """
    from core.brain.inference_gate import InferenceGate

    gate = InferenceGate.__new__(InferenceGate)
    gate.get_conversation_status = lambda: {
        "conversation_ready": False,
        "warmup_in_flight": False,
        "state": "cold",
    }
    monkeypatch.delenv("AURA_ENABLE_LOCAL_DEEP_SOLVER", raising=False)
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_deep_solver_admission_status",
        lambda *_a, **_k: SimpleNamespace(
            admitted=False,
            reason="specialist_not_configured",
            certificate_sha256="",
            admitted_domains=(),
        ),
    )

    assert gate._local_deep_solver_block_reason() == "specialist_not_configured"


def test_primary_foreground_timeout_is_bounded_for_live_desktop_path():
    from interface.routes.chat import (
        _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S,
        _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
        _DESKTOP_COGNITIVE_TURN_TIMEOUT_S,
        _foreground_timeout_for_lane,
    )

    # A ready lane is bounded by min(max-turn ceiling, turn timeout + response
    # reserve), floored at 30s. Compute it from the live constants so tuning the
    # desktop budget can't silently break the bound this test guards. Cold lanes
    # retain a separate 210s outer bound for model startup.
    ready_bound = max(
        30.0,
        min(
            _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S,
            _DESKTOP_COGNITIVE_TURN_TIMEOUT_S + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
        ),
    )
    assert _foreground_timeout_for_lane({"conversation_ready": True, "state": "ready"}) == ready_bound
    assert _foreground_timeout_for_lane({"conversation_ready": False, "state": "warming"}) == 210.0
    assert _foreground_timeout_for_lane({"conversation_ready": False, "state": "unknown"}) == ready_bound


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
async def test_initiative_arbiter_quarantines_prompt_shaped_synthesis_goal():
    from core.agency.initiative_arbiter import InitiativeArbiter

    cognition = SimpleNamespace(
        pending_initiatives=[
            {
                "goal": (
                    "SUBCONSCIOUS SYNTHESIS Concept A: user memory Concept B: idle thought "
                    "Task: 1. Analyze these concepts. 2. If YES, produce a Universal Principle."
                ),
                "urgency": 1.0,
            },
            {"goal": "Investigate thermal surprise", "urgency": 0.4},
        ],
        working_memory=[],
    )
    state = SimpleNamespace(cognition=cognition, identity=SimpleNamespace(core_values=[]))

    selected = await InitiativeArbiter().arbitrate(state)

    assert selected is not None
    assert selected.initiative["goal"] == "Investigate thermal surprise"
    assert len(cognition.pending_initiatives) == 1
    assert cognition.pending_initiatives[0]["goal"] == "Investigate thermal surprise"


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
    from core.autonomy.proactive_presence import ProactivePresence

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


def test_stall_watchdog_rate_limits_boot_grace_suppression_logs(monkeypatch, caplog):
    import logging

    from core.resilience.stall_watchdog import StallWatchdog

    monkeypatch.setenv("AURA_WATCHDOG_BOOT_GRACE_S", "120")
    monkeypatch.setenv("AURA_WATCHDOG_SUPPRESSION_LOG_INTERVAL_S", "60")
    watchdog = StallWatchdog(SimpleNamespace(is_closed=lambda: False), threshold=1.0)
    watchdog._started_at = time.time()

    with caplog.at_level(logging.INFO, logger="Aura.Resilience.Watchdog"):
        assert watchdog._should_suppress_stall(15.0) is True
        assert watchdog._should_suppress_stall(15.0) is True

    assert sum("launch stall during boot grace" in record.message for record in caplog.records) == 1


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


def test_terminal_monitor_classifies_foreground_conversation_failures():
    from core.terminal_monitor import ErrorEntry, TerminalMonitor

    monitor = TerminalMonitor.__new__(TerminalMonitor)
    monitor._actionable_patterns = {
        r"Foreground conversation lane returned no text|conversation lane returned no text": "Investigate foreground conversation lane blank output",
        r"TimeoutError|timed out": "Investigate a timeout",
    }

    no_text = ErrorEntry(
        message="Foreground conversation lane returned no text",
        level="ERROR",
        source="Aura.Chat",
    )
    timed_out = ErrorEntry(
        message="conversation lane timed out before coherent reply",
        level="ERROR",
        source="Aura.Chat",
    )

    assert monitor._classify_error(no_text) == "Investigate foreground conversation lane blank output"
    assert monitor._classify_error(timed_out) == "Investigate a timeout"


def test_chat_live_proof_classifier_requires_explicit_proof_intent():
    from interface.routes.chat import _build_glass_arithmetic_reply, _classify_live_runtime_proof

    assert _classify_live_runtime_proof(
        "Create a simple game of Snake and save it as artifacts/live_runtime/generated/live_snake.html"
    ) is None
    assert _classify_live_runtime_proof(
        "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
    ) is None
    assert _classify_live_runtime_proof(
        "Run a live proof: create a simple game of Snake and save it as artifacts/live_runtime/generated/live_snake.html"
    ) == "snake"
    assert _classify_live_runtime_proof(
        "Live runtime proof: stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
    ) == "novel_topic"
    assert _classify_live_runtime_proof(
        "Run a live proof: open Calculator, copy the result into Notes, export a PDF, and move it into a folder."
    ) == "desktop"
    assert "14'" in _build_glass_arithmetic_reply(
        "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
    )


def test_chat_live_desktop_proof_has_no_specific_calculator_notes_chain():
    source = chat_lane_source()

    assert "_execute_desktop_chain_from_chat" not in source
    assert "tell application \"Calculator\"" not in source
    assert "live_proof_desktop_chain" not in source
    assert "chat.live_runtime_proof.desktop_task" in source


@pytest.mark.asyncio
async def test_api_chat_live_proof_receipt_survives_quality_repair(monkeypatch, tmp_path):
    from interface.routes import chat as chat_module

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "file_operation"
            assert context["origin"] == "user"
            assert context["user_requested_action"] is True
            assert context["agency_capability_token_id"] == "cap-token-test"
            target = tmp_path / params["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(params["content"])
            digest = hashlib.sha256(params["content"].encode("utf-8")).hexdigest()
            return {
                "ok": True,
                "path": params["path"],
                "context": context,
                "effect_verified": True,
                "expected_sha256": digest,
                "sha256": digest,
            }

    class FakeAgencyOrchestrator:
        async def run(self, proposal, *, perceive=None, simulate=None, execute=None, assess=None, **_kwargs):
            state = await perceive() if perceive else {}
            if simulate:
                await simulate(proposal, state)
            exec_result = await execute(proposal, state, "cap-token-test")
            outcome = await assess(proposal, state, exec_result) if assess else {"observed": exec_result}
            return SimpleNamespace(
                blocked_at=None,
                blocked_reason=None,
                proposal_id="AO-test-live-proof",
                will_receipt_id="will-test-live-proof",
                authority_receipt="authority-test-live-proof",
                execution_receipt=str(exec_result),
                outcome_assessment=outcome,
            )

    repair_calls = []

    async def fail_if_repaired(*_args, **_kwargs):
        repair_calls.append("called")
        raise AssertionError("verified live proof replies must not be replaced by quality repair")

    async def no_op_async(*_args, **_kwargs):
        return None

    monkeypatch.chdir(tmp_path)
    chat_module._locks.pop("fg", None)
    monkeypatch.setattr(chat_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status", lambda: {"state": "ready"})
    monkeypatch.setattr(chat_module, "_emit_chat_output_receipt", no_op_async)
    monkeypatch.setattr(_chat_preflight, "_log_exchange", no_op_async)
    monkeypatch.setattr(chat_module, "_repair_final_degraded_reply", fail_if_repaired)
    monkeypatch.setattr(
        chat_module.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                FakeCapabilityEngine()
                if name == "capability_engine"
                else FakeAgencyOrchestrator()
                if name == "agency_orchestrator"
                else default
            )
        ),
    )

    response = await chat_module.api_chat(
        chat_module.ChatRequest(
            message=(
                "Run a live proof: create a simple game of Snake and save it as "
                "artifacts/live_runtime/generated/live_snake.html"
            )
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


@pytest.mark.asyncio
async def test_chat_live_desktop_proof_routes_through_generic_desktop_task(monkeypatch):
    from interface.routes import chat as chat_module

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 6/6 governed computer-use steps.",
                "steps_requested": 6,
                "steps_completed": 6,
                "receipts": [
                    {
                        "index": index,
                        "action": "verified_desktop_step",
                        "critical": True,
                        "ok": True,
                        "effect_verified": True,
                        "effect_evidence": f"step={index};observable_effect=verified",
                    }
                    for index in range(1, 7)
                ],
            }

    class FakeAgencyOrchestrator:
        async def run(self, proposal, *, perceive=None, simulate=None, execute=None, assess=None, **_kwargs):
            state = await perceive() if perceive else {}
            if simulate:
                await simulate(proposal, state)
            exec_result = await execute(proposal, state, "cap-token-desktop")
            outcome = await assess(proposal, state, exec_result) if assess else {"observed": exec_result}
            return SimpleNamespace(
                blocked_at=None,
                blocked_reason=None,
                proposal_id="AO-test-desktop-proof",
                will_receipt_id="will-test-desktop-proof",
                authority_receipt="authority-test-desktop-proof",
                execution_receipt=str(exec_result),
                outcome_assessment=outcome,
            )

    monkeypatch.setattr(
        chat_module.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                FakeCapabilityEngine()
                if name == "capability_engine"
                else FakeAgencyOrchestrator()
                if name == "agency_orchestrator"
                else default
            )
        ),
    )

    result = await chat_module._execute_live_runtime_proof(
        "Run a live proof: open Calculator, copy the result into Notes, export a PDF, and move it into a folder."
    )

    assert result is not None
    assert result["status"] == "live_proof_desktop"
    assert result["data"]["desktop_task"]["ok"] is True
    assert [call["skill_name"] for call in calls] == ["desktop_task"]
    assert calls[0]["params"] == {
        "objective": (
            "Run a live proof: open Calculator, copy the result into Notes, export a PDF, and move it into a folder."
        ),
        "steps": [],
        "desktop_execution_contract": True,
        "allow_heuristic_desktop_plan": True,
        "foreground_request": True,
        "user_requested_action": True,
        "user_explicitly_authorized": True,
        "user_visible_desktop_action": True,
        "local_desktop_action": True,
        "verification_required": True,
    }
    assert calls[0]["context"]["route"] == "chat.live_runtime_proof.desktop_task"
    assert calls[0]["context"]["agency_capability_token_id"] == "cap-token-desktop"
    assert calls[0]["context"]["foreground_request"] is True
    assert calls[0]["context"]["desktop_execution_contract"] is True
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert "Completed 6/6" in result["response"] or "6/6 governed" in result["response"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("desktop_result", "expected_reason"),
    [
        (
            {
                "ok": True,
                "steps_requested": 2,
                "steps_completed": 2,
                "summary": "Completed 2/2 steps.",
            },
            "missing_step_receipts",
        ),
        (
            {
                "ok": True,
                "steps_requested": 1,
                "steps_completed": 1,
                "receipts": [
                    {
                        "ok": True,
                        "critical": True,
                        "effect_verified": False,
                        "effect_evidence": "executor returned",
                    }
                ],
            },
            "step_1_effect_unverified",
        ),
    ],
)
async def test_chat_live_desktop_proof_rejects_shallow_success(
    monkeypatch,
    desktop_result,
    expected_reason,
):
    from interface.routes import chat as chat_module

    async def fake_execute(*_args, **_kwargs):
        return desktop_result

    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", fake_execute)

    result = await chat_module._execute_live_runtime_proof(
        "Run a live proof: open Calculator and put the result in Notes."
    )

    assert result is not None
    assert result["status"] == "live_proof_failed"
    assert result["data"]["verification_reason"] == expected_reason
    assert "effects were not all verified" in result["response"]
    assert "I completed" not in result["response"]


@pytest.mark.asyncio
async def test_live_proof_file_rejects_unbound_success_even_when_file_exists(monkeypatch, tmp_path):
    from interface.routes import chat as chat_module

    async def shallow_write(_skill_name, params, **_kwargs):
        target = tmp_path / params["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params["content"], encoding="utf-8")
        return {"ok": True, "path": params["path"]}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", shallow_write)

    result = await chat_module._write_live_proof_file(
        "artifacts/live_runtime/generated/proof.txt",
        "verified payload",
        objective="Run a live proof.",
    )

    assert result["ok"] is False
    assert result["verification_failure"] == "effect_unverified"


@pytest.mark.asyncio
async def test_chained_live_proof_rejects_failed_observation(monkeypatch, tmp_path):
    from interface.routes import chat as chat_module

    target = tmp_path / "artifacts/live_runtime/generated/chain_note.txt"

    async def verified_write(_path, content, **_kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "absolute_path": str(target),
            "bytes": len(content.encode("utf-8")),
        }

    async def failed_observation(*_args, **_kwargs):
        return {"ok": False, "error": "computer_use unavailable"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_chat_runtime_proof, "_write_live_proof_file", verified_write)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", failed_observation)

    result = await chat_module._execute_live_runtime_proof("Run a chained live proof.")

    assert result is not None
    assert result["status"] == "live_proof_failed"
    assert result["data"]["verification_reason"] == "observation_result_not_ok"
    assert "I completed" not in result["response"]


@pytest.mark.asyncio
async def test_chained_live_proof_accepts_only_matching_pwd_observation(monkeypatch, tmp_path):
    from interface.routes import chat as chat_module

    target = tmp_path / "artifacts/live_runtime/generated/chain_note.txt"

    async def verified_write(_path, content, **_kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "absolute_path": str(target),
            "bytes": len(content.encode("utf-8")),
        }

    async def verified_observation(*_args, **_kwargs):
        return {"ok": True, "output": str(tmp_path.resolve()), "exit_code": 0}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_chat_runtime_proof, "_write_live_proof_file", verified_write)
    monkeypatch.setattr(_chat_capability_inventory, "_execute_governed_live_skill", verified_observation)

    result = await chat_module._execute_live_runtime_proof("Run a chained live proof.")

    assert result is not None
    assert result["status"] == "live_proof_chain"
    assert "I completed the chained live proof" in result["response"]


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
        _conversation_lane_user_message(
            {
                "state": "failed",
                "last_failure_reason": (
                    "memory_pressure_refused_worker_spawn:"
                    "projected_process_tree_rss:8.0GB+35.0GB+reserve3.0GB=46.0GB > limit 38.0GB"
                ),
            }
        ),
    ]

    for sample in samples:
        assert_no_live_reset_boilerplate(sample)

    assert "unified-memory guard" in samples[-1]
    assert "unsafe RAM spike" in samples[-1]


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
    assert len(calls) == 1
    skill, params, execution_context = calls[0]
    assert skill == "file_operation"
    assert params == {"action": "exists", "path": "README.md"}
    assert {
        "origin": "api",
        "route": "intent_router.route_execution",
        "foreground_request": True,
        "user_explicitly_authorized": True,
        "user_requested_action": True,
    }.items() <= execution_context.items()

    # `user_explicitly_authorized` is a caller-supplied boolean, and
    # BeingRuntime correctly ignores it unless a capability token bound to
    # tool_execution/foreground_desktop_action backs it. Nothing was minting
    # that token, so the assertion was dead and its refusal was logged on
    # every desktop turn. The router now asks the authority gateway — the
    # only issuer — to attest what it has already established.
    #
    # Asserted as a PROPERTY rather than pinning the dict exactly: the old
    # equality check made adding the token look like a regression when it is
    # the fix.
    assert execution_context.get("capability_token"), (
        "the router asserts user authority without obtaining the token that "
        "attests it, so the flag is ignored downstream"
    )


@pytest.mark.asyncio
async def test_intent_router_route_execution_merges_live_context_without_downgrading_foreground():
    from core.cognitive.router import IntentRouter

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context))
            return {"ok": True, "skill": skill_name, "params": params, "context": context}

    result = await IntentRouter().route_execution(
        "file_operation",
        {"action": "write", "path": "artifacts/live_runtime/generated/probe.txt", "content": "ok"},
        FakeCapabilityEngine(),
        context={
            "route": "desktop-ui.live_probe",
            "origin": "desktop_ui",
            "foreground_request": False,
            "user_requested_action": False,
        },
    )

    assert result["ok"] is True
    assert calls[0][2]["route"] == "desktop-ui.live_probe"
    assert calls[0][2]["origin"] == "desktop_ui"
    assert calls[0][2]["foreground_request"] is True
    assert calls[0][2]["user_explicitly_authorized"] is True
    assert calls[0][2]["user_requested_action"] is True


def test_legacy_interface_router_delegates_to_canonical_capability_path():

    from interface.router import IntentRouter

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context))
            return {"ok": True, "skill": skill_name, "params": params, "context": context}

    result = asyncio.run(
        IntentRouter().route_execution(
            "forge_skill",
            {"name": "diagnostic_skill"},
            FakeCapabilityEngine(),
        )
    )

    assert result["ok"] is True
    assert len(calls) == 1
    skill, params, execution_context = calls[0]
    assert skill == "forge_skill"
    assert params == {"name": "diagnostic_skill"}
    assert {
        "origin": "api",
        "route": "intent_router.route_execution",
        "foreground_request": True,
        "user_explicitly_authorized": True,
        "user_requested_action": True,
    }.items() <= execution_context.items()
    # See the note in the sibling test: the router now obtains the gateway
    # token that attests the authority flag it asserts.
    assert execution_context.get("capability_token")

    source = Path("interface/router.py").read_text(encoding="utf-8")
    assert "execute_skill_task.delay" not in source
    assert "CanonicalIntentRouter" in source


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
            calls.append(("telemetry", _payload, {}))

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
    assert [call[0] for call in calls[:2]] == ["coding_skill", "file_operation"]
    assert any(call[0] == "telemetry" for call in calls)
    assert_no_live_reset_boilerplate(reply)


def test_explicit_local_html_file_objective_builds_runnable_snake_artifact():
    from interface.routes.chat import _build_explicit_local_file_artifact

    html = _build_explicit_local_file_artifact(
        "Create a simple game of Snake and save it as artifacts/live_runtime/generated/live_snake.html",
        "artifacts/live_runtime/generated/live_snake.html",
    )

    assert html is not None
    assert "<canvas" in html.lower()
    assert "function tick" in html
    assert "addEventListener" in html
    assert "Score" in html
    assert "Aura Generated Page" not in html


@pytest.mark.asyncio
async def test_live_runtime_probe_accepts_generic_desktop_task_contract(monkeypatch):
    from tools.live_runtime_probe import LiveRuntimeProbe

    probe = LiveRuntimeProbe("http://127.0.0.1:8999")

    async def fake_chat(_message):
        return {
            "response": (
                "Desktop task completed 5/5 governed computer-use steps. "
                "Completed 5/5 governed desktop steps."
            ),
            "status": "desktop_objective_completed",
            "conversation_lane": {
                "governed_action_result": True,
                "governed_action_status": "desktop_objective_completed",
            },
            "data": {
                "desktop_result": {
                    "ok": True,
                    "steps_requested": 2,
                    "steps_completed": 2,
                    "receipts": [
                        {
                            "action": "create_folder",
                            "ok": True,
                            "effect_verified": True,
                        },
                        {
                            "action": "write_text_file",
                            "ok": True,
                            "effect_verified": True,
                        },
                    ],
                }
            },
        }

    monkeypatch.setattr(probe, "_chat", fake_chat)

    detail, data = await probe._regular_chat_desktop_chain()

    assert "generic governed desktop_task" in detail
    assert data["status"] == "desktop_objective_completed"
    assert data["conversation_lane"]["governed_action_result"] is True
    assert data["desktop_result"]["steps_completed"] == 2


@pytest.mark.asyncio
async def test_live_runtime_probe_writes_machine_readable_artifact(tmp_path):
    from tools.live_runtime_probe import LiveRuntimeProbe, ProbeResult

    artifact = tmp_path / "live_runtime_probe.json"
    probe = LiveRuntimeProbe(
        "http://127.0.0.1:8999",
        artifact_path=artifact,
    )
    probe.results = [
        ProbeResult(
            name="health",
            ok=True,
            detail="all required probes passed",
            elapsed_s=0.25,
            data={"required_probes": True},
        )
    ]
    probe.events = [{"type": "telemetry"}, {"type": "action_result"}]

    await probe._write_artifact(passed=True)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["passed"] is True
    assert payload["events_collected"] == 2
    assert payload["results"][0]["name"] == "health"
    assert payload["results"][0]["ok"] is True


def test_authority_readiness_fails_when_unity_blocks_consequential_action(monkeypatch):
    from core.executive.authority_gateway import AuthorityGateway

    gateway = object.__new__(AuthorityGateway)
    gateway._capabilities = SimpleNamespace(
        generate_token=lambda *_args, **_kwargs: "token",
        verify_access=lambda *_args, **_kwargs: True,
    )
    will = SimpleNamespace(decide=lambda *_args, **_kwargs: None)

    def fragmented_get(name, default=None):
        if name == "unified_will":
            return will
        if name == "unity_state":
            return SimpleNamespace(level="fragmented")
        if name == "unity_fragmentation_report":
            return SimpleNamespace(safe_to_act=False)
        return default

    monkeypatch.setattr(
        "core.executive.authority_gateway.ServiceContainer.get",
        staticmethod(fragmented_get),
    )

    assert gateway.is_ready() is False


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
async def test_computer_use_clock_returns_limited_payload_when_the_native_read_fails(
    monkeypatch,
):
    """Reading the clock asks no permission, so blocking one proves nothing.

    This patched _require_permissions, which the read_menu_clock branch does
    not call — it reads the system clock and falls back deterministically. The
    patch reached nothing and the native read simply succeeded, so the test
    measured a normal answer against a degraded contract. What is worth
    holding is the fallback: a failed native read still returns a time.
    """
    from core.skills.computer_use import ComputerUseSkill

    skill = ComputerUseSkill()

    def unavailable():
        raise OSError("no window server")

    monkeypatch.setattr(skill, "_read_menu_clock_macos", unavailable)
    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["clock_text"]
    assert result["source"] == "system_clock_fallback"


@pytest.mark.asyncio
async def test_computer_use_clock_needs_no_permission_when_it_works(monkeypatch):
    from core.skills.computer_use import ComputerUseSkill

    skill = ComputerUseSkill()

    async def refuse(*_args, **_kwargs):  # pragma: no cover - asserted unused
        raise AssertionError("reading the clock asked for a permission")

    monkeypatch.setattr(skill, "_require_permissions", refuse)
    monkeypatch.setattr(skill, "_read_menu_clock_macos", lambda: "Mon Aug 25 03:11")
    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["clock_text"] == "Mon Aug 25 03:11"
    assert result["source"] == "macos_system_clock"


def test_every_chokepoint_door_attaches_desktop_receipts():
    """Round-10 lesson, receipts edition: the desktop chokepoint guards
    every reply exit, and every exit that applies it must also attach the
    step receipts to the wire payload. Visible-demo round 3 failed because
    the kernel/deep door applied the chokepoint but dropped the receipts."""

    src = chat_lane_source()
    doors = src.count("await _apply_desktop_objective_chokepoint(")
    attachments = src.count('"desktop_result": _json_safe_payload')
    assert doors >= 2, f"expected both reply doors, found {doors}"
    assert attachments == doors, (
        f"{doors} chokepoint doors but {attachments} receipt attachments — "
        "a reply exit is dropping desktop receipts"
    )

def test_desktop_objective_execution_routes_through_tracked_gate():
    """Every desktop execution routes through the receipt-tracked gate.

    Generic desktop work now has one post-CognitiveEngine execution lane;
    the other tracked call is the universal reply chokepoint. Narrow proof
    and explicit-file lanes do not invoke the generic desktop executor.
    """

    src = chat_lane_source()
    # Module-qualified since the executor moved to its own lane; the guarantee
    # is the same one, and counting the bare name silently found nothing.
    direct_calls = src.count("_execute_desktop_objective_from_chat(") - src.count(
        "def _execute_desktop_objective_from_chat("
    )
    assert direct_calls == 1, (
        f"{direct_calls} direct executor calls — all desktop objective "
        "execution must go through _run_desktop_objective_tracked"
    )
    tracked_calls = src.count("await _run_desktop_objective_tracked(")
    assert tracked_calls >= 3, (
        f"expected the pre-cognitive self-sufficient lane, chokepoint, "
        f"and post-cognitive desktop lane on the tracked gate, "
        f"found {tracked_calls}"
    )


def test_self_sufficient_desktop_objectives_execute_before_freeform_generation():
    """Direct desktop work should not burn a foreground model call first.

    A visible OS action like opening an app, creating a folder, typing text,
    or setting wallpaper is already self-sufficient when desktop_task can
    derive a verified primitive plan. The live user path should execute that
    plan through governance immediately, then report receipts.
    """

    src = chat_lane_source()
    narrow = src.split(
        "async def _execute_narrow_desktop_objective_before_cognition", 1
    )[1].split("desktop_objective_response = await", 1)[0]
    assert "_desktop_objective_self_sufficient_without_cognitive_text" in narrow
    assert "await _run_desktop_objective_tracked(" in narrow


def test_desktop_self_sufficient_classifier_distinguishes_status_report_from_prose_report():
    from interface.routes.chat import (
        _desktop_objective_executable_after_cognitive_attempt,
        _desktop_objective_self_sufficient_without_cognitive_text,
    )

    assert _desktop_objective_self_sufficient_without_cognitive_text(
        "Use my computer to click a Calculator equation, copy the equation body, "
        "put it into Notes, produce a PDF, move that PDF into a Desktop proof folder, "
        "and report the paths."
    )
    assert _desktop_objective_self_sufficient_without_cognitive_text(
        "Please create a folder named 'Aura Live Proof' in my Documents folder "
        "and write a file inside it called live_proof.txt with one sentence about "
        "who you are and the current timestamp. Use your desktop tools and confirm "
        "exactly what you did."
    )
    assert _desktop_objective_self_sufficient_without_cognitive_text(
        'Open my Notes app and write a note saying "Hello :)"'
    )
    assert _desktop_objective_self_sufficient_without_cognitive_text(
        "Use /Users/bryan/Documents/blue_whale_wallpaper.jpg as my desktop wallpaper."
    )
    assert _desktop_objective_self_sufficient_without_cognitive_text(
        "Set system volume to 30%."
    )
    assert _desktop_objective_self_sufficient_without_cognitive_text(
        "Open my Notes app and write a note saying \u201cHello"
    )
    assert not _desktop_objective_self_sufficient_without_cognitive_text(
        "Open Notes and write a report about quantum mechanics."
    )
    # A SELF-SUMMARY is the exception, and it is measured rather than argued.
    #
    # This used to assert the opposite: that "describe who and what you are"
    # needs the cognitive draft, because the deterministic composer was
    # feared to produce filler. On this objective it does not — it composes
    # from live substrate facts, and the note it produced read:
    #
    #     I am Aura: a persistent digital organism — an orchestrated runtime
    #     (my substrate: perception, memory, affect, autonomy, self-repair,
    #     health monitoring) driving local language-model lanes (my voice) on
    #     this machine. ... The honest claim is functional rather than
    #     metaphysical.
    #
    # Routed through cognition instead, twice on 2026-07-28, it came back as
    # a capability denial that was then typed into the note by the hands it
    # denied: "I can describe myself, but I don't actually open apps or write
    # notes", and "I don't have UI control to open apps or write notes
    # directly — that's something you'd do with your hands on the keyboard."
    # Cognition does not know an actuator is about to run, so on precisely
    # this objective it writes a false sentence about the action in progress.
    #
    # A report about quantum mechanics in the same app is novel prose and
    # still needs the draft. What she is, is not novel prose.
    assert _desktop_objective_self_sufficient_without_cognitive_text(
        "Open Notes and describe who and what you are in your own words."
    )
    assert not _desktop_objective_self_sufficient_without_cognitive_text(
        "Research three climate articles, summarize them in Google Docs, and give your opinion."
    )
    assert _desktop_objective_executable_after_cognitive_attempt(
        "Open Notes and describe who and what you are in your own words."
    )
    assert _desktop_objective_executable_after_cognitive_attempt(
        "Research three climate articles, summarize them in Google Docs, and give your opinion."
    )


def test_goal_text_rejects_stale_or_prompt_scaffold_as_actionable():
    from core.goals.goal_text import first_actionable_goal_text, is_actionable_goal_text

    stale = (
        "Unresolved: Stalled goal: Please create a folder named 'Aura Live Proof' "
        "in my Documents folder and write a file inside it."
    )
    scaffold = (
        "SUBCONSCIOUS SYNTHESIS Concept A: User asked about development "
        "Concept B: desktop task Task: predict how self will react."
    )

    assert not is_actionable_goal_text(stale)
    assert not is_actionable_goal_text(scaffold)
    assert first_actionable_goal_text([{"goal": stale}, {"goal": "Research sleep and memory."}]) == (
        "Research sleep and memory."
    )


def test_initiative_synthesizer_purges_stale_persisted_tensions(tmp_path, monkeypatch):
    import json
    import time

    from core.initiative_synthesis import InitiativeSynthesizer

    path = tmp_path / "unresolved_tensions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "content": "Unresolved: Stalled goal: Please create a folder named Aura Live Proof.",
                    "source": "persisted",
                    "category": "stalled_goal",
                    "urgency": 0.9,
                    "created_at": time.time() - 3600,
                    "last_surfaced": 0,
                    "surface_count": 4,
                    "resolved": False,
                    "metadata": {},
                },
                {
                    "content": "Why is biological sleep linked to memory consolidation?",
                    "source": "conversation",
                    "category": "question",
                    "urgency": 0.5,
                    "created_at": time.time() - 3600,
                    "last_surfaced": 0,
                    "surface_count": 0,
                    "resolved": False,
                    "metadata": {},
                },
            ]
        ),
        encoding="utf-8",
    )

    synth = InitiativeSynthesizer()
    monkeypatch.setattr(synth, "_tension_path", lambda: path)

    synth._load_tensions()

    assert [t.content for t in synth._unresolved_tensions] == [
        "Why is biological sleep linked to memory consolidation?"
    ]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert len(persisted) == 1
    assert "biological sleep" in persisted[0]["content"]


def test_tension_engine_quarantines_stale_and_prunes_persisted_tensions(tmp_path):
    import json
    import time

    from core.agency.tension_engine import TensionEngine

    path = tmp_path / "tensions.json"
    now = time.time()
    rows = [
        {
            "id": "stale",
            "category": "broken_expectation",
            "description": "SUBCONSCIOUS SYNTHESIS Concept A: desktop Concept B: proof Task: retry.",
            "severity": 1.0,
            "created_at": now,
            "last_checked_at": now,
            "resolution_attempts": 0,
            "source_subsystem": "test",
            "related_beliefs": [],
            "related_goals": [],
            "resolved": False,
            "resolution": None,
        }
    ]
    for idx in range(2010):
        rows.append(
            {
                "id": f"valid-{idx}",
                "category": "open_question",
                "description": f"Open question {idx}",
                "severity": (idx % 100) / 100,
                "created_at": now - idx,
                "last_checked_at": now - idx,
                "resolution_attempts": 0,
                "source_subsystem": "test",
                "related_beliefs": [],
                "related_goals": [],
                "resolved": False,
                "resolution": None,
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")

    engine = TensionEngine(persist_path=path)

    active = engine.get_active_tensions()
    assert len(active) == 2000
    assert all(t.id != "stale" for t in active)
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2000


def test_desktop_access_permission_route_has_ui_bounded_probe_budgets():

    src = Path("interface/routes/system.py").read_text(encoding="utf-8")

    assert 'AURA_DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S"' in src
    assert "6.0" in src
    assert "AURA_DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S" in src
    # Direct-probe budget deliberately widened 0.6→2.0; still UI-bounded.
    assert "2.0" in src
    assert "timeout=max(0.2, _DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S)" in src
    assert "timeout=max(0.2, _DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S)" in src


def test_stall_watchdog_foreground_suppression_is_bounded(monkeypatch):
    """Jul 9 48%-wedge lesson: a lane perpetually 'warming' suppressed every
    stall dump for the exact window that mattered. Continuous foreground
    business past 300s stops suppressing — a wedge is not a warmup."""
    from types import SimpleNamespace

    from core.container import ServiceContainer
    from core.resilience.stall_watchdog import StallWatchdog

    monkeypatch.setenv("AURA_WATCHDOG_BOOT_GRACE_S", "0")
    monkeypatch.setenv("AURA_WATCHDOG_FOREGROUND_GRACE_S", "75")
    watchdog = StallWatchdog(SimpleNamespace(is_closed=lambda: False), threshold=1.0)
    watchdog._started_at = time.time() - 1000.0  # boot grace long past

    gate = SimpleNamespace(
        get_conversation_status=lambda: {"state": "warming", "warmup_in_flight": True}
    )

    monkeypatch.setattr(
        ServiceContainer, "get", classmethod(lambda cls, name, default=None: gate)
    )

    # Fresh business: suppressed (real warmups stay quiet).
    assert watchdog._should_suppress_stall(8.0) is True
    # Continuously busy past the deadline: suppression ends, dumps resume.
    watchdog._foreground_suppression_started_at = time.time() - 400.0
    assert watchdog._should_suppress_stall(8.0) is False
    # Lane goes quiet: anchor resets, next business gets a fresh window.
    gate.get_conversation_status = lambda: {"state": "ready"}
    watchdog._should_suppress_stall(8.0)
    assert watchdog._foreground_suppression_started_at == 0.0
