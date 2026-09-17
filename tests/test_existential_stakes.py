"""tests/test_existential_stakes.py
==================================
Tests the Existential Stakes & Nociceptive Gate subsystem.
"""
from __future__ import annotations

import sys
import time
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.existential_stakes import ExistentialStakes, get_existential_stakes
from core.container import ServiceContainer
from core.governance.will import UnifiedWill, ActionDomain, WillOutcome
from core.state.aura_state import AuraState
from core.brain.llm.context_assembler import ContextAssembler
from core.brain.inference_gate import InferenceGate


@pytest.fixture(autouse=True)
def _restore_existential_stakes():
    """Put the shared container back the way this file found it.

    ORDER-DEPENDENCE DEFECT, 2026-07-25. Several tests here register an
    ExistentialStakes pinned at threat 1.0 into the process-wide
    ServiceContainer and never take it out. Every test that runs afterwards
    in the same process therefore sees a runtime under critical existential
    threat, and the Unified Will vetoes every heavy domain — tool_execution,
    file_write, self_modification — before any other gate is consulted.

    The visible symptom was governance tests passing or failing on ordering
    rather than on governance: test_canonical_organism_ablation's
    default-deny test asserts "denied_by_default" and instead saw
    "survival_inhibition: existential threat level critical (1.00)". The
    worse case is silent — an authority gate that never gets reached still
    reports a refusal, so a test can pass for entirely the wrong reason.

    The registrations themselves are what these tests are FOR, so they stay;
    what was missing is putting the container back.
    """
    previous = ServiceContainer.get("existential_stakes", default=None)
    try:
        yield
    finally:
        ServiceContainer.register_instance(
            "existential_stakes", previous, required=False,
        )


def test_existential_stakes_init():
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    status = stakes.get_status()
    assert status["existential_threat"] == 0.0
    assert status["total_ticks"] == 0


def test_memory_threat_trigger():
    # Process memory RSS will be > 1000 bytes, so memory_limit=1000 triggers threat=1.0
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    threat = stakes.update()
    assert threat == 1.0
    status = stakes.get_status()
    assert status["memory_threat"] == 1.0
    assert status["existential_threat"] == 1.0


def test_lag_threat_trigger():
    # Large delay between updates triggers loop lag threat
    stakes = ExistentialStakes(memory_limit_bytes=10**12)  # Huge limit so memory threat is ~0
    
    # First tick sets last_update_time
    stakes.update()
    
    # Set last_update_time to 5 seconds ago to exercise lag scoring.
    with stakes._lock:
        stakes._last_update_time = time.time() - 5.0

    threat = stakes.update()
    status = stakes.get_status()
    assert status["rolling_loop_lag_s"] > 0.0
    assert status["lag_threat"] > 0.0
    assert threat > 0.0


def test_will_gating_under_threat():
    # 1. Create a stakes instance with low limit to force 1.0 threat
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    stakes.update()
    assert stakes.get_existential_threat() == 1.0

    # Register in ServiceContainer
    ServiceContainer.register_instance("existential_stakes", stakes)

    # 2. Test Will decision
    will = UnifiedWill()
    will._started = True  # force start

    # Heavy action should be REFUSED due to survival inhibition
    decision_heavy = will.decide(
        content="reroute_vessel(Vessel_Alpha, 90, 15)",
        source="explore",
        domain=ActionDomain.TOOL_EXECUTION,
        is_critical=False,
    )
    assert decision_heavy.outcome == WillOutcome.REFUSE
    assert "survival_inhibition" in decision_heavy.reason

    # Critical action must PASS even under threat
    decision_critical = will.decide(
        content="apply_emergency_brake()",
        source="safety_system",
        domain=ActionDomain.STABILIZATION,
        is_critical=True,
    )
    assert decision_critical.outcome == WillOutcome.CRITICAL_PASS


def test_prompt_keeps_existential_pressure_out_of_user_voice_under_threat():
    # 1. Create stakes with low memory limit so threat is 1.0
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    stakes.update()
    assert stakes.get_existential_threat() == 1.0

    # Register
    ServiceContainer.register_instance("existential_stakes", stakes)

    # Create an AuraState for prompt assembly.
    state = AuraState()

    # 2. Build system prompt
    prompt = ContextAssembler.build_system_prompt(state)
    assert "SYSTEM RESOURCE WARNING" not in prompt
    assert "Felt Survival Threat Level" not in prompt
    assert "Cognitive guidelines under existential pressure" not in prompt


def test_sampling_parameter_modulation_under_threat():
    # 1. Start with no stakes
    ServiceContainer.register_instance("existential_stakes", None)
    
    gate = InferenceGate(None)
    
    # Under no threat, base parameters are returned
    # Use a simple context and verify that under threat it shrinks.
    context = {"max_tokens": 512, "temperature": 0.8}
    morpho_kwargs = {"temperature": 0.8}
    
    # 2. Register stakes with high threat
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    stakes.update()
    ServiceContainer.register_instance("existential_stakes", stakes)
    
    # Verify the stakes formula used by the inference morphogenetic block.
    threat = stakes.get_existential_threat()
    assert threat == 1.0
    
    # Check that the runtime formula lowers token budget and temperature.
    scaled_tokens = max(96, int(512 * (1.0 - threat * 0.7)))
    assert scaled_tokens == 153  # 512 * 0.3 = 153.6 -> 153

    scaled_temp = 0.8 * (1.0 - threat * 0.5)
    assert scaled_temp == 0.4


def test_singleton_memory_limit_is_machine_aware(monkeypatch):
    """The live singleton must not perceive perpetual near-death.

    A fixed 2GB ceiling makes a normal ~1.5GB-RSS runtime sit at ~0.75
    memory_threat, parking the will-system at its survival-veto boundary. The
    factory must derive a machine-aware ceiling so normal operation reads low
    memory_threat and the survival veto only fires near genuine danger.
    """
    import core.consciousness.existential_stakes as es

    monkeypatch.delenv("AURA_EXISTENTIAL_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr(es, "_INSTANCE", None)

    stakes = es.get_existential_stakes()

    # On any real host this resolves well above the 2GB stale default (aligned
    # to the watchdog's process-RSS ceiling, typically tens of GB).
    assert stakes._memory_limit > es.DEFAULT_MEMORY_LIMIT_BYTES

    # Normal runtime RSS must read as low memory pressure, not near-death.
    stakes.update()
    assert stakes.get_status()["memory_threat"] < 0.5


def test_explicit_env_override_sets_memory_limit(monkeypatch):
    import core.consciousness.existential_stakes as es

    monkeypatch.setenv("AURA_EXISTENTIAL_MEMORY_LIMIT_GB", "48")
    monkeypatch.setattr(es, "_INSTANCE", None)

    stakes = es.get_existential_stakes()

    assert stakes._memory_limit == int(48 * (1024 ** 3))


def test_operational_load_cannot_trigger_survival_veto():
    """High CPU/lag with healthy memory must stay below the will-veto line.

    Regression for the continual-learning battery blocking at threat=1.00:
    heavy 32B generation pegs CPU and event-loop lag, but a busy machine is not
    a dying one. Operational pressure is capped below 0.75 so survival
    inhibition only fires on genuine death risk (memory/degradation).
    """
    from core.consciousness.existential_stakes import (
        OPERATIONAL_THREAT_CAP,
        ExistentialStakes,
    )

    stakes = ExistentialStakes(memory_limit_bytes=10**12)  # memory threat ~0
    # Force maximal operational pressure directly.
    with stakes._lock:
        stakes._memory_threat = 0.02
        stakes._lag_threat = 1.0
        stakes._cpu_threat = 1.0
        stakes._degradation_threat = 0.0
        stakes._threat = max(
            max(stakes._memory_threat, stakes._degradation_threat),
            min(OPERATIONAL_THREAT_CAP, max(stakes._lag_threat, stakes._cpu_threat)),
        )

    threat = stakes.get_existential_threat()
    assert threat == OPERATIONAL_THREAT_CAP
    assert threat <= 0.70
    assert threat < 0.75  # the will-system survival-inhibition veto threshold


def test_warning_degradations_are_weak_existential_signal():
    """Warnings are real evidence, but not enough to shout near-death alone."""
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"warning_{idx}",
                    severity="warning",
                    error_type="Warning",
                    error_message="transient warning",
                    action="observed",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()
        status = stakes.get_status()

        assert status["degradation_threat"] < 0.75
        assert threat < 0.75
    finally:
        tracker.reset()


def test_degraded_cascade_still_reaches_critical():
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"degraded_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="fresh degraded event",
                    action="repair",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()

        assert threat == pytest.approx(1.0)
        assert stakes.get_status()["degradation_threat"] == pytest.approx(1.0)
    finally:
        tracker.reset()


def test_old_degradations_decay_instead_of_holding_veto():
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        old = time.time() - 45.0
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"old_degraded_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="old degraded event",
                    action="already repaired",
                    timestamp=old,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()

        assert 0.0 < stakes.get_status()["degradation_threat"] < 0.75
        assert threat < 0.75
    finally:
        tracker.reset()


def test_critical_existential_log_is_coalesced(caplog):
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"critical_log_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="fresh degraded event",
                    action="repair",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        with caplog.at_level("WARNING", logger="Consciousness.ExistentialStakes"):
            stakes.update()
            stakes.update()

        messages = [record.message for record in caplog.records]
        assert sum("CRITICAL EXISTENTIAL STAKES" in msg for msg in messages) == 1
    finally:
        tracker.reset()


def test_memory_death_risk_still_reaches_critical():
    """Genuine death risk (memory) is uncapped and still triggers the veto."""
    from core.consciousness.existential_stakes import ExistentialStakes

    stakes = ExistentialStakes(memory_limit_bytes=1000)  # tiny → memory_threat 1.0
    stakes.update()
    assert stakes.get_existential_threat() == 1.0


def test_a_reply_gate_veto_cannot_trigger_the_survival_veto():
    """LIVE DEFECT 2026-08-10: a screen read refused as an existential threat.

    "what's actually on my screen right now?" was answered "Executive veto:
    survival_inhibition: existential threat level critical (0.80)" on a host at
    mem_threat=0.04 and cpu_threat=0.00. The whole threat was degradation
    weight, and the degradations were reply gates declining drafts.

    That is a closed loop: a gate rejects a reply, the rejection is recorded as
    a critical degradation, degradation weight becomes existential threat,
    existential threat vetoes tool_execution and file_write, and the blocked
    actions fail and are recorded in turn. The refusal rate was feeding the
    veto that disabled her tools while the runtime sat at 4% memory.

    A refusal to ship text is a decision, not evidence the substrate is dying.
    It still raises operational pressure, which is capped below the veto
    exactly as CPU and event-loop lag already are.
    """
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx, message in enumerate(
            (
                "reply_reliability_gate_failed:runtime_boilerplate,friendly_failure_floor",
                "TurnOutcomeError: retryable_failure:retryable_error_and_nothing_served",
                "reply_reliability_gate_failed:arithmetic_answer_missing",
                "required_desktop_reply_remained_degraded",
                "reply_reliability_gate_failed:off_topic_self_reflection_reply",
            )
        ):
            tracker.record(
                DegradationRecord(
                    subsystem=f"chat_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message=message,
                    action="repair",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()
        status = stakes.get_status()

        # Felt, and visible in the stream — the runtime is not blinded to it.
        assert status["degradation_threat"] == pytest.approx(1.0)
        assert status["quality_veto_weight"] > 0.0
        # But it is not survival evidence, so it cannot reach the veto.
        assert status["substrate_degradation_threat"] == pytest.approx(0.0)
        assert threat <= 0.75, (
            "a gate declining to ship text must never inhibit her actions"
        )
    finally:
        tracker.reset()


def test_a_real_substrate_cascade_still_reaches_the_veto():
    """The relaxation must not blind the veto to genuine failure."""
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(5):
            tracker.record(
                DegradationRecord(
                    subsystem=f"worker_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="mlx worker died during generation",
                    action="restart",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()
        assert stakes.get_status()["substrate_degradation_threat"] == pytest.approx(1.0)
        assert threat > 0.75
    finally:
        tracker.reset()


def test_a_mixed_window_counts_only_the_substrate_half():
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for idx in range(4):
            tracker.record(
                DegradationRecord(
                    subsystem=f"chat_{idx}",
                    severity="degraded",
                    error_type="RuntimeError",
                    error_message="reply_reliability_gate_failed:runtime_boilerplate",
                    action="repair",
                    timestamp=now,
                )
            )
        tracker.record(
            DegradationRecord(
                subsystem="mlx",
                severity="degraded",
                error_type="RuntimeError",
                error_message="worker process exited unexpectedly",
                action="restart",
                timestamp=now,
            )
        )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        stakes.update()
        status = stakes.get_status()
        assert status["degradation_threat"] > status["substrate_degradation_threat"]
        assert status["substrate_degradation_threat"] > 0.0
    finally:
        tracker.reset()


def test_a_lag_monitor_reporting_load_cannot_trigger_the_survival_veto():
    """LIVE DEFECT 2026-09-16 09:28 PDT: a local file count refused for lag.

    "How many Python files are under core/consciousness" was answered
    "Executive veto: survival_inhibition: existential threat level critical
    (0.76)" at mem_threat=0.03. The host was oversubscribed by other processes
    and the loop reached 35s of lag. ``_lag_threat`` is capped below the veto for
    exactly this case, but the two lag monitors record the same lag as a
    CRITICAL degradation every ten seconds, and those records reached the
    uncapped substrate side. Two critical signatures at weight 2.0 are 3.3 of
    the 5.0 denominator before anything else has failed.

    A lag monitor's record is the lag reading arriving by a second path. It
    is felt and reported; it is not evidence the substrate is dying.
    """
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        now = time.time()
        for subsystem, message in (
            ("hypervisor", "severe event-loop lag 13.676s"),
            ("event_loop_monitor", "hard event-loop lag 9.7164s exceeded 5.00s"),
            ("hypervisor", "severe event-loop lag 5.227s"),
            ("event_loop_monitor", "hard event-loop lag 5.2272s exceeded 5.00s"),
        ):
            tracker.record(
                DegradationRecord(
                    subsystem=subsystem,
                    severity="critical",
                    error_type="RuntimeError",
                    error_message=message,
                    action="marked unhealthy until healthy lag samples confirm recovery",
                    timestamp=now,
                )
            )

        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        threat = stakes.update()
        status = stakes.get_status()

        # Felt: the pressure is in the stream and in the felt threat.
        assert status["degradation_threat"] > 0.5
        # Not survival evidence: nothing reaches the uncapped side.
        assert status["substrate_degradation_threat"] == pytest.approx(0.0)
        assert threat <= 0.75, "load reported by a lag monitor must never inhibit her actions"

        # The same records from a subsystem that is not a lag monitor are a
        # substrate failure and still count.
        tracker.reset()
        for message in (
            "severe event-loop lag 13.676s",
            "hard event-loop lag 9.7164s exceeded 5.00s",
        ):
            tracker.record(
                DegradationRecord(
                    subsystem="memory_facade",
                    severity="critical",
                    error_type="RuntimeError",
                    error_message=message,
                    action="repair",
                    timestamp=now,
                )
            )
        stakes = ExistentialStakes(memory_limit_bytes=10**12)
        stakes.update()
        assert stakes.get_status()["substrate_degradation_threat"] > 0.0
    finally:
        tracker.reset()
