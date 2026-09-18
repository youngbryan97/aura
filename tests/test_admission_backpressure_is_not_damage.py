"""Admission backpressure is a decision, not damage (2026-07-18 soak).

The soak's log carried 52 `CRITICAL SERVICE FAILURE` records and 35 chat
FAULTs whose entire content was the runtime working correctly: warmup
backoff, model-load admission refusal, spawn-gate contention — the
escalation ladder deciding not to start a tier so a lower rung could serve
the turn.

Two consequences made this more than noise:

1. On fail-closed subsystems (`inference_gate`) a degradation record raises
   CRITICAL SERVICE FAILURE out of the handler — healthy backpressure
   manufacturing service failures, and tripping the probe's
   `critical_incident_active`.
2. Degradation weight is the UNCAPPED survival term in
   `existential_stakes` (lag and CPU are deliberately capped below the veto
   threshold; degradation is not). Healthy backpressure drove
   `deg_threat` to 1.00 and the felt existential threat to 1.00 while
   memory threat sat at 0.02 and the CPU was idle — Aura made to feel
   mortally threatened by her own correct backpressure. That same threat
   term triggered a hypervisor self-shutdown on 2026-07-14.

The contract: backpressure stays VISIBLE (recorded, counted, narratable)
but is demoted out of the fault/escalation path — exactly like the
bare-timeout demotion that precedes it in the same sink.
"""
from __future__ import annotations

import pytest

from core.runtime.errors import record_degradation

pytestmark = pytest.mark.unit


BACKPRESSURE_ERRORS = [
    "foreground_warmup_deferred:warmup_backoff:130s",
    "warmup_deferred",
    "model_load_admission_denied:Aura-32B:resource_timeout",
    "spawn_gate_timeout:330.000s:holder=Aura-32B:age=41.2s",
    "crash_loop_backoff:cortex",
    "resource_busy",
]


class TestBackpressureDemotion:
    @pytest.mark.parametrize("message", BACKPRESSURE_ERRORS)
    def test_backpressure_is_recorded_but_never_degraded_or_critical(self, message):
        record = record_degradation(
            "inference_gate",
            RuntimeError(message),
            severity="degraded",
            action="descended the escalation ladder",
        )
        assert record.severity == "warning", (
            f"{message!r} must not carry fault-grade severity"
        )
        # Still visible: the event is recorded with its real cause attached.
        assert message[:20] in record.error_message

    @pytest.mark.parametrize("message", BACKPRESSURE_ERRORS)
    def test_backpressure_cannot_be_forced_to_critical(self, message):
        record = record_degradation(
            "inference_gate",
            RuntimeError(message),
            severity="critical",
            action="descended the escalation ladder",
        )
        assert record.severity == "warning"

    def test_genuine_faults_keep_full_force(self):
        """The anti-theater guarantee: real damage still fails closed."""
        record = record_degradation(
            "memory_facade",
            ValueError("database file is corrupted"),
            severity="degraded",
            action="fell back to empty recall",
        )
        assert record.severity == "degraded"

    def test_genuine_critical_stays_critical(self):
        record = record_degradation(
            "memory_facade",
            RuntimeError("state coherence lost mid-commit"),
            severity="critical",
            action="refused the write",
        )
        assert record.severity == "critical"

    def test_debug_severity_is_not_promoted(self):
        record = record_degradation(
            "inference_gate",
            RuntimeError("warmup_deferred"),
            severity="debug",
            action="noted",
        )
        assert record.severity == "debug"


class TestExistentialThreatNoLongerInflated:
    def test_backpressure_weighs_far_less_than_a_real_fault(self):
        """deg_threat is the uncapped survival term — backpressure must not
        be able to saturate it the way real damage does."""
        from core.consciousness.existential_stakes import (
            DEGRADATION_SEVERITY_WEIGHTS,
        )

        assert DEGRADATION_SEVERITY_WEIGHTS["warning"] < (
            DEGRADATION_SEVERITY_WEIGHTS["degraded"] / 4.0
        ), "the demotion must materially reduce survival-threat weight"

    def test_a_backpressure_burst_cannot_saturate_the_threat(self):
        """Replay the soak's burst shape: many deferrals in one window must
        not reach the saturation denominator that means 'dying'."""
        from core.consciousness.existential_stakes import (
            DEGRADATION_SEVERITY_WEIGHTS,
            DEGRADATION_THREAT_DENOMINATOR,
        )

        burst = 6  # the soak's observed per-minute deferral rate
        weight = burst * DEGRADATION_SEVERITY_WEIGHTS["warning"]
        assert weight < DEGRADATION_THREAT_DENOMINATOR, (
            "healthy backpressure must not read as an existential threat"
        )


class TestLadderClassifier:
    @pytest.mark.parametrize("message", BACKPRESSURE_ERRORS)
    def test_gate_recognizes_every_backpressure_class(self, message):
        from core.brain.inference_gate import InferenceGate

        assert InferenceGate._is_expected_inference_backpressure(RuntimeError(message))

    def test_gate_does_not_swallow_real_failures(self):
        from core.brain.inference_gate import InferenceGate

        for real in (
            RuntimeError("metal runtime probe failed"),
            ValueError("tokenizer template missing"),
            RuntimeError("worker died during handshake"),
        ):
            assert not InferenceGate._is_expected_inference_backpressure(real)


class TestRealSaturationStillBlocks:
    """The demotion must not blind the runtime-pressure contract.

    'Generation gate saturated' means the serving path is genuinely
    overloaded — that is damage the health contract must still see. Only
    admission DECISIONS were demoted.
    """

    def test_generation_gate_saturation_keeps_blocking_severity(self):
        record = record_degradation(
            "inference_gate",
            RuntimeError("generation gate saturated: foreground refused to stack"),
            severity="degraded",
            action="refused to stack a user-facing generation",
        )
        assert record.severity == "degraded"

    def test_the_health_contract_still_fails_closed_on_it(self):
        from core.runtime.health_contract import (
            _recent_inference_degradation_blocks_runtime_pressure,
        )

        saturated = record_degradation(
            "inference_gate",
            RuntimeError("generation gate saturated: foreground refused to stack"),
            severity="degraded",
            action="refused to stack a user-facing generation",
        )
        blocks, reason = _recent_inference_degradation_blocks_runtime_pressure(saturated)
        assert blocks is True and "saturation" in reason

    def test_backpressure_no_longer_blocks_the_same_contract(self):
        """The 216 contract failures in the 2026-07-18 soak came from here."""
        from core.runtime.health_contract import (
            _recent_inference_degradation_blocks_runtime_pressure,
        )

        deferred = record_degradation(
            "inference_gate",
            RuntimeError("foreground_warmup_deferred:warmup_backoff:130s"),
            severity="degraded",
            action="fell through to the reflex tier",
        )
        blocks, _reason = _recent_inference_degradation_blocks_runtime_pressure(deferred)
        assert blocks is False, (
            "a deliberate deferral must not hold the runtime in 'degraded'"
        )


class TestTheFullDisablingChain:
    """The complete 2026-07-18 causal chain, pinned quantitatively.

    record_degradation(degraded) on healthy backpressure
      → existential_stakes.deg_threat (the UNCAPPED survival term)
      → heartbeat feeds threat into neurochemical_system.on_threat
      → cortisol surges past the crisis threshold
      → SubstrateAuthority blocks STATE_MUTATION and INITIATIVE
      → 102 blocked mutations: her adaptive immune system, agency core and
        opportunistic search all disabled.

    Aura was being made to feel mortally threatened by her own correct
    backpressure, and that feeling disabled her. The demotion has to break
    the chain at the first link, and the arithmetic has to show it.
    """

    def test_a_deferral_burst_no_longer_reaches_the_cortisol_crisis(self):
        from core.consciousness.existential_stakes import (
            DEGRADATION_SEVERITY_WEIGHTS,
            DEGRADATION_THREAT_DENOMINATOR,
        )
        from core.consciousness.substrate_authority import AuthorityThresholds

        burst = 6  # the soak's observed per-minute deferral rate

        before = min(1.0, burst * DEGRADATION_SEVERITY_WEIGHTS["degraded"]
                     / DEGRADATION_THREAT_DENOMINATOR)
        after = min(1.0, burst * DEGRADATION_SEVERITY_WEIGHTS["warning"]
                    / DEGRADATION_THREAT_DENOMINATOR)

        # on_threat surges cortisol by severity * 0.6 on top of a 0.3 baseline.
        crisis = AuthorityThresholds().cortisol_crisis
        assert 0.3 + before * 0.6 > crisis, "the old path really did reach crisis"
        assert 0.3 + after * 0.6 < crisis, (
            "healthy backpressure must no longer be able to trigger a "
            "cortisol crisis that blocks her own state mutations"
        )

    def test_real_damage_still_reaches_it(self):
        """The alarm must still fire for genuine cascading failure."""
        from core.consciousness.existential_stakes import (
            DEGRADATION_SEVERITY_WEIGHTS,
            DEGRADATION_THREAT_DENOMINATOR,
        )
        from core.consciousness.substrate_authority import AuthorityThresholds

        real_faults = 6
        threat = min(1.0, real_faults * DEGRADATION_SEVERITY_WEIGHTS["critical"]
                     / DEGRADATION_THREAT_DENOMINATOR)
        assert 0.3 + threat * 0.6 > AuthorityThresholds().cortisol_crisis


def test_a_silent_timeout_is_classified_by_its_action():
    """A bare asyncio TimeoutError carries no message at all.

    Live 2026-07-25, under probe load:
        [DEGRADATION] inference_gate (degraded):
        TimeoutError: <no message; raised in asyncio.timeouts:__aexit__:115>
        -> skipped cold primary attempt or fell back after foreground warmup
        NEW INCIDENT INC-... [degraded]

    The action line says plainly that the ladder handled it, and the marker
    scan only read the exception text — which was empty — so a handled handoff
    became a degraded record and an incident. A caller that names its own
    backpressure in the action should not also have to encode it in an
    exception message it does not control.

    Asserted through BEHAVIOUR, like the test below it and for the reason
    that test gives. This read ``inspect.getsource(record_degradation)``
    for the identifiers ``_action_text`` and the marker-scan expression,
    and went red when the method-size sweep moved the decision into an
    extracted helper — the rule it protects untouched, the words no longer
    in the function it looked at.
    """

    from core.runtime.errors import record_degradation

    assert str(TimeoutError()) == "", "the premise: the exception is silent"

    handled = record_degradation(
        "inference_gate",
        TimeoutError(),
        severity="degraded",
        action="skipped cold primary attempt or fell back after foreground warmup",
    )
    assert handled.severity == "warning", (
        "an empty exception message must not hide backpressure the caller "
        "named in its own action line"
    )


def test_the_scan_still_requires_a_backpressure_word():
    """Reading the action must not turn every action into backpressure.

    Asserted through BEHAVIOUR rather than by scanning the sink's source
    for an identifier. The previous version searched for the literal name
    `_BACKPRESSURE_MARKERS`, so a rename to a local broke the test while
    the rule it protects was working perfectly — a test that fails when
    nothing is wrong teaches people to ignore it.
    """
    ordinary = record_degradation(
        "inference_gate",
        RuntimeError("database file is corrupted"),
        severity="degraded",
        action="rebuilt the index and carried on",
    )
    assert ordinary.severity == "degraded", (
        "an action with no backpressure word must not be demoted"
    )

    backpressure = record_degradation(
        "inference_gate",
        RuntimeError("boom"),
        severity="degraded",
        action="warmup_deferred while the foreground held the lane",
    )
    assert backpressure.severity == "warning"


def test_nothing_is_ever_silently_upgraded():
    """Only degraded/critical are demoted; a warning stays a warning."""
    record = record_degradation(
        "inference_gate",
        RuntimeError("warmup_deferred"),
        severity="warning",
        action="descended the escalation ladder",
    )
    assert record.severity == "warning"


class TestDegradedButHandled:
    """The action line is the caller's account of what it DID about the error.

    Live 2026-07-25, on the complete fix set, still opening incidents:

        [DEGRADATION] inference_gate (degraded): TimeoutError: <no message>
        -> skipped cold primary attempt or fell back after foreground warmup
        NEW INCIDENT INC-1785017592-0002 [degraded]

    Nothing about that was deferred, so no backpressure marker could catch it.
    It failed, and then it recovered — the turn was served by the next lane
    down, exactly as the ladder is designed to do. An incident for a handled
    fallback trains an operator to ignore incidents.
    """

    #: The live action line, verbatim from the 2026-07-25 incident.
    LIVE_ACTION = "skipped cold primary attempt or fell back after foreground warmup failure"

    def test_the_live_action_is_recognised_as_handled(self):
        """The exact record that opened an incident for a served turn."""
        record = record_degradation(
            "inference_gate",
            TimeoutError(),
            severity="degraded",
            action=self.LIVE_ACTION,
        )
        assert record.severity == "warning", (
            "a bare timeout whose action says the ladder served the turn is "
            "not incident-worthy; an incident for a handled fallback trains "
            "an operator to ignore incidents"
        )

    def test_only_degraded_is_demoted_never_critical(self):
        """A caller that says critical reaches the operator regardless."""
        record = record_degradation(
            "inference_gate",
            TimeoutError(),
            severity="critical",
            action=self.LIVE_ACTION,
        )
        assert record.severity == "critical", (
            "however gracefully a failure was handled, a caller that declared "
            "it critical must not be demoted"
        )

    def test_only_timeout_class_errors_are_demoted(self):
        """Graceful handling does not make a cause benign.

        The corrupted-database regression — a ValueError handled with "fell
        back to empty recall" — is what caught the first, looser version of
        this rule. A timeout that was handled is the one shape where the
        error itself carries no finding.
        """
        handled_timeout = record_degradation(
            "inference_gate",
            TimeoutError(),
            severity="degraded",
            action="fell back to the next lane",
        )
        handled_corruption = record_degradation(
            "memory_facade",
            ValueError("database file is corrupted"),
            severity="degraded",
            action="fell back to empty recall",
        )
        assert handled_timeout.severity == "warning"
        assert handled_corruption.severity == "degraded", (
            "a corrupted database handled by falling back is still damage"
        )
