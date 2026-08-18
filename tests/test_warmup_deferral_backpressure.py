"""Regression contract: RAM-admission warmup deferrals are backpressure, not faults.

Live cascade (July 8): a foreground_warmup_deferred outcome was recorded as a
degradation on the fail-closed inference_gate; record_degradation escalates
warning+ records on fail-closed subsystems to CRITICAL SERVICE FAILURE and
RAISES — out of the very except-handler doing the recording — which failed the
protected recovery lane and surfaced to the user as chat 503s. The classifier
now separates expected backpressure (info log, reroute, NO record) from
genuine warmup faults (full degradation record).
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.unit

classify = InferenceGate._note_foreground_warmup_failure


class RecorderProbe:
    def __init__(self):
        self.calls = []

    def __call__(self, subsystem, error, **kwargs):
        self.calls.append((subsystem, str(error), kwargs))


@pytest.fixture
def gate():
    return InferenceGate.__new__(InferenceGate)  # classifier needs no gate state


def test_memory_deferral_is_backpressure_no_record(gate, monkeypatch):
    import core.brain.inference_gate as gate_mod

    probe = RecorderProbe()
    monkeypatch.setattr(gate_mod, "record_degradation", probe)
    deferred = gate._note_foreground_warmup_failure(
        RuntimeError("foreground_warmup_deferred:memory_pressure:69.2%/19.7GB")
    )
    assert deferred is True
    assert probe.calls == []          # no record → no fail-closed escalation


def test_genuine_warmup_fault_still_records(gate, monkeypatch):
    import core.brain.inference_gate as gate_mod

    probe = RecorderProbe()
    monkeypatch.setattr(gate_mod, "record_degradation", probe)
    deferred = gate._note_foreground_warmup_failure(
        RuntimeError("worker crashed during shader compile")
    )
    assert deferred is False
    assert len(probe.calls) == 1
    subsystem, error_text, kwargs = probe.calls[0]
    assert subsystem == "inference_gate"
    assert "shader compile" in error_text
    assert kwargs.get("severity") == "degraded"


def test_recording_a_deferral_on_fail_closed_gate_would_raise(monkeypatch):
    """The WHY of this contract: prove the cascade the classifier prevents."""
    from core.runtime.errors import record_degradation
    from core.runtime.mode import AuraMode

    import core.runtime.mode as mode_mod

    monkeypatch.setattr(mode_mod, "get_mode", lambda: AuraMode.PRODUCTION)
    import core.runtime.service_registry as registry_mod

    monkeypatch.setattr(
        registry_mod, "get_service_failure_policy", lambda name: "fail-closed"
    )
    # A GENUINE fault on the same fail-closed subsystem, to show the cascade
    # this contract exists to keep backpressure out of.
    #
    # This used to pass the warmup-deferral string itself and assert it
    # raised. That became unreachable the moment the classifier learned to
    # recognise it — which is the very behaviour the rest of this file
    # asserts — so the test was demanding the cascade it was written to
    # prevent, and failed on a clean tree. The counterfactual needs an error
    # the classifier does NOT consider backpressure, or it is asserting that
    # the fix does not work.
    with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
        record_degradation(
            "inference_gate",
            RuntimeError("gate state corrupted: null adapter in the response lane"),
            severity="degraded",
            action="test replay of the July 8 cascade",
        )

    # And the deferral string, on the same subsystem and severity, does not.
    # The pair is the contract: identical handling, opposite outcomes,
    # decided only by whether the condition is the machine being busy.
    record_degradation(
        "inference_gate",
        RuntimeError("foreground_warmup_deferred:memory_pressure:69%/19GB"),
        severity="degraded",
        action="backpressure must never escalate",
    )
