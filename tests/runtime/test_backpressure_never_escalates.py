"""Backpressure is the system working, not a critical service failure.

LIVE, 2026-08-13, on every boot of the desktop:

    FAULT RUNTIME-INFERENCE_GATE [CRITICAL] in inference_gate:
    RuntimeError: warmup_deferred
    RuntimeError: CRITICAL SERVICE FAILURE: Subsystem 'inference_gate' failed
    with failure policy 'fail-closed'. Original error: RuntimeError:
    warmup_deferred
    🚨 [STABILITY v53] Background task 'InferenceGate.deferred_cortex_prewarm'
    crashed

"warmup_deferred" was already in backpressure_markers, and that demotes it from
degraded to WARNING. The fail-closed escalation then accepted warning and
raised it back to critical, so the demotion bought nothing and a lane saying
"not warm yet, try later" was recorded as a subsystem failing closed.

errors.py says why this matters sixty lines above the bug: degradation weight is
the uncapped survival term in existential_stakes, so healthy backpressure drove
felt existential threat to 1.00 with an idle CPU. Timeouts were already exempt
for exactly this reason; backpressure was not.
"""

from __future__ import annotations

import pytest

from core.runtime.errors import record_degradation, recent_degradations


_ESCALATION = "CRITICAL SERVICE FAILURE"


def _last_record(subsystem: str) -> dict:
    rows = [
        row
        for row in recent_degradations(limit=50)
        if str(row.get("subsystem") or "").startswith(subsystem)
    ]
    assert rows, f"no degradation recorded for {subsystem}"
    return rows[-1]


@pytest.mark.parametrize(
    "marker",
    [
        "warmup_deferred",
        "warmup_backoff",
        "admission_deferred",
        "resource_busy",
        "model_load_admission_denied",
        "crash_loop_backoff",
        "chat_dependencies_warming",
    ],
)
def test_backpressure_records_as_warning_and_does_not_escalate(marker: str) -> None:
    record_degradation("inference_gate", RuntimeError(marker), action="deferred")

    record = _last_record("inference_gate")

    assert record.get("severity") == "warning"
    assert _ESCALATION not in str(record)


def test_a_real_fault_on_the_same_subsystem_still_escalates() -> None:
    """The exemption is for backpressure, not for the subsystem."""
    record_degradation(
        "inference_gate",
        RuntimeError("adapter weights are corrupt"),
        action="failed to load",
    )

    record = _last_record("inference_gate")

    assert record.get("severity") != "warning"


def test_the_exemption_is_read_from_the_shared_marker_list() -> None:
    """One definition. Two lists of what counts as backpressure is how this
    bug existed at all — the demotion knew and the escalation did not.

    Asked of the behaviour rather than of the source. This read
    ``inspect.getsource(record_degradation)`` for the text
    ``not _is_admission_backpressure``, and went red when the method-size
    sweep moved the decision into an extracted helper — the invariant
    intact, the grep target gone. A test that reads source for a call site
    fails on every refactor that does not change behaviour, and passes on
    every change that keeps the words and breaks the meaning.

    What it has to hold is that ONE list decides both: every marker the
    demotion honours is a marker the escalation also exempts.
    """

    from core.runtime.errors import backpressure_markers

    markers = list(backpressure_markers())
    assert markers, "the shared marker list is empty; nothing is being demoted"

    for marker in markers:
        record_degradation(
            "inference_gate", RuntimeError(marker), action="deferred"
        )
        record = _last_record("inference_gate")
        assert record.get("severity") == "warning", (
            f"{marker!r} is in the shared list but was not demoted"
        )
        assert _ESCALATION not in str(record), (
            f"{marker!r} was demoted and then escalated anyway, which is the "
            "bug this file exists for"
        )
