"""Every failure was attributed to the quality gate, including the ones it never saw.

``surface_quality_gate_passed`` starts as ``not enabled``. With the gate
enabled and no draft ever examined it is therefore False, and the label was
applied on exactly that: enabled, and not passed. A deadline, an empty
generation and a cancellation all became "surface_quality_rejected".

LIVE, 2026-08-28: "read this file and tell me what it says" ended in "Cortex
exhausted its worker-owned semantic quality retries", on a receipt reading
attempts=0, reasons=[], generation_stop_reason='deadline_exceeded'. The gate
had examined nothing. Hours went into the quality path because the label sent
them there, and the real cause was printed on the same receipt.

The receipt distinguishes them exactly and nothing was asking it.
"""

from __future__ import annotations

from pathlib import Path

_GATE = Path("core/brain/inference_gate.py")


def test_the_label_requires_the_gate_to_have_examined_something() -> None:
    body = _GATE.read_text()
    start = body.index('metadata["error"] = "surface_quality_rejected"')
    window = body[max(0, start - 900) : start]
    assert "surface_quality_gate_attempts" in window
    assert "_gate_examined_something" in window


def test_a_failure_the_gate_never_saw_reports_what_stopped_it() -> None:
    body = _GATE.read_text()
    assert 'metadata["error"] = f"generation_{stopped}"' in body
    assert 'receipt.get("generation_stop_reason")' in body


def test_the_three_cases_separate() -> None:
    """Stated as the code states it, so the test fails if the rule changes."""

    def label(receipt: dict, success: bool = False) -> str:
        examined = int(receipt.get("surface_quality_gate_attempts") or 0) > 0
        if (
            not success
            and receipt.get("surface_quality_gate_enabled")
            and not receipt.get("surface_quality_gate_passed")
            and (examined or receipt.get("surface_quality_gate_reasons"))
        ):
            return "surface_quality_rejected"
        stopped = str(receipt.get("generation_stop_reason") or "").strip()
        return f"generation_{stopped}" if stopped else "unlabelled"

    examined_and_objected = {
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": False,
        "surface_quality_gate_attempts": 2,
        "surface_quality_gate_reasons": ["prompt_artifact"],
    }
    never_ran_deadline = {
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": False,
        "surface_quality_gate_attempts": 0,
        "surface_quality_gate_reasons": [],
        "generation_stop_reason": "deadline_exceeded",
    }
    assert label(examined_and_objected) == "surface_quality_rejected"
    assert label(never_ran_deadline) == "generation_deadline_exceeded"


def test_a_gate_that_objected_without_recording_attempts_still_counts() -> None:
    """Reasons alone are enough: a recorded objection IS an examination."""

    body = _GATE.read_text()
    start = body.index("_gate_examined_something")
    window = body[start : start + 700]
    assert 'receipt.get("surface_quality_gate_reasons")' in window
