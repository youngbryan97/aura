"""The last step before the one canned reply that must never be reachable.

"I couldn't get to an answer I'd stand behind" is served when the worker's
quality gate rejects every draft. A fix added the reason to that log, because
every occurrence said a gate had said no and nothing said what it objected to —
and the reason is exactly what separates a model producing garbage from a gate
that is too strict, which want opposite fixes.

The fix read ``surface_quality_gate_reasons``. The worker writes its objections
under ``semantic_completion_quality_reasons``; the first key is only written on
the telemetry-sanitizer path. So the diagnosis added to end
"rejected_for=no_reasons_reported" reported no_reasons_reported, live, on
2026-08-28.
"""

from __future__ import annotations

from pathlib import Path

_GATE = Path("core/brain/inference_gate.py")


def test_the_refusal_reads_every_key_a_reason_is_kept_under() -> None:
    body = _GATE.read_text()
    start = body.index("_quality_reasons = tuple(")
    window = body[start : start + 1200]
    for key in (
        "surface_quality_gate_reasons",
        "semantic_completion_quality_reasons",
        "telemetry_sanitizer_reasons",
        "surface_quality_rejected_reasons",
    ):
        assert key in window, key


def test_the_worker_really_writes_the_key_the_gate_now_reads() -> None:
    """A reader with no writer is how this got here in the first place."""

    worker = Path("core/brain/llm/mlx_worker.py").read_text()
    assert '"semantic_completion_quality_reasons": quality_reasons' in worker
    assert 'surface_control_state["surface_quality_gate_reasons"]' in worker


def test_reasons_are_deduplicated_and_bounded() -> None:
    """The same objection under two names is one objection."""

    body = _GATE.read_text()
    start = body.index("_quality_reasons = tuple(")
    window = body[start : start + 700]
    assert "dict.fromkeys" in window
    assert "[:120]" in window


def test_when_there_is_no_reason_it_shows_the_draft() -> None:
    """Four keys hold reasons and a path populates none of them.

    A refusal that can name neither its objection nor what it objected to is
    the least diagnosable thing in the runtime, and it sits one step before the
    one canned reply that must never be reachable. The draft is already kept
    for the repair path; nothing was reading it here.
    """

    body = _GATE.read_text()
    assert "surface_quality_rejected_text" in body
    start = body.index("_rejected_draft = \"\"")
    window = body[start : start + 420]
    assert "if not _quality_reasons:" in window
    assert "[:220]" in window


def test_the_worker_keeps_the_draft_the_gate_now_reads() -> None:
    worker = Path("core/brain/llm/mlx_worker.py").read_text()
    assert 'state["surface_quality_rejected_text"] = body' in worker
    assert 'state["surface_quality_rejected_reasons"] = list(reasons)[:8]' in worker
