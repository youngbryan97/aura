"""A correction that reuses the loaded model is not a second model.

LIVE, 2026-08-20. A turn's first draft was wrong and its repair produced the
right answer. The repair never ran:

    Skipping CognitiveEngine desktop repair retry
    (process_tree_rss:32.8GB/40.0GB (level=warning)); live desktop turns stay
    bounded to one foreground generation by default.

``warning`` is a level, not an event. With a 32B resident the process sits
near its ceiling whenever the model is loaded at all, so the repair path was
disabled permanently on this machine. The completion retry had already been
carved out of the same veto — the same observation, made once.
"""

from __future__ import annotations

from pathlib import Path

CHAT = Path(__file__).resolve().parents[1] / "interface" / "routes" / "chat.py"


def _admission_body() -> str:
    source = CHAT.read_text(encoding="utf-8")
    body = source[source.index("def _desktop_secondary_model_repair_allowed") :]
    return body[: body.index("\ndef ", 10)]


def test_the_veto_is_about_allocating_not_about_a_level() -> None:
    body = _admission_body()
    assert 'bool(getattr(snapshot, "warning", False)) and not safe_same_worker_default' in body
    assert 'and not completion_retry' not in body


def test_running_out_of_memory_still_refuses() -> None:
    """The signal that means "do not run more work" is untouched."""
    body = _admission_body()
    assert 'getattr(snapshot, "refuse_heavy_local_generation", False)' in body


def test_repair_and_completion_are_both_same_worker_reasons() -> None:
    body = _admission_body()
    safe = body[body.index("safe_same_worker_reasons = {") :]
    safe = safe[: safe.index("}")]
    assert '"cognitive_engine_repair_retry"' in safe
    assert '"cognitive_engine_completion_retry"' in safe
    assert '"reliability_gate_failed"' in safe
