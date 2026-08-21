"""A capability that cannot run on this host is not offered on it.

LIVE, 2026-08-21. build_app was offered on every build request. It generates
with the local code model, which wants 21.5GB beside a resident 25.3GB cortex
and is refused here:

    ModelLaneControlError: in_process_model_admission_refused:
    lane_budget_exceeded:cortex request 21.5GB + committed 25.3GB > budget

It spent forty to seventy seconds of each turn and failed every time, while
the plain turn wrote the same page in thirty. This is the rule the deep
solver lane already follows: a lane that cannot load is a hole, not a
fallback.
"""

from __future__ import annotations

from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "core" / "capability_engine.py"


def test_the_registry_asks_before_offering() -> None:
    source = ENGINE.read_text(encoding="utf-8")
    body = source[source.index("def _tool_definition_for_skill") :]
    body = body[: body.index("\n    def ", 10)]
    assert 'getattr(target, "available_here", None)' in body
    assert "not offered: it reports it cannot run on this host" in body


def test_a_skill_that_cannot_answer_is_still_offered() -> None:
    """Silence is not a refusal: only an explicit False withholds a skill."""
    source = ENGINE.read_text(encoding="utf-8")
    body = source[source.index('getattr(target, "available_here", None)') :]
    body = body[: body.index("return {")]
    assert "if available() is False:" in body


def test_build_app_reports_from_the_model_s_own_receipt() -> None:
    """The accessor hands back a lazy handle whether or not the model can ever
    load, so asking whether it is None answered True on a host that refuses
    it."""
    from core.skills.build_app import BuildAppSkill

    assert callable(BuildAppSkill.available_here)
    source = Path("core/skills/build_app.py").read_text(encoding="utf-8")
    assert "model.readiness()" in source
    assert "ReadinessState.ABSENT" in source
    assert "ReadinessState.FAILED" in source


def test_an_admission_refusal_marks_the_model_failed() -> None:
    """Which is what makes the check self-correcting: the first refusal is
    recorded, and the skill stops being offered after it."""
    from core.runtime.model_lane_control import ModelLaneControlError

    assert issubclass(ModelLaneControlError, RuntimeError)
    source = Path("core/brain/llm/local_code_model.py").read_text(encoding="utf-8")
    probe = source[source.index("self._set_readiness(ReadinessState.READY") :]
    probe = probe[: probe.index("return self.readiness()")]
    assert "OSError, RuntimeError, ValueError, TimeoutError" in probe
    assert "ReadinessState.FAILED" in probe
