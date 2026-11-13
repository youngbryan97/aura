"""Verifier curriculum (frontier P4) + teacher lane (P5): bounded, governed,
verifier-gated compounding at the edge of competence. Distinct from
core.curriculum (belief-convergence): this loop resolves through the truth
engines and compounds only through the foundry-gated pipes."""
from __future__ import annotations

import asyncio

import pytest

import core.brain.verifier_curriculum as cl_mod
from core.brain.verifier_curriculum import VerifierCurriculumLoop

pytestmark = pytest.mark.unit


def _approve_will(monkeypatch, approved=True, reason="ok"):
    class _D:
        def __init__(self):
            self.reason = reason

        def is_approved(self):
            return approved

    import core.governance.will as will_mod

    monkeypatch.setattr(will_mod, "get_will",
                        lambda: type("W", (), {"decide": lambda self, **k: _D()})())


async def _perfect_solver(prompt: str, task_type: str) -> str:
    """A solver that answers as each task asks: the battery's math tasks want just the integer, and their own
    grader checks it (a worked equation is no check of the answer, core/brain/verifiers/math_engine.py)."""
    import re

    m = re.search(r"Compute (\d+) \* (\d+)", prompt)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return str(a * b)
    names = re.findall(r"([A-Z][a-z]+) is older than", prompt)
    if names and "oldest" in prompt.lower():
        return names[0]
    return "the answer is grounded and clear"


async def _null_solver(prompt: str, task_type: str) -> str:
    return ""


# ── proposal ─────────────────────────────────────────────────────────────────

def test_proposals_are_bounded_and_fresh():
    loop = VerifierCurriculumLoop()
    tasks = loop.propose(4)
    assert len(tasks) == 4
    assert len({t.prompt for t in tasks}) == 4


def test_weakness_targeting_uses_the_gap_report():
    loop = VerifierCurriculumLoop()
    loop.note_gap_report({"classes": [
        {"task_class": "math", "gap": 0.4},
        {"task_class": "factual", "gap": 0.0},
    ]})
    assert loop._weak_classes() == ["math"]
    tasks = loop.propose(4)
    weak = [t for t in tasks if t.source == "weakness"]
    assert weak and all(t.task_type == "math" for t in weak)


# ── the cycle ─────────────────────────────────────────────────────────────────

def test_cycle_refused_without_will_approval(monkeypatch):
    _approve_will(monkeypatch, approved=False, reason="stabilization first")
    loop = VerifierCurriculumLoop()
    report = asyncio.run(loop.run_cycle(_perfect_solver, k=2))
    assert report.refused == 1
    assert report.proposed == 0


def test_cycle_counts_verified_wins_only(monkeypatch):
    _approve_will(monkeypatch)
    loop = VerifierCurriculumLoop()
    loop.note_gap_report({"classes": [{"task_class": "math", "gap": 0.5}]})
    report = asyncio.run(loop.run_cycle(_perfect_solver, k=3, capture=False))
    assert report.proposed == 3
    assert report.verified >= 1
    assert report.captured == 0


def test_cycle_null_solver_verifies_nothing(monkeypatch):
    _approve_will(monkeypatch)
    loop = VerifierCurriculumLoop()
    report = asyncio.run(loop.run_cycle(_null_solver, k=3))
    assert report.solved == 0 and report.verified == 0


def test_cycle_capture_compounds_through_gated_pipes(monkeypatch, tmp_path):
    _approve_will(monkeypatch)
    from core.brain.procedural_memory import ProceduralMemory

    pm = ProceduralMemory(path=tmp_path / "pb.json")
    monkeypatch.setattr("core.brain.procedural_memory.get_procedural_memory",
                        lambda: pm)
    modes = []

    class _RSI:
        def record_win(self, *a, **k):
            modes.append(k.get("mode"))
            return True

    monkeypatch.setattr(
        "core.brain.reasoning_self_improvement.get_reasoning_self_improvement",
        lambda: _RSI())
    loop = VerifierCurriculumLoop()
    loop.note_gap_report({"classes": [{"task_class": "math", "gap": 0.5}]})
    report = asyncio.run(loop.run_cycle(_perfect_solver, k=3))
    assert report.captured >= 1
    assert pm.status()["playbooks"] >= 1
    assert all(m == "curriculum" for m in modes)


# ── P5 teacher lane ───────────────────────────────────────────────────────────

def test_teacher_lane_is_off_by_default(monkeypatch):
    _approve_will(monkeypatch)
    monkeypatch.delenv("AURA_TEACHER_DISTILLATION", raising=False)
    loop = VerifierCurriculumLoop()
    report = asyncio.run(loop.run_teacher_cycle(_perfect_solver, k=2))
    assert report.refused == 1
    assert "disabled" in report.notes[0]


def test_teacher_lane_verifies_and_stamps_provenance(monkeypatch):
    _approve_will(monkeypatch)
    monkeypatch.setenv("AURA_TEACHER_DISTILLATION", "1")
    stamped = []

    class _RSI:
        def record_win(self, *a, **k):
            stamped.append(k.get("mode"))
            return True

    monkeypatch.setattr(
        "core.brain.reasoning_self_improvement.get_reasoning_self_improvement",
        lambda: _RSI())
    loop = VerifierCurriculumLoop()
    loop.note_gap_report({"classes": [{"task_class": "math", "gap": 0.5}]})
    report = asyncio.run(loop.run_teacher_cycle(_perfect_solver, k=3))
    assert report.verified >= 1 and report.captured >= 1
    assert all(m == "teacher_distillation" for m in stamped)


def test_teacher_lane_hard_gates_bad_teacher_answers(monkeypatch):
    _approve_will(monkeypatch)
    monkeypatch.setenv("AURA_TEACHER_DISTILLATION", "1")

    async def bad_teacher(prompt, task_type):
        import re

        m = re.search(r"Compute (\d+) \* (\d+)", prompt)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return f"{a} * {b} = {a * b + 7}"   # checkable AND wrong
        return "confident nonsense"

    loop = VerifierCurriculumLoop()
    loop.note_gap_report({"classes": [{"task_class": "math", "gap": 0.5}]})
    report = asyncio.run(loop.run_teacher_cycle(bad_teacher, k=3))
    assert report.captured == 0, "wrong teacher answers must never mint traces"


# ── status / singleton ────────────────────────────────────────────────────────

def test_status_surface(monkeypatch):
    monkeypatch.delenv("AURA_TEACHER_DISTILLATION", raising=False)
    s = VerifierCurriculumLoop().status()
    assert s["teacher_lane_enabled"] is False
    assert s["proposals_per_cycle"] >= 1


def test_singleton_accessor():
    assert cl_mod.get_verifier_curriculum() is cl_mod.get_verifier_curriculum()


def test_curriculum_deliberation_never_regresses_a_verified_answer(monkeypatch):
    """solve_with_deliberation must keep a verified-correct first answer even
    when later independent attempts are worse — the anti-regression guarantee
    applied to the curriculum loop's own solver."""
    import asyncio
    from dataclasses import dataclass

    import core.brain.verifiers.registry as registry
    from core.brain.verifier_curriculum import VerifierCurriculumLoop

    @dataclass
    class _V:
        ok: bool
        checked: bool
        score: float
        engine: str = "registry"

    attempts = iter(["correct", "wrong", "wrong"])
    verdicts = {
        "correct": _V(True, True, 0.95),
        "wrong": _V(False, True, 0.1),
    }

    async def _solve(_prompt, _task_type):
        return next(attempts)

    async def _fake_verify(answer, *, task_type, context=None):
        return verdicts[answer]

    monkeypatch.setattr(registry, "verify_candidate", _fake_verify)

    loop = VerifierCurriculumLoop()
    answer, verdict = asyncio.run(
        loop.solve_with_deliberation("q", "math", _solve, max_passes=3)
    )

    assert answer == "correct"
    assert verdict.ok is True
