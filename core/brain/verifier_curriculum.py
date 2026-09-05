"""Curriculum loop — self-generated practice at the edge of competence (P4),
with an env-gated frontier-teacher lane (P5).
================================================================================
The last two phases of the frontier-general arc (docs/FRONTIER_GENERAL_ARC.md).

P4 — CURRICULUM: Aura proposes tasks for herself where they pay most —
weakness-targeted (battery classes where her measured score trails) and
playbook-adjacent (variations on shapes she has proven strategies for, one
step harder). Each proposal is solved by the system under training, verified
by the SAME truth engines as everything else, and verified wins compound
through the existing gated pipes (procedural playbooks now; weights only in
foundry-admitted domains). The Absolute-Zero pattern, with the admission
gate keeping unverifiable domains out of her weights.

P5 — TEACHER DISTILLATION (env-gated, default OFF): when the reach/cloud
lane is enabled, a frontier teacher answers HER curriculum — every teacher
answer passes the same verifier hard gate, only admitted domains mint
training traces, and provenance is stamped ``teacher`` so imported
capability is never mislabeled as self-generated. AURA_TEACHER_DISTILLATION
must be explicitly enabled.

The loop is bounded (fixed proposals per cycle), governed (a Will decision
gates each cycle), and honest (every cycle emits a report of proposed /
solved / verified / captured / refused).

STATUS: wired at boot, behind AURA_ENABLE_VERIFIER_CURRICULUM (default on).
--------------------------------------------------------------------------
``aura_main.py`` calls ``boot_verifier_curriculum()`` during startup, which
registers the ``verifier_curriculum`` ServiceContainer key; the loop below is
covered by tests/test_verifier_curriculum.py.

This block previously carried the unwired marker, and kept carrying it after
the boot call landed. (The marker string itself is deliberately not repeated
here: the gate that reads this file scans for it, and an explanatory mention
would read as the claim.) The claim outlived its truth because the
check that would have caught it only scanned ``core/`` and ``interface/`` —
and the boot entrypoint is at the repository root, so a capability wired ONLY
from boot looked uncalled. tests/test_capability_claims_have_call_sites.py now
scans aura_main.py too.

A capability claim needs a call site, not a class — and a claim that it has no
call site needs the same standard.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.VerifierCurriculum")

_TEACHER_FLAG = declare(
    "AURA_TEACHER_DISTILLATION", kind=FlagKind.BOOL, default=False,
    description="Allow the frontier-teacher lane to mint training traces",
    owner="core.brain.curriculum_loop",
)
_PROPOSALS_FLAG = declare(
    "AURA_CURRICULUM_PROPOSALS_PER_CYCLE", kind=FlagKind.INT, default=4,
    description="Curriculum tasks proposed per self-development cycle",
    owner="core.brain.curriculum_loop",
)


@dataclass(frozen=True)
class CurriculumTask:
    prompt: str
    task_type: str
    source: str            # "weakness" | "playbook_adjacent" | "explore"
    difficulty: float      # 0..1 (proposal-time estimate)


@dataclass
class CycleReport:
    proposed: int = 0
    solved: int = 0
    verified: int = 0
    captured: int = 0
    refused: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed, "solved": self.solved,
            "verified": self.verified, "captured": self.captured,
            "refused": self.refused, "by_source": dict(self.by_source),
            "notes": self.notes[:8],
        }


class VerifierCurriculumLoop:
    """Propose → solve → verify → compound, at the edge of competence."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._last_gap_report: dict[str, Any] | None = None

    # ── proposal ─────────────────────────────────────────────────────────
    def note_gap_report(self, report: dict[str, Any]) -> None:
        """Feed the latest frontier-gap run so weakness-targeting has data."""
        self._last_gap_report = dict(report or {})

    def _weak_classes(self) -> list[str]:
        report = self._last_gap_report or {}
        classes = report.get("classes", [])
        weak = [c["task_class"] for c in classes
                if float(c.get("gap", 0.0)) > 0.05]
        return weak or []

    def propose(self, k: int | None = None) -> list[CurriculumTask]:
        """Fresh tasks, biased toward measured weakness, then playbook
        adjacency, then exploration."""
        k = int(k if k is not None else _PROPOSALS_FLAG.value())
        tasks: list[CurriculumTask] = []
        weak = self._weak_classes()

        try:
            from core.brain.frontier_gap import build_battery

            pool = build_battery(seed=self._rng.randrange(10 ** 6),
                                 per_class=max(2, k))
        except (ImportError, RuntimeError, ValueError) as exc:
            record_degradation("curriculum_loop", exc, severity="warning",
                               action="battery templates unavailable; explore-only cycle")
            pool = []

        # 1. weakness-targeted: harder instances of trailing classes
        for item in pool:
            if len(tasks) >= k:
                break
            if item.task_class in weak:
                tasks.append(CurriculumTask(item.prompt, item.task_type,
                                            "weakness", 0.7))

        # 2. playbook-adjacent: shapes she has proven strategies for
        if len(tasks) < k:
            try:
                from core.brain.procedural_memory import get_procedural_memory

                status = get_procedural_memory().status()
                known_types = set(status.get("by_task_type", {}))
            except (ImportError, RuntimeError, AttributeError):
                known_types = set()
            for item in pool:
                if len(tasks) >= k:
                    break
                if item.task_type in known_types and item not in tasks:
                    tasks.append(CurriculumTask(item.prompt, item.task_type,
                                                "playbook_adjacent", 0.5))

        # 3. exploration fills the rest
        for item in pool:
            if len(tasks) >= k:
                break
            task = CurriculumTask(item.prompt, item.task_type, "explore", 0.4)
            if all(t.prompt != task.prompt for t in tasks):
                tasks.append(task)
        return tasks[:k]

    # ── governance ───────────────────────────────────────────────────────
    @staticmethod
    def _authorized() -> tuple[bool, str]:
        try:
            from core.governance.will import ActionDomain, get_will

            decision = get_will().decide(
                content="curriculum self-practice cycle: propose, solve, and "
                        "verify a bounded batch of self-generated tasks",
                source="curriculum_loop",
                domain=ActionDomain.REFLECTION,
                priority=0.2,
            )
            return decision.is_approved(), decision.reason
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("curriculum_loop", exc, severity="warning",
                               action="cycle skipped: the Will was unavailable")
            return False, f"will_unavailable:{type(exc).__name__}"

    # ── revision-gated deliberation (make extra passes safe) ─────────────
    async def solve_with_deliberation(
        self,
        prompt: str,
        task_type: str,
        solve: "Callable[[str, str], Awaitable[str]]",
        *,
        max_passes: int = 3,
    ) -> "tuple[str, Any]":
        """Solve one task over up to ``max_passes`` INDEPENDENT attempts,
        keeping an answer only when the monotonic revision gate says its
        verified evidence clearly beats the one in hand.

        This is the "make a second pass safe" primitive: a later attempt can
        only replace an earlier one on strictly stronger verifier evidence,
        so additional compute cannot regress a correct answer into a wrong
        one — the measured failure mode of naive self-correction. Verifier
        reliability is drawn from the Verifier Foundry so the confidence
        bounds reflect each engine's track record.
        """
        from core.brain.reasoning_revision_gate import deliberate_best_of

        async def _solve(_index: int) -> str:
            return str(await solve(prompt, task_type) or "")

        async def _verify(answer: str) -> Any:
            from core.brain.verifiers.registry import verify_candidate

            return await verify_candidate(answer, task_type=task_type)

        def _reliability(verdict: Any) -> float:
            try:
                from core.runtime.service_access import optional_service

                foundry = optional_service("verifier_foundry", default=None)
                engine = str(getattr(verdict, "engine", "") or "registry")
                if foundry is not None and hasattr(foundry, "weight_for"):
                    return float(foundry.weight_for(engine, task_type))
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                pass
            return 0.5

        result = await deliberate_best_of(
            _solve, _verify,
            max_passes=max_passes,
            reliability_of=_reliability,
            stop_when_verified=True,
        )
        return result.answer, result.verdict

    # ── the cycle ────────────────────────────────────────────────────────
    async def run_cycle(
        self,
        solve: Callable[[str, str], Awaitable[str]],
        *,
        k: int | None = None,
        capture: bool = True,
    ) -> CycleReport:
        """One bounded practice cycle through the real verification stack."""
        report = CycleReport()
        approved, reason = self._authorized()
        if not approved:
            report.refused = 1
            report.notes.append(f"cycle refused: {reason[:120]}")
            return report

        tasks = self.propose(k)
        report.proposed = len(tasks)
        for task in tasks:
            report.by_source[task.source] = report.by_source.get(task.source, 0) + 1
            try:
                answer = await solve(task.prompt, task.task_type)
            except (RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                record_degradation("curriculum_loop", exc, severity="debug",
                                   action="curriculum task solve failed")
                continue
            if not str(answer or "").strip():
                continue
            report.solved += 1

            try:
                from core.brain.verifiers.registry import verify_candidate

                verdict = await verify_candidate(answer, task_type=task.task_type)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("curriculum_loop", exc, severity="warning",
                                   action="curriculum verification failed; win not counted")
                continue
            if not (verdict.ok and verdict.checked):
                continue
            report.verified += 1

            if capture:
                captured = self._capture_win(task, answer, verdict.score)
                report.captured += 1 if captured else 0
        return report

    @staticmethod
    def _capture_win(task: CurriculumTask, answer: str, score: float) -> bool:
        """Compound through the existing gated pipes: playbooks always
        (collapse-free), training traces only via record_win's admission
        gate."""
        captured = False
        try:
            from core.brain.procedural_memory import get_procedural_memory

            get_procedural_memory().record_win(
                objective=task.prompt, task_type=task.task_type, answer=answer,
                strategy=f"curriculum/{task.source}", verifiers=["registry"],
                confidence=max(0.5, float(score)),
            )
            captured = True
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("curriculum_loop", exc, severity="warning",
                               action="playbook capture failed for curriculum win")
        try:
            from core.brain.reasoning_self_improvement import (
                get_reasoning_self_improvement,
            )

            get_reasoning_self_improvement().record_win(
                task.prompt, task.task_type, answer=answer,
                confidence=max(0.7, float(score)), mode="curriculum",
                verified=True,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("curriculum_loop", exc, severity="debug",
                               action="training-trace capture skipped")
        return captured

    # ── P5: the teacher lane (env-gated, provenance-stamped) ────────────
    async def run_teacher_cycle(
        self,
        teacher_generate: Callable[[str, str], Awaitable[str]],
        *,
        k: int | None = None,
    ) -> CycleReport:
        """A frontier teacher answers HER curriculum. Hard-gated three ways:
        the AURA_TEACHER_DISTILLATION flag, the verifier hard gate on every
        teacher answer, and the foundry admission gate on trace capture.
        Provenance is stamped 'teacher' — imported capability is never
        mislabeled as self-generated."""
        report = CycleReport()
        if not bool(_TEACHER_FLAG.value()):
            report.refused = 1
            report.notes.append("teacher lane disabled (AURA_TEACHER_DISTILLATION)")
            return report
        approved, reason = self._authorized()
        if not approved:
            report.refused = 1
            report.notes.append(f"cycle refused: {reason[:120]}")
            return report

        tasks = self.propose(k)
        report.proposed = len(tasks)
        for task in tasks:
            try:
                answer = await teacher_generate(task.prompt, task.task_type)
            except (RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                record_degradation("curriculum_loop", exc, severity="debug",
                                   action="teacher generation failed")
                continue
            if not str(answer or "").strip():
                continue
            report.solved += 1
            try:
                from core.brain.verifiers.registry import verify_candidate

                verdict = await verify_candidate(answer, task_type=task.task_type)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                continue
            if not (verdict.ok and verdict.checked):
                continue
            report.verified += 1
            try:
                from core.brain.reasoning_self_improvement import (
                    get_reasoning_self_improvement,
                )

                if get_reasoning_self_improvement().record_win(
                    task.prompt, task.task_type, answer=answer,
                    confidence=max(0.7, float(verdict.score)),
                    mode="teacher_distillation",   # provenance, never hidden
                    verified=True,
                ):
                    report.captured += 1
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("curriculum_loop", exc, severity="debug",
                                   action="teacher trace capture skipped")
        return report

    def status(self) -> dict[str, Any]:
        return {
            "teacher_lane_enabled": bool(_TEACHER_FLAG.value()),
            "proposals_per_cycle": int(_PROPOSALS_FLAG.value()),
            "weak_classes": self._weak_classes(),
            "has_gap_report": self._last_gap_report is not None,
        }

    def is_alive(self) -> bool:
        return True


_loop: VerifierCurriculumLoop | None = None


def get_verifier_curriculum() -> VerifierCurriculumLoop:
    global _loop
    if _loop is None:
        _loop = VerifierCurriculumLoop()
    return _loop


def boot_verifier_curriculum() -> VerifierCurriculumLoop:
    loop = get_verifier_curriculum()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("verifier_curriculum", loop, required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("curriculum_loop", exc, severity="warning",
                           action="curriculum loop built but not registered")
    return loop
