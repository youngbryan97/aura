"""The conductor: one autonomous loop instead of a hand-driven pipeline (CP243).

Every result this session was produced by a pipeline I hand-designed. That
is the opposite of the thing being aimed at, which is a loop that runs
itself for any query in any domain:

    identify gap -> acquire -> deliberate -> act -> verify -> learn -> retain

This module is that loop, as one reusable conductor. It does NOT hard-code a
domain or a pipeline; it calls the same organ seams every cycle, so the same
code handles a science question, a coding task, or an ordinary one. It is
the answer to "without you hand-designing a pipeline each time": the loop IS
the pipeline, and it is data-driven.

It is deliberately organ-agnostic. Every stage is a seam -- a protocol or a
callback -- so the conductor is testable without the live 32B and wireable
to the live application without change. What plugs in:

* gap detector   -> does the model already know this? (skip acquisition if so)
* producers      -> the workspace spine (retrieval, imagination, ...)
* deliberator    -> the model reasoning over the material, in words
* verifier       -> a programmatic or organ check of the candidate answer
* learner        -> turns a verified outcome into a retained training signal

Two properties are load-bearing, both learned the hard way this session:

* **Every stage degrades honestly.** A missing organ makes its stage report
  "unavailable" and the loop continues without it; nothing is ever
  fabricated to keep a cycle looking successful.
* **Self-correction is real, not decorative.** If verification fails, the
  loop re-acquires and re-deliberates up to a bounded budget, and every
  attempt is recorded. A loop that could not show its retries would be
  indistinguishable from one that never retried.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.runtime.errors import record_degradation

#: Stage failures are reported to the caller as a failed StageResult, which
#: is the loop's contract. They were NOT reaching the degradation ledger, so
#: a plugged-in organ that threw on every cycle was invisible to health and
#: incident tooling — the loop honestly reported "failed" to whoever asked,
#: and nobody was asking. CLAUDE.md states the rule: never a silent swallow.
def _record_stage_degradation(stage: str, exc: BaseException) -> None:
    record_degradation(
        "cognitive_loop",
        exc,
        severity="warning",
        action=f"reported {stage} stage failure and continued the loop",
        enforce_failure_policy=False,
    )


_StageFailure = Callable[[BaseException], "StageResult"]


def _stage_boundary(stage: str, exc: BaseException, on_failure: _StageFailure) -> "StageResult":
    """Turn a collaborator's failure into this loop's own vocabulary.

    Every stage of this loop calls out to an injected, Protocol-typed
    collaborator — a gap detector, a composer, a deliberator, a verifier, a
    learner. Their implementations are not this module's to know, so there is
    no exception set to narrow to: an injected component can raise anything,
    and the loop's job is to keep thinking when one of them does.

    So the breadth is deliberate, and it is stated once here rather than eight
    times inline. The failure is recorded, and the caller is handed a
    StageResult that says which stage failed and what it raised. Crucially,
    each caller's failure result preserves the loop's safety direction rather
    than its convenience: a failed gap check ASSUMES a gap, and a failed
    verification reports NOT verified. A boundary that converted failures into
    the permissive answer would be far worse than no boundary at all.
    """
    _record_stage_degradation(stage, exc)
    return on_failure(exc)


def _run_stage(
    stage: str,
    call: Callable[[], Any],
    on_failure: _StageFailure,
) -> tuple[Any, "StageResult | None"]:
    """Run one stage. Returns (value, None) or (None, failure StageResult)."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001 — see _stage_boundary
        return None, _stage_boundary(stage, exc, on_failure)


async def _run_stage_async(
    stage: str,
    call: Callable[[], Any],
    on_failure: _StageFailure,
) -> tuple[Any, "StageResult | None"]:
    """Async twin of _run_stage; awaits the collaborator if it returns a coroutine.

    The await is INSIDE the boundary on purpose. Awaiting outside it would
    catch only the failures raised before the first suspension point, which is
    the subset least likely to matter.
    """
    import inspect

    try:
        result = call()
        if inspect.isawaitable(result):
            result = await result
        return result, None
    except Exception as exc:  # noqa: BLE001 — see _stage_boundary
        return None, _stage_boundary(stage, exc, on_failure)

COGNITIVE_LOOP_SCHEMA = "aura.cognitive_loop.v1"


class GapDetector(Protocol):
    """Does the model already know the answer, or is knowledge missing?

    Returns True when a gap exists (acquisition is worth doing). The honest
    default when no detector is wired is 'assume a gap' -- acquiring
    knowledge you turned out to already have is cheap; skipping acquisition
    you needed is a wrong answer.
    """

    def has_gap(self, query: str) -> bool: ...


class Deliberator(Protocol):
    """The model reasoning, in words, over the query and gathered material.

    Returns a candidate answer string. This is the token-level deliberation
    the session found to WORK (versus silent latent looping, which did not).
    """

    def deliberate(self, query: str, material: list[str]) -> str: ...


class Verifier(Protocol):
    """A programmatic or organ check of a candidate answer.

    Returns a dict with at least ``{"correct": bool}`` and may include a
    bounded ``feedback`` or ``reason`` string for the next attempt. Reward
    comes from here and only here -- never from the model's own confidence,
    which would strengthen confident mistakes.
    """

    def check(self, query: str, candidate: str) -> dict[str, Any]: ...


@dataclass
class StageResult:
    name: str
    status: str  # "ok" | "unavailable" | "skipped" | "failed"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopResult:
    """Everything one cycle did, so it can be read rather than trusted."""

    query: str
    answer: str | None
    verified: bool
    attempts: int
    stages: list[StageResult]
    learned: bool

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": COGNITIVE_LOOP_SCHEMA,
            "query": self.query,
            "answer": self.answer,
            "verified": self.verified,
            "attempts": self.attempts,
            "learned": self.learned,
            "stages": [
                {"name": s.name, "status": s.status, **s.detail}
                for s in self.stages
            ],
        }


@dataclass
class CognitiveLoop:
    """One autonomous cycle of the full loop, wireable to live organs."""

    composer: Any = None            # WorkspaceComposer (the producer spine)
    deliberator: Deliberator | None = None
    verifier: Verifier | None = None
    gap_detector: GapDetector | None = None
    learner: Callable[[str, str, dict], bool] | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 16:
            raise ValueError("max_attempts must be inside [1, 16]")
        if self.deliberator is None:
            raise ValueError(
                "a loop with no deliberator cannot think; wire the model in"
            )

    def _identify_gap(self, query: str) -> tuple[bool, StageResult]:
        if self.gap_detector is None:
            # No detector -> assume a gap. Acquiring knowledge already held
            # is cheap; skipping acquisition you needed is a wrong answer.
            return True, StageResult("identify_gap", "unavailable",
                                     {"assumed_gap": True})
        gap, failure = _run_stage(
            "identify_gap",
            lambda: bool(self.gap_detector.has_gap(query)),
            # A detector that raised tells us nothing about whether the gap is
            # there, so assume it is — same direction as having no detector.
            lambda exc: StageResult(
                "identify_gap",
                "failed",
                {"assumed_gap": True, "error": type(exc).__name__},
            ),
        )
        if failure is not None:
            return True, failure
        return gap, StageResult("identify_gap", "ok", {"gap": gap})

    def _acquire(self, query: str, gap: bool) -> tuple[list[str], StageResult]:
        if not gap:
            return [], StageResult("acquire", "skipped", {"reason": "no_gap"})
        if self.composer is None:
            return [], StageResult("acquire", "unavailable", {})
        block, failure = _run_stage(
            "acquire",
            lambda: self.composer.compose(query),
            lambda exc: StageResult("acquire", "failed", {"error": type(exc).__name__}),
        )
        if failure is not None:
            return [], failure
        if not isinstance(block, dict):
            return [], StageResult(
                "acquire", "failed", {"error": "invalid_composer_result"}
            )
        raw_lines = block.get("lines", [])
        if not isinstance(raw_lines, list):
            return [], StageResult(
                "acquire", "failed", {"error": "invalid_material_lines"}
            )
        lines = [str(line).strip() for line in raw_lines if str(line).strip()]
        return lines, StageResult("acquire", "ok", {
            "material": len(lines),
            "grounded": block.get("grounded", 0),
            "hypothetical": block.get("hypothetical", 0),
        })

    def _deliberate(self, query: str, material: list[str]) -> tuple[str, StageResult]:
        answer, failure = _run_stage(
            "deliberate",
            lambda: self.deliberator.deliberate(query, material),
            lambda exc: StageResult(
                "deliberate", "failed", {"error": type(exc).__name__}
            ),
        )
        if failure is not None:
            return "", failure
        answer_text = str(answer or "").strip()
        return answer_text, StageResult(
            "deliberate",
            "ok" if answer_text else "failed",
            {"answered": bool(answer_text), **({"error": "empty_answer"} if not answer_text else {})},
        )

    def _verify(self, query: str, candidate: str) -> tuple[bool, StageResult]:
        if not candidate.strip():
            return False, StageResult(
                "verify", "skipped", {"verified": False, "reason": "no_answer"}
            )
        if self.verifier is None:
            # No verifier -> the answer is UNVERIFIED, never assumed correct.
            return False, StageResult("verify", "unavailable", {"verified": False})
        verdict, failure = _run_stage(
            "verify",
            lambda: self.verifier.check(query, candidate),
            # A verifier that raised has not verified anything. Reporting
            # anything but False here is how a system starts trusting answers
            # nothing checked.
            lambda exc: StageResult(
                "verify", "failed", {"verified": False, "error": type(exc).__name__}
            ),
        )
        if failure is not None:
            return False, failure
        if not isinstance(verdict, dict) or "correct" not in verdict:
            return False, StageResult(
                "verify",
                "failed",
                {"verified": False, "error": "invalid_verifier_result"},
            )
        ok = bool(verdict.get("correct"))
        detail: dict[str, Any] = {"verified": ok}
        for key in ("feedback", "reason"):
            value = str(verdict.get(key) or "").strip()
            if value:
                detail[key] = value[:500]
        return ok, StageResult("verify", "ok", detail)

    async def _adeliberate(self, query: str, material: list[str]) -> tuple[str, StageResult]:
        result, failure = await _run_stage_async(
            "deliberate",
            lambda: self.deliberator.deliberate(query, material),
            lambda exc: StageResult(
                "deliberate", "failed", {"error": type(exc).__name__}
            ),
        )
        if failure is not None:
            return "", failure
        answer_text = str(result or "").strip()
        return answer_text, StageResult(
            "deliberate",
            "ok" if answer_text else "failed",
            {"answered": bool(answer_text), **({"error": "empty_answer"} if not answer_text else {})},
        )

    async def _averify(self, query: str, candidate: str) -> tuple[bool, StageResult]:
        if not candidate.strip():
            return False, StageResult(
                "verify", "skipped", {"verified": False, "reason": "no_answer"}
            )
        if self.verifier is None:
            return False, StageResult("verify", "unavailable", {"verified": False})
        verdict, failure = await _run_stage_async(
            "verify",
            lambda: self.verifier.check(query, candidate),
            lambda exc: StageResult(
                "verify", "failed", {"verified": False, "error": type(exc).__name__}
            ),
        )
        if failure is not None:
            return False, failure
        if not isinstance(verdict, dict) or "correct" not in verdict:
            return False, StageResult(
                "verify",
                "failed",
                {"verified": False, "error": "invalid_verifier_result"},
            )
        ok = bool(verdict.get("correct"))
        detail: dict[str, Any] = {"verified": ok}
        for key in ("feedback", "reason"):
            value = str(verdict.get(key) or "").strip()
            if value:
                detail[key] = value[:500]
        return ok, StageResult("verify", "ok", detail)

    def _attempt_budget(self, stages: list[StageResult]) -> int:
        """Retries require a verifier capable of judging the changed answer."""
        if self.verifier is not None or self.max_attempts == 1:
            return self.max_attempts
        stages.append(
            StageResult(
                "retry_control",
                "skipped",
                {
                    "reason": "verifier_unavailable",
                    "configured_attempts": self.max_attempts,
                    "effective_attempts": 1,
                },
            )
        )
        return 1

    @staticmethod
    def _correction_material(
        candidate: str,
        verify_stage: StageResult,
    ) -> str:
        """Turn adjudicated failure into bounded next-attempt context."""
        feedback = str(
            verify_stage.detail.get("feedback")
            or verify_stage.detail.get("reason")
            or "The verifier rejected the prior candidate."
        ).strip()[:500]
        prior = str(candidate or "").strip()[:500]
        return (
            "[verifier correction] The prior candidate was rejected. "
            f"Feedback: {feedback} Prior candidate: {prior}"
        )

    async def arun(self, query: str) -> LoopResult:
        """Async cycle for live organs (the LLM router's generate is async).

        Mirrors ``run`` exactly -- same stages, same self-correction, same
        honest-degradation and never-retain-unverified rules -- but awaits
        the deliberator and verifier when they are coroutines. The sync
        ``run`` stays the tested reference; this is the live path.
        """
        if not str(query).strip():
            raise ValueError("query must be non-empty")
        stages: list[StageResult] = []
        gap, gap_stage = self._identify_gap(query)
        stages.append(gap_stage)
        answer: str | None = None
        verified = False
        attempts = 0
        correction_material = ""
        for attempt in range(self._attempt_budget(stages)):
            attempts = attempt + 1
            material, acq_stage = self._acquire(query, gap)
            if correction_material:
                material = [*material, correction_material]
            deliberated, del_stage = await self._adeliberate(query, material)
            verified, ver_stage = await self._averify(query, deliberated)
            for stage in (acq_stage, del_stage, ver_stage):
                stage.detail["attempt"] = attempts
                stages.append(stage)
            answer = deliberated or answer
            if verified:
                break
            gap = True
            if self.verifier is not None and deliberated:
                correction_material = self._correction_material(deliberated, ver_stage)
        learned = False
        if verified and self.learner is not None and answer is not None:
            outcome, failure = await _run_stage_async(
                "learn",
                lambda: self.learner(query, answer, {"verified": True}),
                lambda exc: StageResult(
                    "learn", "failed", {"retained": False, "error": type(exc).__name__}
                ),
            )
            if failure is not None:
                learned = False
                stages.append(failure)
            else:
                learned = bool(outcome)
                stages.append(StageResult("learn", "ok" if learned else "skipped",
                                          {"retained": learned}))
        elif self.learner is not None:
            stages.append(StageResult("learn", "skipped",
                                      {"reason": "unverified" if not verified else "no_answer"}))
        return LoopResult(query=query, answer=answer, verified=verified,
                          attempts=attempts, stages=stages, learned=learned)

    def run(self, query: str) -> LoopResult:
        """Run one cycle, with bounded self-correction on verification failure."""
        if not str(query).strip():
            raise ValueError("query must be non-empty")
        stages: list[StageResult] = []
        gap, gap_stage = self._identify_gap(query)
        stages.append(gap_stage)

        answer: str | None = None
        verified = False
        attempts = 0
        correction_material = ""
        for attempt in range(self._attempt_budget(stages)):
            attempts = attempt + 1
            material, acq_stage = self._acquire(query, gap)
            if correction_material:
                material = [*material, correction_material]
            deliberated, del_stage = self._deliberate(query, material)
            verified, ver_stage = self._verify(query, deliberated)
            # Tag each stage with the attempt so retries are legible, not
            # silently collapsed into one line.
            for stage in (acq_stage, del_stage, ver_stage):
                stage.detail["attempt"] = attempts
                stages.append(stage)
            answer = deliberated or answer
            if verified:
                break
            # Self-correction: a failed check means try again -- widening
            # acquisition next round -- until the budget is spent.
            gap = True
            if self.verifier is not None and deliberated:
                correction_material = self._correction_material(deliberated, ver_stage)

        # Learn: a verified outcome becomes a retained training signal. An
        # unverified one never does -- retaining unverified answers is how a
        # system trains on its own mistakes.
        learned = False
        if verified and self.learner is not None and answer is not None:
            outcome, failure = _run_stage(
                "learn",
                lambda: self.learner(query, answer, {"verified": True}),
                lambda exc: StageResult(
                    "learn", "failed", {"retained": False, "error": type(exc).__name__}
                ),
            )
            if failure is not None:
                learned = False
                stages.append(failure)
            else:
                learned = bool(outcome)
                stages.append(StageResult("learn", "ok" if learned else "skipped",
                                          {"retained": learned}))
        elif verified and self.learner is not None:
            stages.append(StageResult("learn", "skipped", {"reason": "no_answer"}))
        elif self.learner is not None:
            stages.append(StageResult("learn", "skipped",
                                      {"reason": "unverified"}))

        return LoopResult(
            query=query, answer=answer, verified=verified,
            attempts=attempts, stages=stages, learned=learned,
        )


def loop_health(results: list[LoopResult]) -> dict[str, Any]:
    """Is the loop actually working across a batch, or just running?

    Reports the numbers that decide whether the loop is real: how often it
    verified, how often self-correction rescued a first-attempt failure, and
    how often it retained a signal. A loop that never verifies is running,
    not working.
    """
    if not results:
        raise ValueError("no loop results to assess")
    verified = sum(1 for r in results if r.verified)
    rescued = sum(1 for r in results if r.verified and r.attempts > 1)
    learned = sum(1 for r in results if r.learned)
    return {
        "schema": COGNITIVE_LOOP_SCHEMA,
        "cycles": len(results),
        "verified_rate": round(verified / len(results), 4),
        "self_correction_rescues": rescued,
        "retained": learned,
        # The honest headline: a loop that verifies nothing is not a loop
        # that works, however smoothly it runs.
        "working": bool(verified > 0),
    }


__all__ = [
    "COGNITIVE_LOOP_SCHEMA",
    "CognitiveLoop",
    "Deliberator",
    "GapDetector",
    "LoopResult",
    "StageResult",
    "Verifier",
    "loop_health",
]
