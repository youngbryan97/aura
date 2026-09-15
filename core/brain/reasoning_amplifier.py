"""Inference-time reasoning amplification — verifier-filtered self-consistency.

Aura already samples multiple reasoning paths (the CONSISTENCY strategy) and is wired
into the cognitive engine. What was missing is the frontier piece that actually lifts a
base model's accuracy: **filter out provably-wrong paths before voting**. Today's
reasoning models work this way — many samples, self-verification, and convergence — not
a bigger network.

``amplify`` is the reusable core: given a set of candidate reasoning paths it (1) asks
Aura's own deduction engine whether each path contains a provable non-sequitur or
arithmetic error, (2) keeps the verifier-clean paths, and (3) takes the answer the most
of them converge on (self-consistency). The confidence is the real agreement fraction,
lifted when the winning cluster is verifier-clean. This is pure over already-generated
candidates, so it is deterministically testable; ``DeliberationEngine`` wraps it with
parallel sampling for standalone use.

It is wired into ``reasoning_strategies._consistency`` so it runs on the hard/factual
turns the classifier already routes there — i.e. it is causal, not a shelf ornament.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningAmplifier")

_ANSWER_RE = re.compile(r"(?:final answer|answer|therefore)\s*[:\-]?\s*(.+)$", re.IGNORECASE)


def default_extract_answer(text: str) -> str:
    """Pull the conclusion out of a reasoning trace (last 'answer:' or last line)."""
    t = str(text or "").strip()
    matches = _ANSWER_RE.findall(t)
    if matches:
        return matches[-1].strip().rstrip(".")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[-1] if lines else t).rstrip(".")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower()).strip(" .!?\"'")


class VerifierOutcome(StrEnum):
    """What the verifier established. UNKNOWN is not a pass.

    The distinction this enum exists to force: "I checked and found nothing
    wrong" and "I could not check" are different facts, and a bool cannot hold
    both. Collapsing them is how a crashed verifier came to certify every
    candidate as clean.
    """

    PASS = "pass"        # checked; no provable error found
    FAIL = "fail"        # checked; a provable error was found
    UNKNOWN = "unknown"  # could not check — establishes nothing


async def verify_reasoning_checked(text: str) -> tuple[VerifierOutcome, list[str]]:
    """Audit reasoning for provable non-sequiturs and arithmetic errors.

    Returns UNKNOWN — never PASS — when the verifier itself cannot run.
    """
    try:
        from core.reasoning.symbolic_bridge import SymbolicBridge

        audit = SymbolicBridge().audit_reasoning(text)
        issues: list[str] = []
        for ns in audit.get("non_sequiturs", []) or []:
            issues.append(f"non-sequitur: {ns.get('conclusion')}")
        for ae in audit.get("arithmetic_errors", []) or []:
            issues.append(f"arithmetic: {ae.get('claim')}")
        return (VerifierOutcome.PASS if not issues else VerifierOutcome.FAIL), issues
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        # A verifier that crashed has verified nothing. This used to
        # `return True, []` — reporting the crash as a clean bill of health,
        # which marked every candidate verifier-clean and set verified=True on
        # the result. Absence of a check is not a passed check.
        record_degradation(
            "reasoning_amplifier",
            exc,
            action="verifier unavailable; reasoning reported as UNCHECKED, not verified",
            enforce_failure_policy=False,
        )
        return VerifierOutcome.UNKNOWN, [f"verifier_unavailable: {type(exc).__name__}"]


async def verify_reasoning(text: str) -> tuple[bool, list[str]]:
    """Back-compatible boolean form.

    UNKNOWN maps to False: if we could not check, we must not claim the text is
    clean. Callers that need to tell "unchecked" from "checked and failed"
    should use :func:`verify_reasoning_checked`.
    """
    outcome, issues = await verify_reasoning_checked(text)
    return outcome is VerifierOutcome.PASS, issues


async def _run_verifier(
    verifier: Callable[[str], Awaitable[Any]], candidate: str
) -> tuple[VerifierOutcome, list[str]]:
    """Normalize any verifier — custom or default — into a tri-state outcome.

    Custom verifiers supplied by callers return ``(bool, list[str])``. A crash
    inside one is UNKNOWN here too, rather than propagating (which would take
    down the whole amplification) or being swallowed as a pass.
    """
    try:
        result = await verifier(candidate)
    except (RuntimeError, AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
        record_degradation(
            "reasoning_amplifier",
            exc,
            action="candidate verifier crashed; candidate treated as UNCHECKED",
            enforce_failure_policy=False,
        )
        return VerifierOutcome.UNKNOWN, [f"verifier_crashed: {type(exc).__name__}"]

    if isinstance(result, tuple) and len(result) == 2:
        ok, issues = result
        issue_list = [str(i) for i in (issues or [])]
        if isinstance(ok, VerifierOutcome):
            return ok, issue_list
        return (VerifierOutcome.PASS if ok else VerifierOutcome.FAIL), issue_list

    # An unrecognized return shape establishes nothing.
    return VerifierOutcome.UNKNOWN, [f"verifier_bad_return: {type(result).__name__}"]


@dataclass
class AmplifiedResult:
    answer: str
    confidence: float
    n: int
    valid_n: int
    agreement: float
    verified: bool = False
    issues: list[str] = field(default_factory=list)
    # Did the verifier actually run on anything? False means every candidate
    # came back UNKNOWN, so `verified` and `valid_n` carry no information and
    # the confidence below is unlifted. Without this a caller cannot tell
    # "nothing was wrong" from "nothing was checked".
    verifier_checked: bool = True
    # Whether the selected answer passed, failed, or could not be checked.
    # Aggregate verifier activity does not identify the selected answer.
    answer_verdict: VerifierOutcome = VerifierOutcome.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer[:200],
            "confidence": round(self.confidence, 3),
            "n": self.n,
            "valid_n": self.valid_n,
            "agreement": round(self.agreement, 3),
            "verified": self.verified,
            "verifier_checked": self.verifier_checked,
            "answer_verdict": self.answer_verdict.value,
        }


async def amplify(
    candidates: list[str],
    *,
    extract_answer: Callable[[str], str] | None = None,
    verify: Callable[[str], Awaitable[tuple[bool, list[str]]]] | None = None,
) -> AmplifiedResult:
    """Verifier-filtered self-consistency over candidate reasoning paths."""
    extract = extract_answer or default_extract_answer
    verifier = verify or verify_reasoning_checked
    cands = [c for c in candidates if c and str(c).strip()]
    if not cands:
        return AmplifiedResult(answer="", confidence=0.0, n=0, valid_n=0, agreement=0.0)

    outcomes: list[VerifierOutcome] = []
    all_issues: list[str] = []
    for c in cands:
        outcome, issues = await _run_verifier(verifier, c)
        outcomes.append(outcome)
        all_issues.extend(issues)

    # Only a PASS makes a candidate verifier-clean. An UNKNOWN candidate is not
    # filtered out (we have no grounds to reject it) but it is not certified
    # either — it simply carries no verification evidence.
    valid = [c for c, o in zip(cands, outcomes, strict=True) if o is VerifierOutcome.PASS]
    unrejected = [
        c for c, o in zip(cands, outcomes, strict=True) if o is not VerifierOutcome.FAIL
    ]
    verifier_checked = any(o is not VerifierOutcome.UNKNOWN for o in outcomes)

    # `valid if valid else cands` collapsed two different facts. "Nothing
    # passed because nothing was checkable" and "every candidate was
    # PROVABLY WRONG" both fell back to the full pool, so a unanimously
    # rejected set was clustered and shipped with a confidence computed as
    # though it were merely unverified — the enum above exists precisely to
    # stop that conflation, and this line reintroduced it downstream.
    #
    # Preference order: certified, then merely unchecked, then — only when
    # the verifier rejected everything — the rejected set, because returning
    # nothing at all is worse than returning a flagged answer. That last
    # case is marked so the confidence below can say what it is.
    all_rejected = not unrejected
    pool = valid or unrejected or cands

    # Self-consistency: cluster the pool by extracted answer; the biggest cluster wins.
    clusters: dict[str, list[str]] = {}
    for c in pool:
        clusters.setdefault(_normalize(extract(c)), []).append(c)
    _key, winners = max(clusters.items(), key=lambda kv: len(kv[1]))
    agreement = len(winners) / len(pool)
    verified_winner = bool(valid) and winners[0] in valid
    if verified_winner:
        certainty = 1.0
    elif all_rejected:
        # Every path carried a provable error. Agreement among wrong answers
        # is agreement about being wrong; it must not read like an ordinary
        # unverified result.
        certainty = 0.25
    else:
        certainty = 0.85
    confidence = min(0.98, agreement * certainty)

    if all_rejected:
        logger.warning(
            "🧠 [Amplify] every one of %d paths FAILED verification — "
            "returning the largest cluster with confidence %.2f",
            len(cands), confidence,
        )

    if not verifier_checked:
        logger.warning(
            "🧠 [Amplify] verifier unavailable for all %d paths — answer is "
            "UNVERIFIED (agreement %.0f%%)",
            len(cands), agreement * 100,
        )
    else:
        logger.info(
            "🧠 [Amplify] %d paths, %d verifier-clean → answer agreement %.0f%% (conf %.2f)",
            len(cands), len(valid), agreement * 100, confidence,
        )
    return AmplifiedResult(
        answer=winners[0],
        confidence=round(confidence, 4),
        n=len(cands),
        valid_n=len(valid),
        agreement=round(agreement, 4),
        verified=verified_winner,
        answer_verdict=(VerifierOutcome.PASS if verified_winner else
                        VerifierOutcome.FAIL if all_rejected else VerifierOutcome.UNKNOWN),
        verifier_checked=verifier_checked,
        issues=all_issues[:10],
    )


class DeliberationEngine:
    """Standalone reasoning amplifier: parallel-sample then amplify."""

    def __init__(self, *, n_samples: int = 5, temperatures: list[float] | None = None) -> None:
        self.n_samples = max(1, int(n_samples))
        self.temperatures = temperatures or [0.2, 0.5, 0.7, 0.9, 1.0]

    async def _sample(
        self,
        question: str,
        generate: Callable[[str, float], Awaitable[str]],
        count: int,
        offset: int = 0,
    ) -> list[str]:
        import asyncio

        async def _one(i: int) -> str:
            try:
                return await generate(question, self.temperatures[(offset + i) % len(self.temperatures)])
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("reasoning_amplifier", exc)
                return ""

        return [s for s in await asyncio.gather(*[_one(i) for i in range(count)]) if s]

    async def deliberate(
        self,
        question: str,
        generate: Callable[[str, float], Awaitable[str]],
        *,
        cross_tier: Any | None = None,
        **kw: Any,
    ) -> AmplifiedResult:
        samples = await self._sample(question, generate, self.n_samples)
        result = await amplify(samples, **kw)
        return await self._maybe_cross_tier(question, result, cross_tier)

    async def _maybe_cross_tier(self, question: str, result: AmplifiedResult, cross_tier: Any | None) -> AmplifiedResult:
        """Optionally let a stronger model tier verify/correct the winning answer."""
        if cross_tier is None or not result.answer:
            return result
        try:
            verdict = await cross_tier.verify(question, result.answer)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reasoning_amplifier", exc)
            return result
        if verdict.corrected and verdict.answer:
            result.answer = verdict.answer
            result.verified = True
            result.answer_verdict = VerifierOutcome.PASS
            result.confidence = round(min(0.99, max(result.confidence, 0.9)), 4)
        elif verdict.ok and "verified by strong tier" in verdict.critique:
            result.verified = True
            result.answer_verdict = VerifierOutcome.PASS
            result.confidence = round(min(0.99, result.confidence + 0.1), 4)
        elif not verdict.ok:
            # strong tier flagged doubt → lower confidence, keep the answer
            result.confidence = round(result.confidence * 0.7, 4)
        return result

    async def adaptive_deliberate(
        self,
        question: str,
        generate: Callable[[str, float], Awaitable[str]],
        *,
        min_samples: int = 3,
        max_samples: int = 9,
        batch: int = 2,
        target_agreement: float = 0.67,
        **kw: Any,
    ) -> AmplifiedResult:
        """Spend MORE compute when uncertain — the frontier 'think longer on hard ones'.

        Start with ``min_samples``; if the verifier-clean paths don't yet agree strongly
        enough (and there is budget), draw another ``batch`` and re-amplify. Stops as soon
        as a verifier-clean consensus is reached, capping at ``max_samples``.
        """
        samples = await self._sample(question, generate, min_samples)
        result = await amplify(samples, **kw)
        while (
            len(samples) < max_samples
            and not (result.verified and result.agreement >= target_agreement)
        ):
            extra = await self._sample(question, generate, batch, offset=len(samples))
            if not extra:
                break
            samples.extend(extra)
            result = await amplify(samples, **kw)
            if result.verified and result.agreement >= target_agreement:
                break
        logger.info(
            "🧠 [Adaptive] settled on %d samples (verified=%s, agreement %.0f%%)",
            result.n, result.verified, result.agreement * 100,
        )
        return result

    async def decompose_and_solve(
        self,
        question: str,
        generate: Callable[[str, float], Awaitable[str]],
        decompose: Callable[[str], Awaitable[list[str]]],
        *,
        recombine: Callable[[str, list[tuple[str, str]]], Awaitable[str]] | None = None,
        **kw: Any,
    ) -> AmplifiedResult:
        """Decompose-then-verify: split a hard problem, amplify each part, recombine.

        Each sub-question is solved by verifier-filtered self-consistency, so errors are
        caught *per step* before they compound — the failure mode that sinks long
        single-shot chains. The recombined answer is itself verified.
        """
        try:
            subqs = await decompose(question)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reasoning_amplifier", exc)
            subqs = []
        subqs = [q for q in (subqs or []) if q and str(q).strip()]
        if not subqs:
            return await self.adaptive_deliberate(question, generate, **kw)

        solved: list[tuple[str, str]] = []
        for sub in subqs:
            sub_result = await self.deliberate(sub, generate, **kw)
            solved.append((sub, sub_result.answer))

        if recombine is not None:
            try:
                final = await recombine(question, solved)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("reasoning_amplifier", exc)
                final = ""
        else:
            steps = "\n".join(f"- {q}: {a}" for q, a in solved)
            final_prompt = f"{question}\n\nSub-results:\n{steps}\n\nFinal answer:"
            final = await generate(final_prompt, 0.3)

        # Verify the recombined answer; its confidence reflects the sub-step agreement.
        outcome, issues = await _run_verifier(
            kw.get("verify") or verify_reasoning_checked, final or ""
        )
        ok = outcome is VerifierOutcome.PASS
        sub_conf = sum(1 for _q, a in solved if a) / max(1, len(solved))
        return AmplifiedResult(
            answer=final or (solved[-1][1] if solved else ""),
            confidence=round(min(0.98, sub_conf * (1.0 if ok else 0.8)), 4),
            n=len(subqs),
            valid_n=len(subqs) if ok else 0,
            agreement=round(sub_conf, 4),
            verified=ok and bool(final),
            answer_verdict=outcome if final else VerifierOutcome.UNKNOWN,
            issues=issues,
            verifier_checked=outcome is not VerifierOutcome.UNKNOWN,
        )


_instance: DeliberationEngine | None = None


def get_deliberation_engine() -> DeliberationEngine:
    global _instance
    if _instance is None:
        _instance = DeliberationEngine()
    return _instance
