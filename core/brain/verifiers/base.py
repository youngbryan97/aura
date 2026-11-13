"""Uniform verification result + verifier protocol for the truth engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VerificationResult:
    """The verdict of one truth engine over a candidate answer.

    ``ok`` is the hard gate (no *provable* failure was found). ``checked`` records
    whether a real check actually ran — an engine that found nothing to verify
    (e.g. no code in the candidate) returns ``ok=True, checked=False`` so the
    amplifier does not mistake "nothing to check" for "verified correct".
    ``score`` is a soft 0..1 contribution used to rank surviving candidates.
    """

    domain: str
    ok: bool
    checked: bool
    score: float = 0.5
    engine: str = ""
    issues: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    #: True when verification was IMPOSSIBLE rather than unnecessary — the
    #: sandbox was down, an engine would not import, the infrastructure
    #: failed. `checked=False` alone conflates that with the benign case
    #: ("this candidate contains no code"), and only one of the two should
    #: stop an admission.
    infrastructure_failed: bool = False
    #: The engine did not finish inside the deadline it was given. Not the
    #: same as crashing: the check may well have been right, it just did not
    #: say so in time, and a reader deciding whether to retry needs to know.
    timed_out: bool = False
    #: No engine applies to this task at all. "Nothing here was checkable"
    #: and "there is nothing that could check this" are different facts: the
    #: first says the answer made no claim a verifier could reach, the second
    #: says the task has no verifier.
    not_applicable: bool = False
    #: Whether a pass speaks to the answer itself: the answer compared with a value, or run against examples, that do
    #: not come from the candidate. A check of the working, the citations, the form or a trace can find a provable
    #: fault and so refute an answer; its pass does not make the answer right. LIVE 13 November 2026 (F04): an AIME
    #: draft cut off before any answer was kept as a verified incumbent because the logic audit found no non-sequitur
    #: in its working. Only a pass that reaches the answer is authority to keep an answer or put another in its place
    #: (core/brain/reasoning_amplifier_v2.py); every other pass still ranks candidates.
    reaches_the_answer: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """PASSED, FAILED, or UNCHECKED — the question callers actually mean.

        ``ok`` is "no provable failure was found", which is TRUE when nothing
        was checked. That contract is deliberate and documented above, and it
        is also a trap: every caller that reads ``.ok`` alone treats "we did
        not look" as "we looked and it was fine". Two were doing exactly
        that — a proof obligation reached PROVED on an unchecked verifier,
        and Aura's own introspection rendered "Verifier: pass" for a check
        that never ran.

        Three states, so the unchecked one has to be handled rather than
        collapsed into the passing one.
        """
        if self.infrastructure_failed or self.timed_out:
            return "UNVERIFIABLE"
        if not self.checked:
            return "UNCHECKED"
        return "PASSED" if self.ok else "FAILED"

    @property
    def outcome(self) -> str:
        """Which of six things happened. Only ``verified`` is correctness.

        ``verdict`` folds four of these into two names, and the two it folds
        hardest are the ones a reader acts on differently: an engine that
        crashed and one that ran out of time are both UNVERIFIABLE there, and
        a task no engine covers reads the same as an answer with nothing in it
        to check. The release requirement names all six, and so does this.

        A contradiction outranks everything: a provable failure is final even
        when another engine could not run. After that, an engine that could
        not finish outranks one that passed, the same order ``verdict`` uses,
        because a pass beside a crash is a pass from fewer checks than were
        asked for.
        """
        if self.checked and not self.ok:
            return "contradicted"
        if self.timed_out:
            return "timed_out"
        if self.infrastructure_failed:
            return "verifier_failed"
        if self.checked:
            return "verified"
        if self.not_applicable:
            return "not_applicable"
        return "unsupported"

    @property
    def verification_was_possible(self) -> bool:
        """False when the checking machinery itself failed.

        A gate that admits on "nothing was checked" is often right — most
        candidates contain no code to verify. Admitting because the verifier
        could not RUN is a different decision, and it was indistinguishable
        from the first.
        """
        return not (self.infrastructure_failed or self.timed_out)

    @property
    def conclusively_ok(self) -> bool:
        """True only when a real check ran AND it passed.

        This is what a gate wants. ``ok`` is what a ranking wants.
        """
        return self.verdict == "PASSED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "ok": self.ok,
            "checked": self.checked,
            "verdict": self.verdict,
            "outcome": self.outcome,
            "infrastructure_failed": self.infrastructure_failed,
            "timed_out": self.timed_out,
            "not_applicable": self.not_applicable,
            "reaches_the_answer": self.reaches_the_answer,
            "score": round(float(self.score), 3),
            "engine": self.engine,
            "issues": self.issues[:8],
            "evidence": self.evidence[:8],
        }


@runtime_checkable
class Verifier(Protocol):
    """A domain truth engine.

    ``domains`` lists the task types it claims (e.g. ``("code",)``); ``"*"`` means
    it always runs (logic/non-sequitur checks apply to any prose answer).
    """

    name: str
    domains: tuple[str, ...]

    def handles(self, task_type: str) -> bool: ...

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult: ...


def combine_results(
    domain: str,
    results: list[VerificationResult],
    *,
    weights: dict[str, float] | None = None,
) -> VerificationResult:
    """Fold many engine verdicts into one for a candidate.

    Hard gate: ``ok`` is False if *any* engine that actually checked failed —
    NEVER weighted; a provable failure is final regardless of the engine's
    track record. ``checked`` is True if at least one engine checked.
    ``score`` is the mean of the checked engines (defaulting to neutral 0.5
    when nothing was checkable); when ``weights`` (per-engine measured
    reliability, from the Verifier Foundry) are provided, the soft mean is
    reliability-weighted so a leaky engine's enthusiasm counts for less.
    """
    checked = [r for r in results if r.checked]
    issues: list[str] = []
    evidence: list[str] = []
    engines: list[str] = []
    for r in results:
        issues.extend(r.issues)
        evidence.extend(r.evidence)
        if r.engine:
            engines.append(r.engine)
    infrastructure_failed = any(r.infrastructure_failed for r in results)
    timed_out = any(r.timed_out for r in results)
    # An empty verifier set is not a pass and not an unchecked answer: it is a
    # task nothing covers. Only when every engine says so does the fold.
    not_applicable = all(r.not_applicable for r in results) if results else True
    ok = all(r.ok for r in checked) if checked else True
    if checked and weights:
        wsum = sum(max(0.0, float(weights.get(r.engine or "?", 1.0))) for r in checked)
        if wsum > 0:
            score = sum(r.score * max(0.0, float(weights.get(r.engine or "?", 1.0)))
                        for r in checked) / wsum
        else:
            score = sum(r.score for r in checked) / len(checked)
    else:
        score = sum(r.score for r in checked) / len(checked) if checked else 0.5
    return VerificationResult(
        domain=domain,
        ok=ok,
        checked=bool(checked),
        infrastructure_failed=infrastructure_failed,
        timed_out=timed_out,
        not_applicable=not_applicable,
        reaches_the_answer=ok and any(r.ok and r.reaches_the_answer for r in checked),
        score=round(score, 4),
        engine="+".join(engines),
        issues=issues[:12],
        evidence=evidence[:12],
        detail={"sub": [r.to_dict() for r in results]},
    )
