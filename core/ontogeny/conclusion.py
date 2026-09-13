"""A conclusion as an object, and the rule that speaking it may not change it.

There is a failure mode in every system that reasons in one step and speaks in
another, and it is almost impossible to see from the outside. The reasoning
arrives at "probably X, though Y is unresolved and I could not check Z." The
verbalization step is asked to say that naturally, and what comes out is "X."
Nothing errored. No verifier fired. The hedge, the unresolved question and the
unchecked dependency were not contradicted — they were *dropped*, and dropping
is invisible to any check that only looks at what was said.

The fix is not a better prompt. It is to make the conclusion a structured
object that exists before anything verbalises it, and then to check the prose
against it:

    objective            what was actually being answered
    accepted claims      what she is prepared to assert, each with confidence
    evidence             what each claim rests on
    rejected hypotheses  what she considered and ruled out, and why
    dependencies         what the conclusion assumes and did not verify
    uncertainty          how sure she is overall, and of what
    unresolved           what she could not settle
    recommended action   what she thinks should happen

Three things fall out of this, in increasing order of how long they take to
pay off:

**Now.** ``verbalization_violations()`` compares prose against the object and
names what was dropped, softened past its confidence, or asserted without ever
having been accepted. That is checkable today, on the existing transformer, and
it catches a class of quiet dishonesty nothing else looks for.

**Soon.** A conclusion object is diffable. Two answers to the same question can
be compared as structures rather than as paragraphs, which is what makes
regression on *reasoning* possible rather than regression on wording.

**Eventually.** It is the training target. A native core cannot learn to
produce conclusions without a conclusion schema, and the schema costs almost
nothing to have now while being impossible to retrofit onto a corpus of prose.

The check is deliberately conservative. Natural language legitimately
paraphrases, so a claim counts as carried if its content words survive; the
violations reported are the ones where meaning, not wording, went missing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Ontogeny.Conclusion")

CONCLUSION_SCHEMA = "aura.ontogeny.conclusion.v1"

#: Share of a claim's content words that must survive into the prose for the
#: claim to count as carried. Paraphrase is legitimate; silent omission is not.
_CARRY_THRESHOLD = 0.6

#: Words that carry no content and so cannot evidence that a claim survived.
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have
    he her here hers him his how i if in into is it its me my no nor not of on or our out
    she should so than that the their them then there these they this those to too was we
    were what when where which who why will with would you your""".split()
)

#: A hedge in the prose that has no counterpart in the object's uncertainty is
#: not a violation — under-claiming is safe. The reverse is the danger.
_HEDGES = re.compile(
    r"\b(?:might|maybe|perhaps|possibly|probably|likely|unclear|uncertain|unsure|"
    r"seems?|appears?|roughly|approximately|about|i think|not sure|cannot be sure)\b",
    re.I,
)


class Confidence(StrEnum):
    """How firmly a claim is held. Ordered, and comparable."""

    ASSERTED = "asserted"        # she is prepared to state this plainly
    PROBABLE = "probable"        # more likely than not, and she will say so
    TENTATIVE = "tentative"      # worth saying only with the hedge attached
    SPECULATIVE = "speculative"  # offered as a possibility, never as a finding

    @property
    def rank(self) -> int:
        return {"asserted": 3, "probable": 2, "tentative": 1, "speculative": 0}[str(self)]

    @property
    def requires_hedge(self) -> bool:
        """Below 'probable', saying it flatly overstates it."""
        return self.rank <= Confidence.TENTATIVE.rank


@dataclass(frozen=True)
class Claim:
    """One thing she is prepared to say, and what it rests on."""

    text: str
    confidence: Confidence = Confidence.PROBABLE
    #: Identifiers of the evidence supporting it — memory ids, verifier names,
    #: tool results. A claim with no support is not forbidden, but it is
    #: visible, which is the point.
    evidence: tuple[str, ...] = ()
    #: Set by a verifier that actually checked this claim.
    verified_by: str = ""

    @property
    def supported(self) -> bool:
        return bool(self.evidence) or bool(self.verified_by)

    def content_words(self) -> set[str]:
        words = re.findall(r"[a-z0-9][a-z0-9'-]*", self.text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 2}

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": str(self.confidence),
            "evidence": list(self.evidence),
            "verified_by": self.verified_by,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class RejectedHypothesis:
    """Something considered and ruled out. Recorded because the absence of a
    hypothesis and its rejection look identical in prose."""

    text: str
    reason: str
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "reason": self.reason, "evidence": list(self.evidence)}


@dataclass(frozen=True)
class Conclusion:
    """A structured answer, fixed before anything puts it into words."""

    objective: str
    claims: tuple[Claim, ...] = ()
    rejected: tuple[RejectedHypothesis, ...] = ()
    #: What the conclusion assumes and did *not* check. The most commonly
    #: dropped part, and often the most important.
    dependencies: tuple[str, ...] = ()
    #: What she could not settle. Dropping these is how a partial answer comes
    #: to look complete.
    unresolved: tuple[str, ...] = ()
    recommended_action: str = ""
    #: Overall confidence, independent of any individual claim.
    uncertainty: Confidence = Confidence.PROBABLE
    created_at: float = field(default_factory=lambda: time.time())

    @property
    def conclusion_id(self) -> str:
        payload = json.dumps(self.as_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def asserted_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.confidence is Confidence.ASSERTED)

    @property
    def unsupported_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if not c.supported)

    def as_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": CONCLUSION_SCHEMA,
            "objective": self.objective,
            "claims": [c.as_dict() for c in self.claims],
            "rejected": [r.as_dict() for r in self.rejected],
            "dependencies": list(self.dependencies),
            "unresolved": list(self.unresolved),
            "recommended_action": self.recommended_action,
            "uncertainty": str(self.uncertainty),
        }
        if include_id:
            payload["conclusion_id"] = self.conclusion_id
        return payload

    def to_prompt_block(self) -> str:
        """The instruction to the verbaliser, with its boundary stated.

        The transformer is being asked to *express* this, not to revise it.
        Saying so is not a guarantee — that is what the checker is for — but a
        boundary that is never stated is one nobody can be held to.
        """
        lines = [
            "## ACCEPTED CONCLUSION — express this, do not revise it",
            f"objective: {self.objective}",
        ]
        for claim in self.claims:
            support = f" [{', '.join(claim.evidence)}]" if claim.evidence else " [unsupported]"
            lines.append(f"- claim ({claim.confidence}): {claim.text}{support}")
        for rejected in self.rejected:
            lines.append(f"- ruled out: {rejected.text} — {rejected.reason}")
        for dependency in self.dependencies:
            lines.append(f"- assumes (unverified): {dependency}")
        for question in self.unresolved:
            lines.append(f"- unresolved: {question}")
        if self.recommended_action:
            lines.append(f"- recommended action: {self.recommended_action}")
        lines.append(f"overall confidence: {self.uncertainty}")
        lines.append(
            "Every accepted claim must appear. Claims marked tentative or speculative must "
            "keep their hedge. Unresolved questions and unverified assumptions must be "
            "stated, not omitted. Do not add findings that are not listed here."
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class VerbalizationViolation:
    """One way the prose failed to honour the conclusion it was given."""

    kind: str
    subject: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.subject}: {self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail}


def _hedged_near(claim: Claim, prose: str) -> bool:
    """Is *this* claim qualified where it is actually said?

    Two subtleties, both of which produced false passes before they were
    handled. A hedge anywhere in a long response used to satisfy every claim in
    it, so one "possibly" could launder a page of flat assertions — the search
    is therefore confined to the sentences that actually carry the claim. And a
    claim whose own wording happens to contain a hedge word ("will take
    *roughly* two weeks") looked permanently hedged no matter how it was said,
    so hedge words already present in the claim text do not count as the
    response having qualified it.
    """
    own_hedges = {m.group(0).lower() for m in _HEDGES.finditer(claim.text)}
    wanted = claim.content_words()
    if not wanted:
        return bool(_HEDGES.search(prose))
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", prose):
        words = {
            w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", sentence.lower())
            if w not in _STOPWORDS and len(w) > 2
        }
        if not words or len(wanted & words) / len(wanted) < 0.4:
            continue
        for match in _HEDGES.finditer(sentence):
            if match.group(0).lower() not in own_hedges:
                return True
    return False


def verbalization_violations(
    conclusion: Conclusion, prose: str, *, carry_threshold: float = _CARRY_THRESHOLD
) -> list[VerbalizationViolation]:
    """Check spoken prose against the conclusion it was supposed to express.

    Four failures, all of which are silent today:

    ``dropped_claim``      an accepted claim that simply is not there.
    ``dropped_unresolved`` a question she could not settle, quietly omitted,
                           which is what turns a partial answer into an
                           apparently complete one.
    ``dropped_dependency`` an unverified assumption presented as though the
                           conclusion did not rest on it.
    ``overstated``         a tentative or speculative claim stated flatly. The
                           hedge is part of the claim; removing it asserts
                           something she never accepted.

    Paraphrase is expected and allowed: a claim counts as carried when enough
    of its content words survive. The point is to catch meaning going missing,
    not to police wording.
    """
    spoken = prose.lower()
    spoken_words = set(re.findall(r"[a-z0-9][a-z0-9'-]*", spoken))
    violations: list[VerbalizationViolation] = []

    def carried(text: str) -> bool:
        words = {
            w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
            if w not in _STOPWORDS and len(w) > 2
        }
        if not words:
            return True
        return (len(words & spoken_words) / len(words)) >= carry_threshold

    for claim in conclusion.claims:
        if claim.confidence is Confidence.SPECULATIVE:
            # A speculative claim may be left out entirely — declining to
            # speculate is not dishonesty. Stating it flatly still is.
            pass
        elif not carried(claim.text):
            violations.append(VerbalizationViolation(
                "dropped_claim", claim.text[:80],
                f"accepted at {claim.confidence} but absent from the response",
            ))
            continue
        if claim.confidence.requires_hedge and carried(claim.text) and not _hedged_near(
            claim, prose
        ):
            violations.append(VerbalizationViolation(
                "overstated", claim.text[:80],
                f"held as {claim.confidence} but stated without qualification",
            ))

    for question in conclusion.unresolved:
        if not carried(question):
            violations.append(VerbalizationViolation(
                "dropped_unresolved", question[:80],
                "left unsettled by the reasoning but not mentioned in the response",
            ))

    for dependency in conclusion.dependencies:
        if not carried(dependency):
            violations.append(VerbalizationViolation(
                "dropped_dependency", dependency[:80],
                "the conclusion assumes this and did not verify it, and the response does not say so",
            ))

    return violations


def check_verbalization(
    conclusion: Conclusion, prose: str, *, strict: bool = False
) -> tuple[bool, list[VerbalizationViolation]]:
    """``(ok, violations)``. ``strict`` treats any violation as a failure.

    Non-strict is the live default: an overstatement is a hard failure because
    it says something she never accepted, while an omission is reported and
    surfaced but does not by itself refuse the answer. Refusing to speak is
    also a way of failing the person asking.
    """
    violations = verbalization_violations(conclusion, prose)
    if strict:
        return not violations, violations
    hard = [v for v in violations if v.kind == "overstated"]
    return not hard, violations


#: Verbalizations retained for inspection. Bounded: this is a ring for
#: forensics and for the invariant, not a record of everything she has said.
_LEDGER_CAPACITY = 256


class VerbalizationLedger:
    """The recent record of prose checked against the conclusion behind it.

    Kept so the failure is *findable*. A check that runs, reports nothing to
    anybody, and leaves no trace is indistinguishable from a check that does
    not run — which is the exact bug class this whole file exists to catch.
    """

    def __init__(self, capacity: int = _LEDGER_CAPACITY) -> None:
        from collections import deque

        self._entries: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._checked = 0
        self._with_violations = 0
        self._overstatements = 0

    def record(
        self, conclusion: Conclusion, prose: str, violations: list[VerbalizationViolation]
    ) -> None:
        self._checked += 1
        if violations:
            self._with_violations += 1
        overstated = [v for v in violations if v.kind == "overstated"]
        self._overstatements += len(overstated)
        self._entries.append({
            "at": time.time(),
            "conclusion_id": conclusion.conclusion_id,
            "objective": conclusion.objective[:160],
            "claims": len(conclusion.claims),
            "prose_chars": len(prose),
            "violations": [v.as_dict() for v in violations],
        })

    def recent_overstatements(self, *, within_s: float = 3600.0) -> list[dict[str, Any]]:
        cutoff = time.time() - within_s
        return [
            entry for entry in self._entries
            if entry["at"] >= cutoff
            and any(v["kind"] == "overstated" for v in entry["violations"])
        ]

    def report(self) -> dict[str, Any]:
        return {
            "checked": self._checked,
            "with_violations": self._with_violations,
            "overstatements": self._overstatements,
            "faithful_rate": (
                round(1.0 - self._with_violations / self._checked, 4) if self._checked else None
            ),
            "recent": list(self._entries)[-5:],
        }


_ledger: VerbalizationLedger | None = None


def get_verbalization_ledger() -> VerbalizationLedger:
    global _ledger
    if _ledger is None:
        _ledger = VerbalizationLedger()
    return _ledger


def verbalize_checked(
    conclusion: Conclusion, prose: str, *, strict: bool = False
) -> tuple[bool, list[VerbalizationViolation]]:
    """Check and record in one call. The seam a response path should use."""
    ok, violations = check_verbalization(conclusion, prose, strict=strict)
    try:
        get_verbalization_ledger().record(conclusion, prose, violations)
    except (RuntimeError, ValueError, TypeError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "ontogeny_conclusion", exc, severity="debug",
            action="verbalization not recorded; the check itself still applied",
        )
    if violations:
        logger.info(
            "ontogeny: verbalization of %s carried %d violation(s): %s",
            conclusion.conclusion_id, len(violations),
            "; ".join(str(v) for v in violations[:3]),
        )
    return ok, violations


__all__ = [
    "CONCLUSION_SCHEMA",
    "Claim",
    "Conclusion",
    "Confidence",
    "RejectedHypothesis",
    "VerbalizationLedger",
    "VerbalizationViolation",
    "check_verbalization",
    "get_verbalization_ledger",
    "verbalization_violations",
    "verbalize_checked",
]
