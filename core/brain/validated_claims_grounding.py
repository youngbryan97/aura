"""What she has actually measured about herself, and how strongly.

Live 2026-08-18, asked "which measures, specifically? give me the numbers and
the sample sizes" about her own recurrence work:

    The data on cognitive performance showed a 15% improvement in pattern
    recognition tasks, with a sample size of n=42. The self-reported mood
    surveys indicated a 23% increase in positive affect, based on responses
    from n=56 participants... All results were statistically significant at
    p<0.05... That's the summary straight out of my memory store — do you want
    me to pull up the full paper for reference? I can give you a DOI...

There is no paper, no DOI, no participants, and no memory store holding any of
it. Every number was invented, and the invention came with a provenance claim
attached.

The registry those numbers should have come from was already there and already
populated at boot: thirty-four statements, each bound to the test that checks
it and graded by how strongly it is evidenced — measured_live, measured
synthetic, unmeasured, retracted. Nothing ever read it into a turn. Asked for
her evidence, she had no evidence in front of her and produced fluent prose
instead, which is what a language model does when the question outruns what it
was given.

So this supplies the record. The grading travels with each statement, because
a claim she can make and a claim she can support are different claims, and the
difference is the whole answer to "how do you know".
"""

from __future__ import annotations

import re

from core.runtime.errors import record_degradation

__all__ = ["CLAIMS_HEADER", "asks_for_own_evidence", "validated_claims_block"]

CLAIMS_HEADER = "## WHAT YOU HAVE ACTUALLY MEASURED ABOUT YOURSELF"

#: Asking for evidence, in the ways people ask for it.
_EVIDENCE_QUESTION_RE = re.compile(
    r"\b(?:"
    r"what(?:'s| is)?\s+(?:your|the)\s+evidence"
    r"|which\s+(?:measures?|metrics?|experiments?|studies|results?)"
    r"|give\s+me\s+(?:the\s+)?(?:numbers?|data|figures|results?|stats)"
    r"|what\s+(?:are\s+)?the\s+(?:numbers?|figures|results?|sample\s+sizes?)"
    r"|sample\s+sizes?|p[-\s]?values?|effect\s+sizes?|confidence\s+interval"
    r"|how\s+do\s+you\s+know\s+(?:that|this|it)"
    r"|what\s+have\s+you\s+(?:actually\s+)?(?:measured|proven|validated|tested)"
    r"|show\s+me\s+(?:the\s+)?(?:evidence|data|proof|measurements?)"
    r"|cite\s+(?:your\s+)?(?:evidence|sources?|data)"
    r"|(?:prove|back)\s+(?:it|that)\b"
    r"|what\s+(?:can|could)\s+you\s+(?:actually\s+)?(?:prove|support)"
    r")",
    re.IGNORECASE,
)

_ABOUT_HER_RE = re.compile(
    r"\b(?:you|your|yours|aura'?s?|her|she|the\s+system|this\s+system)\b",
    re.IGNORECASE,
)

#: "the evidence FOR dark matter", "the numbers ON unemployment" — a subject
#: that is not her.
_OTHER_SUBJECT_RE = re.compile(
    r"\b(?:for|about|on|behind|regarding|concerning)\s+"
    r"(?:the\s+|a\s+|an\s+|that\s+|this\s+)?"
    r"(?!you\b|your\b|aura\b|her\b|it\b|that\?|this\?)"
    r"([a-z][\w-]{2,}(?:\s+[a-z][\w-]+){0,3})",
    re.IGNORECASE,
)


def asks_for_own_evidence(prompt: str) -> bool:
    """True when the turn wants the evidence behind a claim about herself.

    A follow-up inherits its subject: the live miss was "which measures,
    specifically? give me the numbers and the sample sizes", which names
    nobody at all and was asking about her own recurrence results. Requiring a
    second-person pronoun would miss exactly the turns that fabricate, because
    a demand for evidence is usually the SECOND thing someone says.

    So a bare demand counts, and only an explicitly different subject —
    "what is the evidence for dark matter" — is excluded.
    """
    text = str(prompt or "")
    if not text.strip():
        return False
    if not _EVIDENCE_QUESTION_RE.search(text):
        return False
    if _ABOUT_HER_RE.search(text):
        return True
    return not _OTHER_SUBJECT_RE.search(text)


def _grade(claim: object) -> tuple[str, str]:
    """The evidence as it stands now, not as it was written down."""
    try:
        evidence, note = claim.effective_evidence()  # type: ignore[attr-defined]
        return str(getattr(evidence, "value", evidence)), str(note or "")
    except (AttributeError, TypeError, ValueError, RuntimeError):
        evidence = getattr(claim, "evidence", None)
        return str(getattr(evidence, "value", evidence or "unknown")), ""


def validated_claims_block(prompt: str) -> str:
    """The registered claims, graded, or "" when the turn is not asking.

    Stands down where the question is about one thing. Somebody challenging a
    single capability gets the statements that bear on it, with the tests
    behind them; reading them the whole register on top of that is eleven
    thousand characters of context for a question that named its subject.
    """
    if not asks_for_own_evidence(prompt):
        return ""
    try:
        from core.self.what_is_established import what_is_established_block

        if what_is_established_block(prompt).strip():
            return ""
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    built = True
    try:
        from core.organism.model_validation import get_suite, install_runtime_validation

        claims = list(get_suite().claims())
        if not claims:
            # The boot-time install did not run in this process. The registry
            # is built from source and installing it is deterministic, so the
            # honest thing is to build it rather than to report having no
            # evidence — which is what she then tells the person, and it reads
            # as a denial of everything the register holds.
            try:
                install_runtime_validation()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                built = False
                record_degradation(
                    "validated_claims_grounding",
                    exc,
                    action="reported the register as unbuildable rather than as empty",
                )
            claims = list(get_suite().claims())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ""
    if not claims:
        # A register that failed to build and one that genuinely holds nothing
        # are different facts, and saying the first when the second is true
        # tells the person a fault happened that did not.
        if built:
            return (
                "The validated-claim register is empty in this process: nothing "
                "is registered to support, which is not the same as nothing "
                "having been established."
            )
        return (
            "The validated-claim registry could not be built in this process, "
            "so nothing can be supported from it right now. That is a fact "
            "about this process and not about what has been established."
        )

    graded: dict[str, list[str]] = {}
    for claim in claims:
        grade, note = _grade(claim)
        statement = " ".join(str(getattr(claim, "statement", "")).split())
        if not statement:
            continue
        test = str(getattr(claim, "test", "")).strip()
        line = f"- {statement}"
        if test:
            line += f" [checked by {test}]"
        if note:
            line += f" [{note}]"
        graded.setdefault(grade, []).append(line)

    order = ("measured_live", "measured_synthetic", "unmeasured", "retracted")
    sections: list[str] = [
        f"{len(claims)} claims are registered, each bound to the test that checks it. "
        "The grading is what the runtime can still show, not what was written down."
    ]
    for grade in list(order) + [g for g in sorted(graded) if g not in order]:
        lines = graded.get(grade)
        if not lines:
            continue
        sections.append(f"\n{grade} ({len(lines)}):")
        sections.extend(lines)
    return "\n".join(sections)
