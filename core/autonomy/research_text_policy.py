"""What a research cycle may say, and what it may believe it learned.

Four small policies, lifted out of ``ResearchCycle`` because none of them
touch the cycle's state and the class was well past the size the ratchet
allows. Each one is a CP126 finding:

* ``69bca04d`` — the resident model was asked to "research" a topic it can
  only recall, and its prose was mined for concrete facts. A finding with
  no external source now carries a prefix that travels with it into
  knowledge, memory and the identity narrative.
* ``0d164a09`` — a generated sentence was appended straight to the
  identity narrative, and past 2,000 characters the oldest prefix was
  sliced off mid-word. Sentences are admitted or refused, and the trim
  lands on a sentence boundary.
* ``0768770c`` — a denied tool, a saturated model, a parse error and a
  network outage all counted identically toward suppressing a goal, so two
  of them permanently retired a valuable initiative.
"""

from __future__ import annotations

__all__ = [
    "MAX_NARRATIVE_CHARS",
    "PARAMETRIC_PREFIX",
    "bounded_narrative",
    "is_transient_failure",
    "label_findings",
    "narrative_admits",
]

#: Prefix carried by anything derived without consulting a source outside
#: the model.
PARAMETRIC_PREFIX = "[unverified — from the model's own training, no source consulted] "

#: Ceiling on the stored narrative. The old cap sliced the oldest 2,000
#: characters off mid-boundary; this trims whole sentences from the front
#: so what remains still reads as something she said.
MAX_NARRATIVE_CHARS = 2000

#: Failures that say nothing about whether the GOAL is researchable.
_TRANSIENT_FAILURE_MARKERS = (
    "timeout",
    "timederror",
    "connection",
    "network",
    "unavailable",
    "ratelimit",
    "rate limit",
    "429",
    "saturat",
    "denied",
    "permission",
    "jsondecode",
    "parse",
    "cancelled",
)

#: Text that rewrites what she IS rather than reporting what she learned.
_NARRATIVE_REFUSALS = (
    "you are",
    "ignore previous",
    "system:",
    "as an ai language model",
    "i am not",
    "i have no",
)


def label_findings(findings: list[str], parametric: bool) -> list[str]:
    """Mark findings that rest on recall rather than a consulted source."""
    if not parametric:
        return findings
    return [
        item if item.startswith(PARAMETRIC_PREFIX) else PARAMETRIC_PREFIX + item
        for item in findings
    ]


def is_transient_failure(error_text: str) -> bool:
    """Whether a failure was about the lane rather than about the goal."""
    from core.conversation.word_markers import names_any

    return names_any(str(error_text or "").lower(), _TRANSIENT_FAILURE_MARKERS)


def narrative_admits(sentence: str) -> tuple[bool, str]:
    """Whether a generated sentence may join the identity narrative.

    Not a constitutional reconciliation — that belongs to the identity
    engine and does not exist as a callable here. This is the bound that
    does exist: one sentence, first person, no instruction-shaped text,
    nothing that rewrites what she is rather than what she learned.
    """
    text = str(sentence or "").strip()
    if len(text) > 400:
        return False, "longer than one sentence"
    lowered = text.lower()
    for marker in _NARRATIVE_REFUSALS:
        if marker in lowered:
            return False, f"contains {marker!r}"
    if not any(pronoun in lowered for pronoun in (" i ", "i ", "my ", "me ")):
        return False, "not written in the first person"
    return True, ""


def bounded_narrative(narrative: str) -> str:
    """Trim from the front on a sentence boundary, never mid-word."""
    text = str(narrative or "")
    if len(text) <= MAX_NARRATIVE_CHARS:
        return text
    tail = text[-MAX_NARRATIVE_CHARS:]
    boundary = tail.find(". ")
    return tail[boundary + 2:] if boundary != -1 else tail
