"""What is actually established about her, with the test behind each one.

Asked to prove she can invent primitives for the language she makes rules out
of, she said she could not — that her representation language was "the static
set of instructions defined by my developers" — two turns after answering a
question with a word she had derived, kept, and recalled across a restart.

Nothing was wrong with the model. There was nothing for it to answer from, so
it answered from what a language model generally believes about itself, and
what it generally believes is a denial.

This is the record it should have read. Every statement here is registered with
the test that checks it, and the grade says what that test actually
establishes: measured live, measured on constructed cases, or not measured. A
claim with no test cannot be registered at all, so nothing in here is a
sentence somebody wrote about her.

What brings it into a turn is grammar rather than a list of topics. Somebody
asking whether she can do something at all, or telling her she cannot, is
asking about the record. Somebody asking her to do something is not, and "can
you open Safari" must not drag it in.
"""

from __future__ import annotations

import logging
import re
from typing import Any

__all__ = [
    "asks_what_is_established",
    "what_is_established_block",
]

logger = logging.getLogger("Aura.WhatIsEstablished")

#: The turn is about her.
_ABOUT_HER = re.compile(r"\b(?:you|your|yours|yourself|you're|youre)\b", re.IGNORECASE)

#: It challenges what she is, rather than asking her to do something. A request
#: to act says what to do; this asks whether the ability is there at all.
_CHALLENGES_IT = re.compile(
    r"\b(?:prove|proof|demonstrate|substantiate|evidence|verify|"
    r"claim|claims|claimed|claiming|"
    r"actually|really|genuinely|truly|"
    r"how\s+do\s+you\s+know|what\s+makes\s+you\s+think|"
    r"capab(?:le|ility|ilities)|architecture|架构)\b",
    re.IGNORECASE,
)

#: Or it tells her she cannot. A denial is a claim about her too, and it is the
#: one she has most often agreed with when she had nothing to check it against.
_DENIES_IT = re.compile(
    # She cannot.
    r"\byou\s+(?:can'?t|cannot|could\s?n[o']?t|will\s+never)\b"
    # She is not able to, is not capable of, does not have.
    r"|\byou(?:'re|\s+are|\s+do|\s+have)?\s*(?:not|never|n'?t)\s+"
    r"(?:\w+\s+){0,2}(?:able|capable|have|possess|do|really)\b"
    # She is unable, she is incapable.
    r"|\byou(?:'re|\s+are)\s+(?:unable|incapable)\b",
    re.IGNORECASE,
)

#: How many statements to show. Enough to answer, few enough to read.
_SHOWN = 6


def asks_what_is_established(prompt: Any) -> bool:
    """True when the turn asks what is established about her, or denies it."""
    text = str(prompt or "").strip()
    if not text or not _ABOUT_HER.search(text):
        return False
    return bool(_CHALLENGES_IT.search(text) or _DENIES_IT.search(text))


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", str(text).lower())}


def _names_a_subject(weights: list[float]) -> bool:
    """Whether the question singles anything out, or asks about the lot.

    Decided from the spread rather than from a number to clear: a question
    naming a subject scores a few statements far above the middle, and one
    asking in general scores everything alike. Comparing the best against the
    middle is the same judgement without a constant in it.
    """
    if not weights or weights[0] <= 0.0:
        return False
    middle = weights[len(weights) // 2]
    return weights[0] > middle * 2.0


def _how_telling(claims: list[Any]) -> dict[str, float]:
    """How much a word narrows the register down.

    A word in nearly every statement says nothing about which one is wanted,
    and the register is written in English, so "that", "which" and "record"
    turn up everywhere. Counting shared words without this ranked six
    statements about reply custody above the one the question was actually
    asking about.

    Worked out from the register itself rather than from a list of words to
    ignore, so it stays right as the register grows.
    """
    how_many: dict[str, int] = {}
    for claim in claims:
        for word in _words(claim.statement):
            how_many[word] = how_many.get(word, 0) + 1
    total = max(1, len(claims))
    return {word: total / seen for word, seen in how_many.items()}


def what_is_established_block(prompt: Any = "") -> str:
    """The registered claims nearest to what was asked, and what backs each.

    Ordered by how much of the question each one is about, because a person
    asking about one thing does not want the whole register read to them.
    """
    if prompt and not asks_what_is_established(prompt):
        return ""
    claims = _registered()
    if not claims:
        return ""
    asked = _words(prompt)
    tells_them_apart = _how_telling(claims)
    scored = [
        (
            sum(tells_them_apart.get(word, 0.0) for word in asked & _words(claim.statement)),
            claim,
        )
        for claim in claims
    ]
    ranked = sorted(scored, key=lambda one: (-one[0], one[1].statement))
    weights = [weight for weight, _claim in ranked]
    if not _names_a_subject(weights):
        # The question does not single anything out — "what have you actually
        # measured about yourself" wants the register, not six statements
        # ranked by words that do not tell them apart. The whole-register
        # reading answers that one.
        return ""
    nearest = [claim for weight, claim in ranked if weight > 0.0][:_SHOWN]
    if not nearest:
        return ""
    lines = [
        f"{len(claims)} statements about this system are registered, each bound "
        "to the test that checks it. These are the ones this question is about:",
    ]
    for claim in nearest:
        grade = getattr(claim.evidence, "value", str(claim.evidence))
        lines.append(f"- {claim.statement}")
        lines.append(f"    checked by: {claim.test} ({grade}), in {claim.owner}")
        if claim.evidence_note:
            lines.append(f"    what that establishes: {claim.evidence_note}")
        for reading in _now(getattr(claim, "live_channels", ()) or ()):
            lines.append(f"    as it stands in this process: {reading}")
    lines.append(
        "These hold in this build. A statement with no test cannot be "
        "registered here, so this record says nothing that nothing checks — "
        "and it is the answer to what this system can do, ahead of anything a "
        "language model believes about itself in general."
    )
    return "\n".join(lines)


def _now(channels: tuple[str, ...]) -> list[str]:
    """What the channels a claim rests on are reading right now.

    A statement about what she can do, with the number it currently stands at,
    answers "what specifically did you add" — which a test name on its own
    does not. A channel that has never been written says so rather than
    reading zero, because those are different facts.
    """
    if not channels:
        return []
    try:
        from core.fsw.telemetry_dictionary import channel_value
    except ImportError:
        return []
    said: list[str] = []
    for name in channels:
        try:
            sample = channel_value(name)
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        if sample is None:
            said.append(f"{name} has not been written in this process")
            continue
        said.append(f"{name} = {getattr(sample, 'value', sample)}")
    return said


def _registered() -> list[Any]:
    """The claims, installing the suite once if this process has not."""
    try:
        from core.organism.model_validation import get_suite, install_runtime_validation

        claims = get_suite().claims()
        if claims:
            return list(claims)
        install_runtime_validation()
        return list(get_suite().claims())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "what_is_established",
            exc,
            severity="info",
            action="answered a question about herself without the record",
        )
        return []
