"""core/conversation/requested_reply_shape.py — what the person asked the REPLY to be.

Extracted from ``response_reliability.py`` for the same reason as
``arithmetic_check``: this parses the USER's message, and that module's job is
deciding what Aura may SAY. Three compiled patterns of request parsing were
being counted as output-filter debt by a ratchet that exists to catch
phrase-banning, which is a category error in the measurement, not debt in the
code.

Nothing here suppresses or rewrites anything. It answers one question — when a
person has said what the reply should look like, which part of the turn is
that? — so a downstream predicate stops guessing.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["is_reply_shape_constraint_segment", "reply_scope_text"]


#: An explicit instruction about what the REPLY should contain: "reply with
#: just the path", "tell me only the filename", "answer with nothing but the
#: number".
_REPLY_SHAPE_CONSTRAINT_RE = re.compile(
    r"\b(?:reply|respond|answer|report(?:\s+back)?|tell\s+me|say)\s+"
    r"(?:back\s+)?(?:with\s+|using\s+)?"
    r"(?:just|only|nothing\s+but|solely|simply)\s+"
    r"(?P<shape>[^.?!]{1,140})",
    re.IGNORECASE,
)

#: A span describing what goes INSIDE a produced artifact, not what the answer
#: is. "containing exactly three lines: line one the date, line two how many
#: subsystems…" is a file specification; the numbers in it are content, not the
#: question.
_ARTIFACT_CONTENT_SPEC_RE = re.compile(
    r"\b(?:containing|contains|with\s+the\s+(?:following|content|contents|text|lines)"
    r"|whose\s+contents?|that\s+says|saying|that\s+reads)\b.*?"
    r"(?=(?:\.\s+(?:then|and\s+then|after\s+that|finally|reply|respond|answer|tell)\b)|$)"
    r"|\bline\s+(?:one|two|three|four|1|2|3|4)\b.*?"
    r"(?=(?:\.\s+(?:then|and\s+then|reply|respond|answer|tell)\b)|$)",
    re.IGNORECASE | re.DOTALL,
)


#: A reply constraint that is itself asking for a quantity. "reply with just the
#: number" narrows the SHAPE without removing the numeric requirement.
_REPLY_SHAPE_WANTS_QUANTITY_RE = re.compile(
    r"\b(?:number|numbers|count|total|sum|quantity|amount|figure|digits?|"
    r"value|result|answer|probability|fraction|percentage|average|how\s+many)\b",
    re.IGNORECASE,
)

_REPLY_SHAPE_SEGMENT_RE = re.compile(
    r"^\s*(?:answer|reply|respond|return|report|present)\b.{0,32}"
    r"\b(?:in|with|using|as)\b.{0,80}"
    r"\b(?:sentence|sentences|paragraph|paragraphs|bullet|bullets|item|items|"
    r"word|words|line|lines|json|table|list|format)\b[^?]*[.!]?\s*$"
    r"|^\s*(?:use|write)\s+(?:exactly\s+)?(?:\w+\s+){0,3}"
    r"(?:sentence|sentences|paragraph|paragraphs|bullet|bullets|item|items|"
    r"word|words|line|lines)\s*[.!]?\s*$",
    re.IGNORECASE,
)


#: Words that describe HOW an answer should read. A clause built only from
#: these introduces no subject matter — it is about the reply, not the world.
_MANNER_WORDS = frozenset(
    """
    brief briefly short shorter shortly concise concisely direct directly
    blunt bluntly honest honestly frank frankly plain plainly plainly simple
    simply straight straightforward clear clearly terse succinct quick quickly
    real actual actually genuine genuinely specific concrete
    generic boilerplate canned formulaic
    preamble preambles caveat caveats disclaimer disclaimers hedge hedges
    hedging fluff filler padding waffle ramble rambling apology apologies apologising
    apologizing collapsing lecture lecturing
    ai assistant answer answers answering reply replies replying respond
    responding response point words word question questions
    """.split()
)

#: The grammar of an instruction about the reply: an imperative aimed at the
#: answer itself, or a prohibition on padding.
_MANNER_OPENER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"be|keep\s+it|stay|answer|reply|respond|say\s+it|make\s+it|"
    r"don'?t|do\s+not|no|skip|avoid|without|just|only"
    r")\b",
    re.IGNORECASE,
)

#: Function words carry no subject matter, so they never disqualify a clause.
_MANNER_FUNCTION_WORDS = frozenset(
    """
    a an the and or but not no nor so if then than that this these those
    to of in on at by for from with within without into onto about as is are
    was were be been being do does did doing done have has had having will
    would can could shall should may might must let get got it its
    me my you your yours we us our i one two three too very much more less
    please s t don dont doesn isn aren wasn weren won
    """.split()
)


def _is_manner_only_clause(segment: str) -> bool:
    """True when every content word in the clause is about the answer's manner.

    LIVE 2026-08-18. "Are you conscious or self-aware? Answer honestly without
    collapsing into a generic AI disclaimer." split into two asks, and the
    second — an instruction about tone — was scored as a question the reply
    had failed to cover. The reply had obeyed it. Only "answer in N sentences"
    was recognised as presentation, so every instruction about MANNER rather
    than FORMAT was counted as dropped subject matter, and a reply was
    penalised precisely for doing what it was told.

    The test is structural rather than another phrase list: strip the function
    words, and if what remains is entirely manner vocabulary, the clause asks
    for no facts.
    """
    if not _MANNER_OPENER_RE.match(segment):
        return False
    # Apostrophes are punctuation inside a word: "don't" is the function word
    # "do not", and leaving the mark on made it read as unknown content.
    words = [
        word.lower().replace("'", "").replace("\u2019", "")
        for word in re.findall(r"[A-Za-z']+", segment)
    ]
    content = [
        word
        for word in words
        if word not in _MANNER_FUNCTION_WORDS and word not in _MANNER_OPENER_WORDS
    ]
    if not content:
        return True
    return all(word in _MANNER_WORDS for word in content)


_MANNER_OPENER_WORDS = frozenset(
    "be keep stay answer reply respond say make don dont do not no skip avoid "
    "without just only please it".split()
)


def is_reply_shape_constraint_segment(segment: Any) -> bool:
    """Whether a whole clause constrains presentation instead of content.

    ``Write a paragraph about yourself`` remains a content request. ``Answer
    in exactly two numbered sentences`` belongs to the output contract checker,
    and so does ``be brief`` or ``answer honestly without the disclaimer`` —
    an instruction about how to speak is not a second thing to speak about.
    """

    text = str(segment or "").strip()
    if _REPLY_SHAPE_SEGMENT_RE.fullmatch(text):
        return True
    return _is_manner_only_clause(text)


def reply_scope_text(user_message: Any) -> str:
    """The part of the turn that constrains the ANSWER, not the artifact.

    LIVE DEFECT, 2026-08-10. Asked to write a file whose three lines were a
    date, a subsystem count, and the word DONE, and to "reply with just the
    path you wrote", the worker rejected six consecutive drafts with
    ``numeric_answer_missing`` — 20 seconds of 32B generation — then the turn
    died:

        compact desktop generation returned no usable text
        TurnOutcomeError: retryable_failure:retryable_error_and_nothing_served
        CRITICAL SERVICE FAILURE: Subsystem 'cognitive_engine' … fail-closed
        reply_reliability_gate_failed:runtime_boilerplate,missing_requested_line_count

    and the person was handed "I couldn't get to an answer I'd stand behind on
    that one … Ask me again in a moment."

    The gate read "line two how many subsystems are heartbeating" and concluded
    the REPLY could only be answered with a quantity. That number belonged in
    the file. The reply had been specified explicitly, in the same sentence, as
    just the path — which is what she produced and what was thrown away.

    Two spans are therefore removed before judging what the answer must be: a
    specification of artifact CONTENT, and everything outside an explicit
    reply-shape constraint when one is present. A file is not an answer, and a
    person who says what the reply should be has already answered the question
    this predicate exists to guess.
    """
    text = str(user_message or "")
    if not text.strip():
        return ""
    without_artifact = _ARTIFACT_CONTENT_SPEC_RE.sub(" ", text)
    constraint = _REPLY_SHAPE_CONSTRAINT_RE.search(text)
    if constraint is None:
        return without_artifact
    shape = constraint.group("shape")
    # A constraint that names a quantity is a numeric reply constraint: "add 14
    # and 9 and reply with just the number" must still be held to producing one.
    # Narrowing to the shape alone would drop the operands and suppress the very
    # guard the person asked for — the suppression cutting the wrong way.
    if _REPLY_SHAPE_WANTS_QUANTITY_RE.search(shape):
        return without_artifact
    # Otherwise the person has said the reply is something that is not a
    # quantity, and that outranks any number appearing elsewhere in the turn.
    return shape
