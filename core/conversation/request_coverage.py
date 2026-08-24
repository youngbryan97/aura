"""Conservative semantic coverage for compound user requests.

This module is shared by both phase-level dialogue validation and the desktop
reliability gate. A prompt contract that detects multiple asks is useful only
if every production response path checks the same contract before surfacing a
reply.
"""

from __future__ import annotations

import re
from typing import Any

from core.conversation.requested_reply_shape import (
    is_reply_shape_constraint_segment,
)

_COVERAGE_STOPWORDS = frozenset(
    {
        "about",
        "actually",
        "again",
        "and",
        "answer",
        "any",
        "anything",
        "are",
        "ask",
        "asked",
        "aura",
        "because",
        "been",
        "being",
        "both",
        "but",
        "can",
        "chatgpt",
        "current",
        "could",
        "correct",
        "did",
        "does",
        "doing",
        "done",
        "for",
        "from",
        "give",
        "had",
        "has",
        "have",
        "her",
        "here",
        "him",
        "his",
        "how",
        "its",
        "just",
        "hey",
        "like",
        "make",
        "many",
        "may",
        "mean",
        "might",
        "more",
        "most",
        "much",
        "must",
        "not",
        "now",
        "one",
        "only",
        "other",
        "out",
        "over",
        "naturally",
        "own",
        "really",
        "right",
        "say",
        "see",
        "separately",
        "she",
        "should",
        "some",
        "something",
        "still",
        "such",
        "take",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "this",
        "those",
        "through",
        "too",
        "use",
        "very",
        "want",
        "was",
        "way",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)

_MIN_COVERAGE_TOKENS = 2

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_COUNT_TOKEN = r"(?:\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_MINIMUM_QUANTITY_RE = re.compile(
    rf"\b(?:at\s+least|no\s+fewer\s+than|a\s+minimum\s+of|minimum(?:\s+of)?)\s+"
    rf"(?P<count>{_COUNT_TOKEN})\s+"
    r"(?P<label>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})",
    re.IGNORECASE,
)
_BOTH_SIDES_RE = re.compile(
    r"\bboth\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?=\s*[,.;]|$)",
    re.IGNORECASE,
)
_SHARED_HEAD_PAIR_RE = re.compile(
    r"\b(?P<left>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?)\s+and\s+"
    r"(?P<right>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?)\s+"
    r"(?P<head>complexit(?:y|ies)|comparison|contrast|tradeoffs?|effects?|results?)\b",
    re.IGNORECASE,
)
_CORRECT_ALTERNATIVE_RE = re.compile(
    r"\b(?:the\s+)?correct\s+alternative\b", re.IGNORECASE
)
_ALTERNATIVE_REQUEST_RE = re.compile(
    r"(?:"
    r"\bcorrect\s+(?:alternative|replacement)\b"
    r"|\b(?:alternative|replacement)\s+"
    r"(?:algorithm|method|approach|procedure|technique|tool|system)\b"
    r"|\b(?:what|which)\s+(?:is|would\s+be)\s+(?:the\s+|an?\s+)?"
    r"(?:alternative|replacement)\s+(?:to|for)\b"
    r"|\b(?:what|which)\s+(?:alternative|replacement)\s+"
    r"(?:should|can|could|would)\b[^.!?;]{0,40}\b(?:use|choose|select|apply|run)\b"
    r"|\b(?:algorithm|method|approach|procedure|technique|tool|system)\b"
    r"[^.!?;]{0,80}\b(?:used?\s+instead|alternative|replacement)\b"
    r"|\b(?:what|which|name|identify|explain)\b[^.!?;]{0,80}"
    r"\b(?:use|choose|select|apply|run|switch\s+to)\b[^.!?;]{0,50}"
    r"\binstead\b"
    r")",
    re.IGNORECASE,
)
_ALTERNATIVE_ACTION_RE = re.compile(
    r"\b(?:use|choose|select|prefer|apply|run|switch(?:es|ed)?\s+to|"
    r"replace(?:s|d)?(?:\s+[^.!?;]{0,60}?)?\s+with)\s+"
    r"(?:the\s+|an?\s+)?(?P<candidate>"
    r"[A-Za-z][A-Za-z0-9+*'-]*(?:\s+[A-Za-z][A-Za-z0-9+*'-]*){0,4}"
    r")",
    re.IGNORECASE,
)
_ALTERNATIVE_NOMINAL_RE = re.compile(
    r"(?P<candidate>[A-Za-z][A-Za-z0-9+*'-]*(?:\s+[A-Za-z][A-Za-z0-9+*'-]*){0,4})"
    r"\s+(?:is|would\s+be|becomes)\s+(?:the\s+|an?\s+)?"
    r"(?:alternative|replacement)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_REVERSE_NOMINAL_RE = re.compile(
    r"\b(?:the\s+)?(?:alternative|replacement)\s+"
    r"(?:is|would\s+be|becomes)\s+(?:the\s+|an?\s+)?"
    r"(?P<candidate>[A-Za-z][A-Za-z0-9+*'-]*"
    r"(?:\s+[A-Za-z][A-Za-z0-9+*'-]*){0,4})",
    re.IGNORECASE,
)
_ALTERNATIVE_CAPABILITY_RE = re.compile(
    r"(?P<candidate>[A-Za-z][A-Za-z0-9+*'-]*"
    r"(?:[ \t]+[A-Za-z][A-Za-z0-9+*'-]*){0,4})[ \t]+"
    r"(?:handles?|supports?|accepts?|works?|operates?|applies?)\b"
    r"(?P<context>[^.!?;]{0,120})",
    re.IGNORECASE,
)

_ALTERNATIVE_PREDICATE_SCAFFOLD = frozenset(
    {
        "cannot",
        "can't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "never",
        "no",
        "not",
        "won't",
    }
)
_ALTERNATIVE_EXPLICIT_RELATION_RE = re.compile(
    r"\b(?:alternative|instead|replacement|replaces?|switch(?:es|ed)?\s+to)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_CONDITION_RE = re.compile(
    r"\b(?:when|if)\b(?P<condition>[^.!?;]+)",
    re.IGNORECASE,
)
_GENERIC_ALTERNATIVE_TOKENS = frozenset(
    {
        "algorithm",
        "another",
        "approach",
        "alternative",
        "answer",
        "care",
        "caution",
        "different",
        "generic",
        "instead",
        "method",
        "other",
        "option",
        "replacement",
        "something",
        "solution",
        "technique",
        "the",
        "safe",
        "safer",
        "appropriate",
        "better",
        "capable",
        "more",
        "reliable",
        "resilient",
        "robust",
        "suitable",
    }
)
_QUALIFIED_COMPLEXITY_REQUEST_RE = re.compile(
    r"\b(?:time|space|runtime|memory|computational|algorithmic)\s+"
    r"complexit(?:y|ies)\b",
    re.IGNORECASE,
)
_PLAIN_COMPLEXITY_REQUEST_RE = re.compile(r"\bcomplexit(?:y|ies)\b", re.IGNORECASE)
_COMPUTATIONAL_CONTEXT_RE = re.compile(
    r"\b(?:algorithm|array|code|computation|data\s+structure|edge|graph|heap|"
    r"input\s+size|queue|runtime|sort|vertex)\w*\b",
    re.IGNORECASE,
)
_ASYMPTOTIC_NOTATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:O|Θ|Ω)\s*\([^\n)]{1,120}(?:\)[^\n)]{0,80})?\)",
    re.IGNORECASE,
)
_NAMED_COMPLEXITY_RE = re.compile(
    r"(?:"
    r"\b(?:time|space|runtime|memory|complexity|work|operations?|steps?)\b"
    r"[^.!?;]{0,48}"
    r"\b(?:constant|linear|logarithmic|quadratic|cubic|polynomial|"
    r"exponential|factorial|amortized|linearly|logarithmically|quadratically|"
    r"cubically|polynomially|exponentially|factorially)\b"
    r"|\b(?:constant|linear|logarithmic|quadratic|cubic|polynomial|"
    r"exponential|factorial|amortized)\s+(?:time|space|runtime|memory)\b"
    r")",
    re.IGNORECASE,
)
_SYMBOLIC_OPERATION_COUNT_RE = re.compile(
    r"\b(?:[A-Za-z]\s+log\s+[A-Za-z]|[A-Za-z]\s*(?:\^|\*\*)\s*\d+|"
    r"[A-Za-z]\s+(?:squared|cubed))\b[^.!?;]{0,32}"
    r"\b(?:operations?|steps?|work)\b",
    re.IGNORECASE,
)
_SCALING_RELATION_RE = re.compile(
    r"\b(?:runtime|memory|operations?|steps?|work|time|space)\b[^.!?;]{0,48}"
    r"\b(?:grows?|scales?|increases?)\b[^.!?;]{0,48}"
    r"\bin\s+(?:direct|inverse)?\s*proportion\s+to\b",
    re.IGNORECASE,
)
_RATE_VARIABLE_RE = re.compile(
    r"\b(?:with|as|in|per|over|against|according\s+to|relative\s+to|"
    r"depending\s+on|given)\b(?P<variable>[^.!?;]{1,80})",
    re.IGNORECASE,
)
_CHANGING_SUBJECT_RE = re.compile(
    r"\b(?:when|while)\b(?P<variable>[^.!?;]{1,80})\b"
    r"(?:expands?|grows?|increases?|changes?|varies?|shrinks?|decreases?)\b",
    re.IGNORECASE,
)
_EXTENT_MARKER_RE = re.compile(
    r"\b(?:size|length|count|number|cardinality|dimension|volume)\b",
    re.IGNORECASE,
)
_COMPUTATIONAL_EXTENT_RE = re.compile(
    r"(?:"
    r"\b(?:input|problem|data)\s+(?:size|length|count|cardinality|dimension|volume)\b"
    r"|\b(?:set|sequence|list|array|graph|tree|heap|queue|record|item|element|"
    r"vertex|edge|token|sample|batch|request)s?\s+"
    r"(?:size|length|count|cardinality|dimension|volume)\b"
    r"|\bnumber\s+of\s+(?:records?|items?|elements?|vertices|edges|tokens?|"
    r"samples?|requests?)\b"
    r"|(?<![A-Za-z0-9_])(?:n|v|e)(?![A-Za-z0-9_])"
    r")",
    re.IGNORECASE,
)
_QUALITATIVE_MODIFIER_RE = re.compile(
    r"(?:"
    r"\b(?:more|less|most|least)\b"
    r"|[A-Za-z]+(?:able|ible|ful|less|ous|ive|ent|ant|ed|er|est)\b"
    r")",
    re.IGNORECASE,
)
_NAMED_ENUMERATION_RE = re.compile(
    r"\b(?:vertices|nodes|items|options|cases|stages|phases)\s+"
    r"(?P<items>[A-Z0-9][A-Z0-9_-]*(?:\s*,\s*[A-Z0-9][A-Z0-9_-]*){1,15})",
)
_NUMBERED_ANSWER_SECTION_RE = re.compile(
    # A section marker starts at a token boundary.  Punctuation is not enough:
    # ``C-D:5. next`` is a weighted edge followed by prose, not answer item 5.
    # Whitespace still admits compact inline lists (``1. ... 2. ...``) and an
    # optional opening parenthesis admits Markdown headings such as ``## (2)``.
    # The existing exponent case remains excluded for the same structural
    # reason.
    r"(?<!\S)(?:\(\s*)?(?P<number>\d{1,2})\s*[.)]\s+"
)
_GRAPH_EDGE_RE = re.compile(
    r"(?:"
    r"\(\s*[A-Z]\s*[,\-]\s*[A-Z]\s*\)"
    r"|\(\s*[A-Z]{2}\s*\)"
    r"|\b[A-Z]\s*(?:->|→|--?|–|—|\bto\b)\s*[A-Z]\b"
    r")\s*(?::|=|\(|\[)?\s*-?\d+(?:\.\d+)?",
    re.IGNORECASE,
)

_RELATION_REQUEST_RE = re.compile(
    r"\b(?:distinguish|differentiate|separate|compare|contrast)\b"
    r"(?P<left>.+?)"
    r"(?:\bfrom\b|\bwith\b|\bversus\b|\bvs\.?\b|\band\b)"
    r"(?P<right>.+)",
    re.IGNORECASE,
)

# Surface forms that prove the same side of a requested distinction.  These
# are deliberately narrow semantic families, not a general synonym table.
# The important case is epistemic provenance: saying the word ``state`` does
# not satisfy "distinguish what you know from what you can only infer".
_COVERAGE_EQUIVALENCE = {
    # Natural check-ins commonly restate one intent twice: "Are you okay?
    # Feeling fine?" is not two independent tasks.  Collapsing these surface
    # forms into one semantic side lets concise direct answers satisfy the
    # request without disabling coverage for genuinely compound turns.
    "okay": "self_condition",
    "fine": "self_condition",
    "feel": "self_condition",
    "feeling": "self_condition",
    "steady": "self_condition",
    "condition": "self_condition",
    "know": "epistemic_known",
    "known": "epistemic_known",
    "knowing": "epistemic_known",
    "observe": "epistemic_known",
    "observed": "epistemic_known",
    "observation": "epistemic_known",
    "observations": "epistemic_known",
    "measure": "epistemic_known",
    "measured": "epistemic_known",
    "measurement": "epistemic_known",
    "measurements": "epistemic_known",
    "confirmed": "epistemic_known",
    "direct": "epistemic_known",
    "directly": "epistemic_known",
    "evidence": "epistemic_known",
    "infer": "epistemic_inferred",
    "inferred": "epistemic_inferred",
    "inference": "epistemic_inferred",
    "inferences": "epistemic_inferred",
    "inferential": "epistemic_inferred",
    "inferentially": "epistemic_inferred",
    "estimate": "epistemic_inferred",
    "estimated": "epistemic_inferred",
    "apparently": "epistemic_inferred",
    "likely": "epistemic_inferred",
    "presumably": "epistemic_inferred",
    "probably": "epistemic_inferred",
    "perhaps": "epistemic_inferred",
    "maybe": "epistemic_inferred",
    "seem": "epistemic_inferred",
    "seems": "epistemic_inferred",
    "seemed": "epistemic_inferred",
    "uncertain": "epistemic_inferred",
    "failed": "failure",
    "fails": "failure",
    "failing": "failure",
    "weighted": "weight",
    "weights": "weight",
    "instead": "alternative",
}

_EPISTEMIC_SIDES = frozenset({"epistemic_known", "epistemic_inferred"})
_CLAUSE_BOUNDARY_RE = re.compile(r"(?:[.!?;]+|\n+)")
_DIRECT_ASSERTION_RE = re.compile(
    r"\b(?:"
    r"i(?:'m|\s+am|\s+have|\s+feel|\s+see|\s+observe|\s+remember|"
    r"\s+can|\s+cannot|\s+do|\s+don't)|"
    r"my\s+[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3}\s+"
    r"(?:is|are|was|were|has|had|feels?|remains?|ended)|"
    r"(?:the|this|that|these|those)\s+[a-z][a-z'-]*"
    r"(?:\s+[a-z][a-z'-]*){0,4}\s+"
    r"(?:is|are|was|were|has|have|shows?|reads?|reports?|contains?|remains?|ended)"
    r")\b",
    re.IGNORECASE,
)


def coverage_tokens(text: Any) -> set[str]:
    """Return distinctive words that can prove an ask was engaged."""

    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text or "").lower())
    tokens: set[str] = set()
    for word in words:
        candidates = (word, *word.split("-")) if "-" in word else (word,)
        for candidate in candidates:
            stem = candidate.split("'", 1)[0]
            if candidate in _COVERAGE_STOPWORDS or stem in _COVERAGE_STOPWORDS:
                continue
            if len(stem) <= 2:
                continue
            tokens.add(_COVERAGE_EQUIVALENCE.get(stem, stem))
    return tokens


def _epistemic_partition_is_covered(body: Any) -> bool:
    """Return whether prose separates asserted evidence from inference.

    A direct assertion is not automatically synonymous with knowledge. It is
    admitted here only as one side of an explicit epistemic partition and only
    when another substantive clause marks itself as inference. This recognizes
    natural discourse such as ``I am steady. Inferentially, that may persist``
    without accepting a wholly speculative answer or requiring magic words.
    """

    direct_witness = False
    inferred_witness = False
    for raw_clause in _CLAUSE_BOUNDARY_RE.split(str(body or "")):
        clause = raw_clause.strip()
        words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", clause)
        if len(words) < 4:
            continue
        tokens = coverage_tokens(clause)
        if "epistemic_inferred" in tokens:
            inferred_witness = True
            continue
        if "epistemic_known" in tokens or _DIRECT_ASSERTION_RE.search(clause):
            direct_witness = True
    return direct_witness and inferred_witness


def requested_epistemic_partition_is_covered(request: Any, body: Any) -> bool:
    """Return whether a requested known/inferred distinction has both sides."""

    match = _RELATION_REQUEST_RE.search(str(request or ""))
    if match is None:
        return True
    left = coverage_tokens(match.group("left"))
    right = coverage_tokens(match.group("right"))
    if (left | right) != _EPISTEMIC_SIDES or left == right:
        return True
    return _epistemic_partition_is_covered(body)


def complete_epistemic_partition_from_evidence(
    request: Any,
    body: Any,
    evidence_body: Any,
) -> str:
    """Append only missing epistemic witnesses from an authoritative answer.

    This is a semantic merge, not a model instruction or a regenerated reply.
    It is deliberately usable only when the evidence answer itself proves both
    sides of the requested distinction.  A model-authored direct answer is
    retained; the smallest evidence clauses needed to satisfy the omitted
    known/inferred predicate are appended in their original wording.
    """

    draft = str(body or "").strip()
    evidence = str(evidence_body or "").strip()
    if requested_epistemic_partition_is_covered(request, draft):
        return draft
    if not draft or not evidence:
        return draft
    if not requested_epistemic_partition_is_covered(request, evidence):
        return draft

    draft_tokens = coverage_tokens(draft)
    needed = set(_EPISTEMIC_SIDES - draft_tokens)
    evidence_clauses = [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY_RE.split(evidence)
        if clause.strip()
    ]
    explicit: dict[str, str] = {}
    fallback_known = ""
    for clause in evidence_clauses:
        explicit_sides = coverage_tokens(clause) & _EPISTEMIC_SIDES
        for side in explicit_sides:
            explicit.setdefault(side, clause)
        if not fallback_known and _DIRECT_ASSERTION_RE.search(clause):
            fallback_known = clause

    additions: list[str] = []
    for side in ("epistemic_known", "epistemic_inferred"):
        if side not in needed:
            continue
        clause = explicit.get(side) or (
            fallback_known if side == "epistemic_known" else ""
        )
        if not clause:
            continue
        additions.append(clause.rstrip(".!?; ") + ".")
        needed.remove(side)
    if needed:
        return draft
    merged = " ".join((draft, *additions)).strip()
    return (
        merged
        if requested_epistemic_partition_is_covered(request, merged)
        else draft
    )


def _relation_sides_are_covered(
    segment: Any,
    body: Any,
    answered: set[str],
) -> bool | None:
    """Return whether both sides of an explicit relation were addressed.

    ``None`` means the segment is not an explicit compare/contrast request.
    A relation is stronger than ordinary lexical engagement: each named side
    needs its own witness in the answer.  Otherwise one shared context word
    can make a reply look complete while the requested distinction is absent.
    """

    match = _RELATION_REQUEST_RE.search(str(segment or ""))
    if match is None:
        return None
    left = coverage_tokens(match.group("left"))
    right = coverage_tokens(match.group("right"))
    if not left or not right:
        return None
    if (left | right) == _EPISTEMIC_SIDES and left != right:
        return _epistemic_partition_is_covered(body)
    return bool(left & answered) and bool(right & answered)


def _count_value(raw: Any) -> int | None:
    text = str(raw or "").strip().lower()
    if text.isdigit():
        return int(text)
    return _NUMBER_WORDS.get(text)


def _numbered_answer_sections(body: Any) -> dict[int, str]:
    """Return numbered response bodies without confusing exponents for items."""

    text = str(body or "")
    matches = list(_NUMBERED_ANSWER_SECTION_RE.finditer(text))
    if len(matches) < 2:
        return {}
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        number = int(match.group("number"))
        if number in sections:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[number] = text[match.end() : end].strip()
    return sections


def merge_numbered_answer_section(
    body: Any,
    number: int,
    completion: Any,
) -> str:
    """Compose model-authored text into one numbered answer section.

    A generation can stop after opening the section it has not finished.  An
    append-only retry must then extend that section rather than add a duplicate
    marker: the coverage reader deliberately treats the first marker as the
    authoritative structural position.  If the marker is absent, insert it
    before the next higher numbered section so the user's order is preserved.
    """

    text = str(body or "").rstrip()
    tail = str(completion or "").strip()
    try:
        target = int(number)
    except (TypeError, ValueError, OverflowError):
        return text
    if target <= 0 or not tail:
        return text

    matches = list(_NUMBERED_ANSWER_SECTION_RE.finditer(text))
    for index, match in enumerate(matches):
        if int(match.group("number")) != target:
            continue
        section_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        section_body = text[match.end() : section_end].rstrip()
        addition = f"{section_body}\n\n{tail}" if section_body else tail
        return f"{text[:match.end()]}{addition}{text[section_end:]}".rstrip()

    marker = f"{target}. {tail}"
    for match in matches:
        if int(match.group("number")) > target:
            prefix = text[: match.start()].rstrip()
            suffix = text[match.start() :].lstrip()
            return f"{prefix}\n\n{marker}\n\n{suffix}".strip()
    if not text:
        return marker
    return f"{text}\n\n{marker}"


def _semantic_group_is_covered(group: set[str], body: Any) -> bool:
    """Require every distinctive term in one explicitly named obligation."""

    if not group:
        return True
    return group <= coverage_tokens(body)


def _alternative_candidate_tokens(candidate: Any) -> set[str]:
    """Return the substantive name carried by a replacement-method witness."""

    return {
        token
        for token in coverage_tokens(candidate)
        if token not in _GENERIC_ALTERNATIVE_TOKENS
    }


def _concept_tokens(text: Any) -> set[str]:
    """Normalize light inflection without introducing a domain ontology."""

    normalized: set[str] = set()
    for token in coverage_tokens(text):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        normalized.add(token)
    return normalized


def _computational_complexity_requested(segment: Any) -> bool:
    text = str(segment or "")
    if _QUALIFIED_COMPLEXITY_REQUEST_RE.search(text):
        return True
    return bool(
        _PLAIN_COMPLEXITY_REQUEST_RE.search(text)
        and _COMPUTATIONAL_CONTEXT_RE.search(text)
    )


def _candidate_is_substantive(candidate: Any) -> bool:
    raw = str(candidate or "").strip()
    tokens = _alternative_candidate_tokens(raw)
    if not tokens:
        return False
    if tokens <= {
        "care",
        "caution",
        "different",
        "other",
        "something",
    }:
        return False
    # An adjective plus a generic method class does not identify a replacement.
    # The grammatical shape admits named and symbolic candidates and concrete
    # noun compounds without an approved-algorithm registry.
    words = re.findall(r"[A-Za-z][A-Za-z0-9+*'-]*", raw)
    while words and words[-1].casefold() in {
        "instead",
        "alternative",
        "replacement",
    }:
        words.pop()
    if not words:
        return False
    # A negative capability clause is not the name of a replacement. Regex
    # backtracking can otherwise reinterpret "Dijkstra's algorithm does not
    # work" as candidate="Dijkstra's algorithm does not", predicate="work".
    # Keep this grammar-level: it rejects auxiliary/negation scaffolding
    # without maintaining a registry of valid algorithms or tools.
    if {word.casefold() for word in words} & _ALTERNATIVE_PREDICATE_SCAFFOLD:
        return False
    has_identity_shape = any(
        any(char.isupper() or char.isdigit() for char in word) or "-" in word
        for word in words
    )
    if has_identity_shape:
        return True
    if _QUALITATIVE_MODIFIER_RE.fullmatch(words[0]):
        return False
    # Lowercase candidates need positive structural identity. Three-token names
    # carry enough internal structure to distinguish them from adjective-plus-
    # category advice. Short names need a symbol, capitalization, or hyphen.
    if len(words) >= 3:
        return True
    return False


def _computational_complexity_is_covered(body: Any) -> bool:
    """Require a quantitative growth witness tied to problem size."""

    text = str(body or "")
    if _ASYMPTOTIC_NOTATION_RE.search(text) or _SYMBOLIC_OPERATION_COUNT_RE.search(text):
        return True
    for clause in _CLAUSE_BOUNDARY_RE.split(text):
        if not clause.strip():
            continue
        has_rate = _NAMED_COMPLEXITY_RE.search(clause)
        has_proportion = _SCALING_RELATION_RE.search(clause)
        if has_rate is None and has_proportion is None:
            continue
        explicit_variable = _RATE_VARIABLE_RE.search(clause)
        changing_subject = _CHANGING_SUBJECT_RE.search(clause)
        variable = explicit_variable or changing_subject
        if variable is None:
            return True
        variable_text = variable.group("variable")
        if _COMPUTATIONAL_EXTENT_RE.search(variable_text):
            return True
        if (
            changing_subject is None
            and has_proportion is None
            and _EXTENT_MARKER_RE.search(variable_text) is None
        ):
            # "quadratic with an array" names the implementation, while
            # "quadratic in font size" names a non-computational variable.
            return True
    return False


def _alternative_is_covered(segment: Any, body: Any) -> bool | None:
    """Require an actual replacement method when the request asks for one.

    A bare occurrence of ``algorithm`` or ``use`` is not an answer to "what
    should be used instead?".  The response must bind the replacement relation
    to a substantive candidate (for example ``use Bellman-Ford`` or
    ``breadth-first search is the alternative``).  This is domain-neutral: the
    candidate is extracted from the response rather than looked up in a table
    of known algorithms.
    """

    request = str(segment or "")
    if _ALTERNATIVE_REQUEST_RE.search(request) is None:
        return None
    response = str(body or "")
    condition = _ALTERNATIVE_CONDITION_RE.search(request)
    condition_tokens = (
        _concept_tokens(condition.group("condition")) if condition is not None else set()
    )

    if _ALTERNATIVE_EXPLICIT_RELATION_RE.search(response):
        for pattern in (
            _ALTERNATIVE_ACTION_RE,
            _ALTERNATIVE_NOMINAL_RE,
            _ALTERNATIVE_REVERSE_NOMINAL_RE,
        ):
            for match in pattern.finditer(response):
                candidate = match.group("candidate")
                if _candidate_is_substantive(candidate):
                    return True

    for match in _ALTERNATIVE_CAPABILITY_RE.finditer(response):
        if not _candidate_is_substantive(match.group("candidate")):
            continue
        if not condition_tokens:
            return True
        if condition_tokens & _concept_tokens(match.group("context")):
            return True
    return False


def _minimum_quantity_is_covered(segment: Any, body: Any) -> bool | None:
    """Prove a requested minimum from an explicit or structured witness.

    This does not infer that mentioning a plural noun satisfies a cardinality.
    Domain readers can supply exact structured counts; graph edges are the first
    such reader because their endpoint/weight syntax is independently parsable.
    ``None`` means no minimum quantity was requested.
    """

    request = _MINIMUM_QUANTITY_RE.search(str(segment or ""))
    if request is None:
        return None
    minimum = _count_value(request.group("count"))
    if minimum is None:
        return False
    label_tokens = coverage_tokens(request.group("label"))
    label_head = next(reversed(sorted(label_tokens, key=len)), "")
    text = str(body or "")

    explicit = re.compile(
        rf"\b(?P<count>{_COUNT_TOKEN})\s+"
        rf"(?:[A-Za-z][A-Za-z'-]*\s+){{0,2}}{re.escape(label_head)}(?:s|es)?\b",
        re.IGNORECASE,
    ) if label_head else None
    if explicit is not None:
        for match in explicit.finditer(text):
            value = _count_value(match.group("count"))
            if value is not None and value >= minimum:
                return True

    if "edge" in label_tokens or "edges" in str(request.group("label")).lower():
        unique_edges = {match.group(0).casefold() for match in _GRAPH_EDGE_RE.finditer(text)}
        return len(unique_edges) >= minimum
    return False


def _strong_segment_obligations_are_covered(segment: Any, body: Any) -> bool:
    """Check constraints whose semantics are stronger than lexical overlap."""

    text = str(segment or "")
    quantity = _minimum_quantity_is_covered(text, body)
    if quantity is False:
        return False

    if _computational_complexity_requested(text) and not _computational_complexity_is_covered(
        body
    ):
        return False

    alternative = _alternative_is_covered(text, body)
    if alternative is False:
        return False

    both = _BOTH_SIDES_RE.search(text)
    if both is not None:
        left = coverage_tokens(both.group("left"))
        right = coverage_tokens(both.group("right"))
        if not (
            _semantic_group_is_covered(left, body)
            and _semantic_group_is_covered(right, body)
        ):
            return False

    shared_head = _SHARED_HEAD_PAIR_RE.search(text)
    if shared_head is not None:
        left = coverage_tokens(shared_head.group("left"))
        right = coverage_tokens(shared_head.group("right"))
        if not (
            _semantic_group_is_covered(left, body)
            and _semantic_group_is_covered(right, body)
        ):
            return False

    if _CORRECT_ALTERNATIVE_RE.search(text) and alternative is not True:
        return False

    enumeration = _NAMED_ENUMERATION_RE.search(text)
    if enumeration is not None:
        requested = [item.strip() for item in enumeration.group("items").split(",")]
        answer_text = str(body or "")
        if any(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])", answer_text)
            is None
            for item in requested
        ):
            return False
    return True


def unanswered_question_parts(body: Any, contract: object | None) -> list[str]:
    """Return substantive asks a reply never engages with at all.

    This intentionally fails open unless the upstream prompt-shape contract
    already classified the turn as requiring single-reply coverage. A segment
    is missing only when it has at least two distinctive words and shares none
    with the reply. The check catches a wholly dropped part without grading
    answer quality or punishing concise prose.
    """

    if not getattr(contract, "requires_single_reply_coverage", False):
        return []
    segments = tuple(getattr(contract, "question_segments", ()) or ())
    if len(segments) < 2:
        return []

    answered = coverage_tokens(body)
    numbered_sections = _numbered_answer_sections(body)
    numbered_markers = re.findall(r"(?:^|\n|\s)\d+\s*[.)]", str(body or ""))
    if len(numbered_markers) >= 2:
        answered.add("numbered")
    try:
        numbered_parts = max(0, int(getattr(contract, "numbered_parts", 0) or 0))
    except (TypeError, ValueError):
        numbered_parts = 0
    numbered_start = max(0, len(segments) - numbered_parts)
    missed: list[str] = []
    for index, segment in enumerate(segments):
        if is_reply_shape_constraint_segment(segment):
            continue
        numbered_section = index - numbered_start + 1
        local_body = str(body or "")
        local_answered = answered
        if numbered_parts >= 3 and index >= numbered_start and numbered_sections:
            local_body = numbered_sections.get(numbered_section, "")
            if not local_body:
                missed.append(str(segment))
                continue
            local_answered = coverage_tokens(local_body)
            # The parsed section marker is itself the structural witness for
            # "numbered". Requiring the section body to repeat that adjective
            # made a natural heading such as ``## (2) Pseudocode`` impossible
            # to satisfy even though the answer was visibly numbered.
            local_answered.add("numbered")
        if not _strong_segment_obligations_are_covered(segment, local_body):
            missed.append(str(segment))
            continue
        local_answered = set(local_answered)
        if (
            _computational_complexity_requested(segment)
            and _computational_complexity_is_covered(local_body)
        ):
            local_answered.add("complexity")
        if _alternative_is_covered(segment, local_body) is True:
            local_answered.add("alternative")
        relation_covered = _relation_sides_are_covered(
            segment, local_body, local_answered
        )
        if relation_covered is False:
            missed.append(str(segment))
            continue
        if relation_covered is True:
            continue
        wanted = coverage_tokens(segment)
        if len(wanted) < _MIN_COVERAGE_TOKENS:
            continue
        overlap = wanted & local_answered
        # Numbered multipart requests carry independent explicit obligations.
        # One shared context word cannot prove one of those obligations was
        # answered: "weights" in a graph example must not satisfy a later ask
        # about negative-weight failure. Ordinary short conversation keeps the
        # one-anchor rule so concise natural answers remain valid.
        required_anchors = (
            min(2, len(wanted))
            if numbered_parts >= 3 and index >= numbered_start
            else 1
        )
        if len(overlap) < required_anchors:
            missed.append(str(segment))
    return missed


__all__ = [
    "complete_epistemic_partition_from_evidence",
    "coverage_tokens",
    "merge_numbered_answer_section",
    "requested_epistemic_partition_is_covered",
    "unanswered_question_parts",
]
