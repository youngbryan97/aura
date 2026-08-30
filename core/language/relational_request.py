"""Shared parsing and coverage for explicit two-subject relations."""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_RELATION_PATTERNS = (
    (
        re.compile(
            r"\b(?:difference|differences|distinction|distinctions|contrast|contrasts)"
            r"\s+between\s+(?P<left>[^?.;\n]{2,80}?)\s+and\s+"
            r"(?P<right>[^?.;\n]{2,80}?)\s*[?.;\n]",
            re.IGNORECASE,
        ),
        True,
    ),
    (
        re.compile(
            r"\b(?:compare|contrast)\s+(?P<left>[^?.;\n]{2,80}?)\s+"
            r"(?:and|with|to|against)\s+(?P<right>[^?.;\n]{2,80}?)\s*[?.;\n]",
            re.IGNORECASE,
        ),
        True,
    ),
    (
        re.compile(
            r"\b(?:distinguish|differentiate|separate)\s+"
            r"(?P<left>[^?.;\n]{2,80}?)\s+(?:from|and)\s+"
            r"(?P<right>[^?.;\n]{2,80}?)\s*[?.;\n]",
            re.IGNORECASE,
        ),
        False,
    ),
    (
        re.compile(
            r"\b(?:how\s+(?:does|do|did|would|will)\s+|what\s+makes\s+)"
            r"(?P<left>[^?.;\n]{2,80}?)\s+"
            r"(?:differ(?:s)?|different)\s+from\s+"
            r"(?P<right>[^?.;\n]{2,80}?)\s*[?.;\n]",
            re.IGNORECASE,
        ),
        False,
    ),
    (
        re.compile(
            r"\b(?P<left>[^?.;\n]{2,80}?)\s+differs?\s+from\s+"
            r"(?P<right>[^?.;\n]{2,80}?)\s*[?.;\n]",
            re.IGNORECASE,
        ),
        True,
    ),
    (
        re.compile(
            r"\b(?P<left>[^?.;\n]{2,60}?)\s+(?:versus|vs\.?)\s+"
            r"(?P<right>[^?.;\n]{2,60}?)\s*[?.;\n]",
            re.IGNORECASE,
        ),
        True,
    ),
)
_INSTRUCTION_SIDE_RE = re.compile(
    r"\b(?:explain|describe|list|enumerate|verify|test|prove|validate|choose|"
    r"recommend|prefer|tell|show|why|how|what|give|walk)\b",
    re.IGNORECASE,
)
_SUBJECT_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "against",
        "also",
        "among",
        "an",
        "and",
        "answer",
        "because",
        "before",
        "being",
        "both",
        "could",
        "design",
        "does",
        "each",
        "every",
        "explain",
        "enumerate",
        "from",
        "have",
        "into",
        "itself",
        "list",
        "more",
        "most",
        "other",
        "should",
        "some",
        "stronger",
        "such",
        "than",
        "that",
        "the",
        "this",
        "their",
        "then",
        "there",
        "these",
        "they",
        "through",
        "those",
        "under",
        "using",
        "verify",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
    }
)


def _subject_tokens(text: Any) -> tuple[str, ...]:
    return tuple(
        token
        for token in (word.casefold() for word in _WORD_RE.findall(str(text or "")))
        if token not in _SUBJECT_STOPWORDS and len(token) > 1
    )


def compared_subjects(
    request: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return the two named subjects of an explicit comparison request."""

    text = str(request or "").strip()
    if not text:
        return None
    probe = text if text.endswith(("?", ".", ";", "\n")) else text + "."
    for pattern, reject_instruction_sides in _RELATION_PATTERNS:
        match = pattern.search(probe)
        if match is None:
            continue
        left_text = match.group("left").strip()
        right_text = match.group("right").strip()
        if reject_instruction_sides and (
            _INSTRUCTION_SIDE_RE.search(left_text) or _INSTRUCTION_SIDE_RE.search(right_text)
        ):
            continue
        left = _subject_tokens(left_text)
        right = _subject_tokens(right_text)
        if left and right and set(left) != set(right):
            return left, right
    return None


def comparison_sides_are_covered(answer: Any, request: Any) -> bool | None:
    """Return side coverage, or ``None`` when no explicit pair was requested."""

    subjects = compared_subjects(request)
    if subjects is None:
        return None
    answer_tokens = set(_subject_tokens(answer))
    left, right = subjects
    return bool(answer_tokens & set(left)) and bool(answer_tokens & set(right))


__all__ = ["compared_subjects", "comparison_sides_are_covered"]
