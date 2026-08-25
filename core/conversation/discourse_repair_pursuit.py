"""Typed conversation state for a user pursuing an unanswered question focus.

A repeated question is not always a new request.  In ordinary dialogue,
"well then how did you know" after an answer to "how'd you know that" rejects
the answer while preserving the interrogative focus.  Treating the previous
assistant text as an ordinary few-shot example anchors the next decode on the
very wording the person just pursued.

This module identifies that relation from question structure and returns a
typed state.  It does not generate a repair sentence.  The consumer can remove
the rejected assistant turn from model history while retaining the user's
questions and the earlier referent-bearing context.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "DiscourseRepairPursuit",
    "QuestionFocus",
    "apply_repair_pursuit_to_history",
    "build_repair_pursuit",
    "question_focus",
]

_SCHEMA = "aura.discourse_repair_pursuit.v1"
_WORD_RE = re.compile(r"[a-z0-9']+")
_CONTRACTIONS = {
    "how'd": "how did",
    "what'd": "what did",
    "who'd": "who did",
    "where'd": "where did",
    "when'd": "when did",
    "why'd": "why did",
}
_PURSUIT_LEAD_RE = re.compile(
    r"^(?:(?:well|but|no|so|then|still|again|actually|okay|ok|wait)\b[\s,;:]*)+",
    re.IGNORECASE,
)
_FOCUS_RE = re.compile(
    r"\b(?P<focus>how\s+(?:many|much|long|old|far|often)|how|why|where|when|who|whom|whose|which|what)\b",
    re.IGNORECASE,
)
_HOW_STATE_RE = re.compile(
    r"\bhow\s+(?:am|are|is|was|were|feel|feels|felt|seem|seems|look|looks)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "i",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "that",
        "the",
        "then",
        "this",
        "those",
        "to",
        "was",
        "were",
        "well",
        "will",
        "would",
        "you",
        "your",
    }
)


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("’", "'").replace("`", "'")
    for source, replacement in _CONTRACTIONS.items():
        text = text.replace(source, replacement)
    return " ".join(text.split()).strip()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_normalize(value).encode("utf-8")).hexdigest()


def _focus_kind(match_text: str, whole_text: str) -> str:
    focus = match_text.casefold()
    if focus == "why":
        return "cause"
    if focus == "where":
        return "location"
    if focus == "when":
        return "time"
    if focus in {"who", "whom", "whose"}:
        return "identity"
    if focus == "which":
        return "selection"
    if focus == "what":
        return "entity_or_description"
    if focus == "how many" or focus == "how much":
        return "quantity"
    if focus == "how long":
        return "duration"
    if focus == "how old":
        return "age"
    if focus == "how far":
        return "distance"
    if focus == "how often":
        return "frequency"
    if _HOW_STATE_RE.search(whole_text):
        return "state_or_degree"
    return "mechanism_or_manner"


@dataclass(frozen=True, slots=True)
class QuestionFocus:
    """The answer-bearing dimension of one open question."""

    kind: str
    terms: tuple[str, ...]
    question: str


@dataclass(frozen=True, slots=True)
class DiscourseRepairPursuit:
    """A later turn that keeps the prior focus open and rejects its answer."""

    active: bool = False
    focus_kind: str = ""
    focus_terms: tuple[str, ...] = ()
    prior_question: str = ""
    current_question: str = ""
    prior_reply_fingerprint: str = ""
    relation: str = ""
    schema: str = _SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def question_focus(value: Any) -> QuestionFocus | None:
    """Return the interrogative focus and stable content terms, if present."""

    normalized = _normalize(value)
    if not normalized:
        return None
    match = _FOCUS_RE.search(normalized)
    if match is None:
        return None
    focus_text = str(match.group("focus") or "").casefold()
    terms = tuple(
        dict.fromkeys(
            token
            for token in _WORD_RE.findall(normalized[match.end() :])
            if token not in _STOPWORDS and token not in focus_text.split()
        )
    )
    return QuestionFocus(
        kind=_focus_kind(focus_text, normalized),
        terms=terms,
        question=" ".join(str(value or "").split())[:420],
    )


def _same_focus(left: QuestionFocus, right: QuestionFocus) -> bool:
    if left.kind != right.kind:
        return False
    left_terms = set(left.terms)
    right_terms = set(right.terms)
    if not left_terms or not right_terms:
        return not left_terms and not right_terms
    overlap = len(left_terms & right_terms)
    return bool(overlap and overlap / min(len(left_terms), len(right_terms)) >= 0.75)


def build_repair_pursuit(
    current_question: Any,
    recent_exchanges: Sequence[dict[str, Any]] | Any,
) -> DiscourseRepairPursuit:
    """Classify a repeated open focus as user-initiated repair pursuit.

    The current turn must repeat the latest user question's focus after an
    assistant answer.  A discourse pursuit lead (``well then``, ``but``,
    ``still``...) or a near-verbatim repeat supplies the rejection evidence.
    Mere topical similarity does not.
    """

    current_text = " ".join(str(current_question or "").split()).strip()
    current_focus = question_focus(current_text)
    if current_focus is None or not isinstance(recent_exchanges, Sequence):
        return DiscourseRepairPursuit()

    prior: dict[str, Any] | None = None
    for candidate in reversed(list(recent_exchanges)):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("user") or "").strip() and str(
            candidate.get("aura") or ""
        ).strip():
            prior = candidate
            break
    if prior is None:
        return DiscourseRepairPursuit()

    prior_question = " ".join(str(prior.get("user") or "").split()).strip()
    prior_reply = " ".join(str(prior.get("aura") or "").split()).strip()
    prior_focus = question_focus(prior_question)
    if prior_focus is None or not _same_focus(prior_focus, current_focus):
        return DiscourseRepairPursuit()

    current_normalized = _normalize(current_text)
    prior_normalized = _normalize(prior_question)
    pursuit_lead = bool(_PURSUIT_LEAD_RE.match(current_normalized))
    repeated = _PURSUIT_LEAD_RE.sub("", current_normalized).strip(" ?!.,") == (
        _PURSUIT_LEAD_RE.sub("", prior_normalized).strip(" ?!.,")
    )
    if not pursuit_lead and not repeated:
        return DiscourseRepairPursuit()

    return DiscourseRepairPursuit(
        active=True,
        focus_kind=current_focus.kind,
        focus_terms=current_focus.terms,
        prior_question=prior_question[:420],
        current_question=current_text[:420],
        prior_reply_fingerprint=_fingerprint(prior_reply),
        relation="user_reopens_same_interrogative_focus",
    )


def apply_repair_pursuit_to_history(
    messages: Sequence[dict[str, str]],
    pursuit: DiscourseRepairPursuit | dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Withdraw the rejected assistant answer, preserving every other turn."""

    if isinstance(pursuit, dict):
        active = bool(pursuit.get("active"))
        fingerprint = str(pursuit.get("prior_reply_fingerprint") or "")
    elif isinstance(pursuit, DiscourseRepairPursuit):
        active = pursuit.active
        fingerprint = pursuit.prior_reply_fingerprint
    else:
        return [dict(message) for message in messages]
    result = [dict(message) for message in messages]
    if not active or not fingerprint:
        return result
    for index in range(len(result) - 1, -1, -1):
        message = result[index]
        if str(message.get("role") or "").casefold() != "assistant":
            continue
        if _fingerprint(message.get("content")) == fingerprint:
            result.pop(index)
        break
    return result
