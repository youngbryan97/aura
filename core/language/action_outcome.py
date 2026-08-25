"""Language substrate for references to completed or failed actions.

People normally ask about an earlier action without repeating its objective:
``what went wrong there?``, ``why did that break?``, or ``how did it go?``.
Those are questions about an action episode, not instructions to execute one.
This module identifies that grammatical relation without naming any tool,
application, or task domain.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["ActionOutcomeQuestion", "action_outcome_question"]


@dataclass(frozen=True, slots=True)
class ActionOutcomeQuestion:
    """The bounded outcome relation expressed by one utterance."""

    asks_about_outcome: bool = False
    asks_about_failure: bool = False
    referential: bool = False


_FAILURE_PREDICATE = (
    r"(?:break|broke|broken|fail|failed|failing|go\s+wrong|went\s+wrong|"
    r"not\s+work|didn['’]?t\s+work|stop|stopped|error(?:ed)?|crash(?:ed)?)"
)
_OUTCOME_PREDICATE = (
    rf"(?:{_FAILURE_PREDICATE}|happen(?:ed)?|turn(?:ed)?\s+out|go|went|finish(?:ed)?|"
    r"complete(?:d)?|succeed(?:ed)?)"
)
_REFERENCE = r"(?:it|that|this|there|the\s+(?:attempt|action|task|run|operation|process))"

_FAILURE_QUESTION_RE = re.compile(
    rf"(?:\b(?:why|how)\b[^?.!;]{{0,80}}\b{_FAILURE_PREDICATE}\b"
    rf"|\bwhat\s+(?:exactly\s+)?(?:went\s+wrong|broke|failed)\b"
    rf"|\b(?:do|did|can|could)\s+you\s+(?:know|tell|explain)\b[^?.!;]{{0,80}}"
    rf"\b(?:why|how)\b[^?.!;]{{0,60}}\b{_FAILURE_PREDICATE}\b)",
    re.IGNORECASE,
)
_OUTCOME_QUESTION_RE = re.compile(
    rf"(?:\b(?:what|why|how)\b[^?.!;]{{0,80}}\b{_OUTCOME_PREDICATE}\b"
    rf"|\bhow\s+did\s+{_REFERENCE}\s+go\b"
    rf"|\bwhat\s+happened\b)",
    re.IGNORECASE,
)
_REFERENTIAL_RE = re.compile(
    rf"\b{_REFERENCE}\b|\b(?:earlier|before|previous(?:ly)?|last\s+(?:time|attempt|run))\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:why|how|what)\s+(?:would|could|should|might)\b"
    r"|\b(?:hypothetically|in\s+theory|suppose|imagine)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def action_outcome_question(text: str) -> ActionOutcomeQuestion:
    """Return the action-outcome relation expressed by ``text``.

    The function abstains on statements such as ``the task failed``. A
    question or explicit request for an explanation is required, because a
    failure report alone may be new evidence rather than a request for recall.
    """

    normalized = _normalize(text)
    if not normalized or _HYPOTHETICAL_RE.search(normalized):
        return ActionOutcomeQuestion()
    asks_failure = bool(_FAILURE_QUESTION_RE.search(normalized))
    asks_outcome = asks_failure or bool(_OUTCOME_QUESTION_RE.search(normalized))
    if not asks_outcome:
        return ActionOutcomeQuestion()
    return ActionOutcomeQuestion(
        asks_about_outcome=True,
        asks_about_failure=asks_failure,
        referential=bool(_REFERENTIAL_RE.search(normalized)),
    )
