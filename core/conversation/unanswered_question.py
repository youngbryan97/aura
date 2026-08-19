"""The question they say went unanswered, read back rather than guessed.

LIVE 2026-08-19. The turn before was "if you had to pick between being useful
and being understood, which would you choose?", answered well. Then:

    you didn't answer my question
    -> "I'm sorry, I got distracted. You asked about my neurochemistry."

Nothing had been said about neurochemistry. The complaint refers to a specific
earlier turn, and no reading was attached to it, so the reply had to guess
which question was meant — and guessing which question a person asked is the
one thing that cannot be recovered from, because the apology makes it sound
settled.

This reading holds the last thing they actually asked, and what she actually
said back, so the turn can be answered from the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.conversation.word_markers import names_any

__all__ = [
    "UNANSWERED_HEADER",
    "UnansweredExchange",
    "complains_the_question_went_unanswered",
    "last_exchange",
    "unanswered_question_block",
]

UNANSWERED_HEADER = "## THE QUESTION THEY SAY YOU DID NOT ANSWER"

#: Ways of saying "go back and answer what I asked". Each names the ACT of
#: answering or the question itself; none of them names its subject, which is
#: exactly why the subject has to come from the transcript.
_COMPLAINT_MARKERS = (
    "you didn't answer",
    "you did not answer",
    "you never answered",
    "you still haven't answered",
    "you avoided",
    "you dodged",
    "you ignored my question",
    "you ignored the question",
    "that's not what i asked",
    "that is not what i asked",
    "not what i asked",
    "that wasn't my question",
    "that was not my question",
    "answer the question",
    "answer my question",
    "i asked you something else",
    "you went off on a tangent",
    "you changed the subject",
    "you didn't address",
    "you did not address",
)


@dataclass(frozen=True)
class UnansweredExchange:
    """What they asked last, and what she said back."""

    question: str
    reply: str


def complains_the_question_went_unanswered(prompt: Any) -> bool:
    """True when the turn says an earlier question is still open."""
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    return names_any(text, _COMPLAINT_MARKERS)


def _pairs() -> list[tuple[str, str]]:
    """(their question, her reply) for this conversation, oldest first."""
    try:
        from core.conversation.grounded_recall import _transcript_own_exchanges

        return list(_transcript_own_exchanges(""))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return []


def last_exchange(*, before: Any = None) -> UnansweredExchange | None:
    """The most recent exchange that is not the complaint itself."""
    complaint = " ".join(str(before or "").split()).strip().lower()
    for question, reply in reversed(_pairs()):
        asked = " ".join(str(question or "").split()).strip()
        if not asked:
            continue
        if complaint and asked.lower() == complaint:
            continue
        return UnansweredExchange(question=asked, reply=" ".join(str(reply or "").split()))
    return None


def unanswered_question_block(prompt: Any) -> str:
    """The open question in their own words, or "" when there is no record."""
    if not complains_the_question_went_unanswered(prompt):
        return ""
    exchange = last_exchange(before=prompt)
    if exchange is None:
        # A named absence beats an invented question. Saying which question
        # they mean is theirs to do when the record cannot.
        return (
            "They say a question of theirs went unanswered, and this "
            "conversation holds no earlier question to check against. Ask "
            "which one they mean rather than naming one."
        )
    lines = [
        "This is the last thing they actually asked, from the transcript:",
        f'- they asked: "{exchange.question[:400]}"',
    ]
    if exchange.reply:
        lines.append(f'- you replied: "{exchange.reply[:400]}"')
    lines.append(
        "Answer THAT question. Do not name a different subject — the one time "
        "this went wrong, an apology was attached to a question nobody had asked."
    )
    return "\n".join(lines)
