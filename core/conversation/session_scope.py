"""Request-local conversation identity shared by chat, memory, and speech."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

MAX_CONVERSATION_ID_CHARS: Final = 128
MAX_CONVERSATION_TURN_ID_CHARS: Final = 128

conversation_session_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_conversation_session",
    default="",
)
conversation_turn_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_conversation_turn",
    default="",
)

#: What the person actually asked on this turn.
#:
#: LIVE DEFECT, 2026-08-19. "what is 7919 * 6367?" was answered with "the live
#: answer lane could not finish preparing". The runtime can compute that
#: exactly, and did not, because the code choosing the failure message had no
#: idea what had been asked — every degraded path returns a sentence about the
#: LANE rather than about the question. A fact the machine holds must not be
#: suppressed by a lane that was not ready to say it.
#:
#: Turn-scoped like the ids above, so a background tick cannot see a
#: foreground question or answer with one.
user_question_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_user_question",
    default="",
)

MAX_USER_QUESTION_CHARS: Final = 4000

#: Evidence actually handed to the model this turn.
#:
#: A MUTABLE set, replaced once per turn, because a ContextVar set inside a
#: child task does not propagate back to the parent — asyncio gives children a
#: COPY of the context. Readings are taken in child tasks and checked in the
#: parent, so the container is shared and the children mutate it.
_TURN_EVIDENCE: contextvars.ContextVar[set[str]] = contextvars.ContextVar(
    "aura_turn_evidence", default=frozenset()
)

LOCAL_CONVERSATION_ID: Final = "local"
# Native windows are an owner surface, not anonymous internal cognition.  The
# HTTP desktop and voice routes already derive this principal key for the same
# machine, so non-HTTP owner surfaces must join it rather than minting a third
# local conversation.
LOCAL_OWNER_CONVERSATION_ID: Final = "127.0.0.1"


def normalize_conversation_id(value: object) -> str:
    normalized = " ".join(str(value or "").strip().split())
    if any(ord(character) < 32 for character in normalized):
        return ""
    return normalized[:MAX_CONVERSATION_ID_CHARS]


def current_conversation_session(default: str = "") -> str:
    return normalize_conversation_id(conversation_session_var.get()) or normalize_conversation_id(
        default
    )


def normalize_conversation_turn_id(value: object) -> str:
    normalized = " ".join(str(value or "").strip().split())
    if any(ord(character) < 32 for character in normalized):
        return ""
    return normalized[:MAX_CONVERSATION_TURN_ID_CHARS]


def current_conversation_turn(default: str = "") -> str:
    return normalize_conversation_turn_id(
        conversation_turn_var.get()
    ) or normalize_conversation_turn_id(default)


@contextmanager
def conversation_session_scope(session_id: str) -> Iterator[str]:
    normalized = normalize_conversation_id(session_id)
    if not normalized:
        raise ValueError("conversation session identity is required")
    token = conversation_session_var.set(normalized)
    try:
        yield normalized
    finally:
        conversation_session_var.reset(token)


__all__ = [
    "LOCAL_CONVERSATION_ID",
    "LOCAL_OWNER_CONVERSATION_ID",
    "MAX_CONVERSATION_ID_CHARS",
    "MAX_CONVERSATION_TURN_ID_CHARS",
    "conversation_session_scope",
    "conversation_session_var",
    "conversation_turn_var",
    "current_conversation_session",
    "current_conversation_turn",
    "normalize_conversation_id",
    "normalize_conversation_turn_id",
    "record_solved_answer",
    "solved_answers",
]


def set_user_question(text: object) -> None:
    """Record what this turn was asked, for anything that needs to answer it."""
    body = " ".join(str(text or "").split())[:MAX_USER_QUESTION_CHARS]
    user_question_var.set(body)
    # A fresh container per turn, so evidence from the previous one cannot be
    # mistaken for evidence in hand now.
    _TURN_EVIDENCE.set(set())
    _TURN_SOLVED.set({})


#: An exact answer worked out before the model was asked. A dict for the same
#: reason the evidence set is one: children get a copy of the context, so the
#: container is shared and the child mutates it.
_TURN_SOLVED: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "aura_turn_solved", default={}
)


def record_solved_answer(name: object, answer: object) -> None:
    """Keep an answer the runtime worked out before generation.

    LIVE, 2026-08-22: the finite-game solver ran after the reply was written,
    by which time the turn was over for lane admission, so its translation
    call was refused as background work — twice, on a turn whose own generation
    had already timed out at 180 seconds. An exact answer is not an improvement
    on a generated one; it is a reason not to generate.
    """
    label = str(name or "").strip()
    body = str(answer or "").strip()
    if not label or not body:
        return
    holder = _TURN_SOLVED.get()
    if isinstance(holder, dict):
        holder[label] = body


def solved_answers() -> dict[str, str]:
    """Everything worked out exactly this turn, newest last."""
    return dict(_TURN_SOLVED.get() or {})


def record_evidence_delivered(name: object) -> None:
    """Note that a reading was actually handed to the model this turn."""
    label = str(name or "").strip()
    if not label:
        return
    holder = _TURN_EVIDENCE.get()
    if isinstance(holder, set):
        holder.add(label)


def evidence_delivered() -> frozenset[str]:
    """What the model was given this turn, as far as anything recorded it."""
    return frozenset(_TURN_EVIDENCE.get() or ())


def current_user_question() -> str:
    """What this turn was asked, or empty outside a turn."""
    return str(user_question_var.get() or "")
