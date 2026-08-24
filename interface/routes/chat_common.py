"""Primitives every chat-lane module shares.

`interface/routes/chat.py` was one 30,000-line module with 462 top-level
functions. Splitting it means several modules need the same logger, the same
recoverable-error tuple and the same request-scoped context variables. They
live here so the import graph runs one way — this module, then the lane
modules, then chat.py — and never back into chat.py.
"""

from __future__ import annotations

from typing import Any

import time

import collections

import dataclasses

from contextvars import ContextVar
from fastapi import APIRouter, Depends, HTTPException, Request
import asyncio
import json
import logging
from core.runtime import resource_psutil as psutil

_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S = 2.0

_CHAT_RECOVERABLE_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    ImportError,
    LookupError,
    json.JSONDecodeError,
    asyncio.InvalidStateError,
    asyncio.QueueEmpty,
    asyncio.QueueFull,
    HTTPException,
    psutil.Error,
)

_CHAT_REQUEST_PRINCIPAL: ContextVar[str] = ContextVar(
    "aura_chat_request_principal",
    default="",
)

_CHAT_REQUEST_SURFACE: ContextVar[str] = ContextVar(
    "aura_chat_request_surface",
    default="",
)

_MAX_CONVERSATION_LOG_EXCHANGES = 500

_conversation_log: list[dict] = []  # In-memory session log for current runtime

_locks = {}

logger = logging.getLogger("Aura.Server.Chat")

from contextvars import ContextVar

_CHAT_DELIVERY_IDEMPOTENCY_KEY: ContextVar[str] = ContextVar(
    "aura_chat_delivery_idempotency_key",
    default="",
)

_CHAT_PENDING_DELIVERY_CLAIM: ContextVar[tuple[str, tuple[str, ...]]] = ContextVar(
    "aura_chat_pending_delivery_claim",
    default=("", ()),
)

_CHAT_SESSION_ID_MAX_CHARS = 64

_INTERNAL_SURFACE_CONTEXT: ContextVar[str] = ContextVar(
    "aura_internal_surface_context",
    default="",
)

_UNSET = object()

import re

_EXPLICIT_NON_EXECUTION_RE = re.compile(
    r"\b(?:do not execute|don't execute|without executing|before executing|"
    r"do not use tools|don't use tools|no tool use|no tools?|"
    r"do not run|don't run|do not open|don't open)\b",
    re.IGNORECASE,
)

_INCOMPLETE_TAIL_WORDS = {
    "a",
    "an",
    "and",
    "because",
    "but",
    "called",
    "create",
    "for",
    "from",
    "if",
    "into",
    "make",
    "named",
    "open",
    "of",
    "or",
    "save",
    "so",
    "than",
    "that",
    "the",
    "then",
    "this",
    "th",
    "to",
    "when",
    "where",
    "while",
    "write",
    "with",
}

_INTERNAL_STATE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:cognitive baseline tick\s*\d+)"
    r"|(?:monitoring internal state)"
    r"|(?:baseline_continuity)"
    r"|(?:In the [\d.]+ (?:seconds|minutes) just passed)"
    r"|(?:Pending initiatives:)"
    r"|(?:Reconcile continuity gap)"
    r"|(?:Drive alert:.*depleted)"
    r"|(?:Phenomenal Surge:)"
    r"|(?:Winner:.*Content:)"
)

_LOCAL_CHOICE_REFERENCE_RE = re.compile(r"\b(?:what|which)\s+one\b", re.IGNORECASE)

_ORGAN_INERT_STREAKS: dict[str, int] = {}

MAX_CHAT_MESSAGE_BYTES = 64 * 1024  # 64KB

import re

_PROMPT_ARTIFACT_PATTERNS = re.compile(
    r"(?im)"
    r"(?:^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice)\s*:)"
    r"|(?:\[ACTIVE GROUNDING EVIDENCE\])"
    r"|(?:\[FETCHED PAGE CONTENT\])"
    r"|(?:\[INTERNAL MEMORY RECALL\])"
    r"|(?:\[(?:RECENT CONTEXT|RECENT COMPLETED CONVERSATION|END RECENT COMPLETED CONVERSATION|CURRENT USER MESSAGE|OPERATIONAL SELF CONTEXT)\])"
)

_TOPIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "being",
        "but",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "huh",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "kind",
        "like",
        "maybe",
        "me",
        "more",
        "most",
        "my",
        "not",
        # Temporal and discourse fillers carry no topic. "now" was the SINGLE
        # overlapping token between "run a real calculation in your Python
        # sandbox and show me the result" and a 53-token reply about felt state —
        # and any single overlap exonerated the reply, so a completely off-topic
        # answer was logged off_topic=False.
        "now",
        "of",
        "on",
        "or",
        "our",
        "part",
        "really",
        "say",
        "says",
        "said",
        "side",
        "so",
        "sort",
        "stand",
        "standing",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "thing",
        "this",
        "those",
        "through",
        "to",
        "under",
        "up",
        "very",
        "was",
        "we",
        "were",
        "wait",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
    # Stance and degree adverbs. These modify how a clause is asserted; they
    # are never what a question is ABOUT.
    #
    # LIVE DEFECT, 2026-08-10. Asked "(1) how many heartbeats are active,
    # (2) your uptime in seconds, (3) the exact action name that keeps getting
    # refused… I will check all three", the degraded composer answered:
    #
    #     I understood you to be asking about heartbeats and actually.
    #
    # "actually" came from "not accept it. don't agree with me. answer only
    # from what you can actually read". _select_anchor_topic_tokens ranks
    # non-priority candidates by -len(token) — longest word first — so an
    # eight-letter adverb outranked every real noun in the question.
    #
    # The category was already recognised here: "really" was in this set. It
    # was simply never filled in.
    | {
        "actually",
        "already",
        "always",
        "basically",
        "certainly",
        "clearly",
        "definitely",
        "especially",
        "essentially",
        "exactly",
        "generally",
        "honestly",
        "instead",
        "literally",
        "maybe",
        "merely",
        "mostly",
        "obviously",
        "particularly",
        "perhaps",
        "possibly",
        "probably",
        "quite",
        "rather",
        "seriously",
        "simply",
        "specifically",
        "truly",
        "usually",
        "very",
    }
)

# Absolute safety ceiling for append-only continuation. The per-request budget
# is derived below from the parsed obligation count; a fixed two-segment budget
# cut off multipart answers after making measurable progress, while applying
# this ceiling to every request would waste work on ordinary conversation.
_MAX_USER_SURFACE_CONTINUATIONS = 12


@dataclasses.dataclass(frozen=True)
class _UserSurfaceObligation:
    """One unanswered work unit carried by the shared prompt-shape reader."""

    segment_index: int
    segment: str
    numbered_label: int | None = None


def _unanswered_user_surface_obligations(
    reply: object,
    prompt_shape: object | None,
) -> tuple[_UserSurfaceObligation, ...]:
    """Return uncovered asks with their original structural identity.

    The language substrate already preserves the exact request segments.  A
    retry scheduler needs the index as well as the text so a numbered answer
    can be assembled without asking the model to rediscover list structure.
    """

    if prompt_shape is None:
        return ()
    try:
        from core.conversation.request_coverage import unanswered_question_parts

        missing = list(unanswered_question_parts(reply, prompt_shape))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return ()
    if not missing:
        return ()
    segments = tuple(getattr(prompt_shape, "question_segments", ()) or ())
    try:
        numbered_parts = max(
            0,
            int(getattr(prompt_shape, "numbered_parts", 0) or 0),
        )
    except (TypeError, ValueError, OverflowError):
        numbered_parts = 0
    numbered_start = max(0, len(segments) - numbered_parts)
    remaining_counts = collections.Counter(str(item) for item in missing)
    obligations: list[_UserSurfaceObligation] = []
    for index, segment in enumerate(segments):
        segment_text = str(segment)
        if remaining_counts[segment_text] <= 0:
            continue
        remaining_counts[segment_text] -= 1
        numbered_label = (
            index - numbered_start + 1
            if numbered_parts and index >= numbered_start
            else None
        )
        obligations.append(
            _UserSurfaceObligation(
                segment_index=index,
                segment=segment_text,
                numbered_label=numbered_label,
            )
        )
    return tuple(obligations)


def _merge_obligation_completion(
    partial: object,
    completion: object,
    obligation: _UserSurfaceObligation,
) -> str:
    """Append one model-authored work unit under its measured list position."""

    head = str(partial or "").rstrip()
    tail = str(completion or "").strip()
    if not tail:
        return head
    label = obligation.numbered_label
    if label is not None:
        # The structural marker belongs to the scheduler.  The prose remains
        # the resident model's own, while the verifier can bind it to the same
        # numbered obligation that the user supplied.
        tail = re.sub(
            rf"^\s*(?:#+\s*)?(?:\({label}\)|{label}[.)])\s*",
            "",
            tail,
            count=1,
        ).strip()
        from core.conversation.request_coverage import merge_numbered_answer_section

        return merge_numbered_answer_section(head, label, tail)
    else:
        addition = tail
    if not head:
        return addition
    return f"{head}\n\n{addition}"


def _user_surface_continuation_budget(prompt_shape: object | None) -> int:
    """Return a finite continuation budget sized to independent obligations.

    A natural EOS ends one authored branch. Some checkpoints reliably continue
    only the current sentence or section when handed that open assistant state,
    so a five-part request can require more than two branch boundaries even
    though no branch is regenerated. Prompt-shape analysis already counts the
    independent asks; use that typed contract rather than request wording.
    """

    def _nonnegative_int(name: str) -> int:
        try:
            return max(0, int(getattr(prompt_shape, name, 0) or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    obligations = max(
        1,
        _nonnegative_int("question_parts"),
        _nonnegative_int("numbered_parts"),
    )
    return min(_MAX_USER_SURFACE_CONTINUATIONS, max(2, obligations + 1))


def _continuation_made_semantic_progress(
    previous: object,
    candidate: object,
    prompt_shape: object | None,
) -> bool:
    """Return whether an append-only retry retired a typed obligation.

    Character growth remains meaningful for a single open sentence. A
    multipart request has a stronger state contract: another segment must
    reduce the set of unanswered parts. This prevents repeated preamble from
    consuming one retry per requested section while making no semantic
    progress.
    """

    try:
        from core.conversation.request_coverage import unanswered_question_parts

        before = frozenset(unanswered_question_parts(previous, prompt_shape))
        after = frozenset(unanswered_question_parts(candidate, prompt_shape))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        before = frozenset()
        after = frozenset()
    if before:
        return after < before
    return len(str(candidate or "").rstrip()) > len(str(previous or "").rstrip())

_ORGAN_ABSENCE_STREAKS: dict[str, int] = {}

_SEARCH_SKILL_NAMES = {"web_search", "search_web", "free_search", "grounded_search"}
