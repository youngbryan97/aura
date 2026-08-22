"""Answering what Aura currently holds in memory.

A question about her own memory has a checkable answer, so it is
built from the live memory services rather than generated. This module
reads that state, decides whether a reply actually carries the evidence it
claims, and repairs one that does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from fastapi import APIRouter, Depends, HTTPException, Request
from pathlib import Path
from core.memory.session_pin_cipher import (
    SESSION_PIN_ENVELOPE_SCHEMA,
    SESSION_PIN_INDEX_CONTENT,
    SessionPinCipher,
    SessionPinCipherError,
)
from core.memory.session_pin_ledger import (
    SESSION_PIN_LEDGER_FILENAME,
    SessionPinLedger,
    SessionPinLedgerError,
)
from core.container import ServiceContainer
from datetime import UTC, datetime
import asyncio
from core.utils.task_tracker import get_task_tracker
import hashlib
import json
import logging
from functools import lru_cache, wraps
from core.utils.intent_normalization import normalize_memory_intent_text
from core.runtime import resource_psutil as psutil
import re
from core.runtime.errors import describe_error, record_degradation
from core.utils.injected_blocks import stamp_runtime_payload
import threading

from interface.routes.chat_common import (
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,
    _CHAT_RECOVERABLE_ERRORS,
    _CHAT_REQUEST_PRINCIPAL,
    _CHAT_REQUEST_SURFACE,
    _MAX_CONVERSATION_LOG_EXCHANGES,
    _conversation_log,
    _locks,
    logger,
)
from core.runtime.lockdep import checked_async_lock


def _get_convo_lock():
    return _locks.setdefault("convo", checked_async_lock("interface.routes.chat_memory_state"))


_session_memory_pins: list[dict] = []

_DURABLE_CONVERSATION_CONTEXT_TIMEOUT_S = 1.5

_DURABLE_CONVERSATION_SESSION_SCAN_LIMIT = 3

_RECENT_CONVERSATION_USER_CHARS = 800

_RECENT_CONVERSATION_AURA_CHARS = 1200

_SESSION_MEMORY_PIN_LEDGER_LIMIT = 500

_CHAT_BLOCKING_MAX_ACTIVE = 8

_chat_blocking_tasks: set[asyncio.Task[Any]] = set()

_chat_blocking_slots = threading.BoundedSemaphore(_CHAT_BLOCKING_MAX_ACTIVE)


class _ChatBlockingBudgetSaturatedError(RuntimeError):
    pass


def _invoke_chat_blocking_with_slot(
    operation: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if not _chat_blocking_slots.acquire(blocking=False):
        raise _ChatBlockingBudgetSaturatedError("chat blocking-work budget saturated")
    try:
        return operation(*args, **kwargs)
    finally:
        _chat_blocking_slots.release()


def _start_bounded_chat_blocking_task(
    operation: Callable[..., Any],
    /,
    *args: Any,
    operation_name: str,
    **kwargs: Any,
) -> asyncio.Task[Any]:
    """Start one owned blocking operation whose worker outlives waiter timeout."""
    task = get_task_tracker().track(
        asyncio.to_thread(
            _invoke_chat_blocking_with_slot,
            operation,
            args,
            kwargs,
        ),
        name=f"ChatBlocking:{operation_name}"[:120],
        owner="interface.routes.chat",
    )
    _chat_blocking_tasks.add(task)
    task.add_done_callback(_chat_blocking_tasks.discard)
    return task


async def _await_bounded_chat_blocking(
    operation: Callable[..., Any],
    /,
    *args: Any,
    timeout_s: float,
    operation_name: str,
    completion_grace_s: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Run bounded synchronous chat work without orphaning a late result.

    Cancelling ``asyncio.to_thread`` does not stop its worker. The former
    ``wait_for(to_thread(...))`` therefore discarded a deterministic result
    while the operation kept consuming its slot. Keep one supervised task,
    allow explicitly recoverable callers a small in-turn completion window,
    and retain ownership until the worker actually exits even after a hard
    timeout or caller cancellation.
    """

    task = _start_bounded_chat_blocking_task(
        operation,
        *args,
        operation_name=operation_name,
        **kwargs,
    )
    try:
        primary_timeout = max(0.05, float(timeout_s))
        await asyncio.wait({task}, timeout=primary_timeout)
        if task.done():
            return task.result()

        completion_grace = max(0.0, float(completion_grace_s))
        if completion_grace:
            logger.info(
                "Bounded chat operation %s reached its %.2fs soft budget; "
                "waiting up to %.2fs for its already-running deterministic result.",
                operation_name,
                primary_timeout,
                completion_grace,
            )
            await asyncio.wait({task}, timeout=completion_grace)
            if task.done():
                logger.info(
                    "Recovered bounded chat operation %s during completion grace.",
                    operation_name,
                )
                return task.result()

        raise TimeoutError(
            f"{operation_name} exceeded {primary_timeout + completion_grace:.2f}s hard budget"
        )
    except _ChatBlockingBudgetSaturatedError as exc:
        record_degradation(
            "chat.event_loop_budget",
            exc,
            severity="warning",
            action=f"rejected saturated bounded chat operation {operation_name}",
            extra={"operation": operation_name},
        )
        raise
    except TimeoutError as exc:
        record_degradation(
            "chat.event_loop_budget",
            exc,
            severity="warning",
            action=f"stopped waiting for bounded chat operation {operation_name}",
            extra={"operation": operation_name, "timeout_s": float(timeout_s)},
        )
        raise


_SECOND_REQUEST_AFTER_PIN_RE = re.compile(
    r"[,;]\s*(?:and|then|also|but)\s+(?:please\s+)?"
    r"(?:tell|show|give|remind|explain|describe|answer|summarize|summarise|"
    r"walk)\s+me\b"
    r"|[.!?;]\s*(?:(?:also|then|and\s+then|separately|secondly|next)\b[\s,—–-]*)?"
    r"(?:tell|show|open|create|write|export|find|search|go|make|change|"
    r"summarize|summarise|explain|give|do|use|launch|click|describe|list|"
    r"compare|walk|think|answer)\b"
    r"|[.!?;]\s*(?:separately|on\s+another\s+note|aside\s+from\s+that|"
    r"unrelatedly|changing\s+topic)\b"
    r"|[.!?;]\s*(?:can|could|would|will|do|does|did|is|are|what|why|how|when|"
    r"where|which|who)\s+you?\b",
    re.IGNORECASE,
)


def _turn_has_substance_beyond_memory_request(user_message: str) -> bool:
    """Whether this turn asks for something the memory template cannot answer.

    The deterministic memory path ends the turn with one sentence. It may do
    that only when remembering IS the turn. A question, or an explicit pivot
    to a second subject, means there is more here than a pin confirmation.
    """
    text = " ".join(str(user_message or "").split())
    if not text:
        return False
    pinned = _extract_session_memory_pin_request(text)
    if not pinned:
        return False
    # Only what comes AFTER the thing being remembered can be a second
    # request. A verbose preamble ("For this live reliability probe, remember…")
    # is still one turn about memory.
    index = text.find(pinned)
    if index < 0:
        return False
    tail = text[index + len(pinned) :]
    if "?" in tail:
        return True
    return bool(_SECOND_REQUEST_AFTER_PIN_RE.search(tail))


def _extract_session_memory_pin_request(user_message: str) -> str | None:
    text = str(user_message or "").strip()
    if not text:
        return None
    original = " ".join(text.split())
    original_matching = original.replace("’", "'").replace("‘", "'")
    original_matching = re.sub(
        r"\bdont'?\b",
        "don't",
        original_matching,
        flags=re.IGNORECASE,
    )

    def _clean_pinned_memory(raw: str) -> str:
        pinned_text = str(raw or "").strip().strip("\"'“”")
        pinned_text = re.sub(
            r"(?:\s*[.!?]\s*|\s+)(?:just\s+)?"
            r"(?:confirm|acknowledge|say\s+ok|reply\s+ok)\b.*$",
            "",
            pinned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        pinned_text = re.sub(
            r"\s*[.!?]\s+(?:(?:also|then|and\s+then)\s+)?"
            r"(?:tell|show|open|create|write|export|find|search|go|"
            r"make|change|summarize|explain|give|do|use|launch|click)\b.*$",
            "",
            pinned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        pinned_text = re.sub(
            r"\s*[.!?]\s+(?:can|could|would|will)\s+you\b.*$",
            "",
            pinned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # A pivot into a second, unrelated request — "Separately —", "Now, on
        # another note," — ends the thing being remembered.
        #
        # LIVE DEFECT, 2026-07-27: "Remember this: my project codename is
        # HELIOTROPE, build 4471. Separately — do you think a system like you
        # can actually prefer one thing over another…" pinned the whole tail,
        # truncated mid-clause, and reported doing so as the entire reply.
        pinned_text = re.sub(
            r"\s*[.!?;]*\s+(?:separately|aside\s+from\s+that|on\s+another\s+note|"
            r"unrelatedly|secondly|changing\s+topic)\b[\s,—–-]*.*$",
            "",
            pinned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # A second request does not have to start a new sentence. "Please
        # remember my dog is called Pixel, and tell me a joke" pinned the joke
        # request as part of the dog's name — every trim above requires a
        # sentence terminator, and a comma is not one.
        pinned_text = re.sub(
            r"\s*[,;]\s*(?:and|then|also|but)\s+(?:please\s+)?"
            r"(?:tell|show|give|remind|explain|describe|answer|summarize|"
            r"summarise|walk)\s+me\b.*$",
            "",
            pinned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # …as does a following question of any shape.
        # The question may arrive behind a discourse marker and a comma —
        # "Remember: I prefer tea. Also, what do you make of the second law?"
        # — which is why the marker is matched here rather than only the
        # bare question word.
        pinned_text = re.sub(
            r"\s*[.!?]\s+"
            r"(?:(?:also|then|next|separately|secondly|and|but|now)\b[\s,—–-]*)?"
            r"(?:do|does|did|is|are|was|were|what|why|how|when|where|"
            r"which|who|should|shall|may|might|must)\b[^.!?]*\?.*$",
            "",
            pinned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        pinned_text = re.sub(
            r"\s+for\s+(?:this\s+)?(?:conversation|chat|session|probe)[.!?]?\s*$",
            "",
            pinned_text,
            flags=re.IGNORECASE,
        )
        return pinned_text.rstrip(" .!?")

    head, sep, tail = text.partition(":")
    normalized = (
        f"{normalize_memory_intent_text(head)}{sep}{tail}"
        if sep
        else normalize_memory_intent_text(text)
    )

    pin_scope = (
        r"(?:\s+(?:for me|for later|for later in this session|"
        r"for later in this conversation|for later in this chat|"
        r"across restart|across restarts|after restart|after a restart|"
        r"across sessions|between sessions))?"
    )
    memory_object = r"(?:(?:this|the)\s+)?(?:phrase|codeword|word|token|detail|note|fact)?"
    # \s* not \s+: "Remember: I prefer tea" has no space before the colon, and
    # requiring one meant that turn pinned nothing at all — silently, because a
    # missing pin has no failure mode, it just never comes back later.
    patterns = (
        rf"^(?:please\s+)?remember\s*{memory_object}{pin_scope}\s*:\s*(.+)$",
        rf"^(?:please\s+)?remember\s+this{pin_scope}\s*:\s*(.+)$",
        rf"^don't forget(?:\s+this)?{pin_scope}\s*:\s*(.+)$",
        rf"^make note of this{pin_scope}\s*:\s*(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        if match:
            pinned = _clean_pinned_memory(match.group(1))
            return pinned[:240] if pinned else None

    # A discourse anchor such as "Remember the uncertainty you just named.
    # How would that change your decision?" asks the cognitive path to use
    # recent context; it is not a memory-write command. Explicit colon/object
    # forms above still pin, including multi-sentence facts.
    if re.search(
        r"[.!?]\s+(?:how|what|why|where|when|who|would|could|can|does|do|is|are)\b",
        original_matching,
        flags=re.IGNORECASE,
    ):
        return None

    prefixed_object = r"(?:(?:this|the)\s+)?(?:phrase|codeword|word|token|detail|note|fact)"
    prefixed_patterns = (
        rf"\b(?:please\s+)?remember\s+{prefixed_object}\s+(.+)$",
        rf"\b(?:please\s+)?remember\s+that{pin_scope}\s+(.+)$",
    )
    for pattern in prefixed_patterns:
        match = re.search(pattern, original_matching, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        pinned = _clean_pinned_memory(match.group(1))
        if re.match(r"^(?:what|when|where|who|why|how)\b", pinned, flags=re.IGNORECASE):
            continue
        return pinned[:240] if pinned else None

    natural_patterns = (
        rf"^(?:please\s+)?remember\s+that{pin_scope}\s+(.+)$",
        r"^(?:please\s+)?remember\s+((?:my|the|our)\s+.+)$",
        rf"^don't forget\s+that{pin_scope}\s+(.+)$",
        rf"^make\s+(?:a\s+)?note\s+that{pin_scope}\s+(.+)$",
    )
    for pattern in natural_patterns:
        match = re.match(pattern, original_matching, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        pinned = _clean_pinned_memory(match.group(1))
        if re.match(r"^(?:what|when|where|who|why|how)\b", pinned, flags=re.IGNORECASE):
            continue
        return pinned[:240] if pinned else None
    return None


def _is_anaphoric_session_memory_pin_request(user_message: str) -> bool:
    """True when the user asks Aura to hold the current thread without restating it."""

    text = normalize_memory_intent_text(_normalize_user_message(user_message)).rstrip(" .!?")
    if not text:
        return False
    if _extract_session_memory_pin_request(user_message):
        return False
    # Anaphoric writes require a command/request at the start of the utterance.
    # Substring matching used to turn capability questions such as "what are
    # you, and will you remember this tomorrow?" into silent writes of the
    # previous exchange. Accept polite request modals, but not predictive
    # questions beginning with "will/do/did you".
    command_text = re.sub(
        r"^(?:can|could|would)\s+you\s+(?:please\s+)?",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    command_text = re.sub(r"^please\s+", "", command_text, count=1, flags=re.IGNORECASE)
    markers = (
        "hold this thought",
        "hold that thought",
        "keep this thought",
        "keep that thought",
        "pin this thought",
        "pin that thought",
        "save this thought",
        "save that thought",
        "remember it",
        "remember this",
        "remember that",
        "dont forget it",
        "don't forget it",
        "dont forget this",
        "don't forget this",
        "dont forget that",
        "don't forget that",
    )

    def _marker_bounded(candidate: str, marker: str) -> bool:
        # The marker must end at a word boundary — a space OR sentence
        # punctuation ("Hold this thought. And remember it." was refused
        # because only "<marker> " matched, never "<marker>.").
        if not candidate.startswith(marker):
            return False
        rest = candidate[len(marker) :]
        return rest == "" or rest[0] in " .,!;:"

    if not any(_marker_bounded(command_text, marker) for marker in markers):
        return False
    # A follow-up question is a discourse anchor, not a memory-write command.
    if re.search(
        r"[.!?]\s+(?:how|what|why|where|when|who|would|could|can|does|do|is|are)\b",
        str(user_message or ""),
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _is_session_memory_recall_request(user_message: str) -> bool:
    text = normalize_memory_intent_text(_normalize_user_message(user_message))
    if not text:
        return False
    markers = (
        "what codeword did i just give you",
        "what codeword did i give you",
        "what was the codeword i gave you",
        "what is the codeword i gave you",
        "what codeword did i ask you to remember",
        "what was the codeword",
        "what is the codeword",
        "what token did i just give you",
        "what token did i give you",
        "what token did i ask you to remember",
        "what phrase did i just ask you to remember",
        "what phrase did i ask you to remember",
        "what note did i ask you to remember",
        "what note did i tell you to remember",
        "what did i ask you to remember",
        "what did i ask you to remember in this conversation",
        "what did i ask you to remember in this chat",
        "what phrase did i tell you to remember",
        "what was the phrase from earlier",
        "what was the phrase from earlier in this probe",
        "what was the phrase from earlier in this conversation",
        "what phrase from earlier",
        "what did i tell you to remember",
        "what did you store for me earlier in this session",
        "what did you pin for later in this session",
    )
    return any(marker in text for marker in markers)


_RECALL_MATCH_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "your",
        "our",
        "his",
        "her",
        "their",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "was",
        "were",
        "is",
        "are",
        "did",
        "do",
        "does",
        "have",
        "has",
        "had",
        "i",
        "you",
        "we",
        "me",
        "gave",
        "give",
        "given",
        "told",
        "tell",
        "said",
        "say",
        "earlier",
        "before",
        "again",
        "that",
        "this",
        "it",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "with",
        "about",
        "number",
        "remember",
        "quick",
        "check",
        "just",
        "back",
        "recall",
        "please",
        "thing",
    }
)


def _content_recall_matches_pin(user_message: str, pinned_content: str) -> bool:
    """Whether a durable pin is plausibly what this recall question is asking for.

    A pin about tea must not answer a question about a codename. Topic words
    shared between the question and the pin are the evidence; without at least
    one, the honest miss is still the right reply.
    """

    def _topic_words(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9][a-z0-9'-]*", str(text or "").lower())
        return {word for word in words if len(word) > 2 and word not in _RECALL_MATCH_STOPWORDS}

    question_words = _topic_words(user_message)
    pin_words = _topic_words(pinned_content)
    return bool(question_words and pin_words and (question_words & pin_words))


def _is_cross_session_memory_recall_request(user_message: str) -> bool:
    """True when the user explicitly asks for a pin from *before a restart* or a
    *previous session*.

    This is the only signal that unlocks cross-session durable recall. A bare
    recall ("what codeword did I give you") stays scoped to the current session,
    so distinct concurrent sessions remain isolated
    (test_session_memory_pin_isolation_by_session_id). But a durable pin must
    survive a reboot — and a reboot starts a *new* session id — so when the user
    references the restart, we let the durable ledger answer across sessions
    (live_boot_proof.exercise_restart_continuity_turn / tasks #22, #28).
    """
    text = normalize_memory_intent_text(_normalize_user_message(user_message))
    if not text:
        return False
    markers = (
        "before restart",
        "before the restart",
        "before a restart",
        "across restart",
        "across the restart",
        "across a restart",
        "after restart",
        "after the restart",
        "before you restarted",
        "before we restarted",
        "after you restarted",
        "before reboot",
        "before the reboot",
        "before you rebooted",
        "after reboot",
        "after the reboot",
        "before you were restarted",
        "from before the restart",
        "previous session",
        "prior session",
        "earlier session",
        "last session",
        "a previous session",
        "from a previous session",
    )
    return any(marker in text for marker in markers)


def _is_session_memory_context_change_request(user_message: str) -> bool:
    text = normalize_memory_intent_text(_normalize_user_message(user_message))
    if not text:
        return False
    return bool(
        re.search(
            r"\bwhat changed\b.*\b(?:conversation|chat|thread)\b.*"
            r"\b(?:after|since|when)\b.*\b(?:gave|told|asked|shared|mentioned)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _session_memory_pin_ledger_path() -> Path:
    from core.config import config

    return config.paths.data_dir / "memory" / SESSION_PIN_LEDGER_FILENAME


@lru_cache(maxsize=1)
def _session_memory_pin_cipher() -> SessionPinCipher:
    """Resolve the Keychain-custodied pin cipher once per runtime."""

    return SessionPinCipher.from_system()


def _session_memory_pin_binding(
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
    inherit_context: bool = True,
) -> tuple[str, str]:
    if inherit_context:
        principal, surface = _chat_memory_identity(
            principal_id=principal_id,
            principal_surface=principal_surface,
        )
    else:
        principal = " ".join(str(principal_id or "").strip().split())[:160]
        surface = str(principal_surface or "").strip().casefold()[:32]
    if principal and surface:
        return principal, surface
    safe_session_id = str(session_id or "")[:64]
    if safe_session_id and not principal:
        digest = hashlib.sha256(safe_session_id.encode("utf-8")).hexdigest()
        return f"session:{digest}", "session"
    return "", ""


def _seal_session_memory_pin_record(
    content: str,
    source: str,
    timestamp: str,
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> dict[str, str]:
    safe_principal_id, safe_surface = _session_memory_pin_binding(
        session_id=session_id,
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    return _session_memory_pin_cipher().seal(
        content=content,
        source=source,
        timestamp=timestamp,
        session_id=str(session_id or "")[:64],
        principal_id=safe_principal_id,
        principal_surface=safe_surface,
    )


def _migrate_session_memory_pin_ledger_locked(
    ledger: SessionPinLedger,
    cipher: SessionPinCipher,
) -> tuple[list[dict[str, str]], bool, int]:
    """Canonicalize a bounded snapshot before append or recall."""

    snapshot = ledger.read_snapshot()
    raw_lines = list(snapshot.lines)
    migrated: list[dict[str, str]] = []
    changed = snapshot.truncated or snapshot.permissions_repair_required
    dropped = 1 if snapshot.truncated else 0
    for line in raw_lines:
        if not line.strip():
            changed = True
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            changed = True
            dropped += 1
            continue
        if not isinstance(record, dict):
            changed = True
            dropped += 1
            continue
        if record.get("schema") == SESSION_PIN_ENVELOPE_SCHEMA:
            # Opening before preserving the row prevents a forged or corrupt
            # envelope from surviving migration as apparently valid state.
            try:
                cipher.open(record)
            except SessionPinCipherError:
                changed = True
                dropped += 1
                continue
            canonical_envelope = {
                key: str(record.get(key) or "")
                for key in (
                    "schema",
                    "key_id",
                    "record_id",
                    "nonce_b64",
                    "ciphertext_b64",
                )
            }
            if canonical_envelope != record:
                changed = True
            migrated.append(canonical_envelope)
            continue
        if record.get("schema") != "aura.session_memory_pin.v2":
            changed = True
            dropped += 1
            continue
        content = str(record.get("content") or "").strip()
        if not content:
            changed = True
            dropped += 1
            continue
        session_id = str(record.get("session_id") or "")[:64]
        principal_id, surface = _session_memory_pin_binding(
            session_id=session_id,
            principal_id=str(record.get("principal_id") or ""),
            principal_surface=str(record.get("principal_surface") or ""),
            inherit_context=False,
        )
        if not principal_id or not surface:
            # Preserve the information under encryption without assigning an
            # unowned historical record to the next caller who happens to ask.
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
            principal_id = f"legacy-unbound:{digest}"
            surface = "legacy_unbound"
        migrated.append(
            cipher.seal(
                content=content,
                source=str(record.get("source") or ""),
                timestamp=str(record.get("timestamp") or ""),
                session_id=session_id,
                principal_id=principal_id,
                principal_surface=surface,
            )
        )
        changed = True
    return migrated, changed, dropped


def _append_session_memory_pin_ledger(
    content: str,
    source: str,
    timestamp: str,
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> bool:
    try:
        path = _session_memory_pin_ledger_path()
        if not str(content or "").strip():
            return False
        cipher = _session_memory_pin_cipher()
        ledger = SessionPinLedger(path)
        with ledger.transaction():
            records, _changed, dropped = _migrate_session_memory_pin_ledger_locked(
                ledger,
                cipher,
            )
            record = _seal_session_memory_pin_record(
                content,
                source,
                timestamp,
                session_id=session_id,
                principal_id=principal_id,
                principal_surface=principal_surface,
            )
            records.append(record)
            ledger.commit_records(records[-_SESSION_MEMORY_PIN_LEDGER_LIMIT:])
        if dropped:
            logger.warning(
                "Dropped malformed or truncated session-memory ledger row(s) "
                "during encrypted migration"
            )
        return True
    except (
        *_CHAT_RECOVERABLE_ERRORS,
        SessionPinCipherError,
        SessionPinLedgerError,
    ) as exc:
        record_degradation("chat.session_memory_pin", exc)
        logger.debug("Durable session memory pin ledger write skipped: %s", exc)
        return False


def _append_session_memory_pin_ledger_guarded(
    content: str,
    source: str,
    timestamp: str,
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> bool:
    """Append the session pin ledger without letting fallback logging crash chat."""

    try:
        return bool(
            _append_session_memory_pin_ledger(
                content,
                source,
                timestamp,
                session_id=session_id,
                principal_id=principal_id,
                principal_surface=principal_surface,
            )
        )
    except TypeError as exc:
        if not any(
            name in str(exc) for name in ("session_id", "principal_id", "principal_surface")
        ):
            record_degradation("chat.session_memory_pin", exc)
            logger.debug("Durable session memory pin ledger append failed: %s", exc)
            return False
        if principal_id or principal_surface:
            record_degradation("chat.session_memory_pin", exc)
            logger.warning(
                "Principal-bound session memory pin append rejected a legacy writer; "
                "the identity binding was not discarded."
            )
            return False
        try:
            return bool(_append_session_memory_pin_ledger(content, source, timestamp))
        except _CHAT_RECOVERABLE_ERRORS as legacy_exc:
            record_degradation("chat.session_memory_pin", legacy_exc)
            logger.debug(
                "Durable session memory pin legacy ledger append skipped: %s",
                legacy_exc,
            )
            return False
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.session_memory_pin", exc)
        logger.debug("Durable session memory pin ledger append failed: %s", exc)
        return False


def _recall_session_memory_pin_from_ledger(
    *,
    session_id: str = "",
    cross_session: bool = False,
    principal_id: str = "",
    principal_surface: str = "",
) -> dict[str, str] | None:
    try:
        path = _session_memory_pin_ledger_path()
        cipher = _session_memory_pin_cipher()
        ledger = SessionPinLedger(path)
        with ledger.transaction():
            records, changed, dropped = _migrate_session_memory_pin_ledger_locked(
                ledger,
                cipher,
            )
            if changed:
                ledger.commit_records(records)
        if dropped:
            logger.warning(
                "Dropped malformed or truncated session-memory ledger row(s) "
                "during encrypted migration"
            )
        if not records:
            return None
        for raw in reversed(records[-_SESSION_MEMORY_PIN_LEDGER_LIMIT:]):
            expected_session_id = str(session_id or "")[:64]
            if raw.get("schema") != SESSION_PIN_ENVELOPE_SCHEMA:
                continue
            try:
                opened = cipher.open(raw)
            except SessionPinCipherError as exc:
                record_degradation("chat.session_memory_pin", exc)
                continue
            recalled = _session_memory_pin_from_record(
                {**opened, "session_memory_pin": True},
                session_id=expected_session_id,
                cross_session=cross_session,
                principal_id=principal_id,
                principal_surface=principal_surface,
                allow_plaintext=True,
            )
            if recalled:
                recalled["storage"] = "durable"
                return recalled
    except (
        *_CHAT_RECOVERABLE_ERRORS,
        SessionPinCipherError,
        SessionPinLedgerError,
    ) as exc:
        record_degradation("chat.session_memory_pin", exc)
        logger.debug("Durable session memory pin ledger recall skipped: %s", exc)
    return None


def _chat_memory_identity(
    *,
    principal_id: str = "",
    principal_surface: str = "",
) -> tuple[str, str]:
    principal = " ".join(str(principal_id or _CHAT_REQUEST_PRINCIPAL.get() or "").strip().split())[
        :160
    ]
    surface = str(principal_surface or _CHAT_REQUEST_SURFACE.get() or "").strip().casefold()[:32]
    return principal, surface


def _cross_session_memory_recall_allowed(user_message: str) -> bool:
    principal_id, principal_surface = _chat_memory_identity()
    return bool(
        principal_surface == "owner"
        and principal_id
        and _is_cross_session_memory_recall_request(user_message)
    )


async def _store_session_memory_pin(
    content: str,
    source: str,
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> bool:
    pinned = str(content or "").strip()
    if not pinned:
        return False
    timestamp = datetime.now(tz=UTC).isoformat()
    safe_session_id = str(session_id or "")[:64]
    safe_principal_id, safe_principal_surface = _session_memory_pin_binding(
        session_id=safe_session_id,
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    ledger_ok = False
    async with _get_convo_lock():
        _session_memory_pins.append(
            {
                "content": pinned[:240],
                "source": str(source or "").strip()[:512],
                "timestamp": timestamp,
                "session_id": safe_session_id,
                "principal_id": safe_principal_id,
                "principal_surface": safe_principal_surface,
            }
        )
        if len(_session_memory_pins) > 100:
            _session_memory_pins.pop(0)
    try:
        sealed_record = await asyncio.to_thread(
            _seal_session_memory_pin_record,
            pinned,
            source,
            timestamp,
            session_id=safe_session_id,
            principal_id=safe_principal_id,
            principal_surface=safe_principal_surface,
        )
        memory_facade = ServiceContainer.get("memory_facade", default=None)
        if memory_facade is None or not hasattr(memory_facade, "add_memory"):
            ledger_ok = await asyncio.to_thread(
                _append_session_memory_pin_ledger_guarded,
                pinned,
                source,
                timestamp,
                session_id=safe_session_id,
                principal_id=safe_principal_id,
                principal_surface=safe_principal_surface,
            )
            return bool(ledger_ok)
        result = memory_facade.add_memory(
            SESSION_PIN_INDEX_CONTENT,
            metadata={
                "source": "session_memory_pin",
                "family": "episodic",
                "kind": "explicit_user_memory_pin",
                "session_memory_pin": True,
                "session_memory_pin_envelope": sealed_record,
                "session_memory_pin_envelope_schema": SESSION_PIN_ENVELOPE_SCHEMA,
                "importance": 0.9,
                "identity_relevant": True,
                "explicit_memory_request": True,
                "provenance_source": "user_explicit",
                "confidence": 1.0,
            },
        )
        if hasattr(result, "__await__"):
            result = await result
        if not bool(result):
            ledger_ok = await asyncio.to_thread(
                _append_session_memory_pin_ledger_guarded,
                pinned,
                source,
                timestamp,
                session_id=safe_session_id,
                principal_id=safe_principal_id,
                principal_surface=safe_principal_surface,
            )
            return bool(ledger_ok)
        ledger_ok = await asyncio.to_thread(
            _append_session_memory_pin_ledger_guarded,
            pinned,
            source,
            timestamp,
            session_id=safe_session_id,
            principal_id=safe_principal_id,
            principal_surface=safe_principal_surface,
        )
        if not ledger_ok:
            logger.warning(
                "Session memory pin accepted by memory facade but ledger append failed; "
                "canonical memory remains authoritative."
            )
        return True
    except (*_CHAT_RECOVERABLE_ERRORS, SessionPinCipherError) as exc:
        record_degradation("chat.session_memory_pin", exc)
        logger.debug("Durable session memory pin write skipped: %s", exc)
        if not ledger_ok:
            ledger_ok = await asyncio.to_thread(
                _append_session_memory_pin_ledger_guarded,
                pinned,
                source,
                timestamp,
                session_id=safe_session_id,
                principal_id=safe_principal_id,
                principal_surface=safe_principal_surface,
            )
        return bool(ledger_ok)


def _session_memory_pin_from_record(
    item: Any,
    *,
    session_id: str = "",
    cross_session: bool = False,
    principal_id: str = "",
    principal_surface: str = "",
    allow_plaintext: bool = False,
) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    envelope = metadata.get("session_memory_pin_envelope") or item.get(
        "session_memory_pin_envelope"
    )
    if envelope is not None:
        if not isinstance(envelope, dict):
            return None
        try:
            opened = _session_memory_pin_cipher().open(envelope)
        except SessionPinCipherError as exc:
            record_degradation("chat.session_memory_pin", exc)
            return None
        item = {**opened, "session_memory_pin": True}
        metadata = {}
        allow_plaintext = True
    if not allow_plaintext:
        # Durable plaintext pins predate the encrypted store. They remain
        # ineligible for recall so a copied database row cannot bypass the
        # principal-bound envelope contract. The JSONL migration preserves
        # usable legacy rows under encryption.
        return None
    expected_session_id = str(session_id or "")[:64]
    expected_principal_id, expected_surface = _session_memory_pin_binding(
        session_id=expected_session_id,
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    record_principal_id = " ".join(
        str(metadata.get("principal_id") or item.get("principal_id") or "").strip().split()
    )[:160]
    record_surface = (
        str(metadata.get("principal_surface") or item.get("principal_surface") or "")
        .strip()
        .casefold()[:32]
    )
    record_session_id = str(
        metadata.get("chat_session_id")
        or metadata.get("session_id")
        or item.get("chat_session_id")
        or item.get("session_id")
        or ""
    )[:64]
    # Default: a pin only belongs to the session that set it, so distinct
    # concurrent sessions stay isolated. ``cross_session`` is the explicit
    # opt-in (the user asked about a pin from *before a restart*), which is the
    # only path that may return another session's durable pin.
    if not expected_principal_id or not expected_surface:
        return None
    if not record_principal_id or not record_surface:
        return None
    if record_principal_id != expected_principal_id or record_surface != expected_surface:
        return None
    if cross_session and (expected_surface != "owner" or not expected_principal_id):
        return None
    if not cross_session and expected_session_id and record_session_id != expected_session_id:
        return None
    content = str(
        metadata.get("session_memory_pin_content") or item.get("session_memory_pin_content") or ""
    ).strip()
    raw = str(item.get("content") or item.get("text") or "").strip()
    if not content and bool(item.get("session_memory_pin")):
        content = raw
    if not content:
        match = re.search(r"\bSession memory pin:\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip().strip("\"'“”").rstrip(" .!?")
    if not content:
        return None
    return {
        "content": content[:240],
        "source": str(
            metadata.get("source_utterance")
            or metadata.get("source")
            or item.get("source")
            or "durable_memory"
        )[:512],
        "timestamp": str(metadata.get("timestamp") or item.get("timestamp") or ""),
        "session_id": record_session_id,
        "principal_id": record_principal_id,
        "principal_surface": record_surface,
        "storage": "durable",
    }


async def _recall_durable_session_memory_pin(
    *,
    session_id: str = "",
    cross_session: bool = False,
    principal_id: str = "",
    principal_surface: str = "",
) -> dict[str, str] | None:
    safe_session_id = str(session_id or "")[:64]
    safe_principal_id, safe_principal_surface = _session_memory_pin_binding(
        session_id=safe_session_id,
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    try:
        ledger_recall = await _await_bounded_chat_blocking(
            _recall_session_memory_pin_from_ledger,
            session_id=safe_session_id,
            cross_session=cross_session,
            principal_id=safe_principal_id,
            principal_surface=safe_principal_surface,
            timeout_s=_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,
            operation_name="session_memory_pin_ledger_recall",
            completion_grace_s=0.75,
        )
    except TimeoutError:
        ledger_recall = None
    if ledger_recall:
        return ledger_recall
    try:
        memory_facade = ServiceContainer.get("memory_facade", default=None)
        if memory_facade is None:
            return None
        search = getattr(memory_facade, "search", None) or getattr(
            memory_facade, "query_memory", None
        )
        if not callable(search):
            return None
        result = search("session memory pin explicit user remember", limit=8)
        records = await result if hasattr(result, "__await__") else result
        for item in list(records or []):
            recalled = _session_memory_pin_from_record(
                item,
                session_id=safe_session_id,
                cross_session=cross_session,
                principal_id=safe_principal_id,
                principal_surface=safe_principal_surface,
            )
            if recalled:
                return recalled
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.session_memory_pin", exc)
        logger.debug("Durable session memory pin recall skipped: %s", exc)
    return None


async def _recall_session_memory_pin(
    *,
    session_id: str = "",
    cross_session: bool = False,
    principal_id: str = "",
    principal_surface: str = "",
) -> dict[str, str] | None:
    safe_session_id = str(session_id or "")[:64]
    safe_principal_id, safe_principal_surface = _session_memory_pin_binding(
        session_id=safe_session_id,
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    async with _get_convo_lock():
        for latest in reversed(_session_memory_pins):
            recalled = _session_memory_pin_from_record(
                {**latest, "session_memory_pin": True},
                session_id=safe_session_id,
                cross_session=cross_session,
                principal_id=safe_principal_id,
                principal_surface=safe_principal_surface,
                allow_plaintext=True,
            )
            if recalled:
                recalled["storage"] = "session"
                return recalled
    return await _recall_durable_session_memory_pin(
        session_id=safe_session_id,
        cross_session=cross_session,
        principal_id=safe_principal_id,
        principal_surface=safe_principal_surface,
    )


async def _build_memory_state_fastpath_reply(
    user_message: str,
    *,
    session_id: str = "",
    owner_session_restored: bool = False,
    as_evidence: bool = False,
) -> tuple[str, str] | None:
    """Return deterministic memory/continuity state.

    Two callers want two different things from this. The cognitive-engine
    lane wants canonical memory state as EVIDENCE to ground a reply it will
    write itself (``as_evidence=True``); the fastpath lane wants the reply
    itself. Only the second has to prove the template covers the whole turn —
    evidence never silences anyone.
    """
    # A template may only answer a turn it FULLY covers.
    #
    # This path returns a deterministic sentence and ends the turn. That is
    # right for "remember X" and wrong for "remember X, and separately, what do
    # you think about Y" — the second half is a real question and the template
    # has nothing to say about it. Live 2026-07-27 the whole message was
    # swallowed as the thing to remember, and the reply was:
    #
    #   I have pinned "my project codename is HELIOTROPE, build 4471.
    #   Separately — do you think a system like you can actually prefer one
    #   thing over another, or is" in this session.
    #
    # Same shape as the self-condition template earlier in the day: a reflex
    # short-circuiting the model on a turn it only partially understood. When
    # the turn carries substantive content beyond the memory request, the
    # deterministic path stands down and the mind answers.
    #
    # Standing down is about the REPLY, not the memory: the fact is still
    # pinned, or "remember X, and separately Y" would answer Y and forget X.
    if not as_evidence and _turn_has_substance_beyond_memory_request(user_message):
        deferred_pin = _extract_session_memory_pin_request(user_message)
        if deferred_pin:
            await _store_session_memory_pin(
                deferred_pin,
                user_message,
                session_id=session_id,
            )
            logger.info(
                "🧷 Pinned an explicit memory item (chars=%d) and left "
                "the reply to the mind: this turn asks for more than the memory "
                "template can answer.",
                len(deferred_pin),
            )
        return None
    session_pin = _extract_session_memory_pin_request(user_message)
    if not session_pin and _is_anaphoric_session_memory_pin_request(user_message):
        exchanges = await _recent_completed_conversation_exchanges(
            current_user_message=user_message,
            session_id=session_id,
            limit=2,
        )
        if exchanges:
            last = exchanges[-1]
            prior_aura = _clip_conversation_text(last.get("aura"), limit=180)
            prior_user = _clip_conversation_text(last.get("user"), limit=120)
            if prior_aura:
                if prior_user:
                    session_pin = f"Current thread: {prior_user} / Aura's thought: {prior_aura}"
                else:
                    session_pin = f"Aura's current thought: {prior_aura}"
            elif prior_user:
                session_pin = f"Current thread: {prior_user}"
    if session_pin:
        durable_ok = await _store_session_memory_pin(
            session_pin,
            user_message,
            session_id=session_id,
        )
        if not durable_ok:
            return (
                f'I can hold "{session_pin}" in this running session, but durable memory storage did not accept the write yet.',
                "session_memory_pin_transient",
            )
        return (
            f"I've pinned \"{session_pin}\" in durable session memory. Ask for it later and I'll pull it back directly.",
            "session_memory_pin",
        )

    if _is_session_memory_context_change_request(user_message):
        remembered = await _recall_session_memory_pin(session_id=session_id)
        if remembered and remembered.get("content"):
            return (
                "The concrete change is that I stored your explicit session note "
                f'"{remembered["content"]}" as durable conversation state, so later turns can refer back to it directly.',
                "session_memory_context_recall",
            )
        return "I don't have a pinned session note to compare against yet.", "session_memory_miss"

    if _is_session_memory_recall_request(user_message):
        cross_session = _cross_session_memory_recall_allowed(user_message)
        remembered = await _recall_session_memory_pin(
            session_id=session_id,
            cross_session=cross_session,
        )
        if remembered and remembered.get("content"):
            storage = str(remembered.get("storage") or "session")
            if cross_session and storage == "durable":
                source_label = "from durable memory across the restart"
            elif storage == "durable":
                source_label = "from durable memory"
            else:
                source_label = "in this session"
            return (
                f'The phrase you asked me to remember {source_label} was "{remembered["content"]}".',
                "session_memory_recall",
            )
        return "I don't have a pinned phrase from this session yet.", "session_memory_miss"

    owner_name_reply = _build_owner_name_recall_reply(
        user_message,
        owner_session_restored=owner_session_restored,
    )
    if owner_name_reply:
        return owner_name_reply, "owner_identity_recall"

    conversation_recall = await _build_conversation_recall_reply(
        user_message,
        session_id=session_id,
    )
    if conversation_recall:
        return conversation_recall, "conversation_recall"

    return None


_OWNER_NAME_RECALL_MARKERS = (
    "do you know my name",
    "do you remember my name",
    "what is my name",
    "what's my name",
    "who am i",
    "who do you think i am",
    "do you know who i am",
)


def _is_owner_name_recall_request(user_message: str) -> bool:
    text = normalize_memory_intent_text(_normalize_user_message(user_message)).rstrip(" ?!.")
    return bool(text and any(marker in text for marker in _OWNER_NAME_RECALL_MARKERS))


def _resolve_primary_operator_name() -> str:
    try:
        identity_kernel = ServiceContainer.get("identity_kernel", default=None)
        if identity_kernel is not None and hasattr(identity_kernel, "get_current_identity"):
            current = identity_kernel.get_current_identity()
            if isinstance(current, dict):
                primary = str(current.get("primary_operator") or "").strip()
                if primary:
                    return primary
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.owner_identity", exc)
        logger.debug("IdentityKernel primary-operator lookup skipped: %s", exc)

    try:
        from core.identity.self_contract import SelfContract

        primary = str(
            SelfContract().get_relationship_constraints().get("primary_operator") or ""
        ).strip()
        if primary:
            return primary
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.owner_identity", exc)
        logger.debug("SelfContract primary-operator lookup skipped: %s", exc)

    return "the verified owner"


def _owner_session_is_verified(*, owner_session_restored: bool = False) -> bool:
    if owner_session_restored:
        return True
    try:
        from core.security.user_recognizer import get_user_recognizer

        recognizer = get_user_recognizer()
        if hasattr(recognizer, "is_session_verified") and recognizer.is_session_verified():
            return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.owner_identity", exc)
        logger.debug("UserRecognizer verification lookup skipped: %s", exc)

    try:
        from core.security.trust_engine import TrustLevel, get_trust_engine

        context = getattr(get_trust_engine(), "_context", None)
        level = getattr(context, "level", None)
        return bool(level == TrustLevel.SOVEREIGN)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.owner_identity", exc)
        logger.debug("TrustEngine verification lookup skipped: %s", exc)
    return False


def _build_owner_name_recall_reply(
    user_message: str,
    *,
    owner_session_restored: bool = False,
) -> str | None:
    if not _is_owner_name_recall_request(user_message):
        return None
    if not _owner_session_is_verified(owner_session_restored=owner_session_restored):
        return (
            "I know the primary operator from my identity contract, but I should not expose "
            "that as the current speaker's identity until this session is owner-verified."
        )

    name = _resolve_primary_operator_name()
    return (
        f"Yes. You're {name}. I know that from the verified owner session and my identity "
        "contract, not from guessing at the last message."
    )


_CONVERSATION_RECALL_LAST_USER_MARKERS = (
    "can you remind me what i said",
    "can you remind me what i asked",
    "do you remember what i said",
    "do you remember what i asked",
    "do you remember my question",
    "what did i just say",
    "what did i just ask",
    "what did i tell you",
    "what was my last question",
    "what was my last message",
    "what did i ask you",
    "what did i say earlier",
    "what did i say before",
    "what was i asking",
    "what was i asking about",
    "what was the last thing i said",
    "what was i saying",
)

_CONVERSATION_RECALL_LAST_AURA_MARKERS = (
    "can you remind me what you said",
    "can you remind me what you answered",
    "do you remember what you said",
    "do you remember your answer",
    "what did you just say",
    "what did you answer",
    "what did you say earlier",
    "what did you say before",
    "what was your last answer",
    "what was your last message",
    "what did you tell me",
    "what was the last thing you said",
    "what were you saying",
)

_CONVERSATION_RECALL_RECENT_PAIR_MARKERS = (
    "summarize our last two messages",
    "summarize the last two messages",
    "summarize my last two messages",
    "recap our last two messages",
    "recap the last two messages",
    "repeat our last two messages",
    "repeat the last two messages",
)

_CONVERSATION_RECALL_TOPIC_MARKERS = (
    "can you remind me what we discussed",
    "can you remind me what we talked about",
    "what did we discuss",
    "what did we just discuss",
    "what have we discussed",
    "what did we talk about",
    "what did we just talk about",
    "what were we talking about",
    "what have we been talking about",
    "what are we talking about",
    "what was the thread",
    "what was the topic",
    "what was this conversation about",
    "what is this conversation about",
    "do you remember what we were discussing",
    "do you remember what we discussed",
    "remind me what we discussed",
    "remind me what we talked about",
    "earlier in this conversation",
    "summarize our conversation",
    "summarize what we have discussed",
)

_CONVERSATION_RECALL_CONTENT_RE = re.compile(
    r"\bearlier\b.{0,120}\bi\s+(?:gave|told|mentioned|said|asked)\b"
    r"|\bi\s+(?:gave|told|mentioned)\s+you\b.{0,120}\b(?:earlier|before|a\s+while\s+(?:ago|back))\b"
    r"|\bwhat\s+(?:was|is|were)\s+(?:the|my|that|it)\b.{0,120}"
    r"\b(?:i\s+(?:gave|told|mentioned|said)|asked\s+you\s+to\s+(?:keep|remember))\b"
    r"|\b(?:what|which)\s+\w[\w\s'-]{0,50}\bdid\s+i\s+"
    r"(?:say|give|tell|mention|pick|choose)\b"
    r"|\basked\s+you\s+to\s+keep\s+in\s+mind\b.{0,80}\bwhat\s+was\b",
    re.IGNORECASE,
)


def _classify_conversation_recall_request(user_message: str) -> str:
    text = normalize_memory_intent_text(_normalize_user_message(user_message)).rstrip(" ?!.")
    if not text:
        return ""
    if any(marker in text for marker in _CONVERSATION_RECALL_LAST_AURA_MARKERS):
        return "last_aura"
    if _CONVERSATION_RECALL_CONTENT_RE.search(text):
        return "content"
    if any(marker in text for marker in _CONVERSATION_RECALL_RECENT_PAIR_MARKERS):
        return "recent_pair"
    if re.search(
        r"\b(?:what\s+(?:were|are)|remind\s+me|tell\s+me|show\s+me|repeat|recap|summarize)"
        r"\b.{0,80}\blast\s+two\s+messages\b",
        text,
    ):
        return "recent_pair"
    if re.search(
        r"\bwhat\b.{0,100}\bdid\s+i\s+just\s+ask\s+you\s+to\s+"
        r"(?:invent|create|make|define|write|draft|describe|summarize|explain)\b",
        text,
    ):
        return "last_user"
    if any(marker in text for marker in _CONVERSATION_RECALL_LAST_USER_MARKERS):
        return "last_user"
    if any(marker in text for marker in _CONVERSATION_RECALL_TOPIC_MARKERS):
        return "topic"
    return ""


def _clip_conversation_text(text: Any, *, limit: int = 420) -> str:
    clipped = " ".join(str(text or "").strip().split())
    if len(clipped) <= limit:
        return clipped
    if limit <= 3:
        return clipped[: max(0, limit)]
    return clipped[: limit - 3].rstrip() + "..."


def _conversation_record_visible_to_principal(
    record: dict[str, Any],
    *,
    principal_id: str,
    principal_surface: str,
) -> bool:
    """Authorize a personal conversation record before it becomes context.

    Legacy unbound records predate principal-aware persistence. Only the local
    owner may adopt or read those records; paired surfaces require an exact
    principal and surface match.
    """

    if not principal_id or not principal_surface:
        return True
    record_principal = " ".join(
        str(record.get("principal_id") or record.get("user_id") or "").strip().split()
    )[:160]
    record_surface = str(record.get("principal_surface") or "").strip().casefold()[:32]
    if record_principal:
        if record_principal != principal_id:
            return False
        if not record_surface:
            return principal_surface == "owner"
        return record_surface == principal_surface
    return principal_surface == "owner"


async def _recent_completed_conversation_exchanges(
    *,
    current_user_message: str,
    session_id: str = "",
    limit: int = 6,
    allow_cross_session: bool = True,
) -> list[dict[str, str]]:
    current = str(current_user_message or "").strip()
    safe_session_id = str(session_id or "")[:64]
    principal_id, principal_surface = _chat_memory_identity()
    async with _get_convo_lock():
        completed = [
            entry
            for entry in _conversation_log
            if str(entry.get("status") or "complete").strip().lower() == "complete"
            and _conversation_record_visible_to_principal(
                entry,
                principal_id=principal_id,
                principal_surface=principal_surface,
            )
            and (not safe_session_id or str(entry.get("session_id") or "")[:64] == safe_session_id)
        ]

    exchanges: list[dict[str, str]] = []
    for entry in reversed(completed):
        user_text = str(entry.get("user") or "").strip()
        aura_text = str(entry.get("aura") or "").strip()
        if current and user_text == current:
            continue
        if not user_text and not aura_text:
            continue
        exchanges.append(
            # Stamped per entry: the cognitive engine promotes these straight
            # into chat roles, so a forged entry mixed into a real list would
            # become an assistant turn Aura never took.
            stamp_runtime_payload(
                {
                    "exchange_id": str(entry.get("id") or ""),
                    "user": _clip_conversation_text(
                        user_text,
                        limit=_RECENT_CONVERSATION_USER_CHARS,
                    ),
                    "aura": _clip_conversation_text(
                        aura_text,
                        limit=_RECENT_CONVERSATION_AURA_CHARS,
                    ),
                    "timestamp": str(entry.get("completed_at") or entry.get("timestamp") or ""),
                    "session_id": str(entry.get("session_id") or "")[:64],
                }
            )
        )
        if len(exchanges) >= max(1, int(limit)):
            break
    exchanges.reverse()

    if len(exchanges) >= max(1, int(limit)):
        return exchanges

    durable = await _load_durable_conversation_exchanges(
        limit=max(1, int(limit)),
        session_id=safe_session_id,
        allow_cross_session=allow_cross_session,
    )
    in_memory_ids = {
        str(entry.get("exchange_id") or "").strip()
        for entry in exchanges
        if str(entry.get("exchange_id") or "").strip()
    }
    in_memory_content_keys = {
        (
            str(entry.get("user") or "").strip(),
            str(entry.get("aura") or "").strip(),
        )
        for entry in exchanges
    }
    merged: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_legacy_keys: set[tuple[str, str]] = set()
    for entry in durable:
        exchange_id = str(entry.get("exchange_id") or "").strip()
        user_text = _clip_conversation_text(
            entry.get("user"),
            limit=_RECENT_CONVERSATION_USER_CHARS,
        )
        aura_text = _clip_conversation_text(
            entry.get("aura"),
            limit=_RECENT_CONVERSATION_AURA_CHARS,
        )
        key = (user_text, aura_text)
        if current and user_text == current:
            continue
        if not user_text and not aura_text:
            continue
        if exchange_id:
            if exchange_id in in_memory_ids or exchange_id in seen_ids:
                continue
            seen_ids.add(exchange_id)
        else:
            if key in in_memory_content_keys or key in seen_legacy_keys:
                continue
            seen_legacy_keys.add(key)
        merged.append(entry)
    merged.extend(exchanges)
    return merged[-max(1, int(limit)) :]


def _durable_session_may_hold_turns(session: dict[str, Any]) -> bool:
    """False only when a session states it holds no turns."""
    if "turn_count" not in session:
        return True
    try:
        return int(session.get("turn_count") or 0) > 0
    except (TypeError, ValueError):
        return True


def _load_durable_conversation_exchanges_sync(
    *,
    limit: int,
    session_id: str = "",
    allow_cross_session: bool = True,
) -> list[dict[str, str]]:
    persistence = ServiceContainer.get("persistence", default=None)
    get_recent_sessions = getattr(persistence, "get_recent_sessions", None)
    get_session_history = getattr(persistence, "get_session_history", None)
    if not callable(get_session_history):
        return []

    safe_session_id = str(session_id or "")[:64]
    fetch_limit = max(8, limit * 6)
    principal_id, principal_surface = _chat_memory_identity()
    scope_kwargs = (
        {
            "principal_id": principal_id,
            "principal_surface": principal_surface,
        }
        if principal_id and principal_surface
        else {}
    )

    current_rows: list[dict[str, Any]] = []
    if safe_session_id:
        history = get_session_history(
            safe_session_id,
            limit=fetch_limit,
            **scope_kwargs,
        )
        current_rows = [item for item in list(history or []) if isinstance(item, dict)]

    # A restart mints a new session id, so the session a live turn belongs to is
    # empty exactly when continuity matters most.
    #
    # LIVE DEFECT, 2026-08-10. This branch used to read the current session and
    # nothing else, and the multi-session scan below ran only when no session id
    # was supplied — which never happens on a live turn. Asked "earlier today
    # you and i had a long conversation, tell me one specific thing from it",
    # she answered "I can't reach that conversation" with all 34 turns of it on
    # disk. Durable storage the recall path could not reach is not memory.
    #
    # Reach back only for the shortfall: a session already holding the whole
    # window keeps exactly the behaviour it had.
    exchanges_here = sum(
        1 for row in current_rows if str(row.get("role") or "").strip().lower() == "user"
    )
    rows: list[dict[str, Any]] = []
    if (
        allow_cross_session
        and exchanges_here < max(1, int(limit))
        and callable(get_recent_sessions)
    ):
        # Sessions with nothing in them are boot artifacts, and a bounded scan
        # that counts them spends its whole window on restarts instead of
        # conversations — three reboots used to hide yesterday entirely.
        try:
            sessions = list(
                get_recent_sessions(
                    limit=_DURABLE_CONVERSATION_SESSION_SCAN_LIMIT,
                    with_turns_only=True,
                    **scope_kwargs,
                )
                or []
            )
        except TypeError:
            # An older persistence object without the filter. A session that
            # does not report a count is unknown, not empty — dropping those
            # would hide every conversation rather than only the boot rows.
            sessions = [
                session
                for session in (
                    get_recent_sessions(
                        limit=_DURABLE_CONVERSATION_SESSION_SCAN_LIMIT,
                        **scope_kwargs,
                    )
                    or []
                )
                if isinstance(session, dict) and _durable_session_may_hold_turns(session)
            ]
        # Oldest first: ordering below is positional, so earlier conversations
        # have to be earlier in the list.
        for session in reversed(sessions):
            if not isinstance(session, dict):
                continue
            durable_session_id = str(session.get("id") or "").strip()
            if not durable_session_id or durable_session_id == safe_session_id:
                continue
            history = get_session_history(
                durable_session_id,
                limit=fetch_limit,
                **scope_kwargs,
            )
            rows.extend(item for item in list(history or []) if isinstance(item, dict))

    rows.extend(current_rows)
    if not rows:
        return []

    identified: dict[tuple[str, str], dict[str, Any]] = {}
    legacy_pending: dict[str, tuple[int, dict[str, Any]]] = {}
    candidates: list[tuple[int, dict[str, str]]] = []

    for position, row in enumerate(rows):
        role = str(row.get("role") or "").strip().lower()
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        row_session_id = str(row.get("session_id") or safe_session_id or "")[:64]
        cid = str(row.get("cid") or "").strip()
        exchange_id, separator, cid_side = cid.rpartition(":")
        canonical_side = "aura" if cid_side == "assistant" else cid_side
        role_side = "aura" if role == "assistant" else role
        identified_row = bool(
            separator
            and exchange_id
            and canonical_side in {"user", "aura"}
            and canonical_side == role_side
        )

        if identified_row:
            key = (row_session_id, exchange_id)
            state = identified.setdefault(
                key,
                {
                    "exchange_id": exchange_id,
                    "session_id": row_session_id,
                    "position": position,
                    "ambiguous": False,
                },
            )
            existing = state.get(canonical_side)
            if existing is not None:
                if str(existing.get("content") or "").strip() != content:
                    state["ambiguous"] = True
            else:
                state[canonical_side] = row
            state["position"] = max(int(state.get("position") or 0), position)
            continue

        # A nonempty but unrecognized CID is correlated data whose identity we
        # do not understand. Never attach it to a neighboring row. Only old
        # records with no CID at all may use the conservative legacy fallback.
        if cid:
            legacy_pending.pop(row_session_id, None)
            continue
        if role == "user":
            legacy_pending[row_session_id] = (position, row)
            continue
        if role in {"aura", "assistant"}:
            pending = legacy_pending.pop(row_session_id, None)
            if pending is None:
                continue
            _user_position, user_row = pending
            candidates.append(
                (
                    position,
                    {
                        "user": _clip_conversation_text(
                            user_row.get("content"),
                            limit=_RECENT_CONVERSATION_USER_CHARS,
                        ),
                        "aura": _clip_conversation_text(
                            content,
                            limit=_RECENT_CONVERSATION_AURA_CHARS,
                        ),
                        "timestamp": str(row.get("created_at") or user_row.get("created_at") or ""),
                        "session_id": row_session_id,
                    },
                )
            )
            continue
        legacy_pending.pop(row_session_id, None)

    for state in identified.values():
        user_row = state.get("user")
        aura_row = state.get("aura")
        if state.get("ambiguous") or user_row is None or aura_row is None:
            continue
        candidates.append(
            (
                int(state.get("position") or 0),
                {
                    "exchange_id": str(state.get("exchange_id") or ""),
                    "user": _clip_conversation_text(
                        user_row.get("content"),
                        limit=_RECENT_CONVERSATION_USER_CHARS,
                    ),
                    "aura": _clip_conversation_text(
                        aura_row.get("content"),
                        limit=_RECENT_CONVERSATION_AURA_CHARS,
                    ),
                    "timestamp": str(
                        aura_row.get("created_at") or user_row.get("created_at") or ""
                    ),
                    "session_id": str(state.get("session_id") or "")[:64],
                },
            )
        )

    candidates.sort(key=lambda item: item[0])
    # Stamped on the way out, so the durable path attests exactly like the
    # live one. The cognitive engine promotes these into chat roles.
    return [stamp_runtime_payload(exchange) for _position, exchange in candidates[-limit:]]


async def _load_durable_conversation_exchanges(
    *,
    limit: int,
    session_id: str = "",
    allow_cross_session: bool = True,
) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _load_durable_conversation_exchanges_sync,
                limit=max(1, int(limit)),
                session_id=str(session_id or "")[:64],
                allow_cross_session=allow_cross_session,
            ),
            timeout=_DURABLE_CONVERSATION_CONTEXT_TIMEOUT_S,
        )
    except (TimeoutError, *_CHAT_RECOVERABLE_ERRORS) as exc:
        record_degradation("chat.conversation_persistence", exc)
        logger.debug("Durable conversation context load skipped: %s", exc)
        return []


async def _recall_durable_conversation_snippets(user_message: str, *, limit: int = 3) -> list[str]:
    try:
        memory_facade = ServiceContainer.get("memory_facade", default=None)
        if memory_facade is None:
            return []
        search = getattr(memory_facade, "search", None) or getattr(
            memory_facade, "query_memory", None
        )
        if not callable(search):
            return []
        query = f"recent conversation continuity {str(user_message or '').strip()[:160]}"
        principal_id, principal_surface = _chat_memory_identity()
        scope_kwargs = (
            {
                "principal_id": principal_id,
                "principal_surface": principal_surface,
            }
            if principal_id and principal_surface
            else {}
        )
        result = search(query, limit=max(1, int(limit)), **scope_kwargs)
        records = await result if hasattr(result, "__await__") else result
        snippets: list[str] = []
        for item in list(records or []):
            if isinstance(item, dict):
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                if metadata and metadata.get("private"):
                    continue
                content = str(
                    item.get("content") or item.get("text") or item.get("summary") or ""
                ).strip()
            else:
                content = str(item or "").strip()
            if content:
                snippets.append(_clip_conversation_text(content, limit=260))
            if len(snippets) >= limit:
                break
        return snippets
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.conversation_recall", exc)
        logger.debug("Durable conversation recall skipped: %s", exc)
        return []


_CONTENT_RECALL_STOPWORDS = frozenset(
    "a an and are as at be before but by chat choose chose did do does earlier "
    "for from gave give had has have i in is it its just keep kind me mention "
    "mentioned mind my name note number of on or pick picked quick remember "
    "reminder said say small so tell that the thing this those to told was "
    "were what which while with you your".split()
)


def _content_recall_keywords(user_message: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z'-]{2,}", str(user_message or "").lower())
    return [tok for tok in tokens if tok not in _CONTENT_RECALL_STOPWORDS]


async def _find_session_content_exchanges(
    user_message: str,
    *,
    session_id: str = "",
    limit: int = _MAX_CONVERSATION_LOG_EXCHANGES,
) -> list[dict[str, str]]:
    """Latest-first session exchanges whose USER turn matches the question's
    content words. Grounded content recall: the answer to "earlier I gave you
    X" is a quote from the transcript, never a durable-memory guess.

    Searches the FULL retained session (the log is bounded at
    _MAX_CONVERSATION_LOG_EXCHANGES), not a recent window: a fact you gave
    100 turns ago must still be recallable. A 40-turn window silently
    "forgot" anything planted earlier in a long conversation — the 200-turn
    endurance soak's retention probes (plant at turn 3, probe at turn 111)
    failed 0/3 purely because the plant had scrolled out of the window while
    still sitting in the log. Keyword matching over <=500 turns is cheap."""
    keywords = _content_recall_keywords(user_message)
    if not keywords:
        return []
    exchanges = await _recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=limit,
    )
    scored: list[tuple[int, int, dict[str, str]]] = []
    for idx, entry in enumerate(exchanges):
        user_text = str(entry.get("user") or "").lower()
        if not user_text:
            continue
        hits = sum(1 for keyword in keywords if keyword in user_text)
        if hits >= min(2, len(keywords)):
            scored.append((hits, idx, entry))
    # Best keyword coverage first; among ties prefer the most recent turn.
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [entry for _, _, entry in scored]


_NON_ANSWER_OPENERS: tuple[str, ...] = (
    "I couldn't get a clear enough answer together",
    "I couldn't put together an answer I trust",
    "I keep circling the same non-answer",
)


def _is_non_answer_surface(text: str) -> bool:
    """True when the text is a refusal notice rather than a reply.

    Live, 2026-08-04: asked what he had said he wanted to learn, she quoted
    his sentence back correctly and then appended
    'and I acknowledged it: "I couldn\'t get a clear enough answer
    together..."'. She had not acknowledged it — she had declined to answer.
    Recalling a refusal as an acknowledgement asserts a thing that did not
    happen, on the one path whose entire purpose is anti-confabulation.
    """
    stripped = " ".join(str(text or "").split())
    return any(stripped.startswith(opener) for opener in _NON_ANSWER_OPENERS)


async def _build_conversation_recall_reply(
    user_message: str,
    *,
    session_id: str = "",
) -> str | None:
    # Positional/temporal recall ("what did I first ask") is a POSITIONAL key the
    # content classifier below can't resolve — the earliest turn rarely shares
    # words with the question. Resolve the ACTUAL earliest completed turn so the
    # Cortex grounds on the real quote (anti-confabulation) via the established
    # conversation_recall_evidence contract, instead of inventing a memory.
    try:
        from core.conversation.grounded_recall import detect_positional_recall

        _position = detect_positional_recall(user_message)
    except (ImportError, AttributeError, ValueError):
        _position = None
    if _position == "first":
        all_exchanges = await _recent_completed_conversation_exchanges(
            current_user_message=user_message,
            session_id=session_id,
            limit=80,
        )
        if all_exchanges:
            first_user = _clip_conversation_text(all_exchanges[0].get("user"), limit=520)
            if first_user:
                return f'The first thing you asked me in this conversation was: "{first_user}"'

    recall_kind = _classify_conversation_recall_request(user_message)
    if not recall_kind:
        return None

    if recall_kind == "content":
        # The asked-for fact lives in THIS session's transcript or nowhere.
        # Quote the matching turn verbatim (anti-confabulation: always true),
        # or say honestly that it isn't there. Durable memory is the wrong
        # lane for "earlier in this conversation" and must not be asserted.
        matches = await _find_session_content_exchanges(user_message, session_id=session_id)
        if matches:
            quoted = _clip_conversation_text(matches[0].get("user"), limit=420)
            reply = f'Earlier in this conversation you told me: "{quoted}"'
            ack = _clip_conversation_text(matches[0].get("aura"), limit=200)
            # A refusal is not an acknowledgement. Quoting one back claims a
            # response that never happened, on the path built to never claim
            # anything that did not.
            if ack and not _is_non_answer_surface(ack):
                reply += f' — and I acknowledged it: "{ack}"'
            return reply
        # Before declaring a miss: is it in the durable ledger from before a
        # restart?
        #
        # A pin is stored under the session id that made it, and a restart
        # mints a new one, so cross-session recall is deliberately gated
        # behind the user naming the restart — concurrent clients must stay
        # isolated. But the person on the desktop does not know the process
        # died. Live 2026-07-27: pinned "my project codename is HELIOTROPE,
        # build 4471", the runtime was killed and relaunched, and "what was
        # the codename I gave you earlier?" got this honest miss — while the
        # fact sat in session_memory_pins.jsonl, written three times under
        # three different session ids.
        #
        # Isolation is about who may READ a pin, and this changes nothing
        # there: the answer names its own provenance rather than pretending
        # the current session holds it. Losing a fact she demonstrably has,
        # and calling that honesty, is the worse failure.
        durable_pin = await _recall_durable_session_memory_pin(
            session_id=session_id,
            cross_session=_cross_session_memory_recall_allowed(user_message),
        )
        durable_content = str((durable_pin or {}).get("content") or "").strip()
        if durable_content and _content_recall_matches_pin(user_message, durable_content):
            return (
                f"Not in this session's turns — but from before the last restart "
                f'I still have it: "{durable_content}".'
            )
        return (
            "I don't find that in this conversation's completed turns, so I "
            "won't guess. If you tell me again I'll hold onto it."
        )

    exchanges = await _recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=6,
    )
    if exchanges:
        last = exchanges[-1]
        if recall_kind == "last_user":
            user_text = _clip_conversation_text(last.get("user"), limit=520)
            if user_text:
                return f'Your last completed message before this was: "{user_text}"'
        if recall_kind == "last_aura":
            aura_text = _clip_conversation_text(last.get("aura"), limit=620)
            if aura_text:
                return f'My last completed reply before this was: "{aura_text}"'
        if recall_kind == "recent_pair":
            recent_pair = exchanges[-2:]
            pair_lines: list[str] = []
            for entry in recent_pair:
                user_text = _clip_conversation_text(entry.get("user"), limit=220)
                aura_text = _clip_conversation_text(entry.get("aura"), limit=260)
                if user_text and aura_text:
                    pair_lines.append(f"- You: {user_text} / Me: {aura_text}")
                elif user_text:
                    pair_lines.append(f"- You: {user_text}")
                elif aura_text:
                    pair_lines.append(f"- Me: {aura_text}")
            if pair_lines:
                return "The last two completed exchanges were:\n" + "\n".join(pair_lines)

        topic_lines: list[str] = []
        for entry in exchanges[-4:]:
            user_text = _clip_conversation_text(entry.get("user"), limit=180)
            aura_text = _clip_conversation_text(entry.get("aura"), limit=180)
            if user_text and aura_text:
                topic_lines.append(f"- You: {user_text} / Me: {aura_text}")
            elif user_text:
                topic_lines.append(f"- You: {user_text}")
            elif aura_text:
                topic_lines.append(f"- Me: {aura_text}")
        if topic_lines:
            return "Recently, this conversation has been about:\n" + "\n".join(topic_lines)

    durable = await _recall_durable_conversation_snippets(user_message, limit=3)
    if durable:
        lines = "\n".join(f"- {snippet}" for snippet in durable)
        return (
            "I do not have a completed prior turn in this live session, but durable memory has:\n"
            + lines
        )

    return "I do not have a completed prior turn to recall yet in this live session."


def _normalize_user_message(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\bdont'?\b", "don't", normalized)
