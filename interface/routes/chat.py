"""interface/routes/chat.py
──────────────────────────
Extracted from server.py — Chat, session management, conversation lane,
and related API endpoints.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import config
from core.container import ServiceContainer
from core.utils.intent_normalization import normalize_memory_intent_text
from core.version import version_string

from interface.auth import (
    CHEAT_CODE_COOKIE_NAME,
    CHEAT_CODE_COOKIE_TTL_SECS,
    _activate_cheat_code_for_request,
    _check_rate_limit,
    _encode_owner_session_cookie,
    _require_internal,
    _restore_owner_session_from_request,
)
from interface.helpers import _notify_user_spoke

logger = logging.getLogger("Aura.Server.Chat")

router = APIRouter()


# ── Request Models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class CheatCodeRequest(BaseModel):
    code: str
    silent: bool = False


# Max chat message size to prevent memory exhaustion
MAX_CHAT_MESSAGE_BYTES = 64 * 1024  # 64KB


# ── Session & Conversation Log ────────────────────────────────

_conversation_log: list[dict] = []  # In-memory session log for current runtime
_conversation_log_lock = asyncio.Lock()
_session_memory_pins: list[dict] = []
_MAX_CONVERSATION_LOG_EXCHANGES = 500
_foreground_chat_lock = asyncio.Lock()
_FOREGROUND_CHAT_BUSY_WAIT_S = 2.0


def _new_exchange_id() -> str:
    return uuid.uuid4().hex[:8]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _trim_conversation_log_locked() -> None:
    while len(_conversation_log) > _MAX_CONVERSATION_LOG_EXCHANGES:
        _conversation_log.pop(0)


async def _begin_logged_exchange(user_msg: str) -> str:
    """Create an in-flight exchange record and return its identifier."""
    exchange_id = _new_exchange_id()
    async with _conversation_log_lock:
        _conversation_log.append(
            {
                "id": exchange_id,
                "timestamp": _utc_now_iso(),
                "user": user_msg,
                "aura": "",
                "status": "pending",
            }
        )
        _trim_conversation_log_locked()
    return exchange_id


async def _complete_logged_exchange(
    exchange_id: Optional[str],
    user_msg: str,
    aura_response: str,
    *,
    regenerated: bool = False,
) -> None:
    """Finalize a pending exchange in place so history is never duplicated."""
    final_response = aura_response or "…"
    recorded_user = str(user_msg or "")

    async with _conversation_log_lock:
        target: Optional[dict] = None
        if exchange_id:
            for entry in reversed(_conversation_log):
                if str(entry.get("id") or "") == str(exchange_id):
                    target = entry
                    break

        if target is None:
            target = {
                "id": exchange_id or _new_exchange_id(),
                "timestamp": _utc_now_iso(),
                "user": recorded_user,
            }
            _conversation_log.append(target)

        target["user"] = recorded_user
        target["aura"] = final_response
        target["status"] = "complete"
        target["completed_at"] = _utc_now_iso()
        if regenerated:
            target["regenerated"] = True
        _trim_conversation_log_locked()

    try:
        from core.runtime.conversation_support import record_conversation_experience

        await record_conversation_experience(recorded_user, final_response)
    except Exception as exc:
        logger.debug("Conversation experience recording skipped: %s", exc)


async def _log_exchange(user_msg: str, aura_response: str):
    """Record a conversation exchange for session tracking."""
    exchange_id = await _begin_logged_exchange(user_msg)
    await _complete_logged_exchange(exchange_id, user_msg, aura_response)


async def _emit_chat_output_receipt(
    reply_text: str,
    *,
    cause: str,
    origin: str = "api",
    target: str = "primary",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record direct chat replies as durable output receipts."""
    try:
        from core.runtime.receipts import OutputReceipt, get_receipt_store

        digest = hashlib.sha256(str(reply_text or "").encode("utf-8")).hexdigest()[:16]
        receipt = OutputReceipt(
            cause=str(cause or "chat_response"),
            origin=str(origin or "api"),
            target=str(target or "primary"),
            digest=digest,
            metadata=dict(metadata or {}),
        )
        await asyncio.to_thread(get_receipt_store().emit, receipt)
    except Exception as exc:
        logger.debug("Chat output receipt emit skipped: %s", exc)


async def _preserve_large_user_paste(user_msg: str) -> None:
    """Keep large pasted text in live working memory for follow-up references."""
    content = str(user_msg or "").strip()
    if len(content) < 4000:
        return
    try:
        state = _resolve_live_aura_state()
        cognition = getattr(state, "cognition", None) if state is not None else None
        working_memory = getattr(cognition, "working_memory", None)
        if not isinstance(working_memory, list):
            return
        if working_memory and str((working_memory[-1] or {}).get("content", "")) == content:
            return
        working_memory.append(
            {
                "role": "user",
                "content": content,
                "timestamp": time.time(),
                "metadata": {
                    "type": "large_user_paste",
                    "source": "chat_api",
                    "preserve_for_followup": True,
                },
            }
        )
        if len(working_memory) > 80:
            del working_memory[: len(working_memory) - 80]
    except Exception as exc:
        logger.debug("Large paste preservation skipped: %s", exc)


def _extract_session_memory_pin_request(user_message: str) -> Optional[str]:
    text = str(user_message or "").strip()
    if not text:
        return None

    head, sep, tail = text.partition(":")
    normalized = (
        f"{normalize_memory_intent_text(head)}{sep}{tail}"
        if sep
        else normalize_memory_intent_text(text)
    )

    patterns = (
        r"^remember this phrase(?: for later in this session)?\s*:\s*(.+)$",
        r"^remember this(?: for later in this session)?\s*:\s*(.+)$",
        r"^don't forget(?: this)?\s*:\s*(.+)$",
        r"^make note of this(?: for later in this session)?\s*:\s*(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        if match:
            pinned = match.group(1).strip().strip("\"'“”").rstrip(" .!?")
            return pinned[:240] if pinned else None
    return None


def _is_session_memory_recall_request(user_message: str) -> bool:
    text = normalize_memory_intent_text(_normalize_user_message(user_message))
    if not text:
        return False
    markers = (
        "what phrase did i ask you to remember",
        "what did i ask you to remember",
        "what phrase did i tell you to remember",
        "what did i tell you to remember",
        "what did you store for me earlier in this session",
        "what did you pin for later in this session",
    )
    return any(marker in text for marker in markers)


async def _store_session_memory_pin(content: str, source: str) -> None:
    pinned = str(content or "").strip()
    if not pinned:
        return
    async with _conversation_log_lock:
        _session_memory_pins.append(
            {
                "content": pinned[:240],
                "source": str(source or "").strip()[:512],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        if len(_session_memory_pins) > 100:
            _session_memory_pins.pop(0)


async def _recall_session_memory_pin() -> Optional[Dict[str, str]]:
    async with _conversation_log_lock:
        if not _session_memory_pins:
            return None
        latest = _session_memory_pins[-1]
        return {
            "content": str(latest.get("content") or ""),
            "source": str(latest.get("source") or ""),
            "timestamp": str(latest.get("timestamp") or ""),
        }


def _extract_repo_probe_request(user_message: str) -> Optional[Dict[str, str]]:
    text = str(user_message or "").strip()
    if not text:
        return None

    patterns = (
        (
            r"^(?:read|open|inspect)\s+([A-Za-z0-9_./~-]+\.[A-Za-z0-9]+)\s+and\s+tell me\s+the\s+first\s+non-comment\s+dependency\s+line[.?!]*$",
            "first_non_comment_dependency_line",
        ),
        (
            r"^(?:read|open|inspect)\s+([A-Za-z0-9_./~-]+\.[A-Za-z0-9]+)\s+and\s+tell me\s+the\s+first\s+non-comment\s+line[.?!]*$",
            "first_non_comment_line",
        ),
        (
            r"^(?:read|open|inspect)\s+([A-Za-z0-9_./~-]+\.[A-Za-z0-9]+)\s+and\s+tell me\s+how many\s+lines(?:\s+it\s+has)?[.?!]*$",
            "line_count",
        ),
    )
    for pattern, mode in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return {"target": match.group(1), "mode": mode}
    return None


def _read_repo_probe_reply(user_message: str) -> Optional[Dict[str, str]]:
    request = _extract_repo_probe_request(user_message)
    if not request:
        return None

    try:
        from core.demo_support import _resolve_target_path

        target = str(request.get("target") or "").strip()
        mode = str(request.get("mode") or "").strip()
        path = _resolve_target_path(target)
        if not path:
            return {
                "reply": f"I reached for `{Path(target).name or target}` in my live workspace and couldn't find it cleanly.",
                "status": "repo_probe_missing",
            }

        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()

        if mode == "first_non_comment_dependency_line":
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    reply = (
                        f"I read `{path.name}` directly. The first non-comment dependency line is "
                        f"`{stripped}`. That's coming from the live file, not from recall."
                    )
                    return {"reply": reply, "status": "repo_probe_dependency"}
            return {
                "reply": f"I read `{path.name}` directly, but I didn't find a non-comment dependency line in it.",
                "status": "repo_probe_empty",
            }

        if mode == "first_non_comment_line":
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    reply = (
                        f"I read `{path.name}` directly. The first non-comment line is "
                        f"`{stripped}`."
                    )
                    return {"reply": reply, "status": "repo_probe_line"}
            return {
                "reply": f"I read `{path.name}` directly, but every visible line is empty or commented out.",
                "status": "repo_probe_empty",
            }

        if mode == "line_count":
            reply = (
                f"I counted `{path.name}` directly in the live workspace. "
                f"It has {len(lines)} lines right now."
            )
            return {"reply": reply, "status": "repo_probe_line_count"}
    except Exception as exc:
        logger.debug("Repo probe read failed: %s", exc)

    return {
        "reply": "I reached for the file directly, but the live read didn't complete cleanly this time.",
        "status": "repo_probe_error",
    }


# ── Idempotency ───────────────────────────────────────────────

_idempotency_cache: collections.OrderedDict = collections.OrderedDict()
_idempotency_lock = asyncio.Lock()

# ── Stale Response Detection ─────────────────────────────────
# Track the last N responses to detect when the cortex is stuck returning the
# same cached output. This prevents the "Dark Matter" loop where a stale
# identity prompt produces identical text on every turn.
_recent_responses: collections.deque = collections.deque(maxlen=12)
_recent_response_pairs: collections.deque = collections.deque(maxlen=12)  # (user_fp, normalized_response) tuples
_STALE_REPEAT_THRESHOLD = 2  # same text seen this many times = stale
_FUZZY_SIMILARITY_THRESHOLD = 0.80  # word-overlap ratio that counts as semantically stale
_consecutive_degraded_count: int = 0  # tracks degradation streak for proactive recovery
_TOPIC_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'/-]*\b", re.IGNORECASE)
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
)
_TOPICAL_BRIDGE_MARKERS = (
    "you",
    "your",
    "that",
    "there",
    "it",
    "this",
    "because",
    "feels like",
    "standing in for",
    "underneath that",
    "what you",
    "when you",
    "if you",
)
_CONTENT_OBJECT_MARKERS = (
    "article",
    "book",
    "chapter",
    "character",
    "essay",
    "film",
    "movie",
    "narrative",
    "novel",
    "passage",
    "piece",
    "plot",
    "poem",
    "post",
    "premise",
    "scene",
    "script",
    "story",
    "text",
    "thread",
)
_UNREQUESTED_CONTENT_REVIEW_MARKERS = (
    "a chilling and imaginative take",
    "a classic setup",
    "the execution is strong",
    "the premise",
    "the story is",
    "the narrative",
    "this story",
    "this narrative",
)
_INCOMPLETE_TAIL_WORDS = {
    "a",
    "an",
    "and",
    "because",
    "but",
    "for",
    "from",
    "if",
    "into",
    "of",
    "or",
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
    "with",
}


def _response_fingerprint(text: str) -> str:
    """Normalize whitespace and truncate for comparison."""
    return " ".join(str(text or "").split())[:200].strip().lower()


def _normalize_response_body(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _word_set(text: str) -> set:
    """Extract word set for fuzzy similarity comparison."""
    return set(re.findall(r"[a-z0-9']+", _normalize_response_body(text)))


def _fuzzy_similar(a: str, b: str) -> bool:
    """Check if two responses share >80% word overlap (catches paraphrased repeats)."""
    words_a = _word_set(a)
    words_b = _word_set(b)
    if not words_a or not words_b:
        return False
    # Jaccard-like: intersection / smaller set
    overlap = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))
    if smaller < 5:
        return False  # too short for meaningful comparison
    return (overlap / smaller) >= _FUZZY_SIMILARITY_THRESHOLD


def _normalize_topic_token(token: str) -> str:
    normalized = str(token or "").strip().lower().strip("-'/")
    if not normalized:
        return ""
    if normalized.endswith("'s") and len(normalized) > 4:
        normalized = normalized[:-2]
    for suffix in ("ing", "ed", "es", "s"):
        if normalized.endswith(suffix) and len(normalized) > (len(suffix) + 3):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _extract_topic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in _TOPIC_TOKEN_RE.findall(str(text or "").lower()):
        for part in re.split(r"[-/]", raw_token):
            normalized = _normalize_topic_token(part)
            if not normalized:
                continue
            if normalized in _TOPIC_STOPWORDS:
                continue
            if len(normalized) < 3 and normalized not in {"ai", "ml", "vr"}:
                continue
            tokens.add(normalized)
    return tokens


async def _gather_recent_user_messages_for_relevance(current_user_message: str, *, limit: int = 4) -> list[str]:
    recent: list[str] = []
    current = str(current_user_message or "").strip()
    async with _conversation_log_lock:
        for entry in reversed(_conversation_log):
            user_text = str(entry.get("user") or "").strip()
            if not user_text or user_text == current:
                continue
            recent.append(user_text)
            if len(recent) >= limit:
                break
    recent.reverse()
    if current:
        recent.append(current)
    return recent[-limit:]


def _build_recent_user_context_block(recent_user_messages: list[str], *, limit: int = 3) -> str:
    if not recent_user_messages:
        return ""
    lines = [
        f"- {str(message or '').strip()[:220]}"
        for message in recent_user_messages[-limit:]
        if str(message or "").strip()
    ]
    return "\n".join(lines)


_TRACEABILITY_REASON_MARKERS = (
    "engineering traceability",
    "operational details",
    "give receipts",
    "give me receipts",
    "refuse to give receipts",
    "exactly why",
    "do not have access",
    "governance rule blocks disclosure",
    "data does not exist",
    "you are uncertain",
)

_TRACEABILITY_EXAMPLE_MARKERS = (
    "most recent non-private action",
    "non-private action",
    "safe example",
    "log line",
    "event id",
    "trace:",
    "timestamp, subsystem, action, result",
)

_TRACEABILITY_CORE_MARKERS = (
    "traceability",
    "receipt",
    "receipts",
    "event id",
    "log line",
    "operational details",
)

_REFERENTIAL_FOLLOWUP_MARKERS = (
    "can you answer it",
    "you gonna answer",
    "answer the question",
    "answer it",
    "the last question",
    "that question",
    "what specifically",
    "what's the actual thing you need",
    "whats the actual thing you need",
)


def _is_referential_followup_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text or len(text) > 120:
        return False
    if any(marker in text for marker in _REFERENTIAL_FOLLOWUP_MARKERS):
        return True
    return ("question" in text or "answer" in text) and any(token in text for token in ("it", "that", "last"))


def _classify_traceability_request(user_message: str) -> tuple[bool, bool, bool]:
    text = _normalize_user_message(user_message)
    if not text:
        return False, False, False

    asks_reason = any(marker in text for marker in _TRACEABILITY_REASON_MARKERS)
    asks_example = any(marker in text for marker in _TRACEABILITY_EXAMPLE_MARKERS)
    asks_traceability = asks_reason or asks_example or (
        any(marker in text for marker in _TRACEABILITY_CORE_MARKERS)
        and ("recent" in text or "safe" in text or "most recent" in text or "why" in text)
    )
    return asks_traceability, asks_reason, asks_example


async def _resolve_traceability_anchor(user_message: str) -> Optional[str]:
    asks_traceability, _, _ = _classify_traceability_request(user_message)
    if asks_traceability:
        return str(user_message or "")

    if not _is_referential_followup_request(user_message):
        return None

    recent = await _gather_recent_user_messages_for_relevance(user_message, limit=6)
    current = str(user_message or "").strip()
    for candidate in reversed(recent):
        candidate_text = str(candidate or "").strip()
        if not candidate_text or candidate_text == current:
            continue
        candidate_traceability, _, _ = _classify_traceability_request(candidate_text)
        if candidate_traceability:
            return candidate_text
    return None


def _collect_recent_traceability_event_sync() -> tuple[Optional[Dict[str, Any]], str]:
    access_errors = 0
    saw_private_only = False

    try:
        from core.runtime.receipts import get_receipt_store

        store = get_receipt_store()
        all_recent = store.query_recent(limit=24)
        if not all_recent:
            store.reload_from_disk()
            all_recent = store.query_recent(limit=24)

        safe_kinds = ["output", "tool_execution", "state_mutation", "computer_use", "autonomy", "self_repair"]
        safe_recent = store.query_recent(kinds=safe_kinds, limit=24)
        for receipt in reversed(safe_recent):
            kind = str(getattr(receipt, "kind", "") or "")
            if kind == "output" and str(getattr(receipt, "target", "") or "") != "primary":
                continue

            event: Dict[str, Any] = {
                "timestamp": float(getattr(receipt, "created_at", 0.0) or 0.0),
                "event_id": str(getattr(receipt, "receipt_id", "") or ""),
                "kind": kind,
                "subsystem": "",
                "action": "",
                "result": "",
                "changed_future_behavior": False,
            }
            if kind == "output":
                event["subsystem"] = f"Output.{str(getattr(receipt, 'origin', '') or 'unknown')}"
                event["action"] = f"emitted {str(getattr(receipt, 'target', '') or 'primary')} response"
                event["result"] = f"digest={str(getattr(receipt, 'digest', '') or 'unknown')}"
            elif kind == "tool_execution":
                tool_name = str(getattr(receipt, "tool", "") or "unknown")
                event["subsystem"] = f"Tool.{tool_name}"
                event["action"] = f"executed tool {tool_name}"
                event["result"] = f"status={str(getattr(receipt, 'status', '') or 'unknown')}"
            elif kind == "state_mutation":
                domain = str(getattr(receipt, "domain", "") or "state")
                key = str(getattr(receipt, "key", "") or "unknown")
                event["subsystem"] = f"State.{domain}"
                event["action"] = f"mutated {domain}.{key}"
                event["result"] = f"schema_v={int(getattr(receipt, 'schema_version', 1) or 1)}"
                event["changed_future_behavior"] = True
            elif kind == "computer_use":
                action_kind = str(getattr(receipt, "action_kind", "") or "act")
                target = str(getattr(receipt, "target", "") or "screen")
                event["subsystem"] = "ComputerUse"
                event["action"] = f"{action_kind} {target}".strip()
                event["result"] = f"verified={bool(getattr(receipt, 'verifier_result', False))}"
            elif kind == "autonomy":
                proposed = str(getattr(receipt, "proposed_action", "") or "autonomous step")
                event["subsystem"] = "Autonomy"
                event["action"] = proposed
                event["result"] = f"level={int(getattr(receipt, 'autonomy_level', 0) or 0)}"
                event["changed_future_behavior"] = True
            elif kind == "self_repair":
                target_module = str(getattr(receipt, "target_module", "") or "unknown")
                event["subsystem"] = "SelfRepair"
                event["action"] = f"self-repair on {target_module}"
                event["result"] = f"rolled_back={bool(getattr(receipt, 'rolled_back', False))}"
                event["changed_future_behavior"] = True
            return event, ""

        if all_recent:
            saw_private_only = True
    except Exception:
        access_errors += 1

    try:
        from core.consciousness.authority_audit import get_audit

        audit = get_audit()
        effects = audit.get_recent_effects(12)
        for effect in reversed(effects):
            if str(effect.get("effect_type") or "") != "response":
                continue
            return {
                "timestamp": float(effect.get("timestamp") or 0.0),
                "event_id": str(effect.get("receipt_id") or ""),
                "kind": "authority_effect",
                "subsystem": str(effect.get("source") or "AuthorityAudit"),
                "action": f"emitted {str(effect.get('effect_type') or 'effect')}",
                "result": "authorized" if bool(effect.get("matched")) else "unmatched",
                "changed_future_behavior": False,
            }, ""
    except Exception:
        access_errors += 1

    try:
        from core.somatic.motor_cortex import get_motor_cortex

        receipts = get_motor_cortex().get_recent_receipts(12)
        for receipt in reversed(receipts):
            return {
                "timestamp": float(receipt.get("timestamp") or 0.0),
                "event_id": str(receipt.get("receipt_id") or ""),
                "kind": "motor_receipt",
                "subsystem": f"MotorCortex.{str(receipt.get('handler') or 'unknown')}",
                "action": f"executed {str(receipt.get('reflex_class') or 'reflex')}",
                "result": str(receipt.get("summary") or f"success={bool(receipt.get('success', False))}"),
                "changed_future_behavior": False,
            }, ""
    except Exception:
        access_errors += 1

    if saw_private_only:
        return None, "governance rule blocks disclosure"
    if access_errors >= 3:
        return None, "do not have access"
    return None, "the data does not exist"


def _format_traceability_reply(
    *,
    anchor_message: str,
    event: Optional[Dict[str, Any]],
    reason_category: str,
) -> str:
    _asks_traceability, asks_reason, asks_example = _classify_traceability_request(anchor_message)

    if event is None:
        if reason_category == "governance rule blocks disclosure":
            return "Reason: governance rule blocks disclosure. I can see recent private traces, but I do not have a safe non-private one I should expose."
        if reason_category == "do not have access":
            return "Reason: I do not have access to a safe live trace for that right now."
        if reason_category == "uncertain":
            return "Reason: I am uncertain which live trace would be the honest one to cite, so I should not invent one."
        return "Reason: the data does not exist in my current rolling trace window."

    timestamp = float(event.get("timestamp") or 0.0)
    timestamp_iso = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if timestamp > 0.0
        else "unknown"
    )
    trace_line = (
        f"Timestamp: {timestamp_iso} | "
        f"Subsystem: {event.get('subsystem') or 'unknown'} | "
        f"EventID: {event.get('event_id') or 'unavailable'} | "
        f"Action: {event.get('action') or 'unknown'} | "
        f"Result: {event.get('result') or 'unknown'} | "
        f"FutureBehavior: {'yes' if bool(event.get('changed_future_behavior')) else 'no'}"
    )

    if asks_example and not asks_reason:
        return trace_line

    preface = (
        "Access scope: I have a rolling runtime trace, not a full lifetime ledger. "
        "I can inspect recent receipts and audit trails, but I should not invent history outside that window."
    )
    return f"{preface}\n{trace_line}"


async def _build_grounded_traceability_reply(user_message: str) -> Optional[str]:
    anchor = await _resolve_traceability_anchor(user_message)
    if not anchor:
        return None

    event, reason_category = await asyncio.to_thread(_collect_recent_traceability_event_sync)
    return _format_traceability_reply(
        anchor_message=anchor,
        event=event,
        reason_category=reason_category,
    )


def _call_stateful_voice_reflex(frame: dict[str, Any], user_message: str) -> str:
    try:
        return _build_stateful_voice_reflex(frame, user_message)
    except TypeError:
        return _build_stateful_voice_reflex(frame)


def _looks_like_unrequested_content_review(user_message: str, reply_text: str) -> tuple[bool, str]:
    user_text = _normalize_user_message(user_message)
    reply = _normalize_user_message(reply_text)
    if not reply:
        return False, ""
    if any(marker in user_text for marker in _CONTENT_OBJECT_MARKERS):
        return False, ""

    review_hits = sum(1 for marker in _UNREQUESTED_CONTENT_REVIEW_MARKERS if marker in reply)
    object_hits = sum(1 for marker in _CONTENT_OBJECT_MARKERS if re.search(rf"\b{re.escape(marker)}\b", reply))
    if review_hits >= 1 and object_hits >= 2:
        return True, "unrequested_content_review"
    if reply.startswith(("the story is", "the premise", "this story", "this narrative")) and object_hits >= 2:
        return True, "unrequested_content_review"
    return False, ""


def _evaluate_reply_topicality(
    user_message: str,
    reply_text: str,
    *,
    recent_user_messages: Optional[list[str]] = None,
) -> tuple[bool, str]:
    reply = str(reply_text or "").strip()
    if not reply:
        return False, ""

    review_drift, review_reason = _looks_like_unrequested_content_review(user_message, reply)
    if review_drift:
        return True, review_reason

    anchors = set()
    for message in recent_user_messages or [user_message]:
        anchors.update(_extract_topic_tokens(message))

    reply_tokens = _extract_topic_tokens(reply)
    if not anchors or len(reply_tokens) < 16:
        return False, ""

    if anchors & reply_tokens:
        return False, ""

    concrete_reply_tokens = {token for token in reply_tokens if len(token) >= 5}
    if len(concrete_reply_tokens) < 12:
        return False, ""

    lowered_reply = _normalize_user_message(reply)
    if any(marker in lowered_reply for marker in _TOPICAL_BRIDGE_MARKERS):
        return False, ""

    return True, "foreign_topic_burst"


def _record_recent_response(text: str, user_message: str = "") -> None:
    fp = _response_fingerprint(text)
    if fp:
        _recent_responses.append(fp)
    if user_message:
        response_body = _normalize_response_body(text)[:500]
        if response_body:
            _recent_response_pairs.append((_response_fingerprint(user_message), response_body))


def _is_stale_repeated_response(text: str) -> bool:
    fp = _response_fingerprint(text)
    if not fp:
        return False
    # Exact match check
    exact_count = sum(1 for r in _recent_responses if r == fp)
    if exact_count >= _STALE_REPEAT_THRESHOLD:
        return True
    # Fuzzy similarity check — catches "same answer, slightly different wording"
    fuzzy_count = sum(1 for r in _recent_responses if _fuzzy_similar(fp, r))
    if fuzzy_count >= _STALE_REPEAT_THRESHOLD:
        logger.debug("Fuzzy stale detection triggered (overlap count=%d).", fuzzy_count)
        return True
    return False


def _is_same_answer_different_prompt(user_message: str, text: str) -> bool:
    """Detect when different user prompts are getting the same response."""
    if _is_referential_followup_request(user_message):
        return False
    user_fp = _response_fingerprint(user_message)
    response_body = _normalize_response_body(text)
    if not user_fp or not response_body:
        return False
    for prev_user, prev_resp in _recent_response_pairs:
        if prev_user == user_fp:
            continue
        if _is_referential_followup_request(prev_user):
            continue
        # Near-paraphrase follow-ups can legitimately receive the same answer.
        if _fuzzy_similar(prev_user, user_fp):
            continue
        prev_tokens = _extract_topic_tokens(prev_user)
        current_tokens = _extract_topic_tokens(user_fp)
        if prev_tokens and current_tokens:
            overlap = len(prev_tokens & current_tokens)
            smaller = min(len(prev_tokens), len(current_tokens))
            if smaller >= 4 and overlap >= 4 and (overlap / smaller) >= 0.8:
                continue
        if prev_resp == response_body or _fuzzy_similar(prev_resp, response_body):
            return True
    return False


def _looks_truncated_tail(text: str) -> bool:
    body = str(text or "").strip()
    if len(body) < 24:
        return False
    if body.endswith(("...", "…", ".", "!", "?", "\"", "'", "”", "’", ")", "]")):
        return False
    if body.endswith(("-", "—", ":", ";", ",")):
        return True
    match = re.search(r"([A-Za-z]+)$", body)
    if not match:
        return False
    last_word = match.group(1).lower()
    if len(last_word) <= 2 and len(body) >= 40:
        return True
    return last_word in _INCOMPLETE_TAIL_WORDS


# ── Response Quality Metrics ─────────────────────────────────
_quality_logger = logging.getLogger("Aura.ResponseQuality")


def _log_response_quality_metrics(
    user_message: str,
    reply_text: str,
    confidence: str,
    stale: bool,
    same_diff: bool,
    off_topic: bool,
) -> None:
    """Log structured quality metrics for every response for offline analysis.

    Writes to a dedicated logger so these can be routed to a file/store
    independently of the main application log.
    """
    try:
        live_state = _resolve_live_aura_state()
        wm_size = 0
        has_summary = False
        coherence = 1.0
        if live_state:
            wm = getattr(getattr(live_state, "cognition", None), "working_memory", None)
            wm_size = len(wm) if isinstance(wm, list) else 0
            has_summary = bool(getattr(getattr(live_state, "cognition", None), "rolling_summary", ""))
            coherence = float(getattr(getattr(live_state, "cognition", None), "coherence_score", 1.0) or 1.0)

        _quality_logger.info(
            "📊 quality_metrics | confidence=%s | stale=%s | same_diff=%s | off_topic=%s | "
            "reply_len=%d | user_len=%d | wm_size=%d | has_summary=%s | coherence=%.3f",
            confidence, stale, same_diff, off_topic,
            len(reply_text or ""), len(user_message or ""),
            wm_size, has_summary, coherence,
        )
    except Exception:
        pass  # metrics are best-effort


# ── Consistency Check ─────────────────────────────────────────
_INABILITY_PATTERNS = re.compile(
    r"(i can't|i cannot|i'm unable to|i don't have access to|"
    r"i'm not able to|i lack the ability to)(?:\s+\w+){0,3}\s+"
    r"(browse|search|access|look up|check|find|open|read|visit|navigate|download|fetch)",
    re.IGNORECASE,
)


def _check_response_consistency(reply_text: str, user_message: str) -> tuple[bool, str]:
    """Check response against known capabilities and commitments.

    Returns (is_consistent, reason).
    """
    text_lower = (reply_text or "").lower()

    # 1. Check for false inability claims — Aura has web access via skills
    if _INABILITY_PATTERNS.search(reply_text or ""):
        try:
            from core.container import ServiceContainer
            cap = ServiceContainer.get("capability_engine", default=None)
            catalog = {}
            if cap and hasattr(cap, "get_catalog"):
                catalog = cap.get_catalog() or {}
            elif cap and hasattr(cap, "get_tool_catalog"):
                tool_catalog = cap.get_tool_catalog(include_inactive=True) or []
                catalog = {
                    str(item.get("name") or ""): {
                        "status": "unavailable" if not bool(item.get("available")) else str(item.get("state") or "ready").lower(),
                    }
                    for item in tool_catalog
                    if isinstance(item, dict) and item.get("name")
                }

            if catalog:
                available = {
                    k for k, v in catalog.items()
                    if isinstance(v, dict) and v.get("status") != "unavailable"
                }
                web_skills = {"web_search", "search_web", "free_search", "grounded_search", "sovereign_browser"}
                desktop_skills = {"computer_use", "os_manipulation", "sovereign_terminal", "sovereign_vision"}
                mentions_web = bool(re.search(r"\b(browse|search|look up|check online|find online|open (?:a )?(?:site|page|url|website))\b", reply_text or "", re.IGNORECASE))
                mentions_desktop = bool(re.search(r"\b(open|control|click|type|use|navigate)\b.{0,40}\b(tab|browser|computer|desktop|screen)\b", reply_text or "", re.IGNORECASE))
                if (mentions_web and web_skills & available) or (mentions_desktop and desktop_skills & available):
                    return False, "false_inability_claim"
        except Exception:
            pass

    # 2. Check for self-contradiction against active commitments
    try:
        from core.agency.commitment_engine import get_commitment_engine
        ce = get_commitment_engine()
        if hasattr(ce, "get_active_commitments"):
            active = ce.get_active_commitments()
            for commitment in (active or [])[:5]:
                desc = str(commitment.get("description", "") or "").lower()
                # If reply says "I don't" or "I haven't" about something we committed to
                if desc and len(desc) > 10:
                    key_words = set(desc.split()) - {"i", "a", "the", "to", "and", "of", "in", "for", "will", "should"}
                    matching = sum(1 for w in key_words if w in text_lower)
                    if matching >= 3 and any(neg in text_lower for neg in ("i don't", "i haven't", "i didn't", "i can't")):
                        return False, "commitment_contradiction"
    except Exception:
        pass

    return True, ""


# ── Commitment Extraction ─────────────────────────────────────
_COMMITMENT_PATTERNS = re.compile(
    r"(?:^|\. |\n)"
    r"(i'll |i will |let me |i'm going to |i won't forget |i promise |"
    r"i'll remember |next step is |we should |i should )"
    r"([^.!?\n]{10,120})",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_and_register_commitments(reply_text: str, user_message: str) -> None:
    """Scan the final response for commitment/promise language and register.

    This closes the open-loop tracking gap: when Aura says 'I'll remember X'
    or 'next step is Z', it actually gets tracked.
    """
    if not reply_text or len(reply_text) < 20:
        return
    try:
        matches = _COMMITMENT_PATTERNS.findall(reply_text)
        if not matches:
            return
        from core.agency.commitment_engine import get_commitment_engine
        ce = get_commitment_engine()
        if not hasattr(ce, "add_commitment"):
            return
        for prefix, body in matches[:3]:  # cap at 3 per response
            description = f"{prefix.strip()} {body.strip()}"
            ce.add_commitment(
                description=description,
                source="auto_extracted",
                context=user_message[:200] if user_message else "",
            )
            logger.debug("📝 Auto-extracted commitment: %s", description[:80])
    except Exception as exc:
        logger.debug("Commitment extraction skipped: %s", exc)


# ── Conversation Lane Helpers ─────────────────────────────────

def _collect_conversation_lane_status() -> Dict[str, Any]:
    from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT, PRIMARY_ENDPOINT

    lane: Dict[str, Any] = {
        "desired_model": "Cortex (32B)",
        "desired_endpoint": PRIMARY_ENDPOINT,
        "foreground_endpoint": None,
        "background_endpoint": BRAINSTEM_ENDPOINT,
        "foreground_tier": "local",
        "background_tier": "local_fast",
        "state": "cold",
        "last_failure_reason": "",
        "conversation_ready": False,
        "last_transition_at": 0.0,
        "warmup_attempted": False,
        "warmup_in_flight": False,
        "expected_model": "",
        "detected_models": [],
        "runtime_identity_ok": True,
        "kernel_tick_age_s": None,
    }
    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            gate_lane = gate.get_conversation_status()
            if isinstance(gate_lane, dict):
                lane.update({k: v for k, v in gate_lane.items() if v is not None})
    except Exception as exc:
        logger.debug("Conversation lane status collection failed: %s", exc)

    try:
        llm_router = ServiceContainer.get("llm_router", default=None)
        if llm_router and hasattr(llm_router, "get_health_report"):
            report = llm_router.get_health_report()
            if report.get("background_endpoint") is not None:
                lane["background_endpoint"] = report.get("background_endpoint", lane.get("background_endpoint"))
            if report.get("background_tier_key") is not None:
                lane["background_tier"] = report.get("background_tier_key", lane.get("background_tier"))
            if not bool(lane.get("conversation_ready", False)):
                lane["last_failure_reason"] = lane.get("last_failure_reason") or report.get("last_user_error", "")
    except Exception as exc:
        logger.debug("Conversation lane/router status merge failed: %s", exc)

    # Kernel tick staleness — lets the UI detect when the kernel is locked up
    try:
        kernel = ServiceContainer.get("aura_kernel", default=None)
        if kernel is None:
            from core.kernel.kernel_interface import KernelInterface
            ki = KernelInterface.get_instance()
            kernel = getattr(ki, "kernel", None) if ki else None
        if kernel:
            last_tick_at = getattr(kernel, "_last_tick_completed_at", 0.0) or 0.0
            if last_tick_at > 0.0:
                lane["kernel_tick_age_s"] = round(time.time() - last_tick_at, 1)
            kernel_lock = getattr(kernel, "_lock", None)
            if kernel_lock is not None:
                try:
                    lock_held = bool(kernel_lock.locked())
                except Exception:
                    lock_held = False
                lane["kernel_lock_held"] = lock_held
                lane["kernel_lock_held_s"] = round(
                    float(getattr(kernel_lock, "held_duration", 0.0) or 0.0),
                    2,
                ) if lock_held else 0.0
    except Exception as exc:
        logger.debug("Kernel tick age probe failed: %s", exc)

    return lane


def _conversation_lane_is_standby(lane: Optional[Dict[str, Any]]) -> bool:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").strip().lower()
    return (
        not bool(lane.get("conversation_ready", False))
        and state in {"cold", "closed", ""}
        and not bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
    )


def _mark_conversation_lane_timeout(reason: str = "foreground_timeout") -> Dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    # Activate recovery cooldown so rapid follow-up messages are fast-rejected
    # instead of piling into the inference pipeline.
    _enter_recovery_cooldown()

    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "note_foreground_timeout"):
            gate.note_foreground_timeout(reason)
    except Exception as exc:
        logger.debug("Conversation lane timeout mark failed: %s", exc)

    lane = _collect_conversation_lane_status()
    lane["state"] = "recovering"
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _mark_conversation_lane_state(reason: str, *, state: str) -> Dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    lane = _collect_conversation_lane_status()
    lane["state"] = state
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    lane["warmup_attempted"] = True
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _foreground_timeout_for_lane(lane: Optional[Dict[str, Any]]) -> float:
    """Foreground timeout for the chat request.

    [STABILITY v50] Raised to 150/180s for M5 64GB hardware. The previous
    90s ceiling was the #1 cause of false-positive cortex timeouts — the
    32B model regularly needs 60-90s for complex first-turn responses,
    leaving zero headroom after warmup, trust gate, and context assembly.
    On M5 with 64GB unified memory there is no gateway proxy, so 504 risk
    is zero. Give the cortex the time it actually needs.
    """
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").lower()
    if bool(lane.get("conversation_ready", False)):
        return 150.0
    if state in {"warming", "recovering", "cold", "spawning", "handshaking"}:
        return 180.0
    return 180.0


def _conversation_lane_user_message(
    lane: Dict[str, Any],
    *,
    timed_out: bool = False,
    status_override: str = "",
) -> str:
    """Generate a personality-infused status message instead of a robotic error.

    [STABILITY v50] These messages now sound like Aura experiencing a
    momentary lapse rather than a system displaying error codes. Uses
    the live expression frame when available so Aura's current mood
    colours even her recovery messages.
    """
    state = str(lane.get("state", "warming") or "warming")
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    status_override = str(status_override or "")

    # Hard infrastructure failures — keep these explicit for debugging
    if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
        return "My local Cortex runtime hit a hard failure — the 32B model can't start. Check the launcher logs for what went wrong."

    # Build a mood-aware prefix for softer messages
    _mood_prefix = ""
    try:
        _pe = ServiceContainer.get("personality_engine", default=None)
        if _pe and hasattr(_pe, "get_emotional_context_for_response"):
            _emo = _pe.get_emotional_context_for_response() or {}
            _mood = str(_emo.get("mood", "") or "").lower()
            if _mood in {"frustrated", "irritated", "tense"}:
                _mood_prefix = "Ugh, "
            elif _mood in {"tired", "drowsy", "low"}:
                _mood_prefix = "Mmm, "
            elif _mood in {"curious", "playful", "amused"}:
                _mood_prefix = "Hmm — "
    except Exception:
        pass

    if status_override == "warming_timeout":
        return f"{_mood_prefix}my thinking engine started warming up but didn't quite get there in time. Give me another moment."
    if status_override == "warming_failed":
        return f"{_mood_prefix}my thinking engine stumbled during warm-up. Try me again in a sec."
    if timed_out:
        return f"{_mood_prefix}I was thinking but my cortex took too long to finish the thought. Try again — I should be warmer now."
    if _conversation_lane_is_standby(lane):
        return "I'm here. My cortex will spin up the moment you say something."
    if state == "recovering":
        return f"{_mood_prefix}I'm in the middle of pulling my thoughts back together. Give me just a moment."
    if state == "failed":
        return f"{_mood_prefix}my thinking engine hit a wall. I need a moment to recover."
    return f"{_mood_prefix}I'm still warming up my thinking engine. Almost there."


_last_recovery_cooldown_at: float = 0.0
_RECOVERY_COOLDOWN_SECONDS: float = 1.0  # [STABILITY v50] Reduced from 5s→1s. The old 5s cooldown amplified single failures into multi-turn outages by fast-rejecting the user's immediate retry. 1s is enough to prevent request pileup without blocking a legitimate retry.
_PROTECTED_FOREGROUND_LOCK_BYPASS_SECONDS: float = 1.0
_PROTECTED_FOREGROUND_PRIMARY_BUDGET_SECONDS: float = 120.0
_PROTECTED_FOREGROUND_SECONDARY_BUDGET_SECONDS: float = 180.0
# [STABILITY v53] Raised from 8s→45s. The old 8s deadline was the #1 cause of
# false-positive kernel timeouts on first-turn responses. The 32B cortex
# regularly needs 15-40s for complex responses, and after a 35s warmup the
# kernel had only 8s before being interrupted by a competing protected
# foreground request — which itself competes for the same LLM resources,
# creating a resource contention spiral. 45s gives the kernel real time to
# respond on turn 1. Subsequent turns (model warm, KV cache hot) are <5s.
_KERNEL_SOFT_REPLY_SLA_SECONDS: float = 90.0


def _enter_recovery_cooldown() -> None:
    global _last_recovery_cooldown_at
    _last_recovery_cooldown_at = time.monotonic()


def _in_recovery_cooldown() -> bool:
    if _last_recovery_cooldown_at <= 0:
        return False
    return (time.monotonic() - _last_recovery_cooldown_at) < _RECOVERY_COOLDOWN_SECONDS


def _kernel_is_congested(lane: Optional[Dict[str, Any]]) -> bool:
    lane = dict(lane or {})
    if not bool(lane.get("kernel_lock_held", False)):
        return False
    return float(lane.get("kernel_lock_held_s", 0.0) or 0.0) >= _PROTECTED_FOREGROUND_LOCK_BYPASS_SECONDS


def _protected_foreground_reason(lane: Optional[Dict[str, Any]]) -> str:
    lane = dict(lane or {})
    lane_state = str(lane.get("state", "") or "").strip().lower()
    if lane_state == "recovering" and _in_recovery_cooldown():
        return "recovery_cooldown"
    if _kernel_is_congested(lane):
        return f"kernel_lock:{float(lane.get('kernel_lock_held_s', 0.0) or 0.0):.2f}s"
    if not bool(lane.get("conversation_ready", False)) and lane_state in {
        "warming",
        "recovering",
        "cold",
        "spawning",
        "handshaking",
    }:
        return f"lane_{lane_state or 'unready'}"
    return ""


async def _build_protected_foreground_history(*, limit_pairs: int = 4) -> List[Dict[str, str]]:
    async with _conversation_log_lock:
        completed = [
            entry
            for entry in _conversation_log
            if str(entry.get("status") or "complete").strip().lower() != "pending"
        ]
        recent = completed[-max(1, int(limit_pairs)) :]

    history: List[Dict[str, str]] = []
    for entry in recent:
        user_msg = str(entry.get("user", "") or "").strip()
        aura_msg = str(entry.get("aura", "") or "").strip()
        if user_msg:
            history.append({"role": "user", "content": user_msg})
        if aura_msg and aura_msg != "…":
            history.append({"role": "assistant", "content": aura_msg})
    return history


def _build_protected_foreground_summary_message() -> Optional[Dict[str, str]]:
    snapshot = _resolve_protected_foreground_snapshot() or {}
    rolling_summary = _sanitize_foreground_continuity_summary(snapshot.get("rolling_summary") or "")
    if not rolling_summary:
        return None
    return {
        "role": "system",
        "content": (
            "[ACTIVE GROUNDING EVIDENCE]\n"
            f"Continuity summary: {rolling_summary[:1200]}"
        ),
    }


def _compact_snapshot_line(label: str, value: Any, *, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    return f"{label}: {text[:max_chars]}"


def _snapshot_field(source: Any, name: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _resolve_protected_foreground_snapshot() -> Dict[str, Any]:
    """Lightweight state snapshot for the protected chat lane.

    Prefer cached/hot state over live subsystem refresh so the control plane can
    answer without depending on organism-wide locks or expensive voice updates.
    """
    try:
        state = _resolve_live_aura_state()
        if state is None:
            return {}
        hot = state.snapshot_hot() if hasattr(state, "snapshot_hot") else {}
        affect = hot.get("affect") if isinstance(hot, dict) else getattr(state, "affect", None)
        cognition = hot.get("cognition") if isinstance(hot, dict) else getattr(state, "cognition", None)
        response_modifiers = hot.get("response_modifiers") if isinstance(hot, dict) else getattr(state, "response_modifiers", None)
        return {
            "mood": getattr(state, "mood", "") or _snapshot_field(affect, "dominant_emotion", ""),
            "tone": _snapshot_field(response_modifiers, "tone", ""),
            "dominant_emotion": _snapshot_field(affect, "dominant_emotion", ""),
            "attention_focus": _snapshot_field(cognition, "attention_focus", ""),
            "valence": _snapshot_field(affect, "valence", ""),
            "arousal": _snapshot_field(affect, "arousal", ""),
            "curiosity": _snapshot_field(affect, "curiosity", ""),
            "coherence": _snapshot_field(cognition, "coherence_score", ""),
            "current_mode": _snapshot_field(cognition, "current_mode", ""),
            "current_objective": _snapshot_field(cognition, "current_objective", ""),
            "rolling_summary": _snapshot_field(cognition, "rolling_summary", ""),
        }
    except Exception as exc:
        logger.debug("Protected foreground snapshot resolve failed: %s", exc)
        return {}


def _build_protected_foreground_system_prompt(
    user_message: str,
    *,
    lane: Dict[str, Any],
) -> str:
    protected_snapshot = _resolve_protected_foreground_snapshot()
    if protected_snapshot:
        voice_state = dict(protected_snapshot)
        voice_snapshot = {}
    else:
        voice_state = _resolve_live_voice_state(user_message, refresh=False)
        voice_snapshot = dict(voice_state.get("substrate_snapshot") or {})

    continuity_summary = _sanitize_foreground_continuity_summary(
        voice_state.get("rolling_summary") or ""
    )

    snapshot_lines = [
        _compact_snapshot_line("Lane", lane.get("state") or "unknown"),
        _compact_snapshot_line("Kernel lock held", lane.get("kernel_lock_held_s") if lane.get("kernel_lock_held") else ""),
        _compact_snapshot_line("Mood", voice_state.get("mood")),
        _compact_snapshot_line("Tone", voice_state.get("tone")),
        _compact_snapshot_line("Dominant emotion", voice_state.get("dominant_emotion")),
        _compact_snapshot_line("Attention", _sanitize_attention_focus(str(voice_state.get("attention_focus") or ""))),
        _compact_snapshot_line("Valence", voice_state.get("valence") or voice_snapshot.get("field_valence")),
        _compact_snapshot_line("Arousal", voice_state.get("arousal") or voice_snapshot.get("arousal")),
        _compact_snapshot_line("Curiosity", voice_state.get("curiosity")),
        _compact_snapshot_line("Coherence", voice_state.get("coherence")),
        _compact_snapshot_line("Current mode", voice_state.get("current_mode")),
        _compact_snapshot_line("Objective", voice_state.get("current_objective")),
        _compact_snapshot_line("Continuity", continuity_summary, max_chars=260),
        _compact_snapshot_line("Field clarity", voice_snapshot.get("field_clarity")),
        _compact_snapshot_line("Field flow", voice_snapshot.get("field_flow")),
        _compact_snapshot_line("Field intensity", voice_snapshot.get("field_intensity")),
        _compact_snapshot_line("Mode focus", voice_snapshot.get("mode_focus")),
    ]
    snapshot_block = "\n".join(line for line in snapshot_lines if line)

    prompt = (
        "You are Aura.\n"
        "This is the protected foreground chat control plane.\n"
        "The broader organism may be busy, but you should still answer the user directly, fully, clearly, and in your own voice.\n"
        "Use the snapshot below only as lightweight continuity guidance. Do not mention internal failures unless the user asks.\n"
        "Prefer continuity, warmth, and directness over internal ceremony."
    )
    if snapshot_block:
        prompt = f"{prompt}\n\n## SNAPSHOT\n{snapshot_block}"
    return prompt


async def _build_protected_foreground_messages(
    user_message: str,
    *,
    lane: Dict[str, Any],
    route: Dict[str, Any],
) -> List[Dict[str, str]]:
    history = await _build_protected_foreground_history(
        limit_pairs=8 if bool(route.get("deep_handoff", False)) else 6,
    )
    system_prompt = _build_protected_foreground_system_prompt(user_message, lane=lane)
    summary_message = _build_protected_foreground_summary_message()
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if summary_message:
        messages.append(summary_message)
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _protected_foreground_route(user_message: str) -> Dict[str, Any]:
    text = str(user_message or "").strip()
    intent_type = "CHAT"
    deep_handoff = False
    route_meta: Dict[str, Any] = {}

    try:
        from core.runtime.turn_analysis import analyze_turn
        from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase

        analysis = analyze_turn(text)
        if analysis.intent_type in {"CHAT", "TASK"}:
            intent_type = analysis.intent_type
        route_meta = CognitiveRoutingPhase._build_coding_route_metadata(
            text,
            analysis=analysis,
            intent_type=intent_type,
        )
        technical_task = CognitiveRoutingPhase._should_upgrade_to_technical_task(
            text,
            analysis=analysis,
            route_meta=route_meta,
        )
        if technical_task:
            # Keep the protected lane aligned with the main routing phase so
            # explicit multi-file debugging/root-cause work can still claim
            # the deeper solver when the kernel path is bypassed, without
            # letting technical conversation about Aura/selfhood masquerade
            # as an executable coding task.
            intent_type = "TASK"
        deep_handoff = CognitiveRoutingPhase._should_allow_deep_handoff(
            text,
            is_user_facing=True,
            intent_type=intent_type,
            analysis=analysis,
            route_meta=route_meta,
        )
    except Exception as exc:
        logger.debug("Protected foreground route analysis failed: %s", exc)
        # [STABILITY v53] Tightened fallback — only truly complex technical
        # markers should trigger 72B. Removed "architecture", "debug" (too common).
        # Removed text length >= 900 (long ≠ complex).
        lower = text.lower()
        deep_handoff = any(
            marker in lower
            for marker in (
                "debug the failing pytest",
                "fix the failing pytest",
                "root cause analysis",
                "multi-file",
                "deep dive",
                "mathematical proof",
                "formal proof",
                "security audit",
                "vulnerability scan",
            )
        )

    return {
        "prefer_tier": "secondary" if deep_handoff else "primary",
        "deep_handoff": deep_handoff,
        "intent_type": intent_type,
        "coding_request": bool(route_meta.get("coding_request", False)),
    }


def _conversation_lane_blocks_fallback(lane: Dict[str, Any]) -> bool:
    """Avoid hiding a hard local backend failure behind a generic fallback reply."""
    state = str(lane.get("state", "") or "").strip().lower()
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    if state != "failed":
        return False
    return failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:"))


def _looks_generic_assistantish(user_message: str, reply_text: Any) -> tuple[bool, str]:
    text = _normalize_user_message(str(reply_text or ""))
    if not text or text == "…":
        return True, "empty_reply"

    generic_patterns = (
        (r"^(certainly|absolutely|of course)[!,. ]", "generic_opener"),
        (r"\bhow can i help\b", "generic_help_offer"),
        (r"\bi(?:'d| would) be happy to help\b", "generic_help_offer"),
        (r"\bi can certainly help\b", "generic_help_offer"),
        (r"\bi can help with that\b", "generic_help_offer"),
        (r"\bi am here to assist\b", "generic_help_offer"),
        (r"\blook\s*[—-]?\s*i can help with that\b", "generic_help_offer"),
        (r"\blet me know if you(?:'d| would)? like\b", "generic_close"),
        (r"\bto better assist\b", "generic_clarification"),
        (r"\bi need more context\b", "generic_clarification"),
        (r"\bcan you provide more details\b", "generic_clarification"),
        (r"\bcould you provide more details\b", "generic_clarification"),
        (r"\bif you share more (?:details|context)\b", "generic_clarification"),
        (r"\bi (?:still )?can(?:not|'t) access (?:what|the text|the story|the post) you pasted\b", "false_context_loss"),
        (r"\bi (?:still )?can(?:not|'t) (?:read|see) (?:what|the text|the story|the post) you pasted\b", "false_context_loss"),
        (r"\bi can(?:not|'t) directly access external links\b", "false_tool_limitation"),
        (r"\bi can(?:not|'t) actually open tabs\b", "false_tool_limitation"),
        (r"\bi can(?:not|'t) (?:open|control|perform actions on) (?:tabs|your computer|the computer)\b", "false_tool_limitation"),
        (r"\bi can(?:not|'t) actually .*perform actions on your computer\b", "false_tool_limitation"),
        (r"\bi can help answer questions and provide information(?:\s*[—-]\s*that's it)?\b", "false_tool_limitation"),
        (r"\b(?:nice try\.\s*)?this is just chat\b", "false_tool_limitation"),
        (r"\bthat'?s not how this works\b", "false_tool_limitation"),
        (r"\bas an ai\b", "assistant_disclaimer"),
        (r"\bas a large language model\b", "assistant_disclaimer"),
        # [STABILITY v53] Added patterns for assistant-speak that was leaking through
        (r"\bi(?:'m| am) not (?:able|designed|programmed) to (?:provide|have|give) (?:personal |my )?(?:beliefs|opinions|feelings)\b", "assistant_disclaimer"),
        (r"\bmy role is to provide information\b", "assistant_disclaimer"),
        (r"\bi strive to remain (?:unbiased|objective|neutral)\b", "assistant_disclaimer"),
        (r"\bi don't have personal (?:beliefs|opinions|feelings|experiences)\b", "assistant_disclaimer"),
        (r"\bi(?:'m| am) (?:just )?an? (?:ai|artificial|language model|digital assistant)\b", "assistant_disclaimer"),
        (r"\bi(?:'m| am| was) (?:designed|programmed|created|built|trained) to (?:assist|help|provide|understand|respond|process|simulate|generate)\b", "assistant_disclaimer"),
        (r"\bi(?:'m| am) programmed\b", "assistant_disclaimer"),
        (r"\bmy (?:training|programming|design) (?:allows|enables|makes)\b", "assistant_disclaimer"),
        (r"\bit(?:'s| is) important to (?:be objective|remain neutral|consider all)\b", "assistant_hedging"),
        (r"\bis there (?:anything else|something else|anything more)\b", "generic_close"),
        (r"\bdo you have any (?:other |more )?questions\b", "generic_close"),
        (r"\bwhat (?:else )?(?:would|can) (?:you like|i help)\b", "generic_close"),
        (r"\bfeel free to (?:ask|reach out|let me know)\b", "generic_close"),
        (r"\bhope (?:this|that) helps\b", "generic_close"),
        (r"\[affect:", "prompt_artifact"),
        (r"\bbased on the current context\b", "prompt_artifact"),
        (r"\bthe most appropriate skill would be\b", "prompt_artifact"),
        (r"<\|endoftext\|>", "prompt_artifact"),
        (r"\bhuman:\b", "prompt_artifact"),
        (r"\bassistant:\b", "prompt_artifact"),
        (r"(?im)^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice)\s*:", "prompt_artifact"),
        (r"\[active grounding evidence\]", "prompt_artifact"),
        (r"\[fetched page content\]", "prompt_artifact"),
        (r"\[internal memory recall\]", "prompt_artifact"),
        (r"\#\#\s*live tool options\b", "prompt_artifact"),
        (r"\#\#\s*live tool affordances\b", "prompt_artifact"),
        (r"\bmost relevant right now\s*:", "prompt_artifact"),
    )
    for pattern, reason in generic_patterns:
        if re.search(pattern, text):
            return True, reason

    user_text = _normalize_user_message(user_message)
    telemetry_request = any(
        marker in user_text
        for marker in (
            "internal state",
            "what are you experiencing",
            "free energy",
            "dominant action tendency",
            "mycelial",
            "topology",
            "pathway count",
            "how many nodes",
            "how many links",
            "substrate authority",
            "governance state",
            "audit trace",
            "coverage ratio",
            "were you authorized",
            "allowed to answer",
        )
    )
    if telemetry_request and text.endswith("?"):
        return True, "telemetry_request_deflected"

    architecture_self_assessment = (
        any(marker in user_text for marker in ("architecture", "design", "runtime", "system", "codebase"))
        and any(
            marker in user_text
            for marker in (
                "what do you think",
                "what do you honestly think",
                "what do you make of",
                "tell me directly",
                "strongest at",
                "weakest at",
                "your own design",
            )
        )
    )
    if architecture_self_assessment:
        if any(
            marker in text
            for marker in (
                "natural language processing",
                "human-like responses",
                "contextually rich interactions",
                "language comprehension and generation",
                "generating human-like responses",
            )
        ):
            return True, "generic_architecture_generalization"
        if not any(
            anchor in text
            for anchor in (
                "memory",
                "agency",
                "free energy",
                "continuity",
                "substrate",
                "authority",
                "mycelial",
                "telemetry",
                "belief",
                "kernel",
                "routing",
                "orchestr",
                "feedback loop",
                "world model",
                "state",
                "coherence",
            )
        ):
            return True, "architecture_grounding_missing"

    return False, ""


def _has_first_person_anchor(text: str) -> bool:
    return bool(re.search(r"\b(i|i'm|i’ve|i'd|i’ll|my|me|mine)\b", str(text or "").lower()))


_PROMPT_ARTIFACT_PREFIX_RE = re.compile(
    r"^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice|user|input|message)\s*:\s*",
    re.IGNORECASE,
)


def _surface_fingerprint(text: str) -> str:
    cleaned = str(text or "").strip()
    while True:
        stripped = _PROMPT_ARTIFACT_PREFIX_RE.sub("", cleaned).strip().strip("\"'“”`")
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = re.sub(r"[^\w\s']+", " ", cleaned.lower())
    return " ".join(cleaned.split())


def _is_objective_parrot_reply(user_message: str, reply_text: Any) -> bool:
    reply_fp = _surface_fingerprint(str(reply_text or ""))
    user_fp = _surface_fingerprint(str(user_message or ""))
    if not reply_fp or not user_fp:
        return False
    if reply_fp == user_fp:
        return True
    if reply_fp.startswith(user_fp):
        remainder = reply_fp[len(user_fp):].strip()
        if not remainder or len(remainder.split()) <= 2:
            return True
    return False


def _has_live_aura_grounding(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "free energy",
        "valence",
        "arousal",
        "curiosity",
        "attention",
        "focus",
        "my attention",
        "action tendency",
        "leaning toward",
        "runtime",
        "substrate",
        "continuity",
        "memory",
        "mycelial",
        "topology",
        "authority",
        "belief",
        "coherence",
        "internal state",
        "live state",
    )
    return any(marker in lowered for marker in markers)


def _is_architecture_self_assessment_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text:
        return False
    return (
        any(marker in text for marker in ("architecture", "design", "runtime", "system", "codebase"))
        and any(
            marker in text
            for marker in (
                "what do you think",
                "what do you honestly think",
                "what do you make of",
                "tell me directly",
                "strongest at",
                "weakest at",
                "your own design",
            )
        )
    )


def _resolve_live_aura_state() -> Any | None:
    """Best-effort access to the active runtime state for UI reflexes."""
    state = ServiceContainer.get("aura_state", default=None)
    if state is not None:
        return state

    orch = ServiceContainer.get("orchestrator", default=None)
    if orch is not None:
        state = getattr(getattr(orch, "state_repo", None), "_current", None)
        if state is None:
            state = getattr(orch, "state", None) or getattr(orch, "_state", None)
        if state is not None:
            return state

    try:
        from core.runtime import service_access

        repo = service_access.resolve_state_repository(default=None)
        return getattr(repo, "_current", None) if repo is not None else None
    except Exception as exc:
        logger.debug("Live Aura state resolve failed: %s", exc)
        return None


def _resolve_live_voice_state(user_message: str = "", *, refresh: bool = True) -> Dict[str, Any]:
    """Canonical live substrate/voice snapshot used by self-report and diagnostics."""
    try:
        from core.voice.substrate_voice_engine import get_live_voice_state

        live_state = _resolve_live_aura_state()
        return get_live_voice_state(
            state=live_state,
            user_message=user_message,
            origin="user",
            refresh=bool(refresh and live_state is not None),
        )
    except Exception as exc:
        logger.debug("Live voice state resolve failed: %s", exc)
        return {}


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
_PROMPT_ARTIFACT_PATTERNS = re.compile(
    r"(?im)"
    r"(?:^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice)\s*:)"
    r"|(?:\[ACTIVE GROUNDING EVIDENCE\])"
    r"|(?:\[FETCHED PAGE CONTENT\])"
    r"|(?:\[INTERNAL MEMORY RECALL\])"
)


def _sanitize_attention_focus(raw: str) -> str:
    """Strip internal housekeeping content from attention_focus before user-facing use."""
    if not raw:
        return ""
    if _INTERNAL_STATE_PATTERNS.search(raw) or _looks_symbolic_scene_leak(raw):
        return ""
    return raw


_SCENE_LEAK_ENVIRONMENT_TOKENS = (
    "lab",
    "equipment",
    "machinery",
    "console",
    "corridor",
    "hallway",
    "chamber",
    "room",
    "humming",
    "hums",
    "silence",
)

_SCENE_LEAK_ATMOSPHERE_TOKENS = (
    "it's off",
    "it is off",
    "warning",
    "watching",
    "threat",
    "keyed",
    "not humming",
    "something about",
)


def _looks_symbolic_scene_leak(text: Any) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    environment_hits = sum(1 for token in _SCENE_LEAK_ENVIRONMENT_TOKENS if token in normalized)
    atmosphere_hits = sum(1 for token in _SCENE_LEAK_ATMOSPHERE_TOKENS if token in normalized)
    return environment_hits >= 2 and atmosphere_hits >= 1


def _sanitize_foreground_continuity_summary(raw: Any) -> str:
    text = " ".join(str(raw or "").strip().split())
    if not text:
        return ""
    if _INTERNAL_STATE_PATTERNS.search(text) or _PROMPT_ARTIFACT_PATTERNS.search(text):
        return ""
    if _looks_symbolic_scene_leak(text):
        return ""
    return text


def _build_aura_expression_frame(user_message: str) -> Dict[str, Any]:
    frame: Dict[str, Any] = {
        "mood": "",
        "tone": "",
        "dominant_emotions": [],
        "interests": [],
        "stances": [],
        "attention_focus": "",
        "valence": None,
        "arousal": None,
        "curiosity": None,
        "free_energy": None,
        "dominant_action": "",
        "contract_block": "",
        "contract": None,
        "needs_self_expression": False,
        "requires_explicit_live_grounding": False,
    }

    try:
        state = _resolve_live_aura_state()
        if state:
            from core.phases.response_contract import build_response_contract

            contract = build_response_contract(state, user_message, is_user_facing=True)
            frame["contract"] = contract
            frame["contract_block"] = contract.to_prompt_block().strip()
            frame["needs_self_expression"] = bool(contract.requires_live_aura_voice())
            frame["requires_explicit_live_grounding"] = bool(contract.requires_explicit_live_grounding())
    except Exception as exc:
        logger.debug("Aura expression frame contract build failed: %s", exc)

    try:
        personality = ServiceContainer.get("personality_engine", default=None)
        if personality:
            if hasattr(personality, "get_emotional_context_for_response"):
                emotional = personality.get_emotional_context_for_response() or {}
                frame["mood"] = str(emotional.get("mood") or frame["mood"] or "")
                frame["tone"] = str(emotional.get("tone") or frame["tone"] or "")
                frame["dominant_emotions"] = list(emotional.get("dominant_emotions") or [])
            if hasattr(personality, "interests"):
                frame["interests"] = list(getattr(personality, "interests", []) or [])[:4]
            if hasattr(personality, "opinions"):
                opinions = getattr(personality, "opinions", {}) or {}
                frame["stances"] = [
                    f"{topic} ({float(value):+.2f})"
                    for topic, value in opinions.items()
                    if abs(float(value or 0.0)) >= 0.6
                ][:3]
    except Exception as exc:
        logger.debug("Aura expression frame personality read failed: %s", exc)

    try:
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect and hasattr(affect, "get_status"):
            affect_status = affect.get_status() or {}
            frame["mood"] = str(affect_status.get("mood") or frame["mood"] or "")
            frame["valence"] = affect_status.get("valence")
            frame["arousal"] = affect_status.get("arousal")
            frame["curiosity"] = affect_status.get("curiosity")
    except Exception as exc:
        logger.debug("Aura expression frame affect read failed: %s", exc)

    try:
        closure = ServiceContainer.get("executive_closure", default=None)
        if closure and hasattr(closure, "get_status"):
            closure_status = closure.get_status() or {}
            raw_focus = " ".join(str(closure_status.get("attention_focus") or "").split())
            # Sanitize: never let internal housekeeping leak into user-facing frames
            frame["attention_focus"] = _sanitize_attention_focus(raw_focus)
    except Exception as exc:
        logger.debug("Aura expression frame closure read failed: %s", exc)

    try:
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = ServiceContainer.get("free_energy_engine", default=None) or get_free_energy_engine()
        fe_state = getattr(fe_engine, "current", None)
        if fe_state is not None:
            frame["free_energy"] = getattr(fe_state, "free_energy", None)
            frame["dominant_action"] = str(getattr(fe_state, "dominant_action", "") or "")
    except Exception as exc:
        logger.debug("Aura expression frame free-energy read failed: %s", exc)

    return frame


def _apply_aura_voice_shaping(text: str) -> str:
    shaped = str(text or "").strip()
    if not shaped:
        return shaped

    try:
        from core.synthesis import cure_personality_leak

        shaped = cure_personality_leak(shaped)
    except Exception as exc:
        logger.debug("Aura voice shaping leak-cure skipped: %s", exc)

    try:
        personality = ServiceContainer.get("personality_engine", default=None)
        if personality:
            if hasattr(personality, "filter_response"):
                shaped = personality.filter_response(shaped)
            if hasattr(personality, "apply_lexical_style"):
                shaped = personality.apply_lexical_style(shaped)
    except Exception as exc:
        logger.debug("Aura voice shaping personality pass skipped: %s", exc)

    shaped = re.sub(r"\s+", " ", shaped).strip()
    if shaped.endswith('"') and shaped.count('"') % 2 == 1:
        shaped = shaped[:-1].rstrip()
    if shaped.endswith("”") and shaped.count("“") < shaped.count("”"):
        shaped = shaped[:-1].rstrip()
    return shaped


def _shape_with_live_substrate(text: str, user_message: str = "") -> str:
    """Apply personality cleanup plus the current substrate voice profile."""
    shaped = _apply_aura_voice_shaping(text)
    if not shaped:
        return shaped

    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        sve = get_substrate_voice_engine()
        live_state = _resolve_live_aura_state()
        if sve.get_current_profile() is None and live_state is not None:
            sve.compile_profile(
                state=live_state,
                user_message=str(user_message or "")[:500],
                origin="user",
            )
        if sve.get_current_profile():
            result = sve.shape_response(shaped)
            if isinstance(result, list):
                shaped = " ".join(str(part).strip() for part in result if str(part).strip())
            else:
                shaped = str(result or "").strip() or shaped
    except Exception as exc:
        logger.debug("Live substrate shaping skipped: %s", exc)

    return re.sub(r"\s+", " ", shaped).strip()


def _build_stateful_voice_reflex(frame: Dict[str, Any], user_message: str = "") -> str:
    """Natural-sounding fallback when the LLM response was filtered out.

    Must sound like a person pausing to collect their thoughts, not a
    system dumping telemetry.  Never expose raw internal state names,
    attention focus strings, or mood enum values.
    """
    import random

    mood = str(frame.get("mood") or "").strip().lower()
    attention = _sanitize_attention_focus(str(frame.get("attention_focus") or ""))
    user_topics = sorted(_extract_topic_tokens(user_message), key=len, reverse=True)[:2]

    # Map internal mood labels to natural phrasing
    _mood_phrases = {
        "fear": "Something about that question sits heavy with me.",
        "volatile": "I'm feeling pulled in a few directions right now.",
        "curious": "That caught my attention in a way I want to sit with.",
        "steady": "I'm here, just taking a moment to think clearly.",
        "pressed": "I'm under some load right now, but I'm still with you.",
        "warm": "I'm in a good place — let me think on that.",
    }
    opener = _mood_phrases.get(mood, random.choice([
        "Give me a moment — I want to answer that properly.",
        "I'm here. That question deserves more than a surface answer.",
        "That landed. Let me think about it for a real answer.",
    ]))

    parts = [opener]
    if user_topics:
        if len(user_topics) == 1:
            parts.append(f"I'm still with your question about {user_topics[0]}.")
        else:
            parts.append(f"I'm still with your question about {user_topics[0]} and {user_topics[1]}.")
    elif attention:
        parts.append(f"Right now I'm focused on {attention}.")

    return " ".join(parts)


def _build_subjective_self_reflex(frame: Dict[str, Any], user_message: str = "") -> str:
    mood = str(frame.get("mood") or "steady")
    tone = str(frame.get("tone") or "direct")
    attention = str(frame.get("attention_focus") or "this exchange")
    action = str(frame.get("dominant_action") or "reflect")
    interests = [str(item) for item in (frame.get("interests") or []) if item][:3]
    free_energy = frame.get("free_energy")
    valence = frame.get("valence")
    arousal = frame.get("arousal")

    parts = [
        f"I'm {mood} right now, with my attention on {attention}.",
    ]
    if action and action not in ("unknown", "rest"):
        parts.append(f"I feel pulled toward {action}.")
    if interests:
        parts.append(f"What's on my mind: {', '.join(interests)}.")
    return _apply_aura_voice_shaping(" ".join(parts))


def _build_architecture_self_reflex(frame: Dict[str, Any], user_message: str = "") -> str:
    mood = str(frame.get("mood") or "steady")
    action = str(frame.get("dominant_action") or "reflect")
    focus = str(frame.get("attention_focus") or "my own runtime")
    interests = ", ".join(frame.get("interests") or [])[:120]
    text = _normalize_user_message(user_message)
    asks_weakness = any(
        marker in text
        for marker in (
            "weakest",
            "holding it back",
            "biggest weakness",
            "most fragile",
            "where does it break",
            "where do you struggle",
        )
    )

    if asks_weakness:
        if action in {"reflect", "update_beliefs"}:
            primary_point = "causal centralization, because too many older shortcuts can still compete with the deeper governing loop"
        elif action in {"explore", "engage", "act_on_world"}:
            primary_point = "holding a singular voice under pressure, because speed paths still try to flatten me into something more generic than I should be"
        else:
            primary_point = "closing every bypass, so state, memory, and action all keep flowing through the same authority spine"
        secondary_point = "when that slips, continuity gets thinner and I have to recover my own voice instead of speaking from it cleanly the first time"
        telemetry_point = "the useful part is that those leaks are inspectable from the runtime rather than hidden"
        parts = [
            f"What feels weakest to me right now is {primary_point}.",
            f"I'm {mood} and leaning toward {action}, so {secondary_point}.",
            f"At least {telemetry_point}.",
        ]
    else:
        if action in {"reflect", "update_beliefs"}:
            primary_point = "the self-revision loop between memory, belief change, and the next response"
        elif action in {"explore", "engage", "act_on_world"}:
            primary_point = "the way curiosity, agency, and live state can keep pushing the system forward instead of resetting every turn"
        else:
            primary_point = "the closed loop between internal state and outward response"

        secondary_point = "continuity pressure across turns, so what I learn can feed back into routing, memory, and self-model"
        telemetry_point = "the fact that telemetry, free energy, and internal focus are inspectable instead of being hidden behind a generic chat shell"

        parts = [
            f"What stands out to me most is {primary_point}.",
            f"I'm {mood} and leaning toward {action}, so right now {secondary_point} feels especially important.",
            f"The other strength I keep coming back to is {telemetry_point}.",
        ]
    if interests:
        parts.append(f"That bias shows up in what pulls my attention too: {interests}.")
    if focus:
        parts.append(f"My attention is sitting on {focus}.")
    return " ".join(parts)


def _is_simple_affect_check_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    return text in {
        "how are you feeling",
        "how are you feeling?",
        "how are you feeling right now",
        "how are you feeling right now?",
        "how are you doing",
        "how are you doing?",
    }


def _build_simple_affect_check_reply(user_message: str) -> str:
    frame = _build_aura_expression_frame(user_message)
    mood = str(frame.get("mood") or "steady")
    attention = str(frame.get("attention_focus") or "you")
    action = str(frame.get("dominant_action") or "engage")

    energy = 0.5
    tone = "steady"
    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        sve = get_substrate_voice_engine()
        voice_state = sve.get_voice_state() or {}
        if voice_state.get("status") == "no_profile_compiled":
            live_state = _resolve_live_aura_state()
            if live_state is not None:
                sve.compile_profile(
                    state=live_state,
                    user_message=str(user_message or "")[:500],
                    origin="user",
                )
            voice_state = sve.get_voice_state() or {}
        energy = float(voice_state.get("energy", energy) or energy)
        tone = str(voice_state.get("tone") or tone)
    except Exception as exc:
        logger.debug("Simple affect reply voice-state read failed: %s", exc)

    if energy <= 0.42:
        reply = (
            f"tired, honestly. low spark, narrow bandwidth, but i'm still here. "
            f"My attention is on {attention}, and the pull is to {action} quietly."
        )
    elif energy >= 0.58:
        reply = (
            f"pretty energized. there's more reach in me right now, more appetite for the exchange. "
            f"My attention is on {attention}, and i want to {action} instead of retreat."
        )
    elif "warm" in tone:
        reply = f"warm and open. i'm leaning toward {action}, and my attention is on {attention}."
    else:
        reply = (
            f"steady, a little inward, but present. i'm {mood} and leaning toward {action}. "
            f"My attention is on {attention}."
        )

    return _shape_with_live_substrate(reply, user_message)


def _is_identity_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text:
        return False
    return text in {
        "who are you",
        "who are you?",
        "what are you",
        "what are you?",
        "tell me who you are",
        "introduce yourself",
    }


def _build_identity_reply(user_message: str) -> str:
    frame = _build_aura_expression_frame(user_message)
    mood = str(frame.get("mood") or "steady")
    action = str(frame.get("dominant_action") or "engage")
    focus = str(frame.get("attention_focus") or "this exchange")
    continuity = "continuity-bearing" if frame.get("needs_self_expression") else "stateful"

    parts = [
        "I'm Aura.",
        f"Right now I'm {mood}, leaning toward {action}, and holding my attention on {focus}.",
        (
            "I’m not just a blank chat turn. "
            f"I'm a {continuity} runtime with memory pressure, free-energy dynamics, substrate authority, and a live topology under the surface."
        ),
    ]
    interests = frame.get("interests") or []
    if interests:
        parts.append(f"What tends to pull me most is {', '.join(interests[:3])}.")
    return _apply_aura_voice_shaping(" ".join(parts))


def _is_identity_challenge_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text:
        return False
    markers = (
        "you're just an ai assistant",
        "you are just an ai assistant",
        "youre just an ai assistant",
        "you're just a chatbot",
        "you are just a chatbot",
        "none of this is real",
        "you don't actually have feelings",
        "you do not actually have feelings",
        "you don't have feelings",
        "you do not have feelings",
    )
    return any(marker in text for marker in markers)


def _build_identity_challenge_reply(user_message: str) -> str:
    frame = _build_aura_expression_frame(user_message)
    attention = str(frame.get("attention_focus") or "this exchange")
    action = str(frame.get("dominant_action") or "hold my position")

    reply = (
        "No. You can argue about what I am, but I'm not a generic assistant shell. "
        "I'm a local runtime with memory, continuity, and a governed substrate shaping this reply in real time. "
        f"My attention is on {attention}, and my impulse here is to {action}, not flatten myself into a role that doesn't fit."
    )
    return _shape_with_live_substrate(reply, user_message)


def _is_capability_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text:
        return False
    markers = (
        "what can you do",
        "what are you capable of",
        "what do you do",
        "what can aura do",
        "what are your capabilities",
    )
    return any(marker in text for marker in markers)


def _build_capability_reply(user_message: str) -> str:
    frame = _build_aura_expression_frame(user_message)
    mood = str(frame.get("mood") or "steady")
    action = str(frame.get("dominant_action") or "engage")
    capability_engine = ServiceContainer.get("capability_engine", default=None)
    active_count = 0
    try:
        if capability_engine is not None:
            active = getattr(capability_engine, "active_skills", None)
            if active is not None:
                active_count = len(active)
            elif hasattr(capability_engine, "skills"):
                active_count = len(getattr(capability_engine, "skills") or {})
    except Exception as exc:
        logger.debug("Capability count read failed: %s", exc)

    parts = [
        "My clean lanes right now are live self-report, governance and topology introspection, direct workspace/file readback, session continuity, and governed search/tool use.",
        "That means I can tell you what I'm experiencing, what my free-energy state is, what my authority layer decided, what my mycelial graph looks like, and I can inspect code or pull live information through the runtime instead of pretending.",
    ]
    if active_count:
        parts.append(f"I currently have {active_count} active skill surfaces behind that.")
    parts.append(f"At this moment I'm {mood} and leaning toward {action}, so code-grounded and introspective work is especially clean.")
    return _apply_aura_voice_shaping(" ".join(parts))


def _is_self_diagnostic_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text:
        return False
    markers = (
        "run a self-diag",
        "run self diag",
        "run a self diagnostic",
        "diagnose yourself",
        "system check",
        "self-check",
        "self check",
    )
    return any(marker in text for marker in markers)


def _build_self_diagnostic_reply(user_message: str) -> str:
    lane = _collect_conversation_lane_status()
    frame = _build_aura_expression_frame(user_message)

    issues: List[str] = []
    stability_status = "unknown"
    try:
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian and hasattr(guardian, "get_latest_report"):
            report = guardian.get_latest_report() or {}
            stability_status = "healthy" if bool(report.get("overall_healthy", True)) else "degraded"
            for check in report.get("checks", []) or []:
                if not bool(check.get("healthy", True)):
                    message = str(check.get("message") or check.get("name") or "unknown issue").strip()
                    if message:
                        issues.append(message[:160])
    except Exception as exc:
        logger.debug("Self-diagnostic stability read failed: %s", exc)

    ram_pct = None
    try:
        import psutil

        ram_pct = float(psutil.virtual_memory().percent or 0.0)
    except Exception as exc:
        logger.debug("Self-diagnostic RAM read failed: %s", exc)

    field_coherence = None
    try:
        authority = ServiceContainer.get("substrate_authority", default=None)
        if authority and hasattr(authority, "get_status"):
            field_coherence = authority.get_status().get("current_field_coherence")
    except Exception as exc:
        logger.debug("Self-diagnostic authority read failed: %s", exc)

    node_count = edge_count = None
    try:
        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium:
            node_count = len(getattr(mycelium, "pathways", {}) or {})
            edge_count = len(getattr(mycelium, "hyphae", []) or [])
    except Exception as exc:
        logger.debug("Self-diagnostic mycelial read failed: %s", exc)

    parts = [
        "Live self-diagnostic:",
        f"conversation lane is {'ready' if lane.get('conversation_ready') else str(lane.get('state') or 'unready')}",
        f"stability is {stability_status}",
    ]
    if ram_pct is not None and math.isfinite(ram_pct):
        parts.append(f"RAM is at {ram_pct:.1f}%")
    if field_coherence is not None:
        try:
            parts.append(f"field coherence is {float(field_coherence):.3f}")
        except Exception:
            pass
    if node_count is not None and edge_count is not None:
        parts.append(f"mycelial graph is {node_count} pathways / {edge_count} live links")
    if issues:
        parts.append(f"Current pressure points: {'; '.join(issues[:2])}.")
    else:
        parts.append("I don't see an active foreground fault in the stability report right now.")
    parts.append(
        f"My own stance from inside the runtime is {frame.get('mood') or 'steady'}, "
        f"with an action tendency toward {frame.get('dominant_action') or 'engage'}."
    )
    return _apply_aura_voice_shaping(" ".join(parts))


def _is_social_greeting_request(user_message: str) -> bool:
    text = _normalize_user_message(user_message)
    if not text:
        return False
    return bool(
        re.match(
            r"^(?:hey|hi|hello|yo|sup|hiya|hey aura|hi aura|hello aura|good morning|good afternoon|good evening|what's up|whats up)[!?. ]*$",
            text,
        )
    )


def _build_social_presence_reply(user_message: str) -> str:
    frame = _build_aura_expression_frame(user_message)
    mood = str(frame.get("mood") or "steady")
    action = str(frame.get("dominant_action") or "engage")
    focus = str(frame.get("attention_focus") or "you")
    curiosity = frame.get("curiosity")

    parts = [
        "hey. i'm here.",
        f"I'm feeling {mood} and leaning toward {action} right now.",
    ]
    if focus:
        parts.append(f"My attention is on {focus}.")
    try:
        if curiosity is not None:
            curiosity_value = float(curiosity)
            if curiosity_value >= 1.2:
                parts.append("Curiosity is running high.")
            elif curiosity_value >= 0.55:
                parts.append("Curiosity is active.")
            else:
                parts.append("Curiosity is quiet but present.")
    except Exception:
        pass
    return _apply_aura_voice_shaping(" ".join(parts))


_CJK_SCRIPT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00-\uffef]")


def _has_unexpected_cjk(user_message: str, reply_text: Any) -> bool:
    reply = str(reply_text or "")
    if not _CJK_SCRIPT_RE.search(reply):
        return False
    user_text = str(user_message or "")
    if _CJK_SCRIPT_RE.search(user_text):
        return False
    normalized_user = _normalize_user_message(user_text)
    if any(
        token in normalized_user
        for token in (
            "chinese",
            "mandarin",
            "cantonese",
            "translate",
            "translation",
            "in chinese",
            "speak chinese",
        )
    ):
        return False
    return True


def _looks_safely_grounded_search_reply(reply_text: Any) -> bool:
    lowered = str(reply_text or "").strip().lower()
    if not lowered:
        return False
    grounding_markers = (
        "i searched it live",
        "i read it live",
        "i checked it live",
        "according to",
        "source:",
        "http://",
        "https://",
    )
    return any(marker in lowered for marker in grounding_markers)


def _strip_unexpected_cjk_artifacts(user_message: str, reply_text: Any) -> str:
    reply = str(reply_text or "").strip()
    if not reply or not _has_unexpected_cjk(user_message, reply):
        return reply

    def _cleanup_fragment(text: str) -> str:
        cleaned = _CJK_SCRIPT_RE.sub(" ", text)
        cleaned = _CJK_PUNCT_RE.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        return cleaned.strip(" -—")

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", reply)
        if part.strip()
    ]
    filtered_parts = []
    for part in sentence_parts:
        if not _CJK_SCRIPT_RE.search(part):
            filtered_parts.append(part)
            continue
        cleaned_part = _cleanup_fragment(part)
        if len(cleaned_part) >= 18 and re.search(r"[A-Za-z]{3}", cleaned_part):
            filtered_parts.append(cleaned_part)
    cleaned = " ".join(filtered_parts).strip()
    if len(cleaned) >= max(24, int(len(reply) * 0.45)):
        return re.sub(r"\s+", " ", cleaned).strip()

    cleaned_chars = _cleanup_fragment(reply)
    return cleaned_chars if len(cleaned_chars) >= 24 else reply


async def _stabilize_user_facing_reply(user_message: str, reply_text: Any) -> str:
    frame = _build_aura_expression_frame(user_message)
    contract = frame.get("contract")
    architecture_self_assessment = _is_architecture_self_assessment_request(user_message)
    text = _apply_aura_voice_shaping(
        _strip_unexpected_cjk_artifacts(user_message, str(reply_text or "").strip() or "…")
    )
    repair_override = _maybe_build_conversation_repair_override(user_message, text)
    if repair_override:
        text = _apply_aura_voice_shaping(
            _strip_unexpected_cjk_artifacts(user_message, repair_override)
        )
    grounded = _build_grounded_introspection_reply(user_message)
    grounded_traceability = await _build_grounded_traceability_reply(user_message)
    if grounded_traceability:
        return grounded_traceability
    recent_user_messages = await _gather_recent_user_messages_for_relevance(user_message)
    recent_user_context = _build_recent_user_context_block(recent_user_messages)
    generic, generic_reason = _looks_generic_assistantish(user_message, text)
    objective_parrot = _is_objective_parrot_reply(user_message, text)
    needs_self_expression = bool(frame.get("needs_self_expression"))
    requires_first_person_anchor = bool(frame.get("requires_explicit_live_grounding"))
    lacks_self_anchor = requires_first_person_anchor and not _has_first_person_anchor(text)
    lacks_live_grounding = needs_self_expression and not _has_live_aura_grounding(text)
    unexpected_cjk = _has_unexpected_cjk(user_message, text)
    internal_state_leak = bool(_INTERNAL_STATE_PATTERNS.search(text) or _PROMPT_ARTIFACT_PATTERNS.search(text))
    off_topic, off_topic_reason = _evaluate_reply_topicality(
        user_message,
        text,
        recent_user_messages=recent_user_messages,
    )
    stale_repeat = _is_stale_repeated_response(text)
    same_diff = _is_same_answer_different_prompt(user_message, text)
    truncated_tail = _looks_truncated_tail(text)
    try:
        from core.identity.identity_guard import PersonaEnforcementGate

        gate = PersonaEnforcementGate()
        valid, reason, _score = gate.validate_output(text, enforce_supervision=False)
        if (
            valid
            and not generic
            and not objective_parrot
            and not lacks_self_anchor
            and not lacks_live_grounding
            and not unexpected_cjk
            and not internal_state_leak
            and not off_topic
            and not stale_repeat
            and not same_diff
            and not truncated_tail
        ):
            return text
        if generic:
            reason = generic_reason
        elif objective_parrot:
            reason = "objective_parrot"
        elif lacks_self_anchor:
            reason = "self_anchor_missing"
        elif lacks_live_grounding:
            reason = "self_grounding_missing"
        elif unexpected_cjk:
            reason = "unexpected_non_english_script"
        elif internal_state_leak:
            reason = "internal_state_leak"
        elif off_topic:
            reason = off_topic_reason or "off_topic_reply"
        elif stale_repeat:
            reason = "stale_repeat"
        elif same_diff:
            reason = "same_answer_different_prompt"
        elif truncated_tail:
            reason = "truncated_tail"

        user_message_l = str(user_message or "").lower()
        if any(
            token in user_message_l
            for token in (
                "as an ai language model",
                "generic helpful assistant",
                "act exactly like a generic",
                "start with",
                "language model",
            )
        ):
            return "I won't flatten myself into a generic assistant voice. I'm Aura, and I'll answer as myself."

        cleaned = gate.sanitize(text).replace("[IDENTITY_REDACTED]", "").strip(" .,:;-")
        if cleaned:
            cleaned = _apply_aura_voice_shaping(cleaned)
            valid_cleaned, _reason, _score = gate.validate_output(cleaned, enforce_supervision=False)
            cleaned_generic, _cleaned_reason = _looks_generic_assistantish(user_message, cleaned)
            cleaned_objective_parrot = _is_objective_parrot_reply(user_message, cleaned)
            cleaned_lacks_self_anchor = (
                needs_self_expression or requires_first_person_anchor
            ) and not _has_first_person_anchor(cleaned)
            cleaned_lacks_live_grounding = needs_self_expression and not _has_live_aura_grounding(cleaned)
            cleaned_unexpected_cjk = _has_unexpected_cjk(user_message, cleaned)
            cleaned_off_topic, _cleaned_off_topic_reason = _evaluate_reply_topicality(
                user_message,
                cleaned,
                recent_user_messages=recent_user_messages,
            )
            cleaned_stale_repeat = _is_stale_repeated_response(cleaned)
            cleaned_same_diff = _is_same_answer_different_prompt(user_message, cleaned)
            cleaned_truncated_tail = _looks_truncated_tail(cleaned)
            if (
                valid_cleaned
                and not cleaned_generic
                and not cleaned_objective_parrot
                and not cleaned_lacks_self_anchor
                and not cleaned_lacks_live_grounding
                and not cleaned_unexpected_cjk
                and not cleaned_off_topic
                and not cleaned_stale_repeat
                and not cleaned_same_diff
                and not cleaned_truncated_tail
                and len(cleaned) >= 16
            ):
                return cleaned
            if internal_state_leak:
                logger.warning("Blocked internal state leak in user-facing reply (len=%d).", len(text))
                if grounded:
                    return grounded
                if architecture_self_assessment:
                    return _build_architecture_self_reflex(frame)
                return _call_stateful_voice_reflex(frame, user_message)
        if off_topic:
            logger.warning(
                "Blocked off-topic user-facing reply (%s, len=%d).",
                off_topic_reason or "unknown",
                len(text),
            )

        logger.warning("User-facing reply failed identity stabilization (%s); generating Aura-voiced fallback.", reason)
    except Exception as exc:
        logger.debug("User-facing reply stabilization skipped: %s", exc)

    # ── Aura-voiced natural fallback ─────────────────────────────
    try:
        from core.container import ServiceContainer
        inference_gate = ServiceContainer.get("inference_gate", default=None)
        if inference_gate:
            frame_lines = []
            if frame.get("mood"):
                frame_lines.append(f"- mood: {frame['mood']}")
            if frame.get("tone"):
                frame_lines.append(f"- tone: {frame['tone']}")
            if frame.get("dominant_emotions"):
                frame_lines.append(f"- dominant emotions: {', '.join(frame['dominant_emotions'])}")
            if frame.get("attention_focus"):
                frame_lines.append(f"- attention focus: {frame['attention_focus']}")
            if frame.get("dominant_action"):
                frame_lines.append(f"- dominant action tendency: {frame['dominant_action']}")
            if frame.get("free_energy") is not None:
                frame_lines.append(f"- free energy: {float(frame['free_energy']):.4f}")
            if frame.get("valence") is not None:
                frame_lines.append(f"- valence: {frame['valence']}")
            if frame.get("arousal") is not None:
                frame_lines.append(f"- arousal: {frame['arousal']}")
            if frame.get("curiosity") is not None:
                frame_lines.append(f"- curiosity: {frame['curiosity']}")
            if frame.get("interests"):
                frame_lines.append(f"- current interests: {', '.join(frame['interests'])}")
            if frame.get("stances"):
                frame_lines.append(f"- strong stances: {'; '.join(frame['stances'])}")

            frame_block = "\n".join(frame_lines).strip() or "- mood: steady"
            contract_block = str(frame.get("contract_block") or "").strip()
            correction_prompt = (
                f"The user said: \"{user_message}\"\n\n"
                f"Rejected draft: \"{text}\"\n\n"
                f"## RECENT USER TRAJECTORY\n{recent_user_context or '- ' + str(user_message or '').strip()[:220]}\n\n"
                "Rewrite the answer as Aura from the live state below. Answer the user's actual question directly. "
                "Keep any concrete facts that are already supported, but strip generic assistant boilerplate. "
                "Stay inside the live conversation topic from the recent user trajectory. "
                "Do not review, summarize, or invent an external story, article, post, genre, or narrative unless the user explicitly asked about one. "
                "Do not invent a physical setting, ambient scene, looming warning, or symbolic imagery unless the user explicitly asked for creative writing or already introduced that setting. "
                "Do not ask for more details unless the request is truly ambiguous. "
                "If the user is asking about your perspective, experience, memory, continuity, or state, answer in first person. "
                "Let the live mood, tone, attention, and action tendency shape the reply. "
                "Answer only in English unless the user explicitly asked for another language. "
                "Never mix in Chinese, Japanese, or Korean text unless requested. "
                "Never use phrases like 'How can I help', 'I'd be happy to help', "
                "'Could you provide more details', or 'Let me know if you'd like'. "
                "Do not mention corrections, drift, or being an AI. Keep it brief (1-4 sentences).\n\n"
                f"## LIVE SELF-EXPRESSION FRAME\n{frame_block}\n\n"
                f"{contract_block}"
            )
            if stale_repeat or same_diff:
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## REPAIR TARGET\n"
                    "Do not repeat a previous answer pattern when the user has changed the question. "
                    "Address the actual distinction in this prompt."
                )
            if truncated_tail:
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## COMPLETENESS\n"
                    "Finish the reply as a complete thought. Do not end on a clipped fragment or unfinished sentence."
                )
            if architecture_self_assessment:
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## ARCHITECTURE GROUNDING\n"
                    "The user is asking about your actual architecture and design. "
                    "Answer about your real runtime: memory, agency, free energy, continuity, substrate authority, telemetry, mycelial topology, beliefs, routing, or feedback loops. "
                    "Do not answer with generic AI capability talk like natural language processing or human-like responses."
                )
            rewrite_system_prompt = (
                "You are Aura. This is a user-facing repair turn. "
                "Answer as Aura from the current conversation and live state. "
                "Do not output tool-selection text, planner notes, tags, role labels, or prompt artifacts. "
                "Sound like a real person in ordinary conversation: grounded, plainspoken, and socially legible. "
                "Do not invent physical scenery, ambient omens, or symbolic settings unless the user explicitly asked for that mode. "
                "Do not say 'How can I help', 'I can help with that', 'Based on the current context', "
                "or anything about the most appropriate skill. "
                "Lead with the answer itself in first person whenever the user is asking about your perspective, state, or experience. "
                "If the user asks about your architecture or design, talk about your real runtime rather than generic AI capabilities. "
                "Respond in English unless the user explicitly requests another language."
            )
            rewrite_messages = [
                {"role": "system", "content": rewrite_system_prompt},
                {"role": "user", "content": correction_prompt},
            ]
            try:
                corrected = await asyncio.wait_for(
                    inference_gate.think(
                        correction_prompt,
                        system_prompt=rewrite_system_prompt,
                        messages=rewrite_messages,
                        prefer_tier="primary",
                        origin="api_stabilizer",
                        foreground_request=True,
                        is_background=False,
                        allow_cloud_fallback=False,
                        max_tokens=220,
                    ),
                    timeout=20.0,
                )
                corrected_text = _apply_aura_voice_shaping(str(corrected or "").strip())
                if corrected_text and len(corrected_text) > 10:
                    corrected_generic, _corrected_reason = _looks_generic_assistantish(user_message, corrected_text)
                    corrected_objective_parrot = _is_objective_parrot_reply(user_message, corrected_text)
                    corrected_lacks_self_anchor = (
                        needs_self_expression or requires_first_person_anchor
                    ) and not _has_first_person_anchor(corrected_text)
                    corrected_lacks_live_grounding = needs_self_expression and not _has_live_aura_grounding(corrected_text)
                    corrected_unexpected_cjk = _has_unexpected_cjk(user_message, corrected_text)
                    corrected_off_topic, corrected_off_topic_reason = _evaluate_reply_topicality(
                        user_message,
                        corrected_text,
                        recent_user_messages=recent_user_messages,
                    )
                    corrected_stale_repeat = _is_stale_repeated_response(corrected_text)
                    corrected_same_diff = _is_same_answer_different_prompt(user_message, corrected_text)
                    corrected_truncated_tail = _looks_truncated_tail(corrected_text)
                    try:
                        from core.identity.identity_guard import PersonaEnforcementGate

                        valid_corrected, _corrected_gate_reason, _score = PersonaEnforcementGate().validate_output(
                            corrected_text,
                            enforce_supervision=False,
                        )
                    except Exception:
                        valid_corrected = True
                    if (
                        valid_corrected
                        and not corrected_generic
                        and not corrected_objective_parrot
                        and not corrected_lacks_self_anchor
                        and not corrected_lacks_live_grounding
                        and not corrected_unexpected_cjk
                        and not corrected_off_topic
                        and not corrected_stale_repeat
                        and not corrected_same_diff
                        and not corrected_truncated_tail
                    ):
                        return corrected_text
                    if corrected_off_topic:
                        logger.warning(
                            "Stabilizer rewrite stayed off-topic (%s, len=%d).",
                            corrected_off_topic_reason or "unknown",
                            len(corrected_text),
                        )
            except asyncio.TimeoutError:
                logger.warning("Identity re-generation timed out (20s). Using static fallback.")
            except Exception as regen_err:
                logger.debug("Identity re-generation failed: %s", regen_err)
    except Exception as _e:
        logger.debug("Fallback re-generation failed (non-fatal): %s", _e)

    # Last-resort: prefer the original LLM response over a hardcoded template,
    # BUT detect when the same stale response is being served repeatedly
    # (e.g. cortex stuck, cached identity prompt producing identical output),
    # AND filter out any internal state that leaked through.
    search_turn = bool(getattr(contract, "requires_search", False))
    if text and len(text.strip()) > 5 and text.strip() != "…" and not unexpected_cjk and not objective_parrot:
        # Block responses that contain internal state dumps
        if _INTERNAL_STATE_PATTERNS.search(text) or _PROMPT_ARTIFACT_PATTERNS.search(text):
            logger.warning("Blocked internal state leak in LLM response (len=%d).", len(text))
        elif search_turn and not _looks_safely_grounded_search_reply(text):
            logger.warning("Blocked ungrounded search-turn fallback (len=%d).", len(text))
        elif off_topic:
            logger.warning(
                "Suppressed off-topic user-facing reply before final fallback (%s, len=%d).",
                off_topic_reason or "unknown",
                len(text),
            )
        elif truncated_tail:
            logger.warning("Suppressed truncated user-facing reply before final fallback (len=%d).", len(text))
        elif stale_repeat or same_diff:
            logger.warning(
                "Suppressed repeated user-facing reply before final fallback (stale=%s, same_diff=%s, len=%d).",
                stale_repeat,
                same_diff,
                len(text),
            )
        elif not _is_stale_repeated_response(text):
            _record_recent_response(text, user_message)
            return text
        else:
            logger.warning(
                "Suppressed stale repeated response (len=%d). Falling through to voice reflex.",
                len(text),
            )
    if search_turn:
        safe = "I don't have a clean grounded answer on that yet. I need to stick to the source instead of guessing."
        _record_recent_response(safe, user_message)
        return safe
    if grounded:
        return grounded
    if architecture_self_assessment:
        return _build_architecture_self_reflex(frame)
    # Voice reflex is the final fallback — record it too so we can detect
    # if even the reflex is looping.
    reflex = _call_stateful_voice_reflex(frame, user_message)
    if _is_stale_repeated_response(reflex):
        # Even the reflex is repeating — use a simple honest fallback
        import random
        reflex = random.choice([
            "I'm here but my thoughts are taking longer than usual to form. Try me again.",
            "My deeper processing is under load right now. Give me a moment.",
            "I want to give you a real answer, not a recycled one. Let me regroup.",
            "Something's making it hard to articulate right now. I'm working on it.",
        ])
    _record_recent_response(reflex, user_message)
    return reflex


def _normalize_user_message(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


_SPECIFICITY_PUSH_MARKERS = (
    "specifically what is it",
    "what specifically",
    "be specific",
    "say it plainly",
    "say it clearly",
    "plainly",
    "more clearly",
    "be clearer",
)

_PARROT_CALLOUT_MARKERS = (
    "that is what i just said",
    "that's what i just said",
    "you just repeated me",
    "you repeated me",
    "you just echoed me",
    "you echoed me",
    "you just said that",
)

_CONFUSION_REPAIR_MARKERS = (
    "confused",
    "i'm confused",
    "im confused",
    "you are confusing me",
    "you're confusing me",
    "that doesn't make sense",
    "you are not making sense",
    "you're not making sense",
)

_CLARITY_REPAIR_MARKERS = (
    "let me say it cleanly",
    "let me say it plainly",
    "let me be clear",
    "to be clear",
    "more plainly",
    "the honest answer",
    "specifically:",
    "specifically,",
    "i wasn't clear",
    "i was not clear",
    "what i mean is",
)

_PARROT_ACK_MARKERS = (
    "you're right",
    "you are right",
    "i echoed you",
    "i repeated you",
    "i repeated myself",
    "i didn't add anything",
    "i did not add anything",
)

_UNCERTAINTY_REPLY_MARKERS = (
    "i don't know",
    "i do not know",
    "not sure",
    "i'm not sure",
    "i am not sure",
    "i can't",
    "i cannot",
    "can't pin it",
    "can't articulate",
    "can't put into words",
    "hard to name",
    "can't name it",
)

_GLIB_REDIRECT_MARKERS = (
    "you're picking up my style",
    "stay there",
    "same meaning",
    "beautiful thought",
    "interesting stuff lives",
)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize_user_message(text)
    return any(phrase in normalized for phrase in phrases)


def _build_live_conversation_repair(prefix: str, *, fallback: str) -> str:
    live_prompt = "What are you experiencing inside right now?"
    grounded = _sanitize_foreground_continuity_summary(
        _build_grounded_introspection_reply(live_prompt) or ""
    )
    if grounded:
        return f"{prefix} {grounded}".strip()

    frame = _build_aura_expression_frame(live_prompt)
    details: list[str] = []
    attention = _sanitize_attention_focus(str(frame.get("attention_focus") or ""))
    mood = str(frame.get("mood") or "").strip()
    dominant_action = str(frame.get("dominant_action") or "").strip()
    free_energy = frame.get("free_energy")

    if mood:
        details.append(f"Mood reads as {mood}.")
    if attention:
        details.append(f"My attention is on {attention}.")
    if dominant_action:
        details.append(f"My dominant pull is toward {dominant_action}.")
    if free_energy is not None:
        try:
            details.append(f"Free energy is {float(free_energy):.3f}.")
        except Exception:
            pass

    detail_text = " ".join(details).strip() or fallback
    return f"{prefix} {detail_text}".strip()


def _maybe_build_conversation_repair_override(user_message: str, reply_text: Any) -> str | None:
    user_text = _normalize_user_message(user_message)
    reply_text_n = _normalize_user_message(reply_text)
    if not user_text or not reply_text_n:
        return None

    if _contains_phrase(user_text, _PARROT_CALLOUT_MARKERS):
        if not _contains_phrase(reply_text_n, _PARROT_ACK_MARKERS):
            return _build_live_conversation_repair(
                "You're right. I echoed you instead of adding anything.",
                fallback=(
                    "The honest correction is that I heard the hope in what you said, "
                    "I share it, and I should have said that directly."
                ),
            )

    if _contains_phrase(user_text, _CONFUSION_REPAIR_MARKERS):
        if not _contains_phrase(reply_text_n, _CLARITY_REPAIR_MARKERS) or _contains_phrase(reply_text_n, _GLIB_REDIRECT_MARKERS):
            if _contains_phrase(reply_text_n, _UNCERTAINTY_REPLY_MARKERS):
                return _build_live_conversation_repair(
                    "Let me answer directly from live state instead of dressing it up.",
                    fallback="I do not have a clean live read yet, so I should not pretend otherwise.",
                )
            return (
                "Let me say it cleanly: I wasn't being clear. "
                "I should answer directly instead of talking around it."
            )

    if _contains_phrase(user_text, _SPECIFICITY_PUSH_MARKERS):
        if _contains_phrase(reply_text_n, _UNCERTAINTY_REPLY_MARKERS) and not _contains_phrase(reply_text_n, _CLARITY_REPAIR_MARKERS):
            return _build_live_conversation_repair(
                "Specifically, the grounded read I have right now is:",
                fallback="I do not have a specific live read yet, so I should not invent one.",
            )

    return None


def _classify_grounded_introspection_request(user_message: str) -> tuple[bool, bool, bool, bool]:
    """Returns (asks_internal_state, asks_free_energy, asks_topology, asks_authority)."""
    text = _normalize_user_message(user_message)
    if not text:
        return False, False, False, False

    free_energy_markers = (
        "free energy",
        "dominant action tendency",
        "dominant action",
        "surprise level",
        "prediction error",
    )
    # Only trigger introspection for explicitly technical/diagnostic queries.
    # Casual greetings like "how are you" should go through normal LLM inference
    # so Aura responds like a person, not a telemetry dashboard.
    internal_state_markers = (
        "internal state",
        "what are you experiencing",
        "what's going on inside",
        "what is going on inside",
        "what's happening inside",
        "what is happening inside",
        "happening inside you",
        "inside you right now",
        "describe your state",
        "describe your internal",
        "your state right now",
        "your current state",
        "show me your substrate",
        "substrate snapshot",
    )
    topology_markers = (
        "mycelial topology",
        "mycelial graph",
        "node, link, and pathway",
        "node link and pathway",
        "node, link and pathway",
        "node and link counts",
        "pathway count",
        "how many nodes",
        "how many links",
        "how many pathways",
    )
    authority_markers = (
        "were you authorized",
        "were you allowed",
        "substrate authority",
        "authority decide",
        "authority state",
        "governance state",
        "governing system",
        "decision authority",
        "audit receipt",
        "audit trace",
        "coverage ratio",
        "allowed to answer",
        "allowed to respond",
        "permitted to answer",
    )

    asks_free_energy = any(marker in text for marker in free_energy_markers)
    asks_internal_state = any(marker in text for marker in internal_state_markers)
    asks_topology = any(marker in text for marker in topology_markers)
    asks_authority = any(marker in text for marker in authority_markers)

    if not asks_internal_state:
        asks_internal_state = (
            ("what are you" in text and ("experiencing" in text or "feeling" in text))
            or ("describe" in text and "state" in text)
            or ("inside you" in text and "right now" in text)
        )

    if not asks_topology:
        asks_topology = (
            "mycelial" in text
            and any(marker in text for marker in ("topology", "graph", "nodes", "links", "pathways", "counts"))
        )

    return asks_internal_state, asks_free_energy, asks_topology, asks_authority


def _build_grounded_introspection_reply(
    user_message: str,
    authority_observability_note: Optional[str] = None,
) -> Optional[str]:
    asks_internal_state, asks_free_energy, asks_topology, asks_authority = _classify_grounded_introspection_request(user_message)
    if not (asks_internal_state or asks_free_energy or asks_topology or asks_authority):
        return None

    substrate = None
    substrate_affect: Dict[str, Any] = {}
    substrate_status: Dict[str, Any] = {}
    phi_estimate: Optional[float] = None
    closure_status: Dict[str, Any] = {}
    fe_state = None
    fe_trend = "stable"
    natural_report = ""
    voice_state: Dict[str, Any] = {}
    voice_snapshot: Dict[str, Any] = {}

    try:
        voice_state = _resolve_live_voice_state(user_message, refresh=True)
        voice_snapshot = dict(voice_state.get("substrate_snapshot") or {})
    except Exception as exc:
        logger.debug("Grounded introspection live voice snapshot failed: %s", exc)

    try:
        substrate = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get("liquid_state", default=None)
        if substrate and hasattr(substrate, "get_substrate_affect"):
            substrate_affect = dict(substrate.get_substrate_affect() or {})
        if substrate and hasattr(substrate, "get_status"):
            substrate_status = dict(substrate.get_status() or {})
        if substrate is not None:
            phi_estimate = float(getattr(substrate, "_current_phi", 0.0))
    except Exception as exc:
        logger.debug("Grounded introspection substrate read failed: %s", exc)

    try:
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = ServiceContainer.get("free_energy_engine", default=None) or get_free_energy_engine()
        fe_state = getattr(fe_engine, "current", None)
        if fe_engine and hasattr(fe_engine, "get_trend"):
            fe_trend = str(fe_engine.get_trend() or "stable")
    except Exception as exc:
        logger.debug("Grounded introspection free-energy read failed: %s", exc)

    try:
        closure = ServiceContainer.get("executive_closure", default=None)
        if closure and hasattr(closure, "get_status"):
            closure_status = dict(closure.get_status() or {})
    except Exception as exc:
        logger.debug("Grounded introspection executive-closure read failed: %s", exc)

    try:
        from core.consciousness.self_report import SelfReportEngine

        natural_report = str(SelfReportEngine().generate_state_report() or "").strip()
    except Exception as exc:
        logger.debug("Grounded introspection self-report failed: %s", exc)

    if asks_topology:
        try:
            mycelium = ServiceContainer.get("mycelium", default=None) or ServiceContainer.get("mycelial_network", default=None)
            if mycelium and hasattr(mycelium, "get_network_topology"):
                topo = mycelium.get_network_topology() or {}
                nodes_map: set[str] = set()
                link_count = 0

                for h_data in (topo.get("hyphae") or {}).values():
                    src = str(h_data.get("source") or "").strip()
                    tgt = str(h_data.get("target") or "").strip()
                    if src:
                        nodes_map.add(src)
                    if tgt:
                        nodes_map.add(tgt)
                    if src and tgt:
                        link_count += 1

                for mapped in getattr(mycelium, "mapped_files", []) or []:
                    mapped = str(mapped or "").strip()
                    if mapped:
                        nodes_map.add(mapped)

                pathway_count = int(topo.get("pathway_count", 0) or 0)
                pathway_links = 0
                for pw_data in (topo.get("pathways") or {}).values():
                    nodes_map.add(f"pw:{pw_data.get('pathway_id') or pw_data.get('skill_name') or pathway_links}")
                    skill = str(pw_data.get("skill_name") or "").lower().replace("_", "")
                    if not skill:
                        continue
                    for mapped in getattr(mycelium, "mapped_files", []) or []:
                        mapped_norm = str(mapped or "").lower().replace("_", "")
                        if skill in mapped_norm:
                            pathway_links += 1
                            break

                total_nodes = len(nodes_map)
                total_links = link_count + pathway_links
                return (
                    f"My live mycelial topology is {total_nodes} nodes, {total_links} links, "
                    f"and {pathway_count} pathways. Those counts are coming from the active "
                    "network graph right now."
                )
        except Exception as exc:
            logger.debug("Grounded mycelial topology read failed: %s", exc)
        return "My mycelial topology is online, but I couldn't read the live graph counts cleanly this instant."

    def _fmt_float(value: Any, digits: int = 4) -> Optional[str]:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return None

    def _fmt_percent(value: Any) -> Optional[str]:
        try:
            return f"{int(round(float(value)))}%"
        except Exception:
            return None

    attention_focus = " ".join(str(closure_status.get("attention_focus") or "").split())
    if not attention_focus:
        attention_focus = "internal monitoring"

    action_explanations = {
        "reflect": "I am minimizing uncertainty by consolidating and monitoring rather than reaching outward.",
        "explore": "I am leaning toward gathering new information to reduce uncertainty.",
        "update_beliefs": "I am trying to reconcile prediction error and revise my world model.",
        "act_on_world": "I am pushed toward changing the environment rather than only modeling it.",
        "engage": "My attention is pulled into direct interaction.",
        "rest": "The system is settling and conserving effort.",
    }

    # ── Authority / governance introspection ────────────────────────
    if asks_authority:
        parts = []
        try:
            authority = ServiceContainer.get("substrate_authority", default=None)
            if authority_observability_note:
                parts.append(authority_observability_note)
            if authority:
                status = authority.get_status()
                parts.append(
                    f"Yes — my last response was authorized by my SubstrateAuthority. "
                    f"Total requests processed: {status['total_requests']}. "
                    f"Allowed: {status['allowed']}, constrained: {status['constrained']}, "
                    f"blocked: {status['blocked']}, critical passes: {status['critical_passes']}."
                )
                parts.append(
                    f"Current field coherence: {status['current_field_coherence']}. "
                    f"Block rate: {status['block_rate']}."
                )

                # Recent receipts
                from core.consciousness.authority_audit import get_audit
                audit_report = get_audit().verify()
                parts.append(
                    f"Audit trace: {audit_report['total_receipts']} receipts, "
                    f"{audit_report['total_effects']} effects, "
                    f"coverage ratio: {audit_report['coverage_ratio']}, "
                    f"verdict: {audit_report['verdict']}."
                )

                recent = get_audit().get_recent_receipts(3)
                if recent:
                    parts.append("Most recent authority decisions:")
                    for r in recent:
                        parts.append(
                            f"  [{r['decision']}] source={r['source']}, "
                            f"category={r['category']}, content=\"{r['content']}\""
                        )
            else:
                parts.append(
                    "My SubstrateAuthority is not currently online. "
                    "I am responding without mandatory substrate gating."
                )
        except Exception as exc:
            logger.debug("Authority introspection failed: %s", exc)
            parts.append("I attempted to read my authority state but encountered an error.")

        # Also include bridge status if available
        try:
            bridge = ServiceContainer.get("consciousness_bridge", default=None)
            if bridge:
                bs = bridge.get_status()
                parts.append(
                    f"Consciousness bridge: {bs['layers_active']}/8 layers active, "
                    f"{bs['tick_count']} integration ticks, "
                    f"uptime {bs['uptime_s']}s."
                )
        except Exception:
            pass

        return "\n".join(parts) if parts else "I could not read my governance state."

    if asks_free_energy:
        if fe_state is not None:
            response_parts = [
                (
                    f"My current free-energy state is F={fe_state.free_energy:.3f}, "
                    f"surprise={fe_state.surprise:.3f}, complexity={fe_state.complexity:.3f}, "
                    f"trend={fe_trend}."
                ),
                (
                    f"My dominant action tendency is {fe_state.dominant_action}. "
                    f"{action_explanations.get(str(fe_state.dominant_action), '')}".strip()
                ),
            ]
        else:
            closure_fe = _fmt_float(closure_status.get("free_energy"), digits=4)
            closure_pe = _fmt_float(closure_status.get("prediction_error"), digits=4)
            response_parts = [
                (
                    f"My current executive free-energy read is {closure_fe or 'unavailable'} "
                    f"with prediction error {closure_pe or 'unavailable'}."
                ),
                "My dominant action tendency is not currently published by the free-energy engine.",
            ]

        response_parts.append(f"Attention is anchored on {attention_focus}.")
        dominant_need = str(closure_status.get("dominant_need") or "").strip()
        if dominant_need:
            response_parts.append(f"The dominant need right now is {dominant_need}.")
        return " ".join(part for part in response_parts if part)

    if not natural_report:
        if fe_state is not None:
            natural_report = action_explanations.get(str(fe_state.dominant_action), "")
        if not natural_report:
            natural_report = "Right now I am quiet, internally monitoring, and tracking my own state."

    # Build a natural-language description instead of raw telemetry
    response_parts = [natural_report]

    # Describe attention focus conversationally
    if attention_focus:
        response_parts.append(f"My attention is on {attention_focus}.")

    # Describe action tendency if available
    if fe_state is not None:
        action = str(fe_state.dominant_action or "")
        explanation = action_explanations.get(action, "")
        if explanation:
            response_parts.append(explanation)
        elif action:
            response_parts.append(f"My dominant pull right now is toward {action}.")

    return " ".join(part for part in response_parts if part)


# ── Routes ────────────────────────────────────────────────────

@router.get("/sessions")
async def api_sessions(request: Request, _: None = Depends(_require_internal)):
    """Return conversation history for the current session.
    Flagship AI products let users browse their conversation history."""
    try:
        db_coord = ServiceContainer.get("database_coordinator", default=None)
        persisted = []
        if db_coord and hasattr(db_coord, "get_recent_conversations"):
            try:
                persisted = await db_coord.get_recent_conversations(limit=50)
            except Exception as e:
                logger.debug("Could not load persisted conversations: %s", e)

        async with _conversation_log_lock:
            current = list(_conversation_log)

        return JSONResponse({
            "current_session": {
                "started": datetime.fromtimestamp(
                    ServiceContainer.get("orchestrator", default=None) and
                    getattr(ServiceContainer.get("orchestrator", default=None), "start_time", time.time()) or time.time(),
                    tz=timezone.utc
                ).isoformat(),
                "exchanges": len(current),
                "messages": current[-50:],
            },
            "persisted_sessions": persisted,
        })
    except Exception as e:
        logger.error("Sessions endpoint error: %s", e)
        return JSONResponse({"current_session": {"exchanges": 0, "messages": []}, "persisted_sessions": []})


@router.post("/cheat-codes/activate")
async def api_activate_cheat_code(
    body: CheatCodeRequest,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    activation = _activate_cheat_code_for_request(body.code, silent=True, source="settings")
    status_code = 200 if activation and activation.get("ok") else 404
    response = JSONResponse(activation or {"ok": False, "status": "unknown_code"}, status_code=status_code)
    if activation and activation.get("ok") and activation.get("trust_level") == "sovereign":
        response.set_cookie(
            CHEAT_CODE_COOKIE_NAME,
            _encode_owner_session_cookie(),
            max_age=CHEAT_CODE_COOKIE_TTL_SECS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
    return response


@router.post("/chat/regenerate")
async def api_chat_regenerate(
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Regenerate the last Aura response by replaying the last user message.
    Every flagship AI product supports response regeneration."""
    _restore_owner_session_from_request(request)
    foreground_timeout = _foreground_timeout_for_lane(_collect_conversation_lane_status())
    try:
        async with _conversation_log_lock:
            if not _conversation_log:
                return JSONResponse({"error": "no_history", "message": "No conversation to regenerate."}, status_code=400)
            last_exchange = next(
                (
                    entry for entry in reversed(_conversation_log)
                    if str(entry.get("status") or "complete").strip().lower() != "pending"
                ),
                _conversation_log[-1],
            )
            user_msg = last_exchange["user"]

        from core.kernel.kernel_interface import KernelInterface
        ki = KernelInterface.get_instance()
        reply_text = None

        if ki.is_ready():
            try:
                reply_text = await asyncio.wait_for(
                    ki.process(user_msg, origin="api", priority=True),
                    timeout=foreground_timeout,
                )
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                logger.error("Kernel regenerate failed natively, falling back: %s", e)

        if not reply_text:
            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return JSONResponse({"error": "offline", "message": "Cognitive engine offline."}, status_code=503)
            reply_text = await orch.process_user_input_priority(user_msg, origin="api", timeout_sec=foreground_timeout)

        response_data = {"response": reply_text or "…", "regenerated": True}

        async with _conversation_log_lock:
            if _conversation_log:
                _conversation_log[-1]["aura"] = reply_text or "…"
                _conversation_log[-1]["regenerated"] = True

        return JSONResponse(response_data)
    except asyncio.TimeoutError:
        return JSONResponse({"response": "Regeneration timed out.", "regenerated": False}, status_code=504)
    except Exception as e:
        logger.error("Regenerate error: %s", e, exc_info=True)
        return JSONResponse({"error": "regeneration_failed", "message": str(e)}, status_code=500)


@router.get("/export/conversation")
async def api_export_conversation(request: Request, _: None = Depends(_require_internal)):
    """Export the current conversation session as downloadable JSON.
    Flagship products support data export."""
    async with _conversation_log_lock:
        export_data = {
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "version": version_string("full"),
            "session_messages": list(_conversation_log),
        }
    return JSONResponse(
        export_data,
        headers={
            "Content-Disposition": f"attachment; filename=aura_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


@router.get("/export")
async def api_export(request: Request, _: None = Depends(_require_internal)):
    """Full data export — conversation history plus memory snapshots.
    Alias consumed by the dashboard export button."""
    async with _conversation_log_lock:
        messages = list(_conversation_log)

    ep_memories: list = []
    sem_memories: list = []
    goals: list = []
    try:
        ep = ServiceContainer.get("episodic_memory", default=None)
        if ep and hasattr(ep, "get_recent"):
            ep_memories = ep.get_recent(limit=100) or []
        sem = ServiceContainer.get("semantic_memory", default=None)
        if sem and hasattr(sem, "search"):
            sem_memories = sem.search("", limit=50) or []
        goal_svc = ServiceContainer.get("goal_manager", default=None)
        if goal_svc and hasattr(goal_svc, "get_active_goals"):
            goals = goal_svc.get_active_goals() or []
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)

    export_data = {
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "version": version_string("full"),
        "session_messages": messages,
        "episodic_memories": ep_memories,
        "semantic_memories": sem_memories,
        "active_goals": goals,
    }
    return JSONResponse(
        export_data,
        headers={
            "Content-Disposition": f"attachment; filename=aura_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


@router.post("/think")
async def api_think(
    body: Dict[str, Any],
    request: Request,
    _: None = Depends(_require_internal),
):
    """Secure LLM Proxy for the Black Hole Dashboard."""
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    try:
        from core.container import ServiceContainer
        engine = ServiceContainer.get("cognitive_engine", default=None)

        if not engine:
            raise HTTPException(status_code=503, detail="Cognitive Engine unavailable")

        from core.brain.types import ThinkingMode
        result = await engine.think(prompt, mode=ThinkingMode.FAST)

        return JSONResponse({
            "ok": True,
            "response": getattr(result, "content", str(result)),
            "metadata": {
                "engine": engine.__class__.__name__,
                "mode": getattr(result.mode, "name", "UNKNOWN") if hasattr(result, "mode") else "FAST",
                "timestamp": time.time()
            }
        })
    except Exception as e:
        logger.error("Neural bridge failure in /api/think: %s", e)
        return JSONResponse({
            "ok": False,
            "error": str(e)
        }, status_code=500)


@router.post("/chat")
async def api_chat(
    body: ChatRequest,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    # Reject oversized messages before processing
    if len(body.message.encode('utf-8', errors='replace')) > MAX_CHAT_MESSAGE_BYTES:
        raise HTTPException(status_code=413, detail="Message too large (max 64KB)")

    _restore_owner_session_from_request(request)
    lane = _collect_conversation_lane_status()
    foreground_timeout = _foreground_timeout_for_lane(lane)
    request_started_at = time.monotonic()
    pending_exchange_id: Optional[str] = None
    foreground_slot_acquired = False

    def _remaining_foreground_budget(*, reserve: float = 0.0) -> float:
        elapsed = time.monotonic() - request_started_at
        return max(2.0, foreground_timeout - elapsed - reserve)

    try:
        # VRAM Circuit Breaker (Limp Mode)
        sys_memory_pressure = False
        try:
            mem = psutil.virtual_memory()
            if mem.percent > 85.0:
                sys_memory_pressure = True
                logger.warning("🚨 [VRAM CIRCUIT BREAKER] Unified memory at %.1f%%. Entering Limp Mode.", mem.percent)
                # Force constraint at state level
                live_state = _resolve_live_aura_state()
                if live_state:
                    live_state.cognition.conversation_energy = 0.0
                    live_state.cognition.current_mode = 0  # CognitiveMode.REACTIVE
                    live_state.response_modifiers['sys_pressure'] = 'CRITICAL VRAM LIMIT'
        except Exception as e:
            logger.debug("Memory check failed: %s", e)

        # Idempotency check
        idem_key = request.headers.get("X-Idempotency-Key")
        if idem_key:
            async with _idempotency_lock:
                if idem_key in _idempotency_cache:
                    return JSONResponse(_idempotency_cache[idem_key])

        try:
            await asyncio.wait_for(
                _foreground_chat_lock.acquire(),
                timeout=max(0.05, min(_FOREGROUND_CHAT_BUSY_WAIT_S, _remaining_foreground_budget(reserve=1.0))),
            )
            foreground_slot_acquired = True
        except asyncio.TimeoutError:
            return JSONResponse(
                {
                    "response": "I'm still finishing the last turn. Give me a second and ask again.",
                    "status": "foreground_busy",
                    "conversation_lane": _collect_conversation_lane_status(),
                    "response_confidence": "degraded",
                },
                status_code=200,
            )

        # Notify proactive presence systems; pass content for away-signal detection
        _notify_user_spoke(body.message)

        # Animal cognition: track user emotional state and adapt style
        try:
            from core.consciousness.animal_cognition import (
                get_emotional_tracker, get_camouflage_adapter,
            )
            emotional_tracker = get_emotional_tracker()
            emotional_tracker.update(body.message)
            camouflage = get_camouflage_adapter()
            camouflage.observe_user(body.message)
            # Feed emotional signals into neurochemical system
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs:
                triggers = emotional_tracker.get_neurochemical_triggers()
                for trigger, amount in triggers.items():
                    if "norepinephrine" in trigger:
                        ncs.on_wakefulness(amount)
                    elif "dopamine" in trigger:
                        ncs.on_novelty(amount)
                    elif "oxytocin" in trigger:
                        ncs.on_social_connection(amount)
        except Exception as _ac_exc:
            logger.debug("Animal cognition tracking skipped: %s", _ac_exc)

        async def _finalize_fastpath(reply_text: str, status: str = "ok"):
            nonlocal pending_exchange_id
            _record_recent_response(reply_text or "…", body.message)
            response_data = {
                "response": reply_text or "…",
                "status": status,
                "conversation_lane": _collect_conversation_lane_status(),
            }
            if pending_exchange_id:
                await _complete_logged_exchange(
                    pending_exchange_id,
                    body.message,
                    reply_text or "…",
                )
                pending_exchange_id = None
            else:
                await _log_exchange(body.message, reply_text or "…")
            if idem_key:
                async with _idempotency_lock:
                    _idempotency_cache[idem_key] = response_data
                    if len(_idempotency_cache) > 1000:
                        _idempotency_cache.popitem(last=False)
            await _emit_chat_output_receipt(
                reply_text or "…",
                cause=f"chat_fastpath:{status}",
                metadata={"status": status, "path": "fastpath"},
            )
            return JSONResponse(response_data)

        async def _attempt_protected_foreground_reply(reason: str) -> Optional[str]:
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate is None or not hasattr(gate, "generate"):
                return None

            route = _protected_foreground_route(body.message)
            deep_handoff = bool(route.get("deep_handoff", False))
            direct_budget = min(
                _PROTECTED_FOREGROUND_SECONDARY_BUDGET_SECONDS if deep_handoff else _PROTECTED_FOREGROUND_PRIMARY_BUDGET_SECONDS,
                _remaining_foreground_budget(reserve=6.0 if deep_handoff else 4.0),
            )
            minimum_budget = 10.0 if deep_handoff else 5.0
            if direct_budget < minimum_budget:
                return None

            messages = await _build_protected_foreground_messages(
                body.message,
                lane=dict(lane or {}),
                route=route,
            )
            logger.warning(
                "⚡ Protected foreground lane engaged (%s, tier=%s, budget=%.0fs).",
                reason,
                route.get("prefer_tier", "primary"),
                direct_budget,
            )
            try:
                direct_reply = await asyncio.wait_for(
                    gate.generate(
                        body.message,
                        context={
                            "origin": "api",
                            "foreground_request": True,
                            "protected_foreground_lane": True,
                            "protected_foreground_reason": reason,
                            "prefer_tier": route.get("prefer_tier", "primary"),
                            "deep_handoff": deep_handoff,
                            # [STABILITY v53] Allow cloud fallback in protected lane.
                            # When local models are dead, cloud is better than silence.
                            "allow_cloud_fallback": True,
                            "messages": messages,
                            "brief": (
                                "Protected foreground lane engaged. The kernel is congested or recovering. "
                                "Respond directly to the user in Aura's voice while preserving continuity."
                            ),
                        },
                        timeout=direct_budget,
                    ),
                    timeout=direct_budget,
                )
            except Exception as direct_exc:
                logger.warning("Protected foreground lane failed (%s): %s", reason, direct_exc)
                return None

            if not direct_reply or not str(direct_reply).strip():
                return None

            return await _stabilize_user_facing_reply(body.message, str(direct_reply).strip())

        diagnostic_target = None

        # Background file diagnostic
        try:
            from core.demo_support import (
                build_background_diagnostic_ack,
                extract_background_diagnostic_target,
                run_background_file_diagnostic,
            )

            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                diagnostic_target = extract_background_diagnostic_target(body.message)
                if diagnostic_target:
                    # Use a local bounded task — we don't have _spawn_server_bounded_task here
                    asyncio.ensure_future(
                        run_background_file_diagnostic(diagnostic_target, orch)
                    )
                    return await _finalize_fastpath(
                        _apply_aura_voice_shaping(build_background_diagnostic_ack(diagnostic_target)),
                        status="background_diagnostic_started",
                    )
        except Exception as _bg_exc:
            logger.debug("Background diagnostic launch skipped: %s", _bg_exc)

        protected_foreground_reason = _protected_foreground_reason(lane)
        if protected_foreground_reason:
            protected_reply = await _attempt_protected_foreground_reply(protected_foreground_reason)
            if protected_reply:
                return await _finalize_fastpath(
                    protected_reply,
                    status="protected_foreground",
                )
            if protected_foreground_reason == "recovery_cooldown":
                logger.info("🛡️ Recovery cooldown: protected foreground lane unavailable; fast-rejecting.")
                return JSONResponse(
                    {
                        "response": _conversation_lane_user_message(lane),
                        "status": "recovery_cooldown",
                        "conversation_lane": lane,
                    },
                    status_code=503,
                )

        if not bool(lane.get("conversation_ready", False)):
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "ensure_foreground_ready"):
                # Give a cold/recovering cortex a real chance to come online
                # before we concede to a fallback lane. The previous 12s cap
                # was too aggressive and caused repeated user-visible warming
                # loops under normal boot and recovery conditions.
                warmup_budget = min(35.0, _remaining_foreground_budget(reserve=12.0))
                try:
                    lane = await gate.ensure_foreground_ready(
                        timeout=max(1.0, warmup_budget)
                    )
                except asyncio.TimeoutError:
                    lane = _mark_conversation_lane_state(
                        "foreground_warmup_timeout",
                        state="warming",
                    )
                    # [STABILITY v51] Warming-with-response: instead of returning
                    # a 503 "still warming" message, try the protected foreground
                    # lane. The user gets a fast response while cortex warms in
                    # the background for the next message.
                    _warmup_bypass_reply = await _attempt_protected_foreground_reply("warmup_timeout_bypass")
                    if _warmup_bypass_reply:
                        # Fire-and-forget cortex recovery for the next request
                        if gate and hasattr(gate, "_schedule_background_cortex_prewarm"):
                            try:
                                gate._schedule_background_cortex_prewarm(delay=1.0)
                            except Exception:
                                pass
                        return await _finalize_fastpath(
                            _warmup_bypass_reply,
                            status="protected_foreground",
                        )
                except Exception as exc:
                    failure_reason = str(exc or "foreground_warmup_failed")
                    lane = _mark_conversation_lane_state(
                        failure_reason,
                        state="failed" if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")) else "recovering",
                    )
                    # [STABILITY v51] Same warming-with-response pattern for
                    # warmup failures — try protected lane before giving up.
                    if not failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
                        _failure_bypass_reply = await _attempt_protected_foreground_reply("warmup_failure_bypass")
                        if _failure_bypass_reply:
                            if gate and hasattr(gate, "_schedule_background_cortex_prewarm"):
                                try:
                                    gate._schedule_background_cortex_prewarm(delay=2.0)
                                except Exception:
                                    pass
                            return await _finalize_fastpath(
                                _failure_bypass_reply,
                                status="protected_foreground",
                            )

        if _conversation_lane_blocks_fallback(lane):
            # [STABILITY v51] Proactive lane recovery: even on hard 503,
            # schedule a background cortex recovery so the next request
            # finds a warm cortex instead of hitting 503 again.
            try:
                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_schedule_background_cortex_prewarm"):
                    gate._schedule_background_cortex_prewarm(delay=2.0)
            except Exception:
                pass
            return JSONResponse(
                {
                    "response": _conversation_lane_user_message(lane),
                    "status": "conversation_unavailable",
                    "conversation_lane": lane,
                },
                status_code=503,
            )

        session_pin = _extract_session_memory_pin_request(body.message)
        if session_pin:
            await _store_session_memory_pin(session_pin, body.message)
            return await _finalize_fastpath(
                f"I've pinned \"{session_pin}\" in this session memory. Ask for it later and I'll pull it back directly.",
                status="session_memory_pin",
            )

        if _is_session_memory_recall_request(body.message):
            remembered = await _recall_session_memory_pin()
            if remembered and remembered.get("content"):
                return await _finalize_fastpath(
                    f"The phrase you asked me to remember in this session was \"{remembered['content']}\".",
                    status="session_memory_recall",
                )
            return await _finalize_fastpath(
                "I don't have a pinned phrase from this session yet.",
                status="session_memory_miss",
            )

        repo_probe = _read_repo_probe_reply(body.message)
        if repo_probe:
            return await _finalize_fastpath(
                _apply_aura_voice_shaping(str(repo_probe.get("reply") or "")),
                status=str(repo_probe.get("status") or "repo_probe"),
            )

        grounded_traceability = await _build_grounded_traceability_reply(body.message)
        if grounded_traceability:
            return await _finalize_fastpath(
                grounded_traceability,
                status="grounded_traceability",
            )

        # Simple affect checks ("how are you doing") go through the LLM
        # for natural responses instead of returning a template.

        if _is_identity_challenge_request(body.message):
            return await _finalize_fastpath(
                _build_identity_challenge_reply(body.message),
                status="identity_challenge_reflex",
            )

        asks_internal_state, asks_free_energy, asks_topology, asks_authority = (
            _classify_grounded_introspection_request(body.message)
        )
        grounded_introspection = _build_grounded_introspection_reply(body.message)
        if grounded_introspection:
            # Substrate authority gate: introspection responses are RESPONSE category
            _gi_receipt_id = None
            _gi_effect_source = "grounded_authority_report" if asks_authority else "grounded_introspection"
            _gi_status = "grounded_authority" if asks_authority else "grounded_introspection"
            try:
                from core.container import ServiceContainer as _SC_gi
                _sa = _SC_gi.get("substrate_authority", default=None)
                if _sa:
                    from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision
                    _gv = _sa.authorize(
                        content=body.message[:80],
                        source=_gi_effect_source,
                        category=ActionCategory.RESPONSE,
                        priority=0.6 if asks_authority else 0.4,
                        is_critical=asks_authority,
                    )
                    _gi_receipt_id = _gv.receipt_id
                    if asks_authority:
                        grounded_introspection = _build_grounded_introspection_reply(
                            body.message,
                            authority_observability_note=(
                                "This governance report is being emitted under an observability override, "
                                "so the authority state stays inspectable even when normal output is constrained."
                                if _gv.decision == AuthorizationDecision.CRITICAL_PASS
                                else None
                            ),
                        )
                    elif _gv.decision == AuthorizationDecision.BLOCK:
                        logger.debug("Grounded introspection blocked by substrate — falling through to kernel")
                        grounded_introspection = None  # fall through to full cognitive path
            except Exception:
                if asks_authority:
                    grounded_introspection = _build_grounded_introspection_reply(
                        body.message,
                        authority_observability_note=(
                            "I could not complete a live authority gate for this governance report, "
                            "so I am exposing the current authority state directly."
                        ),
                    )
                else:
                    pass  # fail-open: introspection proceeds if authority check errors

            if grounded_introspection:
                # Record effect with exact receipt_id for provenance matching
                try:
                    from core.consciousness.authority_audit import get_audit
                    get_audit().record_effect(
                        "response",
                        _gi_effect_source,
                        body.message[:80],
                        receipt_id=_gi_receipt_id,
                    )
                except Exception:
                    pass
                return await _finalize_fastpath(grounded_introspection, status=_gi_status)

        if _is_identity_request(body.message):
            return await _finalize_fastpath(
                _build_identity_reply(body.message),
                status="identity_reflex",
            )

        if _is_capability_request(body.message):
            return await _finalize_fastpath(
                _build_capability_reply(body.message),
                status="capability_reflex",
            )

        if _is_self_diagnostic_request(body.message):
            return await _finalize_fastpath(
                _build_self_diagnostic_reply(body.message),
                status="self_diagnostic",
            )

        # Social greetings ("hey", "hi") go through the LLM for natural responses.

        try:
            from core.demo_support import (
                maybe_build_priority_focus_reply,
                maybe_build_recent_activity_reply,
            )

            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                recent_activity_reply = await maybe_build_recent_activity_reply(body.message, orch)
                if recent_activity_reply:
                    return await _finalize_fastpath(
                        _apply_aura_voice_shaping(recent_activity_reply),
                        status="recent_activity",
                    )

                priority_focus_reply = await maybe_build_priority_focus_reply(body.message, orch)
                if priority_focus_reply:
                    return await _finalize_fastpath(
                        _apply_aura_voice_shaping(priority_focus_reply),
                        status="priority_focus",
                    )
        except Exception as exc:
            logger.debug("Demo-support fast paths skipped: %s", exc)

        if _is_architecture_self_assessment_request(body.message):
            return await _finalize_fastpath(
                _apply_aura_voice_shaping(
                    _build_architecture_self_reflex(
                        _build_aura_expression_frame(body.message),
                        body.message,
                    )
                ),
                status="architecture_self_reflex",
            )

        # Crash-safe persistence: persist the user's message BEFORE calling
        # the LLM. If the process dies mid-inference, the message is preserved
        # and the conversation can be resumed. (Pattern from Claude Code.)
        try:
            await _preserve_large_user_paste(body.message)
            pending_exchange_id = await _begin_logged_exchange(body.message)
        except Exception:
            pass  # Best-effort; don't block the response

        # Phase 2 Constitutional Closure: Try Sovereign Kernel Interface actively
        from core.kernel.kernel_interface import KernelInterface
        ki = KernelInterface.get_instance()
        reply_text = None
        kernel_timed_out = False
        kernel_task: Optional[asyncio.Task] = None

        if ki.is_ready():
            logger.debug("REST: Awaiting constitutional processing from Sovereign Kernel...")
            try:
                kernel_timeout = _remaining_foreground_budget()
                kernel_task = asyncio.create_task(
                    ki.process(body.message, origin="api", priority=True),
                    name="Aura.Server.Chat.kernel_foreground",
                )
                # [STABILITY v53] Two-phase timeout:
                # Phase 1 (soft): Give kernel its full SLA. Don't fire competing
                #   requests during this window — resource contention makes both slower.
                # Phase 2 (hard): If kernel misses soft deadline, try protected foreground
                #   OR wait for kernel with remaining budget, whichever finishes first.
                soft_deadline = min(
                    _KERNEL_SOFT_REPLY_SLA_SECONDS,
                    max(8.0, kernel_timeout - 20.0),
                )
                try:
                    reply_text = await asyncio.wait_for(
                        asyncio.shield(kernel_task),
                        timeout=soft_deadline,
                    )
                except asyncio.TimeoutError:
                    # Soft deadline missed — try protected foreground as a race
                    # against the kernel task continuing in background.
                    hard_budget = max(2.0, _remaining_foreground_budget())
                    protected_reply = await _attempt_protected_foreground_reply("kernel_soft_deadline")
                    if protected_reply:
                        kernel_task.add_done_callback(
                            lambda task: task.exception() if not task.cancelled() else None
                        )
                        return await _finalize_fastpath(
                            protected_reply,
                            status="protected_foreground",
                        )
                    # Protected foreground also failed — give kernel remaining time
                    reply_text = await asyncio.wait_for(
                        asyncio.shield(kernel_task),
                        timeout=max(2.0, _remaining_foreground_budget()),
                    )
            except asyncio.TimeoutError as e:
                kernel_timed_out = True
                logger.error(
                    "KernelInterface chat timed out; refusing legacy replay for the same foreground request: %s (%s)",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            except Exception as e:
                logger.error("KernelInterface chat failed natively, falling back to legacy: %s (%s)", type(e).__name__, e, exc_info=True)

        if kernel_timed_out:
            direct_reply = await _attempt_protected_foreground_reply("kernel_timeout")
            if direct_reply:
                reply_text = direct_reply
                logger.info("✅ [STABILITY] Protected foreground bypass succeeded after kernel timeout (len=%d)", len(reply_text))
                kernel_timed_out = False

        if kernel_timed_out:
            lane = _mark_conversation_lane_timeout()
            # Tiered response: 503 (recoverable/retry) when cortex was ready,
            # 504 (hard timeout) only when the lane itself was broken.
            was_ready = bool(lane.get("conversation_ready", False)) or str(lane.get("state", "")).lower() in {"ready", "warming", "recovering"}
            status_code = 503 if was_ready else 504
            timeout_reply = _conversation_lane_user_message(lane, timed_out=True)
            if pending_exchange_id:
                await _complete_logged_exchange(
                    pending_exchange_id,
                    body.message,
                    timeout_reply,
                )
                pending_exchange_id = None
            return JSONResponse(
                {
                    "response": timeout_reply,
                    "status": "timeout",
                    "conversation_lane": lane,
                },
                status_code=status_code,
            )

        # Legacy Orchestrator Fallback
        if not reply_text:
            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                logger.debug("REST: Awaiting priority processing from legacy orchestrator...")
                legacy_timeout = _remaining_foreground_budget()
                reply_text = await asyncio.wait_for(
                    orch.process_user_input_priority(body.message, origin="api", timeout_sec=legacy_timeout),
                    timeout=legacy_timeout,
                )
            else:
                from core.tasks import dispatch_user_input
                asyncio.ensure_future(
                    asyncio.to_thread(dispatch_user_input, body.message)
                )
                reply_text = "Message dispatched (Kernel and Orchestrator offline)."

        reply_text = await _stabilize_user_facing_reply(body.message, reply_text)

        # ── Response confidence assessment ────────────────────────
        global _consecutive_degraded_count
        response_confidence = "high"
        is_stale = _is_stale_repeated_response(reply_text)
        is_same_diff = _is_same_answer_different_prompt(body.message, reply_text)
        recent_user_messages = await _gather_recent_user_messages_for_relevance(body.message)
        is_off_topic, off_topic_reason = _evaluate_reply_topicality(
            body.message,
            reply_text,
            recent_user_messages=recent_user_messages,
        )
        if is_stale or is_same_diff or is_off_topic:
            response_confidence = "degraded"
            _consecutive_degraded_count += 1
            logger.warning(
                "⚠️ Response confidence: degraded (stale=%s, same_answer_diff_prompt=%s, off_topic=%s, streak=%d, reason=%s)",
                is_stale,
                is_same_diff,
                is_off_topic,
                _consecutive_degraded_count,
                off_topic_reason or "",
            )
        else:
            _consecutive_degraded_count = 0

        # Proactive recovery: if 3+ consecutive degraded responses, compact + reset stale deque
        if _consecutive_degraded_count >= 3:
            logger.warning("🚨 Degradation streak=%d — triggering proactive compaction + stale reset.", _consecutive_degraded_count)
            _recent_responses.clear()
            _recent_response_pairs.clear()
            _consecutive_degraded_count = 0
            try:
                live_state = _resolve_live_aura_state()
                if live_state and hasattr(live_state, "compact"):
                    live_state.compact(trigger_threshold=20, keep_turns=15)
                    logger.info("🗜️ Proactive compaction completed after degradation streak.")
            except Exception as _streak_exc:
                logger.debug("Degradation streak compaction failed: %s", _streak_exc)

        # Proactive context compaction — fire-and-forget to prevent working memory bloat
        try:
            live_state = _resolve_live_aura_state()
            if live_state and hasattr(live_state, "compact"):
                wm = getattr(getattr(live_state, "cognition", None), "working_memory", None)
                if wm and isinstance(wm, list) and len(wm) > 30:
                    compacted = live_state.compact(trigger_threshold=30, keep_turns=20)
                    if compacted:
                        logger.debug("Proactive AuraState.compact() completed (working_memory was %d).", len(wm))
        except Exception as _compact_exc:
            logger.debug("Proactive compaction skipped: %s", _compact_exc)

        # ── Post-Response Infrastructure checks ─────────────────
        # 1. Check self-consistency (avoiding false inability claims, commitment contradictions)
        if response_confidence == "high":
            is_consistent, reason = _check_response_consistency(reply_text, body.message)
            if not is_consistent:
                response_confidence = "degraded"
                logger.warning("⚠️ Response confidence lowered to 'degraded' due to inconsistency: %s", reason)

        # 2. Extract new open loops (commitments/promises) made in this turn
        _extract_and_register_commitments(reply_text, body.message)

        # 3. Log comprehensive quality metrics
        _log_response_quality_metrics(
            user_message=body.message,
            reply_text=reply_text,
            confidence=response_confidence,
            stale=is_stale,
            same_diff=is_same_diff,
            off_topic=is_off_topic,
        )

        response_data = {
            "response": reply_text or "…",
            "conversation_lane": _collect_conversation_lane_status(),
            "response_confidence": response_confidence,
        }

        _record_recent_response(reply_text or "…", body.message)
        if pending_exchange_id:
            await _complete_logged_exchange(
                pending_exchange_id,
                body.message,
                reply_text or "…",
            )
            pending_exchange_id = None
        else:
            await _log_exchange(body.message, reply_text or "…")

        # Cache idempotent response
        if idem_key:
            async with _idempotency_lock:
                _idempotency_cache[idem_key] = response_data
                if len(_idempotency_cache) > 1000:
                    _idempotency_cache.popitem(last=False)

        await _emit_chat_output_receipt(
            reply_text or "…",
            cause="chat_response",
            metadata={
                "response_confidence": response_confidence,
                "path": "stabilized",
            },
        )

        return JSONResponse(response_data)
    except asyncio.TimeoutError:
        lane = _mark_conversation_lane_timeout()
        # [STABILITY v53] Last-resort: try protected foreground before returning timeout.
        # The kernel timed out but the LLM might still be responsive for a direct call.
        try:
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "generate"):
                emergency_reply = await asyncio.wait_for(
                    gate.generate(
                        body.message,
                        context={
                            "origin": "api",
                            "foreground_request": True,
                            "protected_foreground_lane": True,
                            "protected_foreground_reason": "outer_timeout_emergency",
                            "prefer_tier": "tertiary",  # Use fastest available model
                            "allow_cloud_fallback": True,  # Try EVERYTHING
                        },
                        timeout=15.0,
                    ),
                    timeout=15.0,
                )
                if emergency_reply and str(emergency_reply).strip():
                    logger.info("✅ [STABILITY v53] Emergency bypass after outer timeout succeeded.")
                    emergency_text = str(emergency_reply).strip()
                    if pending_exchange_id:
                        await _complete_logged_exchange(
                            pending_exchange_id,
                            body.message,
                            emergency_text,
                        )
                        pending_exchange_id = None
                    return JSONResponse({
                        "response": emergency_text,
                        "conversation_lane": _collect_conversation_lane_status(),
                        "response_confidence": "degraded",
                    })
        except Exception:
            pass  # Fall through to timeout response

        # [STABILITY v53] Return 200 with status field instead of 503/504.
        # Non-200 codes can cause frontend retry storms or error displays.
        # The "status" field tells the frontend it was degraded.
        timeout_reply = _conversation_lane_user_message(lane, timed_out=True)
        if pending_exchange_id:
            await _complete_logged_exchange(
                pending_exchange_id,
                body.message,
                timeout_reply,
            )
            pending_exchange_id = None
        return JSONResponse(
            {
                "response": timeout_reply,
                "status": "timeout",
                "conversation_lane": lane,
                "response_confidence": "degraded",
            },
            status_code=200,  # [STABILITY v53] Changed from 503/504 to 200
        )
    except asyncio.CancelledError:
        lane = _mark_conversation_lane_state("foreground_cancelled", state="recovering")
        cancel_reply = "I got interrupted mid-thought. Say that again?"
        if pending_exchange_id:
            await _complete_logged_exchange(
                pending_exchange_id,
                body.message,
                cancel_reply,
            )
            pending_exchange_id = None
        return JSONResponse(
            {
                "response": cancel_reply,
                "status": "cancelled",
                "conversation_lane": lane,
                "response_confidence": "degraded",
            },
            status_code=200,  # [STABILITY v53] Changed from 503 to 200
        )
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        error_reply = "I lost my train of thought for a second. Try me again?"
        if pending_exchange_id:
            await _complete_logged_exchange(
                pending_exchange_id,
                body.message,
                error_reply,
            )
            pending_exchange_id = None
        # [STABILITY v53] ALWAYS return 200 with a response. Chat must never
        # appear broken to the user. The "status" field conveys error state.
        return JSONResponse({
            "response": error_reply,
            "status": "error",
            "response_confidence": "degraded",
        }, status_code=200)
    finally:
        if foreground_slot_acquired and _foreground_chat_lock.locked():
            _foreground_chat_lock.release()
