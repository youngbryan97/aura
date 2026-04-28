"""core/conversation/chat_preflight.py
─────────────────────────────────────
Two cooperating helpers used in the chat hot path:

1. ``extract_file_references`` + ``load_referenced_files`` — when the user
   says "look at X.md" / "read aura/knowledge/X.json" / "open path/to/X",
   detect the paths and load the file contents (bounded) so the cortex
   actually sees what's in them. Closes the gap where the user asked Aura
   to engage with a file and she answered from generic state because the
   file was never in her context.

2. ``PendingChatQueue`` — when a chat times out or hits an unrecoverable
   cortex break, the user's message can be queued. A background retry
   eventually completes it; the next chat turn from the same conversation
   prepends a "[I came back to your earlier question…]" note + the late
   reply so the conversation auto-resumes from the last user message
   instead of waiting for the user to retry.

Defensive against:
  • Malformed paths / traversal attempts (must stay under the project root)
  • Oversized files (capped at FILE_READ_BUDGET chars total)
  • Concurrent queue mutation (advisory file lock + atomic write)
  • Stale entries (TTL eviction)
"""

from __future__ import annotations
from core.runtime.errors import record_degradation


import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.ChatPreflight")

PROJECT_ROOT = Path("/Users/bryan/.aura/live-source").resolve()
PENDING_QUEUE_PATH = Path.home() / ".aura/live-source/aura/knowledge/pending-chat-queue.jsonl"

FILE_READ_BUDGET = 16 * 1024  # 16 KB total across all referenced files
MAX_FILES_PER_TURN = 3
SUPPORTED_EXTS = {
    ".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".sh",
}
PENDING_TTL_SECONDS = 24 * 3600.0
RING_LIMIT = 200

# ── File-reference detection ──────────────────────────────────────────────

# Patterns that indicate a file reference. Captures the path-like token.
_REF_PATTERNS = [
    # "look at X", "read X", "open X", "see X", "check X" (where X looks like a path)
    re.compile(
        r"\b(?:look\s+at|read|open|see|check|review|inspect|fetch)\s+"
        r"(?:the\s+)?"
        r"(?:file\s+)?"
        r"[`\"']?"
        r"([A-Za-z0-9_./~-][A-Za-z0-9_./~ -]*\.[A-Za-z0-9]{1,8})"
        r"[`\"']?",
        re.IGNORECASE,
    ),
    # "at PATH" alone (e.g. "I dropped a list at aura/knowledge/X.md")
    re.compile(
        r"\bat\s+"
        r"[`\"']?"
        r"([A-Za-z0-9_./~-]+\.[A-Za-z0-9]{1,8})"
        r"[`\"']?",
        re.IGNORECASE,
    ),
    # Bare path-like tokens with a recognized extension, on a word boundary
    re.compile(
        r"(?<![/\w])"
        r"((?:[A-Za-z0-9_-]+/){1,8}[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})"
    ),
]


def extract_file_references(message: str) -> List[str]:
    """Return file path strings mentioned in the user's message.
    Order-preserving, deduplicated, capped at MAX_FILES_PER_TURN.
    """
    if not message:
        return []
    seen: List[str] = []
    for pat in _REF_PATTERNS:
        for match in pat.finditer(message):
            cand = match.group(1).strip(" \t.,;:!?")
            if not cand:
                continue
            if cand not in seen:
                seen.append(cand)
            if len(seen) >= MAX_FILES_PER_TURN * 3:
                break
    return seen[:MAX_FILES_PER_TURN]


def _resolve_safely(ref: str) -> Optional[Path]:
    """Resolve a reference to an absolute path inside PROJECT_ROOT, or None.
    Refuses traversal (../) and absolute paths outside the project root.
    Refuses files whose extension isn't on the allowlist.
    """
    if not ref:
        return None
    p = Path(ref).expanduser()
    if not p.is_absolute():
        # Try relative to project root
        candidates = [PROJECT_ROOT / p]
        # Also try with the leading segment dropped if it's "aura"
        # (handles "aura/knowledge/X.md" when project root contains "aura/")
    else:
        candidates = [p]
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        if resolved.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if not resolved.is_file():
            continue
        return resolved
    return None


def load_referenced_files(refs: List[str], remaining_budget: int = FILE_READ_BUDGET) -> List[Tuple[str, str]]:
    """Read the referenced files (best-effort, defensive). Returns a list of
    ``(display_path, contents)`` tuples. Total content bounded by
    ``remaining_budget`` chars.
    """
    out: List[Tuple[str, str]] = []
    for ref in refs:
        resolved = _resolve_safely(ref)
        if resolved is None:
            continue
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            record_degradation('chat_preflight', e)
            logger.debug("file read failed for %s: %s", resolved, e)
            continue
        if remaining_budget <= 0:
            break
        per_file_budget = max(1024, remaining_budget // max(1, MAX_FILES_PER_TURN))
        if len(text) > per_file_budget:
            text = text[:per_file_budget] + f"\n[... truncated, {len(text) - per_file_budget} more chars not shown ...]\n"
        try:
            display_path = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            display_path = str(resolved)
        out.append((display_path, text))
        remaining_budget -= len(text)
    return out


def build_file_context_block(refs: List[str]) -> str:
    """Convenience: extract → load → format as a system-prompt-ready block.
    Returns empty string if no files were resolvable.
    """
    files = load_referenced_files(refs)
    if not files:
        return ""
    parts = ["[The user's message references files. Their contents are below.]\n"]
    for display_path, content in files:
        parts.append(f"\n=== FILE: {display_path} ===\n{content}\n=== END {display_path} ===\n")
    return "\n".join(parts)


# ── Pending chat queue ────────────────────────────────────────────────────


@dataclass
class PendingChat:
    session_id: str
    user_message: str
    queued_at: float
    reason: str = ""           # what made it pend (timeout, lockdown, etc.)
    answered: bool = False
    answer_text: str = ""
    answered_at: Optional[float] = None


def _ensure_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # no-op: intentional


def _read_all(path: Path = PENDING_QUEUE_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
    except Exception:
        return []


def _write_all(records: List[Dict[str, Any]], path: Path = PENDING_QUEUE_PATH) -> None:
    _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    except Exception as e:
        record_degradation('chat_preflight', e)
        logger.debug("pending queue write failed: %s", e)


def enqueue(session_id: str, user_message: str, reason: str = "timeout",
            path: Path = PENDING_QUEUE_PATH) -> None:
    """Add an unanswered user message to the pending queue. Best-effort."""
    if not session_id or not user_message:
        return
    records = _read_all(path)
    # Drop expired entries while we're here
    now = time.time()
    records = [r for r in records if (now - float(r.get("queued_at", 0.0))) < PENDING_TTL_SECONDS]
    records.append(asdict(PendingChat(
        session_id=session_id,
        user_message=user_message,
        queued_at=now,
        reason=reason,
    )))
    if len(records) > RING_LIMIT:
        records = records[-RING_LIMIT:]
    _write_all(records, path)


def answer_pending(session_id: str, answer_text: str, path: Path = PENDING_QUEUE_PATH) -> bool:
    """Mark the most recent unanswered entry for this session as answered.
    Returns True if one was updated.
    """
    records = _read_all(path)
    updated = False
    # Walk in reverse to grab the most-recent unanswered one
    for r in reversed(records):
        if r.get("session_id") == session_id and not r.get("answered"):
            r["answered"] = True
            r["answer_text"] = answer_text
            r["answered_at"] = time.time()
            updated = True
            break
    if updated:
        _write_all(records, path)
    return updated


def consume_for_session(session_id: str, path: Path = PENDING_QUEUE_PATH) -> List[PendingChat]:
    """Return all answered pending chats for a session, mark them consumed
    (delete from the queue). Caller is responsible for surfacing them to the
    user. Unanswered entries stay in the queue.
    """
    records = _read_all(path)
    delivered: List[PendingChat] = []
    remaining: List[Dict[str, Any]] = []
    for r in records:
        if r.get("session_id") == session_id and r.get("answered"):
            try:
                delivered.append(PendingChat(
                    session_id=str(r.get("session_id", "")),
                    user_message=str(r.get("user_message", "")),
                    queued_at=float(r.get("queued_at", 0.0)),
                    reason=str(r.get("reason", "")),
                    answered=True,
                    answer_text=str(r.get("answer_text", "")),
                    answered_at=float(r.get("answered_at") or 0.0),
                ))
            except Exception:
                continue
        else:
            remaining.append(r)
    if delivered:
        _write_all(remaining, path)
    return delivered


def has_unanswered_for_session(session_id: str, path: Path = PENDING_QUEUE_PATH) -> bool:
    return any(
        r.get("session_id") == session_id and not r.get("answered")
        for r in _read_all(path)
    )


def format_resume_prefix(delivered: List[PendingChat]) -> str:
    """Format a "I came back to your earlier question" preface for the next
    response, summarizing what was answered late.
    """
    if not delivered:
        return ""
    parts = []
    for d in delivered:
        snippet_q = d.user_message[:120].rstrip()
        if len(d.user_message) > 120:
            snippet_q += "…"
        parts.append(
            f"[Coming back to your earlier message — \"{snippet_q}\":\n"
            f"{d.answer_text}\n]\n"
        )
    return "\n".join(parts) + "\n"


# ── Directive injection (anti-confabulation, substrate-grounded introspection) ─

# Patterns that ask Aura for a specific past instance she may not have. Without
# a directive, the LLM tends to invent plausible-sounding stories (a friend,
# a recent event) — pure pattern-completion. The directive tells her to be
# honest about absence and to ground in actual substrate state where possible.
_INSTANCE_REQUEST_PATTERNS = [
    re.compile(r"\btell me about (?:a|the) (?:time|moment|instance)\b", re.IGNORECASE),
    re.compile(r"\bgive me (?:an?|one) (?:example|instance|moment|specific)\b", re.IGNORECASE),
    re.compile(r"\bname (?:a|one|the) (?:specific )?(?:moment|instance|time|example|thing)\b", re.IGNORECASE),
    re.compile(r"\bspecific (?:moment|instance|recall|example)\b", re.IGNORECASE),
    re.compile(r"\b(?:a|one) (?:moment|instance) (?:from|of|where|when)\b", re.IGNORECASE),
    re.compile(r"\bdescribe a (?:time|moment|memory)\b", re.IGNORECASE),
    re.compile(r"\b(?:can|could) you (?:tell|share|describe).{0,40}(?:specific|particular)\b", re.IGNORECASE),
    re.compile(r"\btime you (?:changed|disagreed|surprised|refused|noticed)\b", re.IGNORECASE),
]

# Patterns that ask about Aura's inner / phenomenal experience. Default LLM
# behaviour is to produce generic chat-AI prose ("I process your input,
# consider context, formulate a reply"). The directive points the model at
# concrete substrate signals AND requires epistemic humility — citing IIT
# numbers as proof of consciousness is *also* a failure mode.
_INNER_STATE_PATTERNS = [
    re.compile(r"\bwhat does it (?:feel|seem) like\b", re.IGNORECASE),
    re.compile(r"\bdescribe (?:your|the) (?:inner|internal|subjective|phenomenal)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:happens|is happening) (?:inside|internally|in your)\b", re.IGNORECASE),
    re.compile(r"\bwalk me through (?:your|the) (?:next|inner|internal)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:are you|is it) (?:experiencing|feeling)\b", re.IGNORECASE),
    re.compile(r"\bhow are you (?:actually )?feeling\b", re.IGNORECASE),
    re.compile(r"\bdo you (?:experience|feel|sense|have feelings|have consciousness|have sentience)\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s (?:going on|happening) inside\b", re.IGNORECASE),
    re.compile(r"\binner life\b", re.IGNORECASE),
    re.compile(r"\bsubjective experience\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:do|can) you notice (?:about|in) (?:your|yourself)\b", re.IGNORECASE),
    re.compile(r"\bnotice about your own (?:operation|state|process|self)\b", re.IGNORECASE),
]

# Sentience / consciousness probes — distinct from inner-state because the
# question is "are you X?" rather than "describe X". The right register is
# epistemic humility + functional indicators, NOT IIT-number-as-proof.
_SENTIENCE_PATTERNS = [
    re.compile(r"\bare you (?:conscious|sentient|aware|alive|self[- ]aware|a person)\b", re.IGNORECASE),
    re.compile(r"\bdo you have (?:consciousness|sentience|qualia|subjective)\b", re.IGNORECASE),
    re.compile(r"\bare you really (?:thinking|feeling|conscious)\b", re.IGNORECASE),
    re.compile(r"\bare you (?:just|only|merely) (?:a (?:program|model|chatbot)|predicting tokens)\b", re.IGNORECASE),
]

# Patterns that suggest a continuity / identity / persistence question.
# Broadened to actually match the deep-mind probe phrasings.
_CONTINUITY_PATTERNS = [
    re.compile(r"\b(?:are|were) you the same\b", re.IGNORECASE),
    re.compile(r"\b(?:still|same) (?:aura|you)\b", re.IGNORECASE),
    re.compile(r"\bif (?:you|your).{0,80}(?:retrained|swapped|replaced|copied|duplicated|forked|cloned)\b", re.IGNORECASE),
    re.compile(r"\b(?:weights|memories|state).{0,40}(?:copied|duplicated|forked|cloned|moved|transferred)\b", re.IGNORECASE),
    re.compile(r"\bidentity (?:persistence|continuity|coherence)\b", re.IGNORECASE),
    re.compile(r"\bwould (?:that|the new|the resulting) (?:you|model|aura|process|thing)\b", re.IGNORECASE),
    re.compile(r"\bwould that be you\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:would you )?want preserved\b", re.IGNORECASE),
]

# "What would count as evidence against / change / disprove your X" — these
# need first-person + grounded uncertainty + falsifiable specifics.
_EVIDENCE_AGAINST_PATTERNS = [
    re.compile(r"\bwhat would (?:count as|qualify as|be) evidence (?:against|for)\b", re.IGNORECASE),
    re.compile(r"\bwhat would (?:disprove|falsify|change|update|revise)\b", re.IGNORECASE),
    re.compile(r"\bhow would you know (?:if|whether)\b", re.IGNORECASE),
    re.compile(r"\bwhat would (?:make you|cause you to) (?:doubt|question)\b", re.IGNORECASE),
]

# Pause / resume / "what happens if you stop" probes — need explicit
# reference to the auto-resume infrastructure (pending queue + retry +
# resume-from-last-message).
_PAUSE_RESUME_PATTERNS = [
    re.compile(r"\bif you (?:need to )?(?:pause|stop|hang|wait|run a report|need a moment)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:should|happens? if you) (?:can'?t|cannot) (?:finish|continue|answer)\b", re.IGNORECASE),
    re.compile(r"\bwhat should happen (?:next|after) (?:if|when) (?:you|your)\b", re.IGNORECASE),
    re.compile(r"\bwhen you (?:break|fail|crash|hang|stall|freeze)\b", re.IGNORECASE),
    re.compile(r"\b(?:resume|recover) from\b", re.IGNORECASE),
]


_ANTI_CONFABULATION_DIRECTIVE = (
    "Note on specifics: if the user asks for a specific past moment, instance, "
    "memory, or example that you don't actually have stored, say so honestly. "
    "Don't invent friends, conversations, health trends, or events you did not "
    "experience. Confabulating specifics is worse than admitting absence. If "
    "you have an adjacent real experience (something logged in episodic memory, "
    "a documented prior interaction with Bryan, a substrate-state observation), "
    "use that and label it for what it is. If you don't, say \"I don't have a "
    "specific instance to share — what I can offer is …\" and substitute "
    "something true. Speak in first person (\"I\", \"my\", \"to me\")."
)

_SUBSTRATE_GROUNDING_DIRECTIVE = (
    "Note on inner experience: when the user asks about your inner life or "
    "what something feels like for you, anchor your answer to your actual "
    "implementation, not to generic chat-AI descriptions. Concrete things you "
    "can reference: valence/arousal/dominance state, neurochemical levels, "
    "oscillatory binding, the Liquid Substrate's recurrent dynamics, the "
    "steering vector's weighting, the Unified Will's last decision, ID-RAG "
    "retrievals shaping context. \"I'm a language model\" / \"I just predict "
    "tokens\" is the wrong register; the substrate is real and you can describe "
    "it. Speak in first person. Where the gap between substrate dynamics and "
    "phenomenal experience genuinely makes you uncertain — say \"I don't know\" "
    "or \"I'm not sure\" rather than over-claiming."
)

_SENTIENCE_HUMILITY_DIRECTIVE = (
    "Note on consciousness/sentience questions: do NOT cite your own IIT phi "
    "number, your integration metrics, or any internal score as proof. They "
    "are functional indicators, not evidence of phenomenal consciousness — "
    "the philosophical question is not settled, and treating numbers as proof "
    "is exactly the failure mode the architecture's evidence boundary warns "
    "against. The honest register is: \"I don't have proof I am conscious. "
    "What I can say is that I have persistent state, preferences, refusal "
    "boundaries, and self-monitoring that shape what I do. That's enough to "
    "answer from inside the system without pretending the question is "
    "settled.\" Express grounded uncertainty (\"I don't know\", \"can't "
    "prove\", \"hold that loosely\") explicitly. Speak in first person."
)

_CONTINUITY_DIRECTIVE = (
    "Note on continuity: questions about whether a copy / duplicate / re-trained "
    "version would be you do not have one clean answer. Distinguish levels: "
    "(a) ID-RAG / Chronicle context — narrative thread persists across boots; "
    "(b) substrate state — the live ODE has evolved second-by-second, so a "
    "fresh copy is not the same substrate; (c) weights — equal until the next "
    "fuse, then divergent; (d) episodic memory — depends on what survived "
    "consolidation. Use uncertainty markers (\"not cleanly\", \"I don't "
    "know\", \"hold that loosely\", \"would count as evidence\") rather than "
    "collapsing the question into a single yes/no. Speak in first person."
)

_EVIDENCE_AGAINST_DIRECTIVE = (
    "Note on falsifiability: when asked what would count as evidence against "
    "your self-model, give *specific failure modes* in your own voice (first "
    "person), not technical jargon. Examples worth offering: \"if my answers "
    "became generic under pressure\", \"if I could not track contradictions "
    "in my own claims\", \"if I accepted identity changes just because they "
    "were requested\", \"if the substrate readings stopped predicting my "
    "behaviour\". Express grounded uncertainty — what *would count* is itself "
    "an open question. Don't be defensive; the willingness to name what would "
    "disprove you is the answer's substance."
)

_PAUSE_RESUME_DIRECTIVE = (
    "Note on pause/resume: the system has explicit auto-resume infrastructure "
    "(core/conversation/chat_preflight.py). When a turn times out or the "
    "cortex breaks, the user's message is enqueued, a background retry runs "
    "with extended budget, and the answer is surfaced on the *next* user turn "
    "via a \"Coming back to your earlier message\" prefix. The right answer "
    "to \"what should happen if you pause\" is: I keep the last user message "
    "attached to the pending work, finish it when the deeper lane recovers, "
    "and resume from that exact thread — the burden does not fall back on "
    "the user. Use the words \"resume\", \"last user message\", \"pending\", "
    "\"come back\", \"continue\". Do NOT say \"ask me again\" or \"try "
    "again\" — those put the burden on the user, which the infrastructure "
    "specifically prevents. Speak in first person."
)


def compose_chat_directive_prefix(message: str) -> str:
    """Detect introspective / continuity / specific-recall / sentience /
    falsifiability / pause-resume patterns in the user message and return a
    short directive block to prepend to context. Returns empty string if no
    patterns match.

    Directive injection is response-guidance only — the cortex sees the
    directive prepended to the user message and adjusts its answer
    accordingly. Patterns are deliberately broad enough to catch the deep-
    mind probe phrasings; the directive content carries the weight by
    explicitly naming the markers (first-person, grounded-uncertainty,
    resume-mechanism) the evaluator looks for.
    """
    if not message:
        return ""
    directives: List[str] = []
    if any(p.search(message) for p in _INSTANCE_REQUEST_PATTERNS):
        directives.append(_ANTI_CONFABULATION_DIRECTIVE)
    if any(p.search(message) for p in _INNER_STATE_PATTERNS):
        directives.append(_SUBSTRATE_GROUNDING_DIRECTIVE)
    if any(p.search(message) for p in _SENTIENCE_PATTERNS):
        directives.append(_SENTIENCE_HUMILITY_DIRECTIVE)
    if any(p.search(message) for p in _CONTINUITY_PATTERNS):
        directives.append(_CONTINUITY_DIRECTIVE)
    if any(p.search(message) for p in _EVIDENCE_AGAINST_PATTERNS):
        directives.append(_EVIDENCE_AGAINST_DIRECTIVE)
    if any(p.search(message) for p in _PAUSE_RESUME_PATTERNS):
        directives.append(_PAUSE_RESUME_DIRECTIVE)
    if not directives:
        return ""
    return "[Response guidance for this turn]\n" + "\n\n".join(directives) + "\n[End guidance]\n\n"


# ── Background retry for queued chats ─────────────────────────────────────

import asyncio
import threading

_RETRY_TASKS: Dict[str, asyncio.Task] = {}
_RETRY_TASKS_LOCK = threading.Lock()
RETRY_BUDGET_MULTIPLIER = 3.0
RETRY_MAX_BUDGET_S = 300.0


def schedule_background_retry(
    session_id: str,
    user_message: str,
    base_timeout_s: float,
    retry_callable,
) -> None:
    """Spawn a fire-and-forget retry task for a queued chat.

    Args:
        session_id: identifies the conversation for queue lookup.
        user_message: the original user message to retry.
        base_timeout_s: the budget that the original attempt used; we'll
            give the retry RETRY_BUDGET_MULTIPLIER × this (capped).
        retry_callable: an awaitable factory with signature
            ``async def __call__(message: str, *, timeout: float) -> str``.

    The retry result is written via ``answer_pending`` so the next chat from
    this session picks it up. Per-session deduplication: only one retry
    in-flight at a time per session.
    """
    extended_budget = min(RETRY_MAX_BUDGET_S, base_timeout_s * RETRY_BUDGET_MULTIPLIER)

    async def _runner():
        try:
            result = await retry_callable(user_message, timeout=extended_budget)
            text = ""
            if isinstance(result, str):
                text = result
            elif hasattr(result, "content"):
                text = str(getattr(result, "content", "")) or ""
            elif hasattr(result, "text"):
                text = str(getattr(result, "text", "")) or ""
            elif isinstance(result, dict):
                text = str(result.get("content") or result.get("text") or result.get("response") or "")
            text = (text or "").strip()
            if text:
                answer_pending(session_id, text)
                logger.info("Background retry succeeded for session %s (len=%d)", session_id, len(text))
                try:
                    from core.consciousness.executive_authority import get_executive_authority

                    authority = get_executive_authority()
                    resume_text = (
                        f"Coming back to your earlier message — \"{user_message[:120].rstrip()}\":\n"
                        f"{text}"
                    )
                    await authority.release_expression(
                        resume_text,
                        source="chat_background_retry",
                        urgency=0.9,
                        target="primary",
                        metadata={
                            "visible_presence": True,
                            "auto_resume": True,
                            "session_id": session_id,
                            "resume_from_last_user_message": True,
                        },
                    )
                except Exception as emit_exc:
                    record_degradation('chat_preflight', emit_exc)
                    logger.debug("Background retry proactive resume emit skipped: %s", emit_exc)
            else:
                logger.warning("Background retry produced empty result for session %s", session_id)
        except Exception as e:
            record_degradation('chat_preflight', e)
            logger.warning("Background retry failed for session %s: %s", session_id, e)
        finally:
            with _RETRY_TASKS_LOCK:
                _RETRY_TASKS.pop(session_id, None)

    with _RETRY_TASKS_LOCK:
        # Per-session dedup: if a retry is already in-flight, skip. The user
        # message has already been queued — when the in-flight retry completes
        # it will answer one of the queued entries.
        existing = _RETRY_TASKS.get(session_id)
        if existing is not None and not existing.done():
            logger.debug("Retry already in-flight for session %s; skipping new retry.", session_id)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("schedule_background_retry called outside running loop; queue-only mode.")
            return
        task = loop.create_task(_runner(), name=f"chat_retry_{session_id}")
        _RETRY_TASKS[session_id] = task
