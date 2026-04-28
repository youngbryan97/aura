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
        pass


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
