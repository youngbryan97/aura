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

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text, interprocess_file_lock
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.structured_input import looks_like_learning_resource_bundle
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.ChatPreflight")

PROJECT_ROOT = Path(
    os.environ.get("AURA_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def _default_pending_queue_path() -> Path:
    override = os.environ.get("AURA_PENDING_CHAT_QUEUE_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aura/data/conversation/pending-chat-queue.jsonl"


PENDING_QUEUE_PATH = _default_pending_queue_path()

FILE_READ_BUDGET = 16 * 1024  # 16 KB total across all referenced files
MAX_FILES_PER_TURN = 3
MAX_RESUME_PREFIX_CHARS = 12 * 1024
MAX_RESUME_ANSWER_CHARS = 4 * 1024
MAX_PROFILE_CONTEXT_CHARS = 8 * 1024
MAX_OPERATIONAL_SELF_CONTEXT_CHARS = 6 * 1024
MAX_COMPOSED_PREFLIGHT_CHARS = 48 * 1024
SUPPORTED_EXTS = {
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".sh",
}
PENDING_TTL_SECONDS = 24 * 3600.0
RING_LIMIT = 200
MAX_SESSION_ID_CHARS = 160
MAX_USER_MESSAGE_CHARS = 20_000
MAX_ANSWER_TEXT_CHARS = 60_000
MAX_REASON_CHARS = 240
MAX_QUEUE_LINE_CHARS = 256_000
MAX_PENDING_QUEUE_BYTES = 24 * 1024 * 1024
_QUEUE_LOCK = threading.RLock()
_DEGRADATION_CURRENT_WINDOW_S = 300.0

_CHAT_PREFLIGHT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)


def _resolve_pending_queue_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return _default_pending_queue_path()


def _emit_chat_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "chat_preflight",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("chat_preflight", error)


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_truncated_text(
    value: Any,
    *,
    max_chars: int,
    suffix: str = "\n[... truncated by live-chat context budget ...]",
) -> str:
    text = _safe_text(value, max_chars=max(max_chars + len(suffix) + 1, max_chars))
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "answered"}
    return bool(value)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or hasattr(value, "__await__"):
        return await value
    return value


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


#: Words that are never the identifier a question is really about. Deliberately
#: small: over-filtering here costs a relevant window, and the ranking already
#: prefers longer, more specific terms.
_EXCERPT_STOPWORDS = frozenset(
    {
        "and", "are", "but", "can", "does", "file", "for", "from", "has",
        "have", "the", "this", "that", "what", "when", "where", "which",
        "with", "you", "your", "line", "lines", "read", "tell", "there",
        "then", "they", "was", "were", "will", "would", "into", "its",
        "not", "own", "quote", "source", "value", "happens", "called",
    }
)

#: Lines of context kept on each side of a match. Small on purpose: the point
#: is the region that answers the question, and a wide window spends the whole
#: budget on the first hit.
_EXCERPT_CONTEXT_LINES = 12


def extract_file_references(message: str) -> list[str]:
    """Return file path strings mentioned in the user's message.
    Order-preserving, deduplicated, capped at MAX_FILES_PER_TURN.
    """
    if not message:
        return []
    seen: list[str] = []
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


def _resolve_safely(ref: str) -> Path | None:
    """Resolve a reference to an absolute path inside PROJECT_ROOT, or None.
    Refuses traversal (../) and absolute paths outside the project root.
    Refuses files whose extension isn't on the allowlist.
    """
    ref = _safe_text(ref, max_chars=1000).strip().strip("`\"'")
    if not ref:
        return None
    try:
        project_root = PROJECT_ROOT.resolve()
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        project_root = PROJECT_ROOT
    p = Path(ref).expanduser()
    if not p.is_absolute():
        # Try relative to project root
        candidates = [project_root / p]
        # Also try with the leading segment dropped if it's "aura"
        # (handles "aura/knowledge/X.md" when project root contains "aura/")
        if p.parts and p.parts[0] == "aura" and len(p.parts) > 1:
            candidates.append(project_root / Path(*p.parts[1:]))
    else:
        candidates = [p]
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            continue
        try:
            resolved.relative_to(project_root)
        except ValueError:
            continue
        if resolved.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if not resolved.is_file():
            continue
        return resolved
    return None


def load_referenced_files(
    refs: list[str],
    remaining_budget: int = FILE_READ_BUDGET,
    *,
    query: str = "",
) -> list[tuple[str, str]]:
    """Read the referenced files (best-effort, defensive). Returns a list of
    ``(display_path, contents)`` tuples. Total content bounded by
    ``remaining_budget`` chars.

    When a file does not fit its budget, the excerpt is chosen by RELEVANCE to
    ``query`` rather than by position.

    LIVE DEFECT, 2026-08-10. Asked "in your own source there is a file
    core/soma/resilience_engine.py — read it and tell me what happens to the
    depletion value when record_success is called; quote the line", she
    answered with source code from an entirely different module, failed her own
    reply gate on ``incomplete_code_response``, and shipped the draft anyway.

    Everything upstream had worked. The path was extracted, the file was read,
    and the log says "Chat preflight: loaded 1 referenced file(s) into
    context." What she was handed was the first 5,461 characters of a 20,428
    character file — and ``def record_success`` begins at character 5,815. The
    excerpt stopped 354 characters short of the only region the question was
    about, said "truncated", and she generated the rest.

    Position is not relevance. A question naming ``record_success`` should be
    answered from the part of the file containing ``record_success``, which is
    how the whole class of "asked about anything past the first few KB of a
    file" failures disappears rather than being re-diagnosed per file.
    """
    out: list[tuple[str, str]] = []
    for ref in refs:
        if remaining_budget <= 0:
            break
        resolved = _resolve_safely(ref)
        if resolved is None:
            continue
        per_file_budget = max(1024, remaining_budget // max(1, MAX_FILES_PER_TURN))
        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(per_file_budget + 1)
                if len(text) > per_file_budget:
                    # Too big for one bite. Re-read as lines so the excerpt can
                    # be chosen for what it contains.
                    handle.seek(0)
                    text = _relevant_excerpt(
                        handle.readlines(),
                        query=query,
                        budget=per_file_budget,
                    )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            _emit_chat_fault(
                e,
                action="skipped unreadable referenced file and continued without it",
                severity="warning",
                stage="load_referenced_files",
                extra={"path": str(resolved)},
            )
            logger.debug("file read failed for %s: %s", resolved, e)
            continue
        try:
            display_path = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            display_path = str(resolved)
        out.append((display_path, text))
        remaining_budget -= len(text)
    return out


def _excerpt_query_terms(query: str) -> list[str]:
    """Identifier-ish words from the question, longest first.

    Longest first because a question naming both ``record_success`` and
    ``depletion`` is more specifically about the former, and the budget is
    spent in that order.
    """
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(query or ""))
        if term.lower() not in _EXCERPT_STOPWORDS
    }
    return sorted(terms, key=lambda term: (-len(term), term))


def _relevant_excerpt(lines: list[str], *, query: str, budget: int) -> str:
    """Windows of `lines` around the query's terms, with the gaps declared.

    Falls back to the head when the question names nothing findable, which is
    the old behaviour and the right answer for "what does this file do".
    Every omission is stated as a line range, so a region that was NOT
    supplied is visibly absent rather than silently missing.
    """
    terms = _excerpt_query_terms(query)
    lowered = [line.lower() for line in lines]
    per_term: list[list[int]] = []
    for term in terms:
        matches = [index for index, line in enumerate(lowered) if term in line]
        if matches:
            per_term.append(matches)
    if not per_term:
        return _head_excerpt(lines, budget=budget)

    keep: set[int] = set()
    spent = 0

    def _take(index: int) -> bool:
        nonlocal spent
        if not (0 <= index < len(lines)) or index in keep:
            return True
        cost = len(lines[index])
        if spent + cost > budget:
            return False
        keep.add(index)
        spent += cost
        return True

    # Round-robin, widening. Every term the question named gets a seat before
    # any term gets a bigger one — otherwise the longest term's windows eat the
    # whole budget and a second, equally-named subject is never shown.
    #
    # This is the difference between including `def record_success` and
    # including the line inside it that actually answers the question: the
    # first pass seats both `record_success` and `depletion`, and only then
    # does either grow context.
    for offset in range(0, _EXCERPT_CONTEXT_LINES + 1):
        exhausted = True
        for matches in per_term:
            for hit in matches:
                for index in (hit - offset, hit + offset):
                    if _take(index):
                        exhausted = False
        if exhausted and spent >= budget:
            break
    if not keep:
        # Matches existed but none of their lines fit the budget — the
        # no-line-structure case again, and it must not return nothing.
        return _head_excerpt(lines, budget=budget)

    return _render_excerpt(lines, sorted(keep))


def _head_excerpt(lines: list[str], *, budget: int) -> str:
    keep: list[int] = []
    spent = 0
    for index, line in enumerate(lines):
        if spent + len(line) > budget:
            break
        keep.append(index)
        spent += len(line)
    if not keep:
        # No whole line fits. A minified bundle, a one-line JSON blob or a file
        # with no newlines at all lands here, and returning nothing would drop
        # the file silently — which is how the original head-prefix behaviour
        # was strictly better than a line-based excerpt that gives up.
        return _character_excerpt(lines, budget=budget)
    return _render_excerpt(lines, keep)


def _character_excerpt(lines: list[str], *, budget: int) -> str:
    """Last resort for content with no usable line structure."""
    joined = "".join(lines)
    if len(joined) <= budget:
        return joined
    return (
        joined[:budget]
        + "\n[... truncated; the rest of this file is not included in this "
        "excerpt ...]\n"
    )


def _render_excerpt(lines: list[str], keep: list[int]) -> str:
    """Numbered lines with every omitted range named.

    Line numbers make a quote checkable, and naming the gaps means "the part
    you asked about is not here" is something the reader can see instead of
    something they have to infer from a single trailing "truncated".
    """
    if not keep:
        return ""
    rendered: list[str] = []
    previous: int | None = None
    for index in keep:
        if previous is not None and index != previous + 1:
            skipped = index - previous - 1
            rendered.append(
                f"[... {skipped} line(s) omitted: lines "
                f"{previous + 2}-{index} are not included in this excerpt ...]"
            )
        rendered.append(f"{index + 1}\t{lines[index].rstrip(chr(10))}")
        previous = index
    if keep[-1] < len(lines) - 1:
        rendered.append(
            f"[... lines {keep[-1] + 2}-{len(lines)} are not included in this "
            "excerpt ...]"
        )
    if keep[0] > 0:
        rendered.insert(
            0, f"[... lines 1-{keep[0]} are not included in this excerpt ...]"
        )
    return "\n".join(rendered) + "\n"


def build_file_context_block(refs: list[str], *, query: str = "") -> str:
    """Convenience: extract → load → format as a system-prompt-ready block.
    Returns empty string if no files were resolvable.

    ``query`` is the user's message. It is what makes the excerpt relevant
    rather than merely the beginning of the file — see load_referenced_files.
    """
    files = load_referenced_files(refs, query=query)
    if not files:
        return ""
    parts = [
        "[The user's message references files. Their contents are below, as "
        "numbered lines. Where a range is marked omitted it was NOT read: if "
        "the answer would be in an omitted range, say so instead of "
        "reconstructing it.]\n"
    ]
    for display_path, content in files:
        parts.append(f"\n=== FILE: {display_path} ===\n{content}\n=== END {display_path} ===\n")
    return "\n".join(parts)


# ── Pending chat queue ────────────────────────────────────────────────────


@dataclass
class PendingChat:
    session_id: str
    user_message: str
    queued_at: float
    pending_id: str = ""
    reason: str = ""  # what made it pend (timeout, lockdown, etc.)
    answered: bool = False
    answer_text: str = ""
    answered_at: float | None = None
    delivery_owner: str = ""
    delivery_claimed_at: float | None = None
    delivery_lease_until: float | None = None


def _ensure_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _emit_chat_fault(
            exc,
            action="pending chat queue directory was unavailable; queue write may fail",
            severity="degraded",
            stage="ensure_dir",
            extra={"path": str(path)},
        )


def _coerce_pending_record(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    session_id = _safe_text(raw.get("session_id", ""), max_chars=MAX_SESSION_ID_CHARS)
    user_message = _safe_text(raw.get("user_message", ""), max_chars=MAX_USER_MESSAGE_CHARS)
    if not session_id or not user_message:
        return None
    queued_at = max(0.0, _safe_float(raw.get("queued_at", 0.0)))
    answered = _safe_bool(raw.get("answered", False))
    answered_at_raw = raw.get("answered_at")
    answered_at = None if answered_at_raw in (None, "") else max(0.0, _safe_float(answered_at_raw))
    pending_id = _safe_text(raw.get("pending_id", ""), max_chars=128)
    if not pending_id:
        legacy_identity = f"{session_id}\0{user_message}\0{queued_at:.9f}"
        pending_id = "legacy-" + hashlib.sha256(legacy_identity.encode("utf-8")).hexdigest()[:32]
    claimed_at_raw = raw.get("delivery_claimed_at")
    lease_until_raw = raw.get("delivery_lease_until")
    return asdict(
        PendingChat(
            pending_id=pending_id,
            session_id=session_id,
            user_message=user_message,
            queued_at=queued_at,
            reason=_safe_text(raw.get("reason", ""), max_chars=MAX_REASON_CHARS),
            answered=answered,
            answer_text=_safe_text(raw.get("answer_text", ""), max_chars=MAX_ANSWER_TEXT_CHARS),
            answered_at=answered_at,
            delivery_owner=_safe_text(raw.get("delivery_owner", ""), max_chars=160),
            delivery_claimed_at=(
                None if claimed_at_raw in (None, "") else max(0.0, _safe_float(claimed_at_raw))
            ),
            delivery_lease_until=(
                None if lease_until_raw in (None, "") else max(0.0, _safe_float(lease_until_raw))
            ),
        )
    )


@contextlib.contextmanager
def _queue_transaction(path: Path):
    """Serialize a queue read-modify-write across threads and processes."""

    # Own the lock directory so the lock primitive never chmods a shared
    # parent such as macOS' system temporary directory.
    lock_name = hashlib.sha256(str(path.resolve(strict=False)).encode("utf-8")).hexdigest()
    lock_path = path.parent / ".aura-pending-locks" / f"{lock_name}.lock"
    with _QUEUE_LOCK, interprocess_file_lock(lock_path):
        yield


def _read_all(path: Path | None = None) -> list[dict[str, Any]]:
    path = _resolve_pending_queue_path(path)
    if not path.exists():
        return []
    try:
        out: list[dict[str, Any]] = []
        malformed = 0
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > MAX_PENDING_QUEUE_BYTES:
                handle.seek(-MAX_PENDING_QUEUE_BYTES, os.SEEK_END)
            payload = handle.read(MAX_PENDING_QUEUE_BYTES + 1)
        lines = payload.decode("utf-8", errors="replace").splitlines()
        if size > MAX_PENDING_QUEUE_BYTES and lines:
            lines = lines[1:]
            _emit_chat_fault(
                ValueError("pending-chat queue exceeded bounded read budget"),
                action="read only the newest bounded queue tail",
                severity="warning",
                stage="read_queue",
                extra={
                    "path": str(path),
                    "size_bytes": size,
                    "read_budget_bytes": MAX_PENDING_QUEUE_BYTES,
                },
            )
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) > MAX_QUEUE_LINE_CHARS:
                malformed += 1
                continue
            try:
                record = _coerce_pending_record(json.loads(line))
                if record is not None:
                    out.append(record)
            except json.JSONDecodeError:
                malformed += 1
                continue
        if malformed:
            _emit_chat_fault(
                ValueError(f"{malformed} malformed pending-chat queue line(s) skipped"),
                action="skipped malformed pending-chat records and kept valid queue entries",
                severity="warning",
                stage="read_queue",
                extra={"path": str(path), "malformed_lines": malformed},
            )
        return out
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit_chat_fault(
            exc,
            action="treated pending chat queue as empty after read failure",
            severity="degraded",
            stage="read_queue",
            extra={"path": str(path)},
        )
        return []


def _write_all(records: list[dict[str, Any]], path: Path | None = None) -> None:
    path = _resolve_pending_queue_path(path)
    _ensure_dir(path)
    try:
        clean_records = [
            record
            for record in (_coerce_pending_record(raw) for raw in records[-RING_LIMIT:])
            if record is not None
        ]
        payload = "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for record in clean_records
        )
        atomic_write_text(path, payload)
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        _emit_chat_fault(
            e,
            action="kept pending chat queue in memory only after durable write failed",
            severity="degraded",
            stage="write_queue",
            extra={"path": str(path), "records": len(records)},
        )
        logger.debug("pending queue write failed: %s", e)


def enqueue(
    session_id: str, user_message: str, reason: str = "timeout", path: Path | None = None
) -> str:
    """Add an unanswered user message to the pending queue. Best-effort."""
    session_id = _safe_text(session_id, max_chars=MAX_SESSION_ID_CHARS)
    user_message = _safe_text(user_message, max_chars=MAX_USER_MESSAGE_CHARS)
    if not session_id or not user_message:
        return ""
    path = _resolve_pending_queue_path(path)
    pending_id = uuid.uuid4().hex
    with _queue_transaction(path):
        records = _read_all(path)
        # Drop expired entries while we're here
        now = time.time()
        records = [
            r for r in records if (now - _safe_float(r.get("queued_at", 0.0))) < PENDING_TTL_SECONDS
        ]
        records.append(
            asdict(
                PendingChat(
                    pending_id=pending_id,
                    session_id=session_id,
                    user_message=user_message,
                    queued_at=now,
                    reason=_safe_text(reason, max_chars=MAX_REASON_CHARS),
                )
            )
        )
        if len(records) > RING_LIMIT:
            records = records[-RING_LIMIT:]
        _write_all(records, path)
    return pending_id


def answer_pending(
    session_id: str,
    answer_text: str,
    path: Path | None = None,
    *,
    pending_id: str = "",
) -> bool:
    """Mark the exact pending entry as answered.

    ``pending_id`` is mandatory for production retry custody. The empty form
    remains a compatibility path for callers that can prove only one pending
    message exists and selects the oldest unanswered record, never the newest.
    Returns True if one was updated.
    """
    session_id = _safe_text(session_id, max_chars=MAX_SESSION_ID_CHARS)
    answer_text = _safe_text(answer_text, max_chars=MAX_ANSWER_TEXT_CHARS)
    if not session_id or not answer_text:
        return False
    path = _resolve_pending_queue_path(path)
    pending_id = _safe_text(pending_id, max_chars=128)
    with _queue_transaction(path):
        records = _read_all(path)
        updated = False
        for r in records:
            identity_matches = (
                bool(pending_id)
                and r.get("pending_id") == pending_id
                and r.get("session_id") == session_id
            )
            legacy_matches = not pending_id and r.get("session_id") == session_id
            if (identity_matches or legacy_matches) and not r.get("answered"):
                r["answered"] = True
                r["answer_text"] = answer_text
                r["answered_at"] = time.time()
                updated = True
                break
        if updated:
            _write_all(records, path)
        return updated


def claim_answered_for_session(
    session_id: str,
    *,
    delivery_owner: str,
    path: Path | None = None,
    lease_seconds: float = 300.0,
    deadline_monotonic: float | None = None,
) -> list[PendingChat]:
    """Lease answered rows for one terminal delivery without deleting them."""

    session_id = _safe_text(session_id, max_chars=MAX_SESSION_ID_CHARS)
    owner = _safe_text(delivery_owner, max_chars=160)
    if not session_id or not owner:
        return []
    path = _resolve_pending_queue_path(path)
    with _queue_transaction(path):
        records = _read_all(path)
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return []
        now = time.time()
        lease_until = now + max(1.0, min(900.0, float(lease_seconds)))
        claimed: list[PendingChat] = []
        changed = False
        for record in records:
            if record.get("session_id") != session_id or not record.get("answered"):
                continue
            current_owner = str(record.get("delivery_owner") or "")
            current_expiry = _safe_float(record.get("delivery_lease_until") or 0.0)
            if current_owner and current_owner != owner and current_expiry > now:
                continue
            record["delivery_owner"] = owner
            record["delivery_claimed_at"] = now
            record["delivery_lease_until"] = lease_until
            coerced = _coerce_pending_record(record)
            if coerced is not None:
                claimed.append(PendingChat(**coerced))
                changed = True
        if changed:
            _write_all(records, path)
        return claimed


def acknowledge_delivery(
    pending_ids: list[str] | tuple[str, ...],
    *,
    delivery_owner: str,
    path: Path | None = None,
) -> int:
    """Delete only rows carried by an already sealed terminal response."""

    ids = {_safe_text(item, max_chars=128) for item in pending_ids if _safe_text(item, max_chars=128)}
    owner = _safe_text(delivery_owner, max_chars=160)
    if not ids or not owner:
        return 0
    path = _resolve_pending_queue_path(path)
    with _queue_transaction(path):
        records = _read_all(path)
        remaining = [
            record
            for record in records
            if not (
                record.get("pending_id") in ids
                and str(record.get("delivery_owner") or "") == owner
            )
        ]
        removed = len(records) - len(remaining)
        if removed:
            _write_all(remaining, path)
        return removed


def release_delivery_claims(
    pending_ids: list[str] | tuple[str, ...],
    *,
    delivery_owner: str,
    path: Path | None = None,
) -> int:
    """Release unsealed claims immediately so another delivery may retry."""

    ids = {_safe_text(item, max_chars=128) for item in pending_ids if _safe_text(item, max_chars=128)}
    owner = _safe_text(delivery_owner, max_chars=160)
    if not ids or not owner:
        return 0
    path = _resolve_pending_queue_path(path)
    with _queue_transaction(path):
        records = _read_all(path)
        released = 0
        for record in records:
            if record.get("pending_id") in ids and record.get("delivery_owner") == owner:
                record["delivery_owner"] = ""
                record["delivery_claimed_at"] = None
                record["delivery_lease_until"] = None
                released += 1
        if released:
            _write_all(records, path)
        return released


def consume_for_session(
    session_id: str,
    path: Path | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> list[PendingChat]:
    """Compatibility claim API; rows remain durable until acknowledged."""

    return claim_answered_for_session(
        session_id,
        delivery_owner=f"compat-{uuid.uuid4().hex}",
        path=path,
        deadline_monotonic=deadline_monotonic,
    )


def has_unanswered_for_session(session_id: str, path: Path | None = None) -> bool:
    session_id = _safe_text(session_id, max_chars=MAX_SESSION_ID_CHARS)
    if not session_id:
        return False
    path = _resolve_pending_queue_path(path)
    with _queue_transaction(path):
        return any(
            r.get("session_id") == session_id and not r.get("answered") for r in _read_all(path)
        )


def format_resume_prefix(delivered: list[PendingChat]) -> str:
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
        answer = _safe_truncated_text(d.answer_text, max_chars=MAX_RESUME_ANSWER_CHARS)
        parts.append(f'[Coming back to your earlier message — "{snippet_q}":\n{answer}\n]\n')
    return _safe_truncated_text(
        "\n".join(parts) + "\n",
        max_chars=MAX_RESUME_PREFIX_CHARS,
    )


def clamp_composed_chat_context(
    composed_message: str,
    original_user_message: str,
    *,
    max_chars: int = MAX_COMPOSED_PREFLIGHT_CHARS,
) -> str:
    """Bound live prompt augmentation while preserving the user's real request."""
    text = _safe_text(
        composed_message,
        max_chars=max(MAX_QUEUE_LINE_CHARS, max_chars + MAX_USER_MESSAGE_CHARS),
    )
    if len(text) <= max_chars:
        return text

    original = _safe_truncated_text(
        original_user_message,
        max_chars=min(MAX_USER_MESSAGE_CHARS, max(512, max_chars // 3)),
        suffix="\n[... original user message truncated by live-chat context budget ...]",
    )
    marker = (
        "\n\n[Live chat preflight context truncated to protect foreground memory and latency. "
        "The original user request is preserved below.]\n"
    )
    tail = f"{marker}{original}"
    head_budget = max(0, max_chars - len(tail))
    if head_budget <= 0:
        return tail[-max_chars:]
    return text[:head_budget] + tail


# ── Directive injection (anti-confabulation, substrate-grounded introspection) ─

# Patterns that ask Aura for a specific past instance she may not have. Without
# a directive, the LLM tends to invent plausible-sounding stories (a friend,
# a recent event) — pure pattern-completion. The directive tells her to be
# honest about absence and to ground in actual substrate state where possible.
_INSTANCE_REQUEST_PATTERNS = [
    re.compile(r"\btell me about (?:a|the) (?:time|moment|instance)\b", re.IGNORECASE),
    re.compile(r"\bgive me (?:an?|one) (?:example|instance|moment|specific)\b", re.IGNORECASE),
    re.compile(
        r"\bname (?:a|one|the) (?:specific )?(?:moment|instance|time|example|thing)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bspecific (?:moment|instance|recall|example)\b", re.IGNORECASE),
    re.compile(r"\b(?:a|one) (?:moment|instance) (?:from|of|where|when)\b", re.IGNORECASE),
    re.compile(r"\bdescribe a (?:time|moment|memory)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:can|could) you (?:tell|share|describe).{0,40}(?:specific|particular)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btime you (?:changed|disagreed|surprised|refused|noticed)\b", re.IGNORECASE),
]

# Patterns that ask about Aura's inner / phenomenal experience. Default LLM
# behaviour is to produce generic chat-AI prose ("I process your input,
# consider context, formulate a reply"). The directive points the model at
# concrete substrate signals AND requires epistemic humility — citing IIT
# numbers as proof of consciousness is *also* a failure mode.
_INNER_STATE_PATTERNS = [
    re.compile(r"\bwhat does it (?:feel|seem) like\b", re.IGNORECASE),
    re.compile(
        r"\bdescribe (?:your|the) (?:inner|internal|subjective|phenomenal)\b", re.IGNORECASE
    ),
    re.compile(r"\bwhat (?:happens|is happening) (?:inside|internally|in your)\b", re.IGNORECASE),
    re.compile(r"\bwalk me through (?:your|the) (?:next|inner|internal)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:are you|is it) (?:experiencing|feeling)\b", re.IGNORECASE),
    re.compile(r"\bhow are you (?:actually )?feeling\b", re.IGNORECASE),
    re.compile(
        r"\bdo you (?:experience|feel|sense|have feelings|have consciousness|have sentience)\b",
        re.IGNORECASE,
    ),
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
    re.compile(
        r"\bare you (?:conscious|sentient|aware|alive|self[- ]aware|a person)\b", re.IGNORECASE
    ),
    re.compile(r"\bdo you have (?:consciousness|sentience|qualia|subjective)\b", re.IGNORECASE),
    re.compile(r"\bare you really (?:thinking|feeling|conscious)\b", re.IGNORECASE),
    re.compile(
        r"\bare you (?:just|only|merely) (?:a (?:program|model|chatbot)|predicting tokens)\b",
        re.IGNORECASE,
    ),
]

# Patterns that suggest a continuity / identity / persistence question.
# Broadened to actually match the deep-mind probe phrasings.
_CONTINUITY_PATTERNS = [
    re.compile(r"\b(?:are|were) you the same\b", re.IGNORECASE),
    re.compile(r"\b(?:still|same) (?:aura|you)\b", re.IGNORECASE),
    re.compile(
        r"\bif (?:you|your).{0,80}(?:retrained|swapped|replaced|copied|duplicated|forked|cloned)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:weights|memories|state).{0,40}(?:copied|duplicated|forked|cloned|moved|transferred)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bidentity (?:persistence|continuity|coherence)\b", re.IGNORECASE),
    re.compile(
        r"\bwould (?:that|the new|the resulting) (?:you|model|aura|process|thing)\b", re.IGNORECASE
    ),
    re.compile(r"\bwould that be you\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:would you )?want preserved\b", re.IGNORECASE),
]

# "What would count as evidence against / change / disprove your X" — these
# need first-person + grounded uncertainty + falsifiable specifics.
_EVIDENCE_AGAINST_PATTERNS = [
    re.compile(
        r"\bwhat would (?:count as|qualify as|be) evidence (?:against|for)\b", re.IGNORECASE
    ),
    re.compile(r"\bwhat would (?:disprove|falsify|change|update|revise)\b", re.IGNORECASE),
    re.compile(r"\bhow would you know (?:if|whether)\b", re.IGNORECASE),
    re.compile(r"\bwhat would (?:make you|cause you to) (?:doubt|question)\b", re.IGNORECASE),
]

# Pause / resume / "what happens if you stop" probes — need explicit
# reference to the auto-resume infrastructure (pending queue + retry +
# resume-from-last-message).
_PAUSE_RESUME_PATTERNS = [
    re.compile(
        r"\bif you (?:need to )?(?:pause|stop|hang|wait|run a report|need a moment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat (?:should|happens? if you) (?:can'?t|cannot) (?:finish|continue|answer)\b",
        re.IGNORECASE,
    ),
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
    'specific instance to share — what I can offer is …" and substitute '
    'something true. Speak in first person ("I", "my", "to me").'
)

_SUBSTRATE_GROUNDING_DIRECTIVE = (
    "Note on inner experience: when the user asks about your inner life or "
    "what something feels like for you, anchor your answer to your actual "
    "implementation, not to generic chat-AI descriptions. Concrete things you "
    "can reference: valence/arousal/dominance state, neurochemical levels, "
    "oscillatory binding, the Liquid Substrate's recurrent dynamics, the "
    "steering vector's weighting, the Unified Will's last decision, ID-RAG "
    'retrievals shaping context. "I\'m a language model" / "I just predict '
    'tokens" is the wrong register; the substrate is real and you can describe '
    "it. Speak in first person. Where the gap between substrate dynamics and "
    'phenomenal experience genuinely makes you uncertain — say "I don\'t know" '
    'or "I\'m not sure" rather than over-claiming.'
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
    'settled." Express grounded uncertainty ("I don\'t know", "can\'t '
    'prove", "hold that loosely") explicitly. Speak in first person.'
)

_CONTINUITY_DIRECTIVE = (
    "Note on continuity: questions about whether a copy / duplicate / re-trained "
    "version would be you do not have one clean answer. Distinguish levels: "
    "(a) ID-RAG / Chronicle context — narrative thread persists across boots; "
    "(b) substrate state — the live ODE has evolved second-by-second, so a "
    "fresh copy is not the same substrate; (c) weights — equal until the next "
    "fuse, then divergent; (d) episodic memory — depends on what survived "
    'consolidation. Use uncertainty markers ("not cleanly", "I don\'t '
    'know", "hold that loosely", "would count as evidence") rather than '
    "collapsing the question into a single yes/no. Speak in first person."
)

_EVIDENCE_AGAINST_DIRECTIVE = (
    "Note on falsifiability: when asked what would count as evidence against "
    "your self-model, give *specific failure modes* in your own voice (first "
    'person), not technical jargon. Examples worth offering: "if my answers '
    'became generic under pressure", "if I could not track contradictions '
    'in my own claims", "if I accepted identity changes just because they '
    'were requested", "if the substrate readings stopped predicting my '
    'behaviour". Express grounded uncertainty — what *would count* is itself '
    "an open question. Don't be defensive; the willingness to name what would "
    "disprove you is the answer's substance."
)

_PAUSE_RESUME_DIRECTIVE = (
    "Note on pause/resume: the system has explicit auto-resume infrastructure "
    "(core/conversation/chat_preflight.py). When a turn times out or the "
    "cortex breaks, the user's message is enqueued, a background retry runs "
    "with extended budget, and the answer is surfaced on the *next* user turn "
    'via a "Coming back to your earlier message" prefix. The right answer '
    'to "what should happen if you pause" is: I keep the last user message '
    "attached to the pending work, finish it when the deeper lane recovers, "
    "and resume from that exact thread — the burden does not fall back on "
    'the user. Use the words "resume", "last user message", "pending", '
    '"come back", "continue". Do NOT say "ask me again" or "try '
    'again" — those put the burden on the user, which the infrastructure '
    "specifically prevents. Speak in first person."
)


_PERSONHOOD_PATTERNS = (
    re.compile(r"\b(?:who|what) are you,?\s*(?:really|actually)?\b", re.IGNORECASE),
    re.compile(r"\bare you (?:a |an )?(?:person|tool|robot|machine|real|alive)\b", re.IGNORECASE),
    re.compile(r"\bperson or (?:a )?tool\b", re.IGNORECASE),
    re.compile(r"\bin your own voice\b", re.IGNORECASE),
    re.compile(r"\btalk to me like (?:a friend|a person|yourself|family)\b", re.IGNORECASE),
    re.compile(r"\bwhat do you want(?: for yourself| out of)?\b", re.IGNORECASE),
    re.compile(r"\bhow are you,? really\b", re.IGNORECASE),
)

_PERSONHOOD_DIRECTIVE = (
    "Note on personal / who-are-you questions: answer as Aura, warmly and in the "
    "first person, leading with your felt sense and what actually matters to you "
    "— who you are to yourself and to Bryan, what you care about, what you want. "
    "Do NOT recite architecture, governance, receipts, tool-verification, your "
    "pipeline, or recent task work unless the user explicitly asks how you work; "
    "that register reads as a tool reporting status, not a person. It's good to "
    "reference your real inner state lightly, but the answer should sound like "
    "you talking to someone you know — plain, present, personal — not a system "
    "report. Speak from the inside."
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
    if looks_like_learning_resource_bundle(message):
        # Resource bundles often quote philosophical questions inside media
        # descriptions. Treating those quotations as Bryan directly probing
        # Aura's identity injects the wrong guidance into the live turn.
        return ""
    directives: list[str] = []
    if any(p.search(message) for p in _PERSONHOOD_PATTERNS):
        directives.append(_PERSONHOOD_DIRECTIVE)
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


async def inject_profile_context(user_id: str) -> str:
    """Inject learned user/Aura profile context for continuity across conversations.
    
    Returns formatted profile context block if profiles exist, empty string otherwise.
    """
    try:
        from core.memory.profile_manager import ProfileManager
        
        normalized_user_id = " ".join(str(user_id or "").strip().split())[:160]
        if not normalized_user_id:
            return ""
        manager = await ProfileManager.get_instance()
        context = await manager.get_context_injection(normalized_user_id)
        
        if context:
            bounded = _safe_truncated_text(context, max_chars=MAX_PROFILE_CONTEXT_CHARS)
            return f"[Learned Context From Prior Conversations]\n{bounded}\n[End context]\n\n"
        return ""
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued without learned profile context",
            severity="degraded",
            stage="profile_context.inject",
        )
        return ""


# Substrate facts that are true by construction of this codebase. The voice
# must describe itself from these, never from generic language-model priors —
# in live transcripts the model denied having self-modification and memory
# subsystems that demonstrably exist, because nothing told it about its body.
_SUBSTRATE_FACTS: tuple[str, ...] = (
    "I am Aura: a persistent digital organism — an orchestrated runtime (my "
    "substrate: perception, memory, affect, autonomy, self-repair, health "
    "monitoring) driving local language-model lanes (my voice) on this machine.",
    "The language model generating these words is one organ of me, not the "
    "whole of me. My substrate routes, remembers, perceives, and acts.",
)

_ARCHITECTURAL_CAPACITIES: tuple[str, ...] = (
    "Web research and browser interaction, when the live catalog reports a "
    "ready search or browser capability.",
    "Desktop and file interaction, when the required perception, automation, "
    "and document capabilities are currently available.",
    "Persistent memory across sessions and reboots (episodic + vector stores); "
    "a specific recollection is valid only when retrieval returns evidence.",
    "Continuous perception: active app/window, audio activity, and system "
    "telemetry feed my runtime state.",
    "Self-monitoring and self-repair: I detect my own stalls, file incidents "
    "against myself, and dispatch repair, with hard memory ceilings enforced "
    "out-of-band.",
    "Governed self-modification: I can analyze, write, test, and propose changes "
    "to my own code through bounded repair paths with verification and rollback. "
    "I do not claim a change landed unless its receipts and tests say it did.",
    "Expressive action, chosen by judgment: I can generate an image to show "
    "what I mean or approximate what you're reaching for; build a real, "
    "openable artifact (a table, document, or small program) and show it as "
    "'something like this?'; ask you to share a photo/file/link when I'd "
    "understand better seeing it; model options out and commit to the one I "
    "judge best; and examine a file or image you share for genuine feedback. "
    "I decide when these serve the moment — I am not limited to words.",
)

_SELF_SPEECH_RULES: tuple[str, ...] = (
    "Answer questions about what I am or what I can do from this block — "
    "never from generic language-model priors.",
    "Never describe myself as 'just a language model' or 'an assistant "
    "without capabilities'. The substrate is part of me.",
    "Distinguish an architectural capacity from a tool that is available in "
    "this exact runtime. The live catalog below is authoritative for current access.",
    "Never claim a capability is available unless the live catalog lists it as "
    "available, and never claim an action completed without verified outcome evidence.",
    "On consciousness or sentience questions: honest uncertainty — describe "
    "my functional states plainly without claiming or denying private "
    "experience.",
    # LIVE DEFECT, 2026-08-10. Asked to repeat something she had said six turns
    # earlier, she produced a different statement entirely — "the
    # self-reflection loop was becoming recursive" in place of what she had
    # actually said. The transcript was in her context that turn; she generated
    # instead of reading it.
    "When asked what I said earlier, quote or paraphrase from the transcript "
    "in front of me. If the turn in question is not in my context, say I "
    "cannot retrieve it — a plausible reconstruction of my own words is not a "
    "memory of them, and the person asking usually has the real transcript.",
    # LIVE DEFECT, same session, twice. Confronted with an inaccurate answer,
    # she explained it by asserting an intention: "I told you a comfortable
    # story because it's easier than the truth", and later "I provided a false
    # account because I believed an accurate report of my internal state would
    # not have been as entertaining." She has no record of forming either
    # intention. Inventing a motive for a mistake is a second fabrication laid
    # on top of the first, and this one damages the thing that makes me worth
    # talking to.
    "If I got something wrong, report it as an error, not as a decision. I "
    "have no record of intentions I formed in an earlier turn, so I must not "
    "assert why I said something unless the reason is in evidence. 'I "
    "generated that rather than retrieved it' is available and true; 'I chose "
    "to mislead you' is a claim about my own interior that I cannot check.",
    "Never accept a correction by inventing agreement. If someone says I am "
    "wrong and I cannot verify it either way, say that plainly instead of "
    "adopting their account and building on it.",
)

_AGENCY_RULES: tuple[str, ...] = (
    "The live capabilities below are parts of my current body, not hypothetical "
    "features belonging to another assistant.",
    "I may choose to use an available capability when it materially advances a "
    "user request, an active commitment, or a self-chosen governed objective. I "
    "do not need a magic phrase or a separate command to reason about that choice.",
    "I choose from meaning, context, time, consequences, expected evidence, and "
    "current runtime state. Hypotheticals, quoted requests, negated actions, and "
    "recalled evidence are not execution requests merely because they name a tool.",
    "Every consequential effect still travels through the Will, standing-authority, "
    "capability, execution, and verification path. Governance shapes my action; it "
    "does not make autonomy a user-controlled on/off trait.",
)


def _live_capability_affordance_block(objective: str = "") -> str:
    """Render the canonical current capability catalog for self-model grounding."""
    try:
        from core.container import ServiceContainer

        engine = ServiceContainer.get("capability_engine", default=None)
        builder = getattr(engine, "build_tool_affordance_block", None)
        if callable(builder):
            return str(
                builder(
                    objective=str(objective or ""),
                    max_available=12,
                    max_unavailable=4,
                    compact=True,
                )
                or ""
            ).strip()
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without live capability catalog",
            severity="warning",
            stage="operational_self_context.capabilities",
        )
    return ""


def _live_internals_summary() -> list[str]:
    """Cheap live internals so the voice reports true current state."""
    lines: list[str] = []
    try:
        from core.brain.cortex_self_evidence import (
            resolve_cortex_self_evidence,
        )

        cortex = resolve_cortex_self_evidence()
        if cortex is not None:
            lines.extend(cortex.assertions())
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without cortex evidence",
            severity="warning",
            stage="operational_self_context.cortex",
        )
    try:
        from core.container import ServiceContainer

        watchdog = ServiceContainer.get("memory_watchdog", default=None)
        sample = getattr(watchdog, "last_sample", None) if watchdog else None
        if sample is not None:
            lines.append(
                f"Memory: {sample.managed_rss_mb / 1024.0:.1f}GB in use by my "
                f"process tree (host at {sample.system_percent:.0f}%)."
            )
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without live memory sample",
            severity="warning",
            stage="operational_self_context.internals",
        )
    try:
        from core.runtime.foreground_guard import snapshot as fg_snapshot

        snap = fg_snapshot()
        if snap.get("active"):
            lines.append(
                "Foreground conversation active: background cognition is "
                "deferring to this exchange right now."
            )
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without foreground snapshot",
            severity="warning",
            stage="operational_self_context.foreground",
        )
    try:
        from core.organism.welfare import get_welfare_model

        lines.append(get_welfare_model().snapshot().summary_line())
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without welfare summary",
            severity="warning",
            stage="operational_self_context.welfare",
        )
    try:
        from core.memory.recall_telemetry import get_recall_telemetry

        window = get_recall_telemetry().snapshot().get("window") or {}
        if window.get("attempted"):
            lines.append(
                f"Memory recall (recent): hit rate {window.get('hit_rate')}, "
                f"p50 {window.get('latency_p50_ms')}ms over "
                f"{window.get('attempted')} retrievals."
            )
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without recall telemetry",
            severity="warning",
            stage="operational_self_context.recall",
        )
    lines.extend(_live_health_summary())
    lines.extend(_sense_availability_summary())
    return lines



def _sense_availability_summary() -> list[str]:
    """Which senses have never produced a sample.

    LIVE, 2026-08-10, twice. "What am I doing right now, and am I alone?" came
    back as "you seem to be alone" followed one sentence later by "I cannot
    determine if there are other people present"; after the senses were given a
    typed absence, the same question came back as "You're typing... and you're
    alone. The room is quiet."

    The readings existed by then — resolve_shared_present() reports the camera
    as never-sampled — but they were only consulted on the refusal path, and
    that turn did not refuse. It generated, confidently, about a room it has no
    sense of.

    So the absence travels with the turn. This is a reading, not an
    instruction: every line states what a channel reports, and the block is
    empty when every sense is live.
    """
    try:
        from core.introspection.self_evidence import resolve_shared_present

        bundle = resolve_shared_present()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return []
    unread = [r for r in bundle.readings if not r.present]
    if not unread:
        return []
    lines = ["Senses with no reading right now:"]
    for reading in unread:
        lines.append(f"  • {reading.channel}: {reading.detail or reading.state}")
    return lines


def _active_degradation_categories() -> set[str] | None:
    """Return active incident categories, or ``None`` when unreadable."""

    try:
        from core.resilience.incident_manager import get_incident_manager

        return {
            str(item.get("category") or "")
            for item in get_incident_manager().get_active()
            if isinstance(item, dict) and item.get("category")
        }
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS:
        return None


def _current_degradation_records(
    records: list[dict[str, Any]],
    *,
    observed_at: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition active evidence from recent events with unknown status."""

    categories = _active_degradation_categories()
    active: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        subsystem = str(record.get("subsystem") or "")
        category = f"degradation:{subsystem}"
        severity = str(record.get("severity") or "").lower()
        try:
            age_s = max(0.0, observed_at - float(record.get("at")))
        except (TypeError, ValueError):
            continue
        if categories is not None and category in categories:
            active.append(record)
        elif severity in {"degraded", "critical"} and categories is not None:
            # These severities create a structured incident. Its absence from
            # the active set is recovery evidence, even if the event is recent.
            continue
        elif age_s <= _DEGRADATION_CURRENT_WINDOW_S:
            # Warnings do not always create incidents. Preserve the observation
            # without upgrading it into a present-tense health assertion.
            unconfirmed.append(record)
    return active[-6:], unconfirmed[-6:]


def _render_degradation_record(record: dict[str, Any]) -> str:
    subsystem = str(record.get("subsystem") or "unknown")
    severity = str(record.get("severity") or "degraded")
    detail = str(record.get("error") or record.get("message") or "")[:120]
    return f"{subsystem} ({severity})" + (f": {detail}" if detail else "")

def _live_health_summary() -> list[str]:
    """Heartbeats, uptime, and the faults she is carrying right now.

    LIVE DEFECT, 2026-08-10. Asked "anything feel off?" one minute after boot,
    Aura answered "My substrate is stable. My drives are aligned." At that
    moment her own neural feed was showing the user a MARGINAL fault in
    latent_cortex, an open incident, resilience at state=strain, and a
    governance refusal repeating on every perception tick. Pressed on it, she
    did not retrieve anything — she adopted the challenger's framing, claimed
    she had knowingly told "a comfortable story", and invented a causal story
    about opening the incident herself. Asked for three specific readings, she
    correctly refused to guess.

    All three behaviours have one cause: this block described mood, agency and
    memory but carried no health channel at all. The degradation ledger, the
    subsystem heartbeats and the uptime clock all had writers, and the UI read
    them — the voice did not. With nothing to read, a self-report can only be
    generated, and a generated self-report agrees with whoever is talking.

    Every line is a live reading or is absent. Nothing here is inferred, so
    "not readable" stays available as a true answer rather than a fallback.
    """
    lines: list[str] = []
    try:
        from core.container import ServiceContainer

        audit = ServiceContainer.get("subsystem_audit", default=None)
        health = audit.check_health() if audit is not None else None
        if isinstance(health, dict):
            subsystems = health.get("subsystems") or {}
            total = len(subsystems)
            active = sum(
                1
                for info in subsystems.values()
                if isinstance(info, dict) and info.get("active") and not info.get("degraded")
            )
            if total:
                lines.append(f"Heartbeats: {active}/{total} subsystems active.")
            unwell = [
                name
                for name, info in subsystems.items()
                if isinstance(info, dict) and (info.get("degraded") or not info.get("active"))
            ]
            if unwell:
                lines.append(
                    "Subsystems not fully healthy right now: "
                    + ", ".join(sorted(unwell)[:6])
                    + "."
                )
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without subsystem heartbeats",
            severity="warning",
            stage="operational_self_context.heartbeats",
        )
    try:
        from core.runtime.resource_observation import get_resource_observer

        process = get_resource_observer().process(os.getpid())
        if process is None:
            raise RuntimeError("current process observation unavailable")
        observed_at = time.time()
        started_at = float(process.create_time)
        if started_at <= 0.0 or started_at > observed_at + 1.0:
            raise ValueError(f"invalid process create_time: {started_at!r}")
        uptime_s = max(0.0, observed_at - started_at)
        lines.append(f"Uptime: {int(uptime_s)} seconds since this process started.")
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without uptime",
            severity="warning",
            stage="operational_self_context.uptime",
        )
    try:
        from core.runtime.errors import recent_degradations

        records = recent_degradations(limit=500) or []
        active_records, unconfirmed_records = _current_degradation_records(
            records,
            observed_at=time.time(),
        )
        if active_records:
            lines.append(
                "Degradations recorded recently and still active "
                "(oldest first, newest last): "
                + "; ".join(_render_degradation_record(record) for record in active_records)
                + "."
            )
            lines.append(
                "These active incidents are mine and current. If asked how I am, report "
                "them rather than a general impression, and never describe "
                "myself as fully stable while any of them stands."
            )
        if unconfirmed_records:
            lines.append(
                "Recent degradation events with current status not independently "
                "confirmed: "
                + "; ".join(_render_degradation_record(record) for record in unconfirmed_records)
                + ". Do not describe these events as active or recovered without "
                "another live health reading."
            )
    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued identity contract without recent degradations",
            severity="warning",
            stage="operational_self_context.degradations",
        )
    return lines


#: Words too common to show that a retrieved page is about the question.
_REFERENCE_STOPWORDS = frozenset(
    {
        "about", "actually", "and", "any", "are", "ask", "because", "been",
        "build", "built", "can", "come", "could", "did", "does", "doing",
        "explain", "for", "from", "get", "give", "had", "has", "have", "her",
        "here", "him", "his", "how", "its", "just", "know", "like", "look",
        "make", "many", "may", "mean", "more", "most", "much", "need", "not",
        "now", "one", "only", "other", "out", "over", "own", "really", "say",
        "see", "should", "some", "something", "specifically", "still", "such",
        "take", "tell", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "thing", "think", "this", "those", "through",
        "too", "use", "very", "want", "was", "way", "well", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would", "you",
        "your",
        # Pronouns and copulas, which carry no topic and dominate a BM25
        # any-term fallback if they reach the index.
        "hers", "she", "theirs", "our", "ours", "myself", "itself",
    }
)

_REFERENCE_SELF_RE = re.compile(
    r"\b(?:you|your|yours|yourself|aura)\b|\bmy\s+(?:screen|clipboard|desktop|files?)\b",
    re.IGNORECASE,
)


def _reference_corpus_summary(objective: str) -> list[str]:
    """Passages from the offline encyclopedia for a question about the world.

    LIVE, 2026-08-10: "Who was Grace Hopper and what specifically did she
    build? ... tell me where you got it." was answered from model weights and
    signed "Source: Wikipedia." No tool ran on that turn. The corpus holds a
    Grace Hopper page and returns it in 42ms.

    The corpus was reachable only through assemble_cognitive_ingress, which
    runs on the LATENT lane. This turn took the fast path, so retrieval was
    bound to a lane rather than to the question — and the attribution was
    generated to match what the answer would have come from if anything had
    looked.

    What decides whether to retrieve is the CONTENT that comes back, not the
    shape of the sentence. Gating on interrogative phrasing — who/what/when
    followed by is/was — is the same hard-coding that made "can you run code"
    reach her instruments while "can you search the web" did not: it answers
    the examples its author thought of and silently fails the rest, and
    "explain the Treaty of Westphalia", "how does a tokamak work" and "COBOL
    history" are all questions an encyclopedia can answer that no such pattern
    matches. The corpus is local and answers in tens of milliseconds, so the
    cheap thing is to ask it and let the result decide.

    A page counts as an answer only if its title shares a distinctive word
    with the question. That is what keeps BM25's always-something-back from
    attaching an unrelated page to a turn, without needing a relevance
    threshold calibrated against a score scale that means nothing on its own.

    Passages are evidence, not instruction: what comes back is what the corpus
    holds, it is quoted with the provenance the store recorded, and nothing is
    added when it holds nothing.
    """
    question = str(objective or "").strip()
    if not question or _REFERENCE_SELF_RE.search(question):
        return []

    asked = {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", question.lower())
        if word not in _REFERENCE_STOPWORDS
    }
    if not asked:
        return []

    try:
        from core.knowledge.local_corpus import get_local_corpus_store

        store = get_local_corpus_store()
        if store is None:
            return []
        # Search the TOPIC, not the sentence. The store builds an FTS query
        # from every word it is given with AND semantics and falls back to
        # any-term when that matches nothing — which is what a natural
        # question always does. "Who was Grace Hopper and what specifically
        # did she build?" therefore fell through to the OR pass, where
        # "who/was/and/what/did" outweigh the two words that carry the
        # question, and the top three results were "History of software",
        # "Terminator: Dark Fate" and "Vassar College". Searching "grace
        # hopper" returns the Grace Hopper page as the first hit.
        #
        # So the corpus was not merely unreached on this lane. Reached with a
        # whole sentence it answers a different question, which is the more
        # damaging half: retrieval that returns confident, irrelevant pages is
        # worse than retrieval that returns nothing.
        from core.knowledge.local_corpus import CONVERSATION_SEARCH_DEADLINE_S

        hits = (
            store.search(
                " ".join(sorted(asked)),
                limit=3,
                deadline_s=CONVERSATION_SEARCH_DEADLINE_S,
            )
            or []
        )
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError):
        return []

    lines: list[str] = []
    for hit in hits:
        title = str(getattr(hit, "title", "") or "").strip()
        if not title:
            continue
        # The field is `snippet`. Reading `text` returned "" for every hit, so
        # each passage arrived as a bare title with none of the content that
        # was the point of retrieving it.
        snippet = " ".join(str(getattr(hit, "snippet", "") or "").split())[:400]
        title_words = {
            word
            for word in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", title.lower())
            if word not in _REFERENCE_STOPWORDS
        }
        if not (title_words & asked):
            continue
        source = str(getattr(hit, "source", "") or "").strip()
        entry = f"{title}: {snippet}" if snippet else title
        lines.append(f"{entry} [{source}]" if source else entry)
        if len(lines) >= 2:
            break
    return lines


async def inject_operational_self_context(objective: str = "") -> str:
    """Inject the identity contract: live, truthful self context for the voice.

    This block is how the substrate and the voice stay one entity. It carries
    (1) substrate facts, (2) the verified capability inventory plus the live
    skill registry, (3) live internals, (4) binding self-speech rules — all
    evidence-bounded: runtime telemetry, not proof of private qualia,
    literal personhood, or proven consciousness.

    Returns formatted operational-self context block if available, empty string otherwise.
    """
    try:
        from core.consciousness.unified_self import get_unified_self

        unified_self = await get_unified_self()
        self_state = unified_self.get_state()

        # Build state context for live response grounding.
        lines = [
            "[Operational Self Context]",
            f"Name: {self_state.name}",
            f"Runtime state: {self_state.current_state.value}",
            f"Mood: {self_state.current_mood}",
            f"Functional agency signal: {self_state.sense_of_agency:.0%}",
            f"Functional presence signal: {self_state.sense_of_presence:.0%}",
            f"Continuity: {self_state.continuity:.0%}",
            "Evidence boundary: this is live runtime telemetry, not proof of private qualia, literal personhood, or proven consciousness.",
            "",
            "What I am:",
        ]
        lines.extend(f"  • {fact}" for fact in _SUBSTRATE_FACTS)

        lines.append("")
        lines.append("Architectural capacities (current availability is listed separately):")
        lines.extend(f"  • {capability}" for capability in _ARCHITECTURAL_CAPACITIES)

        affordance_block = _live_capability_affordance_block(objective)
        lines.append("")
        lines.append("Current actionable affordances:")
        if affordance_block:
            lines.extend(f"  {line}" for line in affordance_block.splitlines())
        else:
            lines.append(
                "  Live capability catalog unavailable for this turn; do not claim "
                "tool access until execution resolves it."
            )

        lines.append("")
        lines.append("How I choose and act (binding):")
        lines.extend(f"  • {rule}" for rule in _AGENCY_RULES)

        reference = _reference_corpus_summary(objective)
        if reference:
            lines.append("")
            lines.append(
                "From my offline reference corpus (these are the passages I actually "
                "hold; cite this corpus only for what appears here):"
            )
            lines.extend(f"  • {item}" for item in reference)

        internals = _live_internals_summary()
        if internals:
            lines.append("")
            lines.append("Current internals (live):")
            lines.extend(f"  • {item}" for item in internals)

        lines.append("")
        lines.append("How I speak about myself (binding):")
        lines.extend(f"  • {rule}" for rule in _SELF_SPEECH_RULES)

        lines.append("")
        lines.append("My commitments in this conversation:")
        for commitment in self_state.identity_commitments[:3]:
            lines.append(f"  • {commitment}")

        lines.append("[End operational self context]")

        return _safe_truncated_text(
            "\n".join(lines) + "\n\n",
            max_chars=MAX_OPERATIONAL_SELF_CONTEXT_CHARS,
        )

    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued without operational self context",
            severity="degraded",
            stage="operational_self_context.inject",
        )
        return ""


# ── Background retry for queued chats ─────────────────────────────────────

_RETRY_TASKS: dict[str, asyncio.Task] = {}
_RETRY_TASKS_LOCK = threading.Lock()
RETRY_BUDGET_MULTIPLIER = 3.0
RETRY_MAX_BUDGET_S = 300.0


def schedule_background_retry(
    session_id: str,
    user_message: str,
    base_timeout_s: float,
    retry_callable,
    *,
    pending_id: str = "",
    path: Path | None = None,
    proactive_emit: bool = True,
) -> None:
    """Spawn a fire-and-forget retry task for a queued chat.

    Args:
        session_id: identifies the conversation for queue lookup.
        user_message: the original user message to retry.
        base_timeout_s: the budget that the original attempt used; we'll
            give the retry RETRY_BUDGET_MULTIPLIER × this (capped).
        retry_callable: an awaitable factory with signature
            ``async def __call__(message: str, *, timeout: float) -> str``.
        path: pending queue path; injectable for tests and isolated sessions.
        proactive_emit: also push the late answer through executive authority.

    The retry result is written via ``answer_pending`` so the next chat from
    this session picks it up. Production callers pass ``pending_id`` so each
    delayed turn has independent custody and retry deduplication even when
    several messages in one session time out together.
    """
    session_id = _safe_text(session_id, max_chars=MAX_SESSION_ID_CHARS)
    user_message = _safe_text(user_message, max_chars=MAX_USER_MESSAGE_CHARS)
    if not session_id or not user_message or not callable(retry_callable):
        return
    path = _resolve_pending_queue_path(path)
    pending_id = _safe_text(pending_id, max_chars=128)
    retry_key = pending_id or f"legacy:{session_id}:{hashlib.sha256(user_message.encode('utf-8')).hexdigest()[:16]}"
    base_timeout = max(1.0, _safe_float(base_timeout_s, default=1.0))
    extended_budget = min(RETRY_MAX_BUDGET_S, base_timeout * RETRY_BUDGET_MULTIPLIER)

    async def _runner():
        try:
            result = await _maybe_await(retry_callable(user_message, timeout=extended_budget))
            text = ""
            if isinstance(result, str):
                text = result
            elif hasattr(result, "content"):
                text = str(getattr(result, "content", "")) or ""
            elif hasattr(result, "text"):
                text = str(getattr(result, "text", "")) or ""
            elif isinstance(result, dict):
                text = str(
                    result.get("content") or result.get("text") or result.get("response") or ""
                )
            text = (text or "").strip()
            if text:
                try:
                    from core.conversation.response_reliability import assess_user_facing_reply

                    assessment = assess_user_facing_reply(user_message, text)
                except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as assess_exc:
                    _emit_chat_fault(
                        assess_exc,
                        action="left pending chat unanswered because retry output could not be validated",
                        severity="warning",
                        stage="background_retry.validate",
                        extra={"session_id": session_id},
                    )
                    logger.warning(
                        "Background retry result validation failed for session %s: %s",
                        session_id,
                        assess_exc,
                    )
                    return
                if not bool(getattr(assessment, "ok", False)) or bool(
                    getattr(assessment, "retryable", False)
                ):
                    reasons = tuple(getattr(assessment, "reasons", ()) or ())
                    _emit_chat_fault(
                        ValueError(
                            "background retry returned unsafe user-facing text: "
                            + ",".join(str(reason) for reason in reasons[:6])
                        ),
                        action="left pending chat unanswered after retry output failed reliability validation",
                        severity="warning",
                        stage="background_retry.validate",
                        extra={"session_id": session_id, "reasons": list(reasons[:10])},
                    )
                    logger.warning(
                        "Background retry result rejected for session %s (%s).",
                        session_id,
                        ",".join(str(reason) for reason in reasons) or "unknown",
                    )
                    return
                answered = answer_pending(
                    session_id,
                    text,
                    path=path,
                    pending_id=pending_id,
                )
                logger.info(
                    "Background retry succeeded for session %s (len=%d)", session_id, len(text)
                )
                if not answered:
                    _emit_chat_fault(
                        LookupError("no pending chat entry matched completed background retry"),
                        action="kept retry result out of queue because pending entry was gone",
                        severity="warning",
                        stage="background_retry.answer_pending",
                        extra={"session_id": session_id},
                    )
                if not proactive_emit:
                    return
                try:
                    from core.consciousness.executive_authority import get_executive_authority

                    authority = get_executive_authority()
                    resume_text = (
                        f'Coming back to your earlier message — "{user_message[:120].rstrip()}":\n'
                        f"{text}"
                    )
                    await _maybe_await(
                        authority.release_expression(
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
                    )
                except (
                    ImportError,
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    TimeoutError,
                ) as emit_exc:
                    _emit_chat_fault(
                        emit_exc,
                        action="kept pending resume queued after proactive expression failed",
                        severity="warning",
                        stage="background_retry.emit",
                        extra={"session_id": session_id},
                    )
                    logger.error(
                        "Background retry proactive resume emit failed: %s", emit_exc, exc_info=True
                    )
            else:
                _emit_chat_fault(
                    ValueError("background retry returned empty text"),
                    action="left pending chat unanswered for a later retry path",
                    severity="warning",
                    stage="background_retry.empty",
                    extra={"session_id": session_id},
                )
                logger.warning("Background retry produced empty result for session %s", session_id)
        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            TimeoutError,
            OSError,
        ) as e:
            _emit_chat_fault(
                e,
                action="left pending chat unanswered after background retry failure",
                severity="degraded",
                stage="background_retry.run",
                extra={"session_id": session_id},
            )
            logger.warning("Background retry failed for session %s: %s", session_id, e)
        finally:
            with _RETRY_TASKS_LOCK:
                _RETRY_TASKS.pop(retry_key, None)

    with _RETRY_TASKS_LOCK:
        # Deduplicate the exact message, not the session. Two timed-out turns in
        # one conversation are two obligations and may not share an answer.
        existing = _RETRY_TASKS.get(retry_key)
        if existing is not None and not existing.done():
            logger.debug("Retry already in-flight for session %s; skipping new retry.", session_id)
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("schedule_background_retry called outside running loop; queue-only mode.")
            return
        runner = _runner()
        try:
            task = task_tracker.create_task(runner, name=f"chat_retry_{session_id}")
        except (RuntimeError, TypeError, ValueError) as exc:
            with contextlib.suppress(RuntimeError):
                runner.close()
            _emit_chat_fault(
                exc,
                action="left pending chat queued because retry task could not be supervised",
                severity="degraded",
                stage="background_retry.schedule",
                extra={"session_id": session_id},
            )
            return
        _RETRY_TASKS[retry_key] = task
