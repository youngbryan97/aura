from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ConversationPersistence")

DEFAULT_PERSIST_DIR = state_root() / "data" / "conversations"


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


MAX_HISTORY_IN_MEMORY = _env_int(
    "AURA_CONVERSATION_CACHE_MAX_MESSAGES",
    500,
    low=50,
    high=10_000,
)
MAX_SESSIONS_ON_DISK = _env_int(
    "AURA_CONVERSATION_CACHE_MAX_SESSIONS",
    200,
    low=20,
    high=10_000,
)
SAVE_EVERY_N_MESSAGES = 3
SESSION_SUMMARY_MIN_MESSAGES = 5
MAX_MESSAGE_CONTENT_CHARS = 20_000
MAX_ORIGIN_CHARS = 64
MAX_SESSION_ID_CHARS = 64
MAX_SUMMARY_SOURCE_MESSAGES = 20
MAX_SUMMARY_CHARS = 500
SUMMARY_TIMEOUT_S = 30.0

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_CONVERSATION_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_SUMMARY_ERRORS = (
    AttributeError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_conversation_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "conversation_persistence",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "conversation_persistence",
                error,
                severity=severity,
                action=action,
            )
        except TypeError:
            logger.debug(
                "Conversation persistence degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_text(value: object, *, default: str = "", max_chars: int = 4096) -> str:
    try:
        text = str(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        text = default
    return text.replace("\x00", "")[:max_chars]


def _safe_timestamp(value: object, *, default: float | None = None) -> float:
    fallback = time.time() if default is None else float(default)
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if timestamp <= 0:
        return fallback
    return timestamp


def _safe_int(value: object, *, default: int = 0, minimum: int = 0, maximum: int = 10_000) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(minimum, min(maximum, number))


def _safe_session_id(value: object) -> str:
    text = _safe_text(value, max_chars=MAX_SESSION_ID_CHARS).strip()
    return text if _SESSION_ID_RE.fullmatch(text) else ""


def _safe_role(value: object) -> str:
    role = _safe_text(value, default="user", max_chars=32).lower().strip()
    if role in {"user", "assistant", "system", "tool", "unknown"}:
        return role
    return "unknown"


def _session_id_from_path(path: Path) -> str:
    name = path.name
    if not name.startswith("session_") or not name.endswith(".json"):
        return ""
    return _safe_session_id(name[len("session_") : -len(".json")])


@dataclass
class ConversationMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    origin: str = "chat"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConversationMessage:
        if not isinstance(d, dict):
            raise ValueError("conversation message must be a dictionary")
        return cls(
            role=_safe_role(d.get("role", "user")),
            content=_safe_text(d.get("content", ""), max_chars=MAX_MESSAGE_CONTENT_CHARS),
            timestamp=_safe_timestamp(d.get("timestamp")),
            session_id=_safe_session_id(d.get("session_id", "")),
            origin=_safe_text(d.get("origin", "chat"), default="chat", max_chars=MAX_ORIGIN_CHARS) or "chat",
        )

    @classmethod
    def from_orchestrator_entry(cls, entry: Any, session_id: str = "") -> ConversationMessage:
        """Convert from supported orchestrator history formats."""
        safe_session_id = _safe_session_id(session_id)
        if isinstance(entry, dict):
            return cls.from_dict({**entry, "session_id": safe_session_id})
        if hasattr(entry, "role") and hasattr(entry, "content"):
            return cls(
                role=_safe_role(getattr(entry, "role", "user")),
                content=_safe_text(getattr(entry, "content", ""), max_chars=MAX_MESSAGE_CONTENT_CHARS),
                session_id=safe_session_id,
            )
        return cls(role="unknown", content=_safe_text(entry, max_chars=MAX_MESSAGE_CONTENT_CHARS), session_id=safe_session_id)


@dataclass
class SessionRecord:
    session_id: str
    started_at: float
    ended_at: float | None
    message_count: int
    summary: str | None
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, fallback_session_id: str = "") -> SessionRecord:
        if not isinstance(d, dict):
            raise ValueError("conversation session must be a dictionary")

        raw_messages = d.get("messages", [])
        if not isinstance(raw_messages, list):
            raw_messages = []

        messages: list[dict[str, Any]] = []
        for item in raw_messages[-MAX_HISTORY_IN_MEMORY * 2 :]:
            try:
                messages.append(ConversationMessage.from_dict(item).to_dict())
            except (TypeError, ValueError):
                continue

        default_started = messages[0]["timestamp"] if messages else time.time()
        message_count = _safe_int(d.get("message_count"), default=len(messages), maximum=1_000_000)
        return cls(
            session_id=_safe_session_id(d.get("session_id", "")) or _safe_session_id(fallback_session_id),
            started_at=_safe_timestamp(d.get("started_at"), default=default_started),
            ended_at=None if d.get("ended_at") is None else _safe_timestamp(d.get("ended_at")),
            message_count=max(message_count, len(messages)),
            summary=_safe_text(d.get("summary", ""), max_chars=MAX_SUMMARY_CHARS) or None,
            messages=messages,
        )


class ConversationPersistence:
    """
    JSON-backed compatibility persistence for recent conversation continuity.

    The canonical transcript store lives under ``core.conversation.persistence``.
    This class remains as a bounded boot-context cache for older call sites and
    must never be allowed to corrupt state, block shutdown, or silently lose the
    latest user turn.
    """

    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR).expanduser()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = uuid.uuid4().hex[:16]
        self._orchestrator = None
        self._message_count_since_save = 0
        self._current_session_messages: list[ConversationMessage] = []
        self._save_task: asyncio.Task | None = None
        self._last_save_ok = True
        self._last_error_at = 0.0

        logger.info(
            "ConversationPersistence initialized (dir=%s, session=%s)",
            self.persist_dir,
            self.session_id,
        )

    def attach(self, orchestrator) -> ConversationPersistence:
        """Attach to orchestrator via middleware registration when available."""
        self._orchestrator = orchestrator

        if hasattr(orchestrator, "middleware") and isinstance(orchestrator.middleware, list):
            orchestrator.middleware.append(self._middleware_hook)
            logger.info("ConversationPersistence attached as middleware")
        elif hasattr(orchestrator, "register_middleware"):
            orchestrator.register_middleware(self._middleware_hook)
            logger.info("ConversationPersistence attached via register_middleware")
        else:
            original_process = getattr(orchestrator, "process_user_input", None)
            if callable(original_process):
                persistence = self

                async def persisting_process(message: str, origin: str = "user", *args, **kwargs):
                    persistence._record(
                        ConversationMessage(
                            role="user",
                            content=_safe_text(message, max_chars=MAX_MESSAGE_CONTENT_CHARS),
                            session_id=persistence.session_id,
                            origin=_safe_text(origin, default="user", max_chars=MAX_ORIGIN_CHARS),
                        )
                    )
                    result = original_process(message, origin, *args, **kwargs)
                    response = await result if asyncio.iscoroutine(result) else result
                    if response and _safe_text(response).strip():
                        persistence._record(
                            ConversationMessage(
                                role="assistant",
                                content=_safe_text(response, max_chars=MAX_MESSAGE_CONTENT_CHARS).strip(),
                                session_id=persistence.session_id,
                                origin=_safe_text(origin, default="user", max_chars=MAX_ORIGIN_CHARS),
                            )
                        )
                    persistence._maybe_save()
                    return response

                orchestrator.process_user_input = persisting_process
                logger.warning("ConversationPersistence using compatibility process_user_input wrapper")
            else:
                _record_conversation_degradation(
                    RuntimeError("orchestrator has no middleware or process_user_input hook"),
                    action="attached without persistence hook; session cache remains manually callable",
                    severity="warning",
                )

        return self

    async def _middleware_hook(
        self,
        message: str,
        response: str,
        *,
        origin: str = "user",
        receipt_id: str = "",
        **kwargs,
    ) -> None:
        """Middleware callback receiving input/output pairs with optional Will receipt."""
        del receipt_id, kwargs
        safe_origin = _safe_text(origin, default="user", max_chars=MAX_ORIGIN_CHARS)
        self._record(
            ConversationMessage(
                role="user",
                content=_safe_text(message, max_chars=MAX_MESSAGE_CONTENT_CHARS),
                session_id=self.session_id,
                origin=safe_origin,
            )
        )

        assistant_text = _safe_text(response, max_chars=MAX_MESSAGE_CONTENT_CHARS).strip()
        if assistant_text:
            self._record(
                ConversationMessage(
                    role="assistant",
                    content=assistant_text,
                    session_id=self.session_id,
                    origin=safe_origin,
                )
            )
        self._maybe_save()

    def load_recent(self, max_messages: int = 20) -> list[dict[str, Any]]:
        """
        Load the most recent conversation messages from the last valid session.
        Returns records ready to inject into conversation_history.
        """
        sessions = self._list_sessions()
        if not sessions:
            logger.info("No previous conversation sessions found")
            return []

        latest = sessions[-1]
        session_id = _safe_session_id(latest.get("session_id", ""))
        if not session_id:
            return []

        session_path = self._session_path(session_id)
        if not session_path.exists():
            return []

        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
            record = SessionRecord.from_dict(data, fallback_session_id=session_id)
            limit = _safe_int(max_messages, default=20, minimum=1, maximum=MAX_HISTORY_IN_MEMORY)
            messages = record.messages[-limit:]

            logger.info(
                "Loaded %d messages from session %s (started %s)",
                len(messages),
                record.session_id,
                time.strftime("%Y-%m-%d %H:%M", time.localtime(record.started_at)),
            )

            if self._orchestrator and hasattr(self._orchestrator, "conversation_history"):
                existing = self._orchestrator.conversation_history or []
                if not existing:
                    orch_format = [
                        {"role": m.get("role", "user"), "content": m.get("content", "")}
                        for m in messages
                        if m.get("content")
                    ]
                    self._orchestrator.conversation_history = orch_format
                    logger.info(
                        "Injected %d messages from previous session into conversation_history",
                        len(orch_format),
                    )

            return messages

        except _CONVERSATION_ERRORS as exc:
            self._last_error_at = time.time()
            self._quarantine_session_file(
                session_path,
                exc,
                action="quarantined unreadable latest session and started with empty boot context",
            )
            return []

    def get_session_context(self, max_sessions: int = 3) -> str:
        """Build a brief context string summarizing recent sessions."""
        limit = _safe_int(max_sessions, default=3, minimum=1, maximum=MAX_SESSIONS_ON_DISK)
        sessions = self._list_sessions()[-limit:]
        if not sessions:
            return ""

        lines = ["Recent conversation history:"]
        for session in sessions:
            started = time.strftime("%b %d %H:%M", time.localtime(session["started_at"]))
            summary = session.get("summary") or f"{session.get('message_count', 0)} messages"
            lines.append(f"  - {started}: {summary}")

        return "\n".join(lines)

    def _record(self, message: ConversationMessage) -> None:
        """Record a sanitized message to the current session."""
        sanitized = ConversationMessage.from_dict(message.to_dict())
        if not sanitized.session_id:
            sanitized.session_id = self.session_id
        self._current_session_messages.append(sanitized)
        self._message_count_since_save += 1

        if len(self._current_session_messages) > MAX_HISTORY_IN_MEMORY * 2:
            self._current_session_messages = self._current_session_messages[-MAX_HISTORY_IN_MEMORY:]

    def _maybe_save(self) -> None:
        """Coalesce non-blocking saves after enough new messages accumulate."""
        if self._message_count_since_save < SAVE_EVERY_N_MESSAGES:
            return
        self._message_count_since_save = 0

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                if self._save_task is not None and not self._save_task.done():
                    return
                self._save_task = get_task_tracker().create_task(
                    self.save_async(),
                    name="conversation_persistence.save_async",
                )
                return
        except RuntimeError as _exc:
            logger.debug("Suppressed %s in core.memory.conversation_persistence: %s", type(_exc).__name__, _exc)

        self.save_sync()

    async def save_async(self) -> bool:
        """Save current session without blocking the event loop."""
        try:
            return await asyncio.to_thread(self.save_sync)
        except _CONVERSATION_ERRORS as exc:
            self._last_error_at = time.time()
            self._last_save_ok = False
            _record_conversation_degradation(
                exc,
                action="kept conversation in memory after asynchronous session save failed",
                severity="degraded",
            )
            logger.error("Async conversation save failed: %s", exc)
            return False

    def save_sync(self) -> bool:
        """Save current session to disk atomically."""
        record = self._build_record(summary=None)
        if record is None:
            return True

        try:
            atomic_write_text(
                self._session_path(record.session_id),
                json.dumps(record.to_dict(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            self._last_save_ok = True
            logger.debug("Session saved: %s (%d messages)", self.session_id, record.message_count)
            return True
        except _CONVERSATION_ERRORS as exc:
            self._last_error_at = time.time()
            self._last_save_ok = False
            _record_conversation_degradation(
                exc,
                action="kept conversation in memory after atomic session save failed",
                severity="degraded",
                extra={"session_id": self.session_id, "messages": len(self._current_session_messages)},
            )
            logger.error("Session save failed: %s", exc)
            return False

    async def end_session(self, generate_summary: bool = True) -> bool:
        """Save final state and optionally generate a deterministic or cognitive summary."""
        if not self._current_session_messages:
            return True

        pending = self._save_task
        if pending is not None and not pending.done():
            try:
                await pending
            except _CONVERSATION_ERRORS as exc:
                _record_conversation_degradation(
                    exc,
                    action="continued final session save after pending background save failed",
                    severity="warning",
                )

        summary = None
        if generate_summary and len(self._current_session_messages) >= SESSION_SUMMARY_MIN_MESSAGES:
            summary = await self._generate_summary()

        record = self._build_record(summary=summary)
        if record is None:
            return True

        try:
            from core.runtime.atomic_writer import async_atomic_write_text

            await async_atomic_write_text(
                self._session_path(record.session_id),
                json.dumps(record.to_dict(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except _CONVERSATION_ERRORS as exc:
            self._last_error_at = time.time()
            self._last_save_ok = False
            _record_conversation_degradation(
                exc,
                action="kept final conversation in memory after shutdown save failed",
                severity="critical",
                extra={"session_id": self.session_id, "messages": record.message_count},
            )
            logger.error("Final session save failed: %s", exc)
            return False

        self._last_save_ok = True
        logger.info(
            "Session ended: %s (%d messages, summary=%s)",
            self.session_id,
            record.message_count,
            "yes" if summary else "no",
        )
        self._rotate_old_sessions()
        return True

    async def _generate_summary(self) -> str | None:
        """Generate a session summary with a deterministic fallback."""
        messages_text = self._summary_source_text()
        if not messages_text:
            return None

        engine = getattr(self._orchestrator, "cognitive_engine", None)
        think = getattr(engine, "think", None)
        if not callable(think):
            return self._extractive_summary()

        prompt = (
            "Summarize this conversation in 2-3 sentences, capturing the main topics "
            "discussed and any important facts or decisions. Be specific and factual.\n\n"
            f"{messages_text}"
        )

        try:
            result = think(objective=prompt, thinking_mode="FAST")
            summary = await asyncio.wait_for(result, timeout=SUMMARY_TIMEOUT_S) if asyncio.iscoroutine(result) else result
            summary_text = _safe_text(summary, max_chars=MAX_SUMMARY_CHARS).strip()
            if summary_text:
                logger.info("Session summary generated (%d chars)", len(summary_text))
                return summary_text
            raise ValueError("cognitive summary was empty")
        except _SUMMARY_ERRORS as exc:
            fallback = self._extractive_summary()
            _record_conversation_degradation(
                exc,
                action="used deterministic session summary after cognitive summary failed",
                severity="warning",
                extra={"fallback_summary_chars": len(fallback or "")},
            )
            logger.warning("Summary generation failed; deterministic summary used: %s", exc)
            return fallback

    def _build_record(self, *, summary: str | None) -> SessionRecord | None:
        if not self._current_session_messages:
            return None
        snapshot = [ConversationMessage.from_dict(message.to_dict()) for message in self._current_session_messages]
        return SessionRecord(
            session_id=self.session_id,
            started_at=snapshot[0].timestamp,
            ended_at=time.time(),
            message_count=len(snapshot),
            summary=_safe_text(summary, max_chars=MAX_SUMMARY_CHARS) or None,
            messages=[message.to_dict() for message in snapshot],
        )

    def _summary_source_text(self) -> str:
        return "\n".join(
            f"{message.role.upper()}: {_safe_text(message.content, max_chars=200)}"
            for message in self._current_session_messages[-MAX_SUMMARY_SOURCE_MESSAGES:]
            if message.content
        )

    def _extractive_summary(self) -> str | None:
        messages = [message for message in self._current_session_messages if message.content]
        if not messages:
            return None

        user_topics = [
            _safe_text(message.content, max_chars=90).strip().replace("\n", " ")
            for message in messages
            if message.role == "user"
        ]
        assistant_notes = [
            _safe_text(message.content, max_chars=90).strip().replace("\n", " ")
            for message in messages
            if message.role == "assistant"
        ]

        parts: list[str] = []
        if user_topics:
            parts.append("User topics: " + "; ".join(user_topics[-3:]))
        if assistant_notes:
            parts.append("Assistant responses: " + "; ".join(assistant_notes[-2:]))
        if not parts:
            parts.append(f"{len(messages)} conversation messages were exchanged.")
        return _safe_text(". ".join(parts), max_chars=MAX_SUMMARY_CHARS)

    def _list_sessions(self) -> list[dict[str, Any]]:
        """List valid session files, sorted by start time."""
        sessions = []
        for path in sorted(self.persist_dir.glob("session_*.json")):
            fallback_session_id = _session_id_from_path(path)
            if not fallback_session_id:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = SessionRecord.from_dict(data, fallback_session_id=fallback_session_id)
                if not record.session_id:
                    raise ValueError("session id is invalid")
                sessions.append(
                    {
                        "session_id": fallback_session_id,
                        "started_at": record.started_at,
                        "message_count": record.message_count,
                        "summary": record.summary,
                    }
                )
            except _CONVERSATION_ERRORS as exc:
                self._last_error_at = time.time()
                self._quarantine_session_file(
                    path,
                    exc,
                    action="quarantined unreadable session file while listing conversation history",
                )

        return sorted(sessions, key=lambda session: session["started_at"])

    def _rotate_old_sessions(self) -> None:
        """Delete oldest sessions if retention exceeds the configured bound."""
        sessions = self._list_sessions()
        if len(sessions) <= MAX_SESSIONS_ON_DISK:
            return

        for session in sessions[: len(sessions) - MAX_SESSIONS_ON_DISK]:
            try:
                self._session_path(session["session_id"]).unlink(missing_ok=True)
                logger.debug("Rotated old session: %s", session["session_id"])
            except _CONVERSATION_ERRORS as exc:
                self._last_error_at = time.time()
                _record_conversation_degradation(
                    exc,
                    action="left old conversation session in place after rotation delete failed",
                    severity="warning",
                    extra={"session_id": session.get("session_id", "")},
                )

    def _session_path(self, session_id: str) -> Path:
        safe_session_id = _safe_session_id(session_id)
        if not safe_session_id:
            raise ValueError("invalid conversation session id")
        return self.persist_dir / f"session_{safe_session_id}.json"

    def _quarantine_session_file(self, path: Path, error: BaseException, *, action: str) -> None:
        quarantine_name = f"corrupt_{int(time.time())}_{path.name}"
        try:
            target = self.persist_dir / quarantine_name
            path.replace(target)
            extra = {"source": path.name, "quarantine": target.name}
        except _CONVERSATION_ERRORS as quarantine_exc:
            error = quarantine_exc
            action = f"{action}; quarantine move failed and file was skipped"
            extra = {"source": path.name}

        _record_conversation_degradation(
            error,
            action=action,
            severity="degraded",
            extra=extra,
        )

    def list_sessions_summary(self) -> str:
        """Human-readable session history."""
        sessions = self._list_sessions()
        if not sessions:
            return "No previous sessions."

        lines = [f"{'Date':<20} {'Messages':<10} Summary", "-" * 60]
        for session in sessions[-10:]:
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(session["started_at"]))
            summary = (session.get("summary") or "")[:40]
            lines.append(f"{date:<20} {session['message_count']:<10} {summary}")

        return "\n".join(lines)
