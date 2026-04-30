from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.ConversationPersistence")

DEFAULT_PERSIST_DIR = Path.home() / ".aura" / "data" / "conversations"
MAX_HISTORY_IN_MEMORY = 50       # Messages kept in active session
MAX_SESSIONS_ON_DISK = 20        # Old sessions kept before rotation
SAVE_EVERY_N_MESSAGES = 3        # Save frequency
SESSION_SUMMARY_MIN_MESSAGES = 5 # Minimum messages before summarizing a session


@dataclass
class ConversationMessage:
    role: str                          # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    origin: str = "chat"               # "chat" | "voice" | "autonomous"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConversationMessage":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_orchestrator_entry(cls, entry: Any, session_id: str = "") -> "ConversationMessage":
        """Convert from whatever format the orchestrator uses."""
        if isinstance(entry, dict):
            return cls(
                role=entry.get("role", "user"),
                content=entry.get("content", str(entry)),
                session_id=session_id,
            )
        elif hasattr(entry, "role") and hasattr(entry, "content"):
            return cls(
                role=entry.role,
                content=entry.content,
                session_id=session_id,
            )
        else:
            return cls(role="unknown", content=str(entry), session_id=session_id)


@dataclass
class SessionRecord:
    session_id: str
    started_at: float
    ended_at: Optional[float]
    message_count: int
    summary: Optional[str]           # LLM-generated summary (if available)
    messages: List[Dict[str, Any]]   # Full messages for recent sessions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=d["session_id"],
            started_at=d["started_at"],
            ended_at=d.get("ended_at"),
            message_count=d.get("message_count", 0),
            summary=d.get("summary"),
            messages=d.get("messages", []),
        )


class ConversationPersistence:
    """
    Persists conversation history across sessions.
    Attaches to an orchestrator and intercepts message processing
    to save/load history transparently.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = str(uuid.uuid4())[:8]
        self._orchestrator = None
        self._message_count_since_save = 0
        self._current_session_messages: List[ConversationMessage] = []
        self._save_task: Optional[asyncio.Task] = None

        logger.info(
            "ConversationPersistence initialized (dir=%s, session=%s)",
            self.persist_dir, self.session_id
        )

    # ── Attachment ────────────────────────────────────────────────────────────

    def attach(self, orchestrator) -> "ConversationPersistence":
        """Attach to orchestrator and hook into its message pipeline."""
        self._orchestrator = orchestrator

        # Patch the message processing to trigger saves
        original_process = getattr(orchestrator, 'process_user_input', None)
        if original_process:
            persistence = self  # capture self

            async def persisting_process(message: str, origin: str = "user"):
                # Record user message
                user_msg = ConversationMessage(
                    role="user",
                    content=message,
                    session_id=persistence.session_id,
                    origin=origin,
                )
                persistence._record(user_msg)

                # Call original
                response = await original_process(message, origin)

                # Record assistant response
                if response and response.strip():
                    assistant_msg = ConversationMessage(
                        role="assistant",
                        content=response.strip(),
                        session_id=persistence.session_id,
                        origin=origin,
                    )
                    persistence._record(assistant_msg)
                    persistence._maybe_save()

                return response

            orchestrator.process_user_input = persisting_process
            logger.info("ConversationPersistence attached to process_user_input")

        return self

    # ── Load on Boot ──────────────────────────────────────────────────────────

    def load_recent(self, max_messages: int = 20) -> List[Dict[str, Any]]:
        """
        Load the most recent conversation messages from the last session.
        Returns them as a list ready to inject into conversation_history.
        """
        sessions = self._list_sessions()
        if not sessions:
            logger.info("No previous conversation sessions found")
            return []

        # Get most recent session
        latest = sessions[-1]
        session_path = self.persist_dir / f"session_{latest['session_id']}.json"

        if not session_path.exists():
            return []

        try:
            data = json.loads(session_path.read_text())
            record = SessionRecord.from_dict(data)
            messages = record.messages[-max_messages:]

            logger.info(
                "Loaded %d messages from session %s (started %s)",
                len(messages),
                record.session_id,
                time.strftime("%Y-%m-%d %H:%M", time.localtime(record.started_at))
            )

            # Inject into orchestrator if attached
            if self._orchestrator and hasattr(self._orchestrator, 'conversation_history'):
                existing = self._orchestrator.conversation_history or []
                if not existing:
                    # Convert to orchestrator format
                    orch_format = [
                        {"role": m.get("role", "user"), "content": m.get("content", "")}
                        for m in messages
                        if m.get("content")
                    ]
                    self._orchestrator.conversation_history = orch_format
                    logger.info(
                        "Injected %d messages from previous session into conversation_history",
                        len(orch_format)
                    )

            return messages

        except Exception as exc:
            logger.error("Could not load previous session: %s", exc)
            return []

    def get_session_context(self, max_sessions: int = 3) -> str:
        """
        Build a brief context string summarizing recent sessions.
        Useful for injecting into the system prompt on boot.
        """
        sessions = self._list_sessions()[-max_sessions:]
        if not sessions:
            return ""

        lines = ["Recent conversation history:"]
        for s in sessions:
            started = time.strftime("%b %d %H:%M", time.localtime(s["started_at"]))
            summary = s.get("summary") or f"{s.get('message_count', 0)} messages"
            lines.append(f"  - {started}: {summary}")

        return "\n".join(lines)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record(self, message: ConversationMessage):
        """Record a message to the current session."""
        self._current_session_messages.append(message)
        self._message_count_since_save += 1

        # Keep in-memory list bounded
        if len(self._current_session_messages) > MAX_HISTORY_IN_MEMORY * 2:
            self._current_session_messages = self._current_session_messages[-MAX_HISTORY_IN_MEMORY:]

    def _maybe_save(self):
        """Save if we've accumulated enough new messages."""
        if self._message_count_since_save >= SAVE_EVERY_N_MESSAGES:
            self._message_count_since_save = 0
            # Schedule async save without blocking
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.save_async())
                else:
                    self.save_sync()
            except RuntimeError:
                self.save_sync()

    # ── Saving ────────────────────────────────────────────────────────────────

    async def save_async(self):
        """Save current session asynchronously."""
        try:
            await asyncio.get_running_loop().run_in_executor(None, self.save_sync)
        except Exception as exc:
            logger.error("Async save failed: %s", exc)

    def save_sync(self):
        """
        Save current session to disk atomically.
        Atomic = write to temp file, then rename (prevents corruption on crash).
        """
        if not self._current_session_messages:
            return

        record = SessionRecord(
            session_id=self.session_id,
            started_at=self._current_session_messages[0].timestamp,
            ended_at=time.time(),
            message_count=len(self._current_session_messages),
            summary=None,  # Summary generated on session end
            messages=[m.to_dict() for m in self._current_session_messages],
        )

        session_path = self.persist_dir / f"session_{self.session_id}.json"
        temp_path = session_path.with_suffix(".json.tmp")

        try:
            atomic_write_text(temp_path, json.dumps(record.to_dict(), indent=2))
            temp_path.replace(session_path)  # Atomic rename
            logger.debug("Session saved: %s (%d messages)", self.session_id, record.message_count)
        except Exception as exc:
            logger.error("Session save failed: %s", exc)
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                import logging
                logger.debug("Exception caught during execution", exc_info=True)

    async def end_session(self, generate_summary: bool = True):
        """
        Called on shutdown. Saves final state and optionally generates a summary.
        """
        if not self._current_session_messages:
            return

        if (generate_summary and
                self._orchestrator and
                len(self._current_session_messages) >= SESSION_SUMMARY_MIN_MESSAGES):
            summary = await self._generate_summary()
        else:
            summary = None

        record = SessionRecord(
            session_id=self.session_id,
            started_at=self._current_session_messages[0].timestamp,
            ended_at=time.time(),
            message_count=len(self._current_session_messages),
            summary=summary,
            messages=[m.to_dict() for m in self._current_session_messages],
        )

        session_path = self.persist_dir / f"session_{self.session_id}.json"
        temp_path = session_path.with_suffix(".json.tmp")
        atomic_write_text(temp_path, json.dumps(record.to_dict(), indent=2))
        temp_path.replace(session_path)

        logger.info(
            "Session ended: %s (%d messages, summary=%s)",
            self.session_id,
            record.message_count,
            "yes" if summary else "no",
        )

        # Rotate old sessions
        self._rotate_old_sessions()

    async def _generate_summary(self) -> Optional[str]:
        """Ask the LLM to summarize the current session."""
        if not self._orchestrator or not hasattr(self._orchestrator, 'cognitive_engine'):
            return None

        messages_text = "\n".join(
            f"{m.role.upper()}: {m.content[:200]}"
            for m in self._current_session_messages[-20:]
        )

        prompt = (
            "Summarize this conversation in 2-3 sentences, capturing the main topics "
            "discussed and any important facts or decisions. Be specific and factual.\n\n"
            f"{messages_text}"
        )

        try:
            engine = self._orchestrator.cognitive_engine
            summary = await asyncio.wait_for(
                engine.think(objective=prompt, thinking_mode="FAST"),
                timeout=30.0
            )
            if summary and summary.strip():
                logger.info("Session summary generated (%d chars)", len(summary))
                return summary.strip()[:500]  # Cap summary length
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)

        return None

    # ── Session Management ────────────────────────────────────────────────────

    def _list_sessions(self) -> List[Dict[str, Any]]:
        """List all session files, sorted by start time."""
        sessions = []
        for f in self.persist_dir.glob("session_*.json"):
            if f.suffix == ".tmp":
                continue
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "session_id": data.get("session_id", ""),
                    "started_at": data.get("started_at", 0),
                    "message_count": data.get("message_count", 0),
                    "summary": data.get("summary"),
                })
            except Exception:
                import logging
                logger.debug("Exception caught during execution", exc_info=True)

        return sorted(sessions, key=lambda s: s["started_at"])

    def _rotate_old_sessions(self):
        """Delete oldest sessions if we have too many."""
        sessions = self._list_sessions()
        if len(sessions) <= MAX_SESSIONS_ON_DISK:
            return

        to_delete = sessions[:len(sessions) - MAX_SESSIONS_ON_DISK]
        for s in to_delete:
            path = self.persist_dir / f"session_{s['session_id']}.json"
            try:
                path.unlink()
                logger.debug("Rotated old session: %s", s["session_id"])
            except Exception:
                import logging
                logger.debug("Exception caught during execution", exc_info=True)

    def list_sessions_summary(self) -> str:
        """Human-readable session history."""
        sessions = self._list_sessions()
        if not sessions:
            return "No previous sessions."

        lines = [f"{'Date':<20} {'Messages':<10} Summary"]
        lines.append("-" * 60)
        for s in sessions[-10:]:
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["started_at"]))
            summary = (s.get("summary") or "")[:40]
            lines.append(f"{date:<20} {s['message_count']:<10} {summary}")

        return "\n".join(lines)
