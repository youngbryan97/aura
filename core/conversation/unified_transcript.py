"""core/conversation/unified_transcript.py

Unified Conversation Transcript
================================

A single, channel-aware conversation store that both voice and text I/O
write to. This replaces the fragmented parallel histories:
  - orchestrator.conversation_history (list of dicts)
  - ConversationContext.history (list of Messages)
  - ConversationPersistence (separate SQLite store)

Every message is tagged with its channel (voice, text, system, visual) and
delivery modality (spoke, typed, generated_image, etc.). This allows Aura
to reference what she said/heard/showed regardless of modality.

Usage:
    transcript = UnifiedTranscript.get_instance()
    transcript.add("user", "Hey, show me that image from earlier", channel="voice")
    transcript.add("aura", "Here it is!", channel="text", modality="typed")
    
    # Get last 20 messages across ALL channels for LLM context
    context = transcript.get_context_window(20)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from core.conversation.session_scope import (
    LOCAL_CONVERSATION_ID,
    current_conversation_session,
    normalize_conversation_id,
)
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.UnifiedTranscript")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ChannelType = Literal["voice", "text", "system", "visual", "internal"]
ModalityType = Literal["spoke", "typed", "generated_image", "sent_link",
                        "system_event", "internal_thought", "streamed"]


@dataclass
class TranscriptEntry:
    """A single entry in the unified conversation transcript."""
    role: str                   # "user", "aura", "system"
    content: str                # The actual message content
    channel: ChannelType        # How it arrived/was delivered
    modality: ModalityType = "typed"  # Specific delivery mechanism
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_id: str = LOCAL_CONVERSATION_ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_llm_format(self) -> str:
        """Format for LLM context injection. Includes channel tag so
        Aura knows the modality of prior exchanges."""
        channel_tag = f"[{self.channel.upper()}]" if self.channel != "text" else ""
        role_label = "User" if self.role == "user" else "Aura"
        if self.role == "system":
            return f"[SYSTEM] {self.content}"
        return f"{role_label}{' ' + channel_tag if channel_tag else ''}: {self.content}"


# ---------------------------------------------------------------------------
# Core transcript
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


# Rolling store window. Prompt injection still uses explicit request limits.
_MAX_HISTORY_DEFAULT = _env_int(
    "AURA_UNIFIED_TRANSCRIPT_MAX_HISTORY",
    500,
    low=50,
    high=100_000,
)


class UnifiedTranscript:
    """Thread-safe, channel-aware conversation transcript.
    
    Singleton pattern — all subsystems write to the same instance.
    Registered in ServiceContainer as "unified_transcript".
    """

    _instance: UnifiedTranscript | None = None
    _lock_class = threading.Lock()

    def __init__(self):
        self._entries: list[TranscriptEntry] = []
        self._max_history = _MAX_HISTORY_DEFAULT
        self._lock = threading.Lock()
        self._listeners: list = []
        logger.info("📝 UnifiedTranscript ONLINE")

    @classmethod
    def get_instance(cls) -> UnifiedTranscript:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock_class:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _conversation_id(
        conversation_id: str | None = None,
        *,
        channel: ChannelType = "text",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        details = metadata or {}
        explicit = normalize_conversation_id(conversation_id)
        scoped = current_conversation_session()
        described = normalize_conversation_id(
            details.get("conversation_id")
            or details.get("chat_session_id")
            or details.get("session_id")
        )
        if explicit or scoped or described:
            # An authenticated request scope is the security boundary. A
            # subsystem may provide a more specific label outside that scope,
            # but it cannot switch a live request into another conversation.
            return scoped or explicit or described
        if channel in {"system", "internal"}:
            return "local-system"
        return LOCAL_CONVERSATION_ID

    def entries_for_conversation(
        self,
        conversation_id: str | None = None,
    ) -> list[TranscriptEntry]:
        identity = self._conversation_id(conversation_id)
        with self._lock:
            return [entry for entry in self._entries if entry.conversation_id == identity]

    def preceding_turns(
        self,
        *,
        before_content: str = "",
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        """The last user request and the last thing Aura said, in that order.

        This is what a message like "Can you do it now?" or "From the grant
        research funds manager" needs to mean anything. Both were said
        to Aura live on 2026-08-03 and both were answered as though the
        conversation had just started, because every router reads one message
        at a time.

        ``before_content`` skips the current turn when it has already been
        written to the transcript, so a message never resolves against itself.
        """
        entries = self.entries_for_conversation(conversation_id)

        if before_content:
            needle = str(before_content).strip()
            for index in range(len(entries) - 1, -1, -1):
                if entries[index].role == "user" and entries[index].content.strip() == needle:
                    entries = entries[:index]
                    break

        last_user = ""
        last_aura = ""
        for entry in reversed(entries):
            content = str(getattr(entry, "content", "") or "").strip()
            if not content:
                continue
            if not last_user and entry.role == "user":
                last_user = content
            elif not last_aura and entry.role == "aura":
                last_aura = content
            if last_user and last_aura:
                break
        return last_user, last_aura

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        role: str,
        content: str,
        channel: ChannelType = "text",
        modality: ModalityType = "typed",
        metadata: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> TranscriptEntry:
        """Add a message to the transcript. Thread-safe."""
        details = dict(metadata or {})
        entry = TranscriptEntry(
            role=role,
            content=content,
            channel=channel,
            conversation_id=self._conversation_id(
                conversation_id,
                channel=channel,
                metadata=details,
            ),
            modality=modality,
            metadata=details,
        )
        with self._lock:
            # Durable delivery recovery may replay either half of an exchange.
            # A correction must use replace_aura_reply, never append a second answer.
            exchange_id = str(details.get("exchange_id") or "").strip()
            if exchange_id:
                for existing in reversed(self._entries):
                    if (
                        existing.conversation_id == entry.conversation_id
                        and existing.channel == channel
                        and existing.role == role
                        and str(existing.metadata.get("exchange_id") or "").strip() == exchange_id
                    ):
                        if existing.content != content:
                            raise ValueError("transcript exchange replay changed its content")
                        return existing
            self._entries.append(entry)
            # Prune if over max
            if len(self._entries) > self._max_history:
                self._entries = self._entries[-self._max_history:]

        # Notify listeners (EventBus, UI, etc.)
        for listener in self._listeners:
            try:
                listener(entry)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('unified_transcript', e)
                capture_and_log(e, {'module': __name__})

        logger.debug(
            "📝 Transcript +%s [%s/%s]: %.60s",
            role, channel, modality, content
        )
        return entry

    def add_voice_input(self, content: str, **kwargs) -> TranscriptEntry:
        """Convenience: add user voice input."""
        return self.add("user", content, channel="voice", modality="spoke", **kwargs)

    def add_voice_output(self, content: str, **kwargs) -> TranscriptEntry:
        """Convenience: add Aura's spoken response."""
        return self.add("aura", content, channel="voice", modality="spoke", **kwargs)

    def add_text_input(self, content: str, **kwargs) -> TranscriptEntry:
        """Convenience: add user text input."""
        return self.add("user", content, channel="text", modality="typed", **kwargs)

    def add_text_output(self, content: str, **kwargs) -> TranscriptEntry:
        """Convenience: add Aura's typed response."""
        return self.add("aura", content, channel="text", modality="typed", **kwargs)

    def add_visual(self, content: str, modality: ModalityType = "generated_image",
                   **kwargs) -> TranscriptEntry:
        """Convenience: add visual output (image, link, etc.)."""
        return self.add("aura", content, channel="visual", modality=modality, **kwargs)

    def add_system(self, content: str, **kwargs) -> TranscriptEntry:
        """Convenience: add system event."""
        return self.add("system", content, channel="system", modality="system_event",
                        **kwargs)

    def replace_aura_reply(
        self,
        *,
        exchange_id: str,
        expected_content: str,
        replacement_content: str,
        revision: int,
        conversation_id: str | None = None,
    ) -> bool:
        """CAS-replace one delivered Aura reply in the bounded live transcript."""

        safe_exchange_id = str(exchange_id or "").strip()
        if not safe_exchange_id or int(revision) < 2:
            return False
        identity = self._conversation_id(conversation_id)
        with self._lock:
            matches = [
                entry
                for entry in self._entries
                if entry.conversation_id == identity
                and entry.role == "aura"
                and str(entry.metadata.get("exchange_id") or "") == safe_exchange_id
            ]
            if len(matches) != 1 or matches[0].content != expected_content:
                return False
            entry = matches[0]
            entry.content = str(replacement_content)
            entry.metadata = {
                **entry.metadata,
                "regenerated": True,
                "revision": int(revision),
            }
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context_window(
        self,
        n: int = 20,
        *,
        conversation_id: str | None = None,
    ) -> list[TranscriptEntry]:
        """Get the last N messages for one conversation across its channels.
        This is the primary interface for LLM context assembly.
        """
        return self.entries_for_conversation(conversation_id)[-n:]

    def get_context_string(self, n: int = 20, *, conversation_id: str | None = None) -> str:
        """Get the last N messages formatted for LLM injection."""
        entries = self.get_context_window(n, conversation_id=conversation_id)
        return "\n".join(e.to_llm_format() for e in entries)

    def get_by_channel(
        self,
        channel: ChannelType,
        n: int = 20,
        *,
        conversation_id: str | None = None,
    ) -> list[TranscriptEntry]:
        """Get last N messages from a specific channel."""
        filtered = [
            entry
            for entry in self.entries_for_conversation(conversation_id)
            if entry.channel == channel
        ]
        return filtered[-n:]

    def get_last_aura_message(
        self,
        *,
        conversation_id: str | None = None,
    ) -> TranscriptEntry | None:
        """Get Aura's most recent message (any channel)."""
        for entry in reversed(self.entries_for_conversation(conversation_id)):
            if entry.role == "aura":
                return entry
        return None

    def get_entry_count(self, *, conversation_id: str | None = None) -> int:
        """Total entries in the selected conversation."""
        return len(self.entries_for_conversation(conversation_id))

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def on_entry(self, callback):
        """Register a listener called on every new entry."""
        self._listeners.append(callback)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Snapshot for telemetry."""
        with self._lock:
            channels = {}
            for e in self._entries:
                channels[e.channel] = channels.get(e.channel, 0) + 1
            return {
                "total_entries": len(self._entries),
                "max_history": self._max_history,
                "channels": channels,
                "last_entry_age": time.time() - self._entries[-1].timestamp
                if self._entries else None,
            }
