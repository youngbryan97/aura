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

from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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

# Rolling window — older entries are pruned
_MAX_HISTORY_DEFAULT = 50


class UnifiedTranscript:
    """Thread-safe, channel-aware conversation transcript.
    
    Singleton pattern — all subsystems write to the same instance.
    Registered in ServiceContainer as "unified_transcript".
    """

    _instance: Optional["UnifiedTranscript"] = None
    _lock_class = threading.Lock()

    def __init__(self):
        self._entries: List[TranscriptEntry] = []
        self._max_history = _MAX_HISTORY_DEFAULT
        self._lock = threading.Lock()
        self._listeners: List = []
        logger.info("📝 UnifiedTranscript ONLINE")

    @classmethod
    def get_instance(cls) -> "UnifiedTranscript":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock_class:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        role: str,
        content: str,
        channel: ChannelType = "text",
        modality: ModalityType = "typed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptEntry:
        """Add a message to the transcript. Thread-safe."""
        entry = TranscriptEntry(
            role=role,
            content=content,
            channel=channel,
            modality=modality,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries.append(entry)
            # Prune if over max
            # Prune if over max
            if len(self._entries) > self._max_history:
                self._entries = self._entries[-self._max_history:]

        # Notify listeners (EventBus, UI, etc.)
        for listener in self._listeners:
            try:
                listener(entry)
            except Exception as e:
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

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context_window(self, n: int = 20) -> List[TranscriptEntry]:
        """Get the last N messages across ALL channels.
        This is the primary interface for LLM context assembly.
        """
        with self._lock:
            return list(self._entries[-n:])

    def get_context_string(self, n: int = 20) -> str:
        """Get the last N messages formatted for LLM injection."""
        entries = self.get_context_window(n)
        return "\n".join(e.to_llm_format() for e in entries)

    def get_by_channel(self, channel: ChannelType, n: int = 20) -> List[TranscriptEntry]:
        """Get last N messages from a specific channel."""
        with self._lock:
            filtered = [e for e in self._entries if e.channel == channel]
            return filtered[-n:]

    def get_last_aura_message(self) -> Optional[TranscriptEntry]:
        """Get Aura's most recent message (any channel)."""
        with self._lock:
            for entry in reversed(self._entries):
                if entry.role == "aura":
                    return entry
        return None

    def get_entry_count(self) -> int:
        """Total entries in transcript."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def on_entry(self, callback):
        """Register a listener called on every new entry."""
        self._listeners.append(callback)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Snapshot for telemetry."""
        with self._lock:
            channels = {}
            for e in self._entries:
                channels[e.channel] = channels.get(e.channel, 0) + 1
            return {
                "total_entries": len(self._entries),
                "channels": channels,
                "last_entry_age": time.time() - self._entries[-1].timestamp
                if self._entries else None,
            }
