"""
Telemetry Events — Structured event system for Aura observability

Ported from gemini-cli/telemetry patterns. Provides typed event classes
for all critical operations with structured JSON logging.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("Aura.Telemetry")


class EventType(Enum):
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    CHAT_COMPRESSION = "chat_compression"
    TOOL_OUTPUT_TRUNCATED = "tool_output_truncated"
    TOOL_OUTPUT_DISTILLED = "tool_output_distilled"
    RETRY_ATTEMPT = "retry_attempt"
    RESEARCH_PHASE = "research_phase"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    CONTEXT_COMPRESSED = "context_compressed"
    BROWSER_ACTION = "browser_action"
    SHELL_EXECUTION = "shell_execution"


@dataclass
class TelemetryEvent:
    """Base telemetry event."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class LLMRequestEvent(TelemetryEvent):
    """Tracks an LLM request."""
    model: str = ""
    task_tier: str = ""
    prompt_tokens: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = EventType.LLM_REQUEST


@dataclass
class LLMResponseEvent(TelemetryEvent):
    """Tracks an LLM response."""
    model: str = ""
    response_tokens: int = 0
    latency_ms: float = 0.0
    had_thought: bool = False

    def __post_init__(self) -> None:
        self.event_type = EventType.LLM_RESPONSE


@dataclass
class ChatCompressionEvent(TelemetryEvent):
    """Tracks a chat compression operation."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    status: str = ""
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = EventType.CHAT_COMPRESSION


@dataclass
class RetryAttemptEvent(TelemetryEvent):
    """Tracks a retry attempt."""
    attempt: int = 0
    max_attempts: int = 0
    error_type: str = ""
    delay_ms: float = 0.0
    operation: str = ""

    def __post_init__(self) -> None:
        self.event_type = EventType.RETRY_ATTEMPT


@dataclass
class ToolOutputEvent(TelemetryEvent):
    """Tracks tool output distillation/truncation."""
    tool_name: str = ""
    original_size: int = 0
    final_size: int = 0
    method: str = ""  # "passthrough", "structural", "llm_summary"
    saved_path: str = ""

    def __post_init__(self) -> None:
        self.event_type = EventType.TOOL_OUTPUT_DISTILLED


@dataclass
class ResearchPhaseEvent(TelemetryEvent):
    """Tracks deep research pipeline phases."""
    phase: str = ""
    loop_count: int = 0
    query_count: int = 0
    is_sufficient: bool = False

    def __post_init__(self) -> None:
        self.event_type = EventType.RESEARCH_PHASE


# ── Token Usage Tracker ──────────────────────────────────────────────────────

class TokenUsageTracker:
    """Tracks cumulative token usage per conversation."""

    def __init__(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._request_count += 1

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    def summary(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self._request_count,
        }

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0


# ── Event Emitter ────────────────────────────────────────────────────────────

class TelemetryEmitter:
    """Central event emission point. Logs structured events."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[TelemetryEvent], Any]] = []
        self.token_tracker = TokenUsageTracker()

    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event."""
        logger.debug("TELEMETRY: %s", event.to_json())
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # no-op: intentional

    def add_listener(self, listener: Callable[[TelemetryEvent], Any]) -> None:
        self._listeners.append(listener)


# Global singleton
_emitter: TelemetryEmitter | None = None

def get_telemetry() -> TelemetryEmitter:
    global _emitter
    if _emitter is None:
        _emitter = TelemetryEmitter()
    return _emitter
