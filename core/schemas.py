"""core/schemas.py
Strict Pydantic payloads for all internal state passing in the new Zenith architecture.
"""

import time
from typing import Any, ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from core.runtime.numeric_guards import bounded_float


class WebsocketMessage(BaseModel):
    """Base schema for any message sent down the websocket."""
    model_config = ConfigDict(extra='allow') # allow extra fields to prevent stripping
    
    type: str = Field(..., description="The type of the message (e.g. 'thought', 'telemetry')")

class TelemetryPayload(WebsocketMessage):
    """Telemetry as it reaches the UI.

    CP126 (high): "Telemetry omits upper and finite bounds."

    Every field carried ``ge=0.0`` and nothing else, so ``inf`` and ``nan``
    validated cleanly — pydantic's ``ge`` uses the same comparison semantics
    as everything else, and ``nan >= 0.0`` is False only for the lower bound
    it never trips, while ``inf >= 0.0`` is simply True. Both then reached
    the browser, where a NaN renders as "NaN" in a gauge and an unbounded
    energy silently rescales every chart on the page.

    These are percentages and normalised scores with known ranges, so the
    ranges are stated and enforced — but by CLAMPING, not by rejecting.

    Rejecting is the wrong failure mode here. This payload is built inside
    the heartbeat and published to the websocket; a ValidationError there
    does not protect the UI, it silently kills the telemetry stream and the
    dashboard freezes on its last good frame while the runtime looks fine.
    A gauge pinned at 100 is a visible, self-explaining wrong; a frozen
    dashboard is an invisible one.
    """

    type: str = "telemetry"
    energy: float = 100.0
    curiosity: float = 50.0
    frustration: float = 0.0
    confidence: float = 100.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0

    # Consciousness Fields (v6) — normalised [0, 1] scores.
    gwt_winner: str = "--"
    coherence: float = 0.0
    vitality: float = 0.0
    surprise: float = 0.0

    #: field -> (minimum, maximum, default when the value is unusable).
    _BOUNDS: ClassVar[dict[str, tuple[float, float, float]]] = {
        "energy": (0.0, 100.0, 100.0),
        "curiosity": (0.0, 100.0, 50.0),
        "frustration": (0.0, 100.0, 0.0),
        "confidence": (0.0, 100.0, 100.0),
        "cpu_usage": (0.0, 100.0, 0.0),
        "ram_usage": (0.0, 100.0, 0.0),
        "coherence": (0.0, 1.0, 0.0),
        "vitality": (0.0, 1.0, 0.0),
        "surprise": (0.0, 1.0, 0.0),
    }

    @field_validator(
        "energy", "curiosity", "frustration", "confidence",
        "cpu_usage", "ram_usage", "coherence", "vitality", "surprise",
        mode="before",
    )
    @classmethod
    def _bound_gauge(cls, value: Any, info: Any) -> float:
        """Clamp to the field's range; non-finite input takes the default.

        ``ge``/``le`` alone would not have closed this: pydantic compares,
        and every comparison with NaN is False, so a NaN slips past a lower
        bound it never trips. The value has to be rejected explicitly.
        """
        minimum, maximum, fallback = cls._BOUNDS[info.field_name]
        return bounded_float(value, default=fallback, minimum=minimum, maximum=maximum)
    narrative: str = ""
    
    consciousness: dict[str, Any] = Field(default_factory=dict)
    mycelial: dict[str, Any] = Field(default_factory=dict)
    
class CognitiveThoughtPayload(WebsocketMessage):
    type: str = "thought"
    content: str
    urgency: str = "NORMAL"
    cognitive_phase: str | None = None

class ChatStreamChunkPayload(WebsocketMessage):
    type: str = "chat_stream_chunk"
    chunk: str

class ChatThoughtChunkPayload(WebsocketMessage):
    type: str = "chat_thought_chunk"
    content: str

class AuraMessagePayload(WebsocketMessage):
    """Used for non-streaming responses, autonomic messages, and reflexes."""
    type: str = "aura_message"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class ActionResultPayload(WebsocketMessage):
    type: str = "action_result"
    tool: str
    result: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class UserMessagePayload(WebsocketMessage):
    type: str = "user_message"
    content: str

class ErrorPayload(WebsocketMessage):
    type: str = "error"
    message: str
class ChatStreamEvent(BaseModel):
    """Internal event for structured chat streaming."""
    type: str  # "token", "thought", "meta", "error", "end"
    content: str | None = None
    metadata: dict[str, Any] | None = None

class ToolInvocation(BaseModel):
    name: str = Field(..., description="The tool to invoke (python_sandbox, web_search)")
    payload: str = Field(..., description="The script or query for the tool")

class ShardResponse(BaseModel):
    """Strict schema for autonomous cognitive shards."""
    model_config = ConfigDict(extra='allow') # allow extra fields like 'thought' from LLMs
    
    analysis: str = Field(..., description="Internal cognitive monologue/analysis.", validation_alias=AliasChoices('analysis', 'thought'))
    action_type: str = Field(..., description="One of: 'observation', 'tool_use', 'conclusion', 'thought'")
    tools: list[ToolInvocation] = Field(default_factory=list, description="Array of tools to execute simultaneously.")
    tool_name: str | None = Field(None, description="[Legacy] The tool to invoke")
    tool_payload: str | None = Field(None, description="[Legacy] The script or query for the tool")
    conclusion: str = Field(..., description="Final takeaway or message.")

class IPCMessage(BaseModel):
    """Strictly validated payload for inter-process communication and task queues."""
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
    
    priority: int = Field(default=20)
    timestamp: float = Field(default_factory=time.monotonic)
    sequence: int = Field(default=0)
    payload: Any = Field(...)
    origin: str = Field(default="background")

    def __lt__(self, other: object) -> bool:
        other_key = _ipc_sort_key(other)
        if other_key is None:
            return False
        return _ipc_sort_key(self) < other_key


def _ipc_sort_key(message: object) -> tuple[int, float, int] | None:
    try:
        priority = int(message.priority)
        timestamp = float(message.timestamp)
        sequence = int(getattr(message, "sequence", 0))
    # not a failure: a reading that cannot be taken, or that is not a
    # number when it is, leaves this at the value below.
    except (AttributeError, TypeError, ValueError):
        return None
    return priority, timestamp, sequence
