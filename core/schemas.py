"""core/schemas.py
Strict Pydantic payloads for all internal state passing in the new Zenith architecture.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

class WebsocketMessage(BaseModel):
    """Base schema for any message sent down the websocket."""
    model_config = ConfigDict(extra='allow') # allow extra fields to prevent stripping
    
    type: str = Field(..., description="The type of the message (e.g. 'thought', 'telemetry')")

class TelemetryPayload(WebsocketMessage):
    type: str = "telemetry"
    energy: float = Field(default=100.0, ge=0.0)
    curiosity: float = Field(default=50.0, ge=0.0)
    frustration: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=100.0, ge=0.0)
    cpu_usage: float = Field(default=0.0, ge=0.0)
    ram_usage: float = Field(default=0.0, ge=0.0)
    
    # Consciousness Fields (v6)
    gwt_winner: str = "--"
    coherence: float = Field(default=0.0, ge=0.0)
    vitality: float = Field(default=0.0, ge=0.0)
    surprise: float = Field(default=0.0, ge=0.0)
    narrative: str = ""
    
    consciousness: Dict[str, Any] = Field(default_factory=dict)
    mycelial: Dict[str, Any] = Field(default_factory=dict)
    
class CognitiveThoughtPayload(WebsocketMessage):
    type: str = "thought"
    content: str
    urgency: str = "NORMAL"
    cognitive_phase: Optional[str] = None

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
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ActionResultPayload(WebsocketMessage):
    type: str = "action_result"
    tool: str
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserMessagePayload(WebsocketMessage):
    type: str = "user_message"
    content: str

class ErrorPayload(WebsocketMessage):
    type: str = "error"
    message: str
class ChatStreamEvent(BaseModel):
    """Internal event for structured chat streaming."""
    type: str  # "token", "thought", "meta", "error", "end"
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ToolInvocation(BaseModel):
    name: str = Field(..., description="The tool to invoke (python_sandbox, web_search)")
    payload: str = Field(..., description="The script or query for the tool")

class ShardResponse(BaseModel):
    """Strict schema for autonomous cognitive shards."""
    model_config = ConfigDict(extra='allow') # allow extra fields like 'thought' from LLMs
    
    analysis: str = Field(..., description="Internal cognitive monologue/analysis.", validation_alias=AliasChoices('analysis', 'thought'))
    action_type: str = Field(..., description="One of: 'observation', 'tool_use', 'conclusion', 'thought'")
    tools: List[ToolInvocation] = Field(default_factory=list, description="Array of tools to execute simultaneously.")
    tool_name: Optional[str] = Field(None, description="[Legacy] The tool to invoke")
    tool_payload: Optional[str] = Field(None, description="[Legacy] The script or query for the tool")
    conclusion: str = Field(..., description="Final takeaway or message.")
