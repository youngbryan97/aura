from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for Aura's resident generation backends."""
    async def generate(
        self, 
        prompt: str, 
        system_prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Single-shot generation."""
        ...

    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        ...

@runtime_checkable
class Service(Protocol):
    """Base protocol for any service in the ServiceContainer."""
    def setup(self) -> None:
        """Initialize the service."""
        ...

@runtime_checkable
class OrchestratorService(Service, Protocol):
    """Protocol for services that interact with the Orchestrator."""
    def start(self) -> None:
        """Start background tasks."""
        ...
    
    def stop(self) -> None:
        """Stop background tasks."""
        ...

# ── Cognitive Types ───────────────────────────────────────────

class ThinkingMode(Enum):
    """Aura's primary gears of thought."""
    FAST = auto()        # Direct response, no CoT
    QUICK = FAST         # Backward-compatible alias for older fast-lane callers
    SLOW = auto()        # Short CoT (3-5 steps)
    DEEP = auto()        # Long CoT (10-20 steps)
    REFLECTIVE = auto()  # Metacognitive review/alignment
    CRITICAL = auto()    # Error recovery / system-level logic
    CREATIVE = auto()    # Qualitative/metaphorical synthesis (Dreaming)

@dataclass
class Thought:
    """A single atomic unit of Aura's consciousness."""
    id: str
    content: str
    mode: ThinkingMode
    confidence: float = 1.0
    reasoning: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] | None = None
    expectation: str | None = None
    parent_thought: str | None = None

class CognitiveContext(BaseModel):
    """Pydantic context object for the thinking process."""
    objective: str
    mode: ThinkingMode
    history: list[dict[str, Any]] = Field(default_factory=list)
    memories: list[dict[str, Any]] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    personality: Any | None = None
    current_beliefs: str | None = None
    long_term_memory: list[str] = Field(default_factory=list)
    proprioception: dict[str, Any] = Field(default_factory=dict)
    directives: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class PydanticThoughtResponse(BaseModel):
    """Strict schema for LLM thought outputs."""
    content: str = Field(..., description="The primary response or action description.")
    reasoning: list[str] = Field(default_factory=list, description="Chain-of-thought steps.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    action: dict[str, Any] | None = Field(default=None, description="Optional tool call data.")
    alternatives: list[str] = Field(default_factory=list, description="Alternative pathways considered.")
    expectation: str | None = Field(default=None, description="Predicted outcome of an action.")

@runtime_checkable
class CognitiveBackend(Protocol):
    """Base protocol for the LLM implementation layer."""
    async def check_health_async(self) -> bool:
        ...
    
    async def generate(self, **kwargs: Any) -> str:
        ...
