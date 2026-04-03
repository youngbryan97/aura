from typing import Protocol, runtime_checkable, Optional, Dict, List, Any, AsyncIterator
from enum import Enum, auto
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict

@runtime_checkable
class LLMClient(Protocol):
    """Protocol for any LLM backend (Gemini, Local MLX)."""
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
    reasoning: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    action: Optional[Dict[str, Any]] = None
    expectation: Optional[str] = None
    parent_thought: Optional[str] = None

class CognitiveContext(BaseModel):
    """Pydantic context object for the thinking process."""
    objective: str
    mode: ThinkingMode
    history: List[Dict[str, Any]] = Field(default_factory=list)
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)
    personality: Optional[Any] = None
    current_beliefs: Optional[str] = None
    long_term_memory: List[str] = Field(default_factory=list)
    proprioception: Dict[str, Any] = Field(default_factory=dict)
    directives: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class PydanticThoughtResponse(BaseModel):
    """Strict schema for LLM thought outputs."""
    content: str = Field(..., description="The primary response or action description.")
    reasoning: List[str] = Field(default_factory=list, description="Chain-of-thought steps.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    action: Optional[Dict[str, Any]] = Field(default=None, description="Optional tool call data.")
    alternatives: List[str] = Field(default_factory=list, description="Alternative pathways considered.")
    expectation: Optional[str] = Field(default=None, description="Predicted outcome of an action.")

@runtime_checkable
class CognitiveBackend(Protocol):
    """Base protocol for the LLM implementation layer."""
    async def check_health_async(self) -> bool:
        ...
    
    async def generate(self, **kwargs: Any) -> str:
        ...