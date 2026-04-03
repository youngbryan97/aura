from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time

class ExecutionPlan(BaseModel):
    goal: str
    plan_steps: List[str]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list) # Placeholder for ToolCall model if needed
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    plan_hash: Optional[str] = None

class Insight(BaseModel):
    id: str
    title: str
    content: str
    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: float
    source: str
    tags: List[str] = Field(default_factory=list)
    impact_score: float = Field(default=0.5)

class SystemHealthState(BaseModel):
    uptime: float
    memory_usage: float
    active_shards: int
    unresolved_refinements: int
    cognitive_stability: float = Field(ge=0.0, le=1.0)
    timestamp: float
