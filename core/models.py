import time
from typing import Any

from pydantic import BaseModel, Field

ToolCallPayload = dict[str, Any]


class ExecutionPlan(BaseModel):
    goal: str
    plan_steps: list[str]
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    plan_hash: str | None = None


class Insight(BaseModel):
    id: str
    title: str
    content: str
    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: float
    source: str
    tags: list[str] = Field(default_factory=list)
    impact_score: float = Field(default=0.5)


class SystemHealthState(BaseModel):
    uptime: float
    memory_usage: float
    active_shards: int
    unresolved_refinements: int
    cognitive_stability: float = Field(ge=0.0, le=1.0)
    timestamp: float
