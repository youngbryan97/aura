"""Reusable option policy schema."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Option:
    name: str
    description: str
    initiation_conditions: list[str]
    termination_conditions: list[str]
    expected_effects: list[str]
    failure_conditions: list[str]
    risk_tags: set[str]
    policy_name: str
    max_steps: int
    cooldown_steps: int = 0


@dataclass
class OptionRun:
    option_name: str
    started_seq: int
    ended_seq: int | None = None
    status: str = "running"
    steps: list[str] = field(default_factory=list)
    outcome_score: float = 0.0
    started_at: float = field(default_factory=time.time)


__all__ = ["Option", "OptionRun"]
