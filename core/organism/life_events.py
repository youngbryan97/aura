"""core/organism/life_events.py
Typed Pydantic event schemas for all canonical organism loop transitions.
Supports verification, hash linking, and secure receipt logging.
"""
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    LOCAL_ONLY = "local_only"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class EventBase(BaseModel):
    event_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    source: str

    previous_event_hash: str | None = None


class PerceptionEvent(EventBase):
    modality: str
    raw_reference: str | None = None
    parsed_content: dict[str, Any]
    confidence: float
    uncertainty: float
    privacy_class: PrivacyClass = PrivacyClass.LOCAL_ONLY
    allowed_uses: list[str] = Field(default_factory=lambda: ["reasoning", "state_update"])
    linked_entities: list[str] = Field(default_factory=list)
    linked_current_goals: list[str] = Field(default_factory=list)
    receipt_id: str | None = None


class BodyStateChanged(EventBase):
    energy_budget: float
    thermal_pressure: float
    memory_pressure: float
    model_capacity: str
    sensor_health: dict[str, bool]
    actuator_health: dict[str, bool]
    governance_integrity: float
    attention_saturation: float
    uncertainty_load: float
    error_rate: float
    repair_need: bool
    user_interruptibility: float


class BeliefUpdated(EventBase):
    belief_id: str
    content: str
    confidence: float
    decay_policy: str
    contradictions: list[str]
    supporting_evidence: list[str]
    downstream_uses: list[str]


class GoalCreated(EventBase):
    goal_id: str
    owner: str
    origin: str
    status: str
    priority: float
    urgency: float
    importance: float
    success_criteria: list[str]
    failure_criteria: list[str]
    allowed_tools: list[str]
    forbidden_tools: list[str]
    risk_class: str
    dependencies: list[str]


class AttentionSelected(EventBase):
    focus_id: str
    target_object: str
    reason_for_attention: str
    salience_score: float
    deadline: float | None = None
    estimated_cost: float


class PlanProposed(EventBase):
    plan_id: str
    goal_id: str
    steps: list[dict[str, Any]]
    tools_required: list[str]
    permissions_required: list[str]
    risks: list[str]
    expected_observations: list[str]
    fallbacks: list[str]
    verification_method: str
    abort_conditions: list[str]
    estimated_cost: float


class ActionRequested(EventBase):
    action_id: str
    channel: str
    params: dict[str, Any]
    risk_score: float
    requires_approval: bool


class ActionApproved(EventBase):
    action_id: str
    approved_by: str
    posture: str
    capability_token: str | None = None
    authority_receipt_id: str | None = None


class ActionExecuted(EventBase):
    action_id: str
    receipt_id: str
    channel: str
    status: str
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None


class ConsequenceVerified(EventBase):
    action_id: str
    expected_evidence: str
    observed_evidence: str
    success: bool
    side_effects: list[str]
    mismatch_description: str | None = None


class MemoryWritten(EventBase):
    memory_id: str
    type: str
    content: dict[str, Any]
    sensitivity: PrivacyClass = PrivacyClass.LOCAL_ONLY
    linked_goals: list[str] = Field(default_factory=list)
    linked_beliefs: list[str] = Field(default_factory=list)


class WelfareUpdated(EventBase):
    welfare_index: float
    energy: float
    stress: float
    distress_level: float
    sleep_debt: float


class ValueUpdated(EventBase):
    value_id: str
    statement: str
    priority: float
    hard_limit: bool
    conflicts: list[str]


class IdentityUpdated(EventBase):
    active_version: str
    active_modules: list[str]
    disabled_modules: list[str]
    known_limitations: list[str]
    active_permissions: list[str]
    capability_boundaries: list[str]


class RepairProposed(EventBase):
    repair_id: str
    subsystem: str
    issue_description: str
    patch_diff: str
    rollback_plan: str
