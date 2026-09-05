"""core/organism/life_events.py
Typed Pydantic event schemas for all canonical organism loop transitions.
Supports verification, hash linking, and secure receipt logging.
"""
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    LOCAL_ONLY = "local_only"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class EventBase(BaseModel):
    event_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    source: str

    previous_event_hash: Optional[str] = None


class PerceptionEvent(EventBase):
    modality: str
    raw_reference: Optional[str] = None
    parsed_content: Dict[str, Any]
    confidence: float
    uncertainty: float
    privacy_class: PrivacyClass = PrivacyClass.LOCAL_ONLY
    allowed_uses: List[str] = Field(default_factory=lambda: ["reasoning", "state_update"])
    linked_entities: List[str] = Field(default_factory=list)
    linked_current_goals: List[str] = Field(default_factory=list)
    receipt_id: Optional[str] = None


class BodyStateChanged(EventBase):
    energy_budget: float
    thermal_pressure: float
    memory_pressure: float
    model_capacity: str
    sensor_health: Dict[str, bool]
    actuator_health: Dict[str, bool]
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
    contradictions: List[str]
    supporting_evidence: List[str]
    downstream_uses: List[str]


class GoalCreated(EventBase):
    goal_id: str
    owner: str
    origin: str
    status: str
    priority: float
    urgency: float
    importance: float
    success_criteria: List[str]
    failure_criteria: List[str]
    allowed_tools: List[str]
    forbidden_tools: List[str]
    risk_class: str
    dependencies: List[str]


class AttentionSelected(EventBase):
    focus_id: str
    target_object: str
    reason_for_attention: str
    salience_score: float
    deadline: Optional[float] = None
    estimated_cost: float


class PlanProposed(EventBase):
    plan_id: str
    goal_id: str
    steps: List[Dict[str, Any]]
    tools_required: List[str]
    permissions_required: List[str]
    risks: List[str]
    expected_observations: List[str]
    fallbacks: List[str]
    verification_method: str
    abort_conditions: List[str]
    estimated_cost: float


class ActionRequested(EventBase):
    action_id: str
    channel: str
    params: Dict[str, Any]
    risk_score: float
    requires_approval: bool


class ActionApproved(EventBase):
    action_id: str
    approved_by: str
    posture: str
    capability_token: Optional[str] = None
    authority_receipt_id: Optional[str] = None


class ActionExecuted(EventBase):
    action_id: str
    receipt_id: str
    channel: str
    status: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None


class ConsequenceVerified(EventBase):
    action_id: str
    expected_evidence: str
    observed_evidence: str
    success: bool
    side_effects: List[str]
    mismatch_description: Optional[str] = None


class MemoryWritten(EventBase):
    memory_id: str
    type: str
    content: Dict[str, Any]
    sensitivity: PrivacyClass = PrivacyClass.LOCAL_ONLY
    linked_goals: List[str] = Field(default_factory=list)
    linked_beliefs: List[str] = Field(default_factory=list)


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
    conflicts: List[str]


class IdentityUpdated(EventBase):
    active_version: str
    active_modules: List[str]
    disabled_modules: List[str]
    known_limitations: List[str]
    active_permissions: List[str]
    capability_boundaries: List[str]


class RepairProposed(EventBase):
    repair_id: str
    subsystem: str
    issue_description: str
    patch_diff: str
    rollback_plan: str
