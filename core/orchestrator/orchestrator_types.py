from __future__ import annotations
from core.runtime.errors import record_degradation


from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Union
import asyncio
import logging

if TYPE_CHECKING:
    from core.brain.inference_gate import InferenceGate
    from core.bus.actor_bus import ActorBus
    from core.orchestrator.flow_control import CognitiveFlowController
    from core.tagged_reply_queue import TaggedReplyQueue
    from core.utils.queues import PriorityBackpressuredQueue

logger = logging.getLogger(__name__)

class SystemStatus(BaseModel):
    """System status tracking"""
    model_config = ConfigDict(validate_assignment=True)
    
    initialized: bool = False
    running: bool = False
    healthy: bool = False
    start_time: Optional[float] = None
    uptime: float = 0.0
    cycle_count: int = 0
    last_error: Optional[str] = None
    skills_loaded: int = 0
    dependencies_ok: bool = False
    is_processing: bool = False
    is_throttled: bool = False
    agency: float = 0.8
    curiosity: float = 0.5
    last_active: Optional[float] = None
    acceleration_factor: float = 1.0 # Phase 21: Cognitive Acceleration
    singularity_threshold: bool = False # Phase 21: Convergence State
    temporal_drift_s: float = 0.0 # Phase 22: Temporal Synchronization
    is_idle: bool = False
    message: str = "Standby"
    last_heartbeat: Optional[float] = None
    
    # Subsystem Aggregation (v5.0 Hardening)
    memory_status: Optional[Dict[str, Any]] = None
    agency_status: Optional[Dict[str, Any]] = None
    cognition_status: Optional[Dict[str, Any]] = None
    liquid_state_status: Optional[Dict[str, Any]] = None
    health_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    def add_error(self, error: str):
        self.last_error = error
        self.healthy = False

    @field_validator("singularity_threshold", mode="before")
    @classmethod
    def _coerce_singularity_threshold(cls, value: Any) -> bool:
        return bool(value)

class OrchestratorState(BaseModel):
    """Unified state model for the Orchestrator."""
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    boredom: int = 0
    stealth_mode: bool = False
    cycle_count: int = 0
    history_length: int = 0
    thoughts_snapshot: List[Dict[str, Any]] = Field(default_factory=list)
    active_objectives: List[str] = Field(default_factory=list)
    mycelial_cohesion: float = 1.0
    health_score: float = 1.0


@dataclass
class OrchestratorComponents:
    """Typed container for all subsystem references that the orchestrator
    lazy-loads during boot.  Replaces the ~30 ``Optional[Any]`` class
    attributes scattered across ``main.py``.

    **Usage** – instantiate once in ``__init__``, then access via
    ``self.components.<name>`` instead of ``self._<name>``.
    """

    # ── Core cognitive pipeline ────────────────────────────
    inference_gate: Optional[InferenceGate] = None
    capability_engine: Optional[Any] = None
    cognitive_engine: Optional[Any] = None

    # ── Coordination ───────────────────────────────────────
    actor_bus: Optional[ActorBus] = None
    supervisor_tree: Optional[Any] = None
    kernel_interface: Optional[Any] = None

    # ── Subsystems (alphabetical) ──────────────────────────
    agency_core: Optional[Any] = None
    autonomic_core: Optional[Any] = None
    ears: Optional[Any] = None
    global_workspace: Optional[Any] = None
    goal_hierarchy: Optional[Any] = None
    healing_service: Optional[Any] = None
    identity: Optional[Any] = None
    intent_router: Optional[Any] = None
    knowledge_graph: Optional[Any] = None
    liquid_state: Optional[Any] = None
    memory_manager: Optional[Any] = None
    memory_optimizer: Optional[Any] = None
    meta_cognition: Optional[Any] = None
    meta_learning: Optional[Any] = None
    metabolic_monitor: Optional[Any] = None
    personality_engine: Optional[Any] = None
    project_store: Optional[Any] = None
    scratchpad_engine: Optional[Any] = None
    self_healer: Optional[Any] = None
    self_model: Optional[Any] = None
    singularity_monitor: Optional[Any] = None
    state_machine: Optional[Any] = None
    strategic_planner: Optional[Any] = None
    subsystem_audit: Optional[Any] = None
    world_state: Optional[Any] = None

    # ── Telemetry / Monitoring ─────────────────────────────
    event_loop_monitor: Optional[Any] = None
    integrity_monitor: Optional[Any] = None

    # ── Sensory ────────────────────────────────────────────
    sensory_actor: Optional[Any] = None
    last_sensory_heartbeat: float = 0.0


def _bg_task_exception_handler(task: asyncio.Task):
    """Callback for background tasks to log exceptions instead of losing them."""
    try:
        exc = task.exception()
        if exc:
            logger.warning("Background task %s failed: %s", task.get_name(), exc)
            try:
                from ..container import ServiceContainer
                immune = ServiceContainer.get("immune_system", None)
                if immune:
                    immune.on_error(exc, {"task": task.get_name()})
            except Exception as e:
                record_degradation('orchestrator_types', e)
                logger.debug("Immune system unavailable for background task error logging: %s", e)
    except asyncio.CancelledError as exc:
        logger.debug('Task was cancelled: %s', exc)
    except asyncio.InvalidStateError as exc:
        logger.debug('Task in invalid state: %s', exc)
