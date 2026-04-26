from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def clamp01(value: float) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.0


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return repr(value)


def stable_digest(*parts: Any, length: int = 16) -> str:
    raw = "|".join(str(p) for p in parts).encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:length]


class SignalKind(str, Enum):
    TASK = "task"
    USER_NEED = "user_need"
    ERROR = "error"
    EXCEPTION = "exception"
    DANGER = "danger"
    RESOURCE_PRESSURE = "resource_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    NOVELTY = "novelty"
    CURIOSITY = "curiosity"
    REPAIR = "repair"
    GROWTH = "growth"
    SOCIAL = "social"
    INHIBITION = "inhibition"
    HOMEOSTASIS = "homeostasis"
    HEARTBEAT = "heartbeat"


class CellRole(str, Enum):
    STEM = "stem"
    SENSOR = "sensor"
    EFFECTOR = "effector"
    MEMORY = "memory"
    REPAIR = "repair"
    ROUTER = "router"
    GOVERNOR = "governor"
    ORGAN = "organ"


class CellLifecycle(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    HIBERNATING = "hibernating"
    QUARANTINED = "quarantined"
    APOPTOTIC = "apoptotic"
    DEAD = "dead"


@dataclass(frozen=True)
class MorphogenSignal:
    """A local field perturbation.

    Signals are intentionally generic.  They can encode a task, exception,
    health event, resource pressure, or curiosity/novelty cue.  Runtime code
    turns these signals into field gradients and local cell decisions.
    """

    kind: SignalKind | str
    source: str
    subsystem: str
    intensity: float = 0.5
    payload: Dict[str, Any] = field(default_factory=dict)
    target_cell_id: Optional[str] = None
    ttl_ticks: int = 6
    timestamp: float = field(default_factory=time.time)
    signal_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "intensity", clamp01(self.intensity))
        if not self.signal_id:
            object.__setattr__(
                self,
                "signal_id",
                "sig_" + stable_digest(self.kind, self.source, self.subsystem, self.timestamp),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "kind": self.kind.value if isinstance(self.kind, SignalKind) else str(self.kind),
            "source": self.source,
            "subsystem": self.subsystem,
            "intensity": round(float(self.intensity), 5),
            "payload": json_safe(self.payload),
            "target_cell_id": self.target_cell_id,
            "ttl_ticks": int(self.ttl_ticks),
            "timestamp": float(self.timestamp),
        }


@dataclass
class CellManifest:
    """Gene-expression profile for a code-cell.

    This is the contract you want every Aura module/skill/service to expose.
    The morphogenetic runtime uses it for routing, resource budgets, local
    decisions, organ formation and safe pruning.
    """

    name: str
    role: CellRole | str = CellRole.STEM
    subsystem: str = "generic"
    capabilities: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)
    emits: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    protected: bool = False
    criticality: float = 0.5
    baseline_energy: float = 0.35
    max_energy: float = 1.0
    activation_threshold: float = 0.18
    replication_threshold: float = 0.82
    differentiation_threshold: float = 0.72
    hibernation_threshold: float = 0.08
    max_parallel_tasks: int = 1
    timeout_s: float = 4.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def canonical_id(self) -> str:
        return f"cell_{stable_digest(self.name, self.role, self.subsystem)}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value if isinstance(self.role, CellRole) else str(self.role)
        return json_safe(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellManifest":
        payload = dict(data or {})
        role = payload.get("role", CellRole.STEM)
        try:
            payload["role"] = CellRole(role)
        except Exception:
            payload["role"] = str(role)
        return cls(**payload)


@dataclass
class CellState:
    lifecycle: CellLifecycle | str = CellLifecycle.ACTIVE
    health: float = 1.0
    energy: float = 0.35
    age_ticks: int = 0
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_activation_at: float = 0.0
    born_at: float = field(default_factory=time.time)
    last_error: str = ""
    lineage_id: str = ""
    generation: int = 0
    specialisation_score: float = 0.0
    confidence: float = 0.5
    quarantined_until: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lifecycle"] = (
            self.lifecycle.value if isinstance(self.lifecycle, CellLifecycle) else str(self.lifecycle)
        )
        return json_safe(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellState":
        payload = dict(data or {})
        lifecycle = payload.get("lifecycle", CellLifecycle.ACTIVE)
        try:
            payload["lifecycle"] = CellLifecycle(lifecycle)
        except Exception:
            payload["lifecycle"] = str(lifecycle)
        return cls(**payload)


@dataclass(frozen=True)
class MorphogenesisConfig:
    enabled: bool = True
    tick_interval_s: float = 1.0
    max_cells: int = 256
    max_organs: int = 48
    max_signals_per_tick: int = 128
    max_cell_actions_per_tick: int = 64
    field_decay: float = 0.08
    field_diffusion: float = 0.18
    signal_decay_per_tick: int = 1
    energy_recovery_per_tick: float = 0.035
    health_recovery_per_tick: float = 0.006
    failure_health_penalty: float = 0.08
    success_health_reward: float = 0.025
    dormant_after_idle_ticks: int = 900
    dead_after_apoptotic_ticks: int = 30
    quarantine_s: float = 300.0
    organ_min_coactivations: int = 6
    organ_min_members: int = 2
    organ_edge_threshold: float = 0.62
    snapshot_every_ticks: int = 15
    episode_every_events: int = 10
    adaptive_immunity_bridge: bool = True
    strict_no_source_patch_apply: bool = True
    require_governance_for_mutation: bool = True
    runtime_name: str = "morphogenesis"

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))
