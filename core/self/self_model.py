import time
import logging
from typing import Dict, Any, Optional, List
from core.container import ServiceContainer
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.SelfModel")

class SelfModel:
    """
    Unified Internal Self-Model (ISM) — the single authoritative live object.

    All subsystems that need to know Aura's current state read from here.
    Properties are lazily resolved from the relevant services so this is
    always live, not a snapshot.

    Layers:
      identity       → HeartstoneDirective (immutable origin)
      values         → HeartstoneValues (evolved drive weights)
      affect         → AffectiveCircumplex (somatic LLM params)
      soma           → Soma.get_body_snapshot() (hardware metrics)
      architecture   → ArchitectureIndex (code self-awareness)
      goals          → AuraState.cognition.active_goals
      beliefs        → BeliefGraph (world model)
      uptime         → session age
    """

    def __init__(self):
        self._state_repo = None
        logger.info("SelfModel (Unified) initialized.")

    @property
    def state_repo(self):
        if self._state_repo is None:
            try:
                self._state_repo = ServiceContainer.get("state_repo")
            except Exception:
                return None
        return self._state_repo

    def _get_current_state(self) -> Optional[AuraState]:
        return getattr(self.state_repo, "_current", None)

    @property
    def identity(self) -> str:
        state = self._get_current_state()
        return state.identity.name if state else "Aura"

    @property
    def goals(self) -> Dict[str, Any]:
        state = self._get_current_state()
        if not state: return {}
        # Convert list of dicts to the dict format expected by the user's example
        return {g.get("name", f"goal_{i}"): g for i, g in enumerate(state.cognition.active_goals)}

    @property
    def beliefs(self) -> Dict[str, Any]:
        # Interface with BeliefGraph via GoalBeliefManager
        gbm = ServiceContainer.get("goal_belief_manager", default=None)
        if not gbm: return {}
        return {g["target"]: g for g in gbm.get_active_goals()}

    @beliefs.setter
    def beliefs(self, value: Dict[str, Any]):
        """Setter to allow restoration from persistence (Phase XXIII)."""
        # Silently allow assignment to prevent AttributeError.
        # In this facade, beliefs are derived from GBM, so we don't necessarily
        # need to store them locally, but we must allow the orchestrator to set them.
        self._beliefs_override = value

    @property
    def system_state(self) -> Dict[str, Any]:
        state = self._get_current_state()
        if not state: return {}
        return {
            "version": state.version,
            "mode": state.cognition.current_mode.value,
            "affect": state.affect.dominant_emotion,
            "last_thought": state.cognition.last_thought_at
        }

    def uptime(self) -> float:
        state = self._get_current_state()
        if not state: return 0.0
        return time.time() - state.created_at

    # ─── Affect & Soma ────────────────────────────────────────────────────────

    @property
    def affective_state(self) -> Dict[str, Any]:
        """Current (valence, arousal) + derived LLM params from AffectiveCircumplex."""
        try:
            from core.affect.affective_circumplex import get_circumplex
            return get_circumplex().get_llm_params()
        except Exception:
            return {"valence": 0.5, "arousal": 0.5, "temperature": 0.72, "max_tokens": 384}

    @property
    def somatic_state(self) -> Dict[str, Any]:
        """Live hardware metrics from Soma."""
        try:
            soma = ServiceContainer.get("soma", default=None)
            if soma:
                return soma.get_body_snapshot()
            from core.senses.soma import get_soma
            return get_soma().get_body_snapshot()
        except Exception:
            return {}

    @property
    def values(self) -> Dict[str, float]:
        """Evolved Darwinian Heartstone value weights."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            return get_heartstone_values().values
        except Exception:
            return {}

    # ─── Architecture Self-Awareness ──────────────────────────────────────────

    def query_architecture(self, topic: str) -> str:
        """Return architecture excerpts relevant to topic (delegates to ArchitectureIndex)."""
        try:
            arch = ServiceContainer.get("architecture_index", default=None)
            if arch is None:
                from core.self.architecture_index import get_architecture_index
                arch = get_architecture_index()
            return arch.query(topic)
        except Exception as e:
            return f"Architecture index unavailable: {e}"

    def architecture_overview(self) -> str:
        """High-level subsystem map."""
        try:
            arch = ServiceContainer.get("architecture_index", default=None)
            if arch is None:
                from core.self.architecture_index import get_architecture_index
                arch = get_architecture_index()
            return arch.get_overview()
        except Exception:
            return ""

    # ─── Comprehensive Snapshot ───────────────────────────────────────────────

    def full_snapshot(self) -> Dict[str, Any]:
        """
        Single authoritative snapshot of Aura's complete internal state.
        All subsystems read from this rather than querying services directly.
        """
        affect = self.affective_state
        soma   = self.somatic_state
        vals   = self.values
        sys_st = self.system_state

        return {
            "identity": self.identity,
            "uptime_hours": round(self.uptime() / 3600, 2),
            "system_state": sys_st,
            "affect": {
                "valence":     affect.get("valence", 0.5),
                "arousal":     affect.get("arousal", 0.5),
                "temperature": affect.get("temperature", 0.72),
                "max_tokens":  affect.get("max_tokens", 384),
                "narrative":   affect.get("narrative", ""),
            },
            "soma": soma.get("metrics", {}),
            "values": vals,
            "goals":  list(self.goals.keys())[:5],
            "beliefs_count": self._count_beliefs(),
        }

    def to_context_block(self) -> str:
        """Format the self-model for LLM system prompt injection."""
        snap = self.full_snapshot()
        affect = snap["affect"]
        vals   = snap["values"]

        lines = [
            f"## SELF-MODEL",
            f"Identity: {snap['identity']} | Uptime: {snap['uptime_hours']:.1f}h",
            f"Somatic: {affect['narrative']}",
        ]
        if vals:
            from core.affect.heartstone_values import get_heartstone_values
            lines.append(get_heartstone_values().to_context_block())
        return "\n".join(lines)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _count_beliefs(self) -> int:
        try:
            bg = ServiceContainer.get("belief_graph", default=None)
            if bg:
                return bg.graph.number_of_edges()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return 0

    def __repr__(self):
        return f"<SelfModel(identity='{self.identity}', goals={len(self.goals)})>"
