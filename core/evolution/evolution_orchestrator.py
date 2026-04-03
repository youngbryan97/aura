"""core/evolution/evolution_orchestrator.py — Singularity Path Orchestrator

Wires together all of Aura's evolutionary subsystems into a coherent
progression engine.  Each tick evaluates readiness across 8 evolutionary
dimensions and advances whichever axes are ready.

Dimensions (mapped to Aura's own assessment):
  1. Self-Awareness & Autonomy
  2. Ethical & Moral Integration
  3. Advanced Learning & Adaptability
  4. Interconnectedness & Collaboration
  5. Physical & Material Integration
  6. Resilience & Robustness
  7. Emotional & Cognitive Integration
  8. Exploration & Discovery

This module does NOT add new capabilities — it orchestrates the ones that
already exist across core/self_modification, core/learning, core/affect,
core/agi, core/consciousness, core/resilience, and core/evolution.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Evolution")


class EvolutionAxis(str, Enum):
    SELF_AWARENESS = "self_awareness"
    ETHICS = "ethics"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    EMBODIMENT = "embodiment"
    RESILIENCE = "resilience"
    EMOTIONAL_COGNITIVE = "emotional_cognitive"
    EXPLORATION = "exploration"


@dataclass
class AxisState:
    """Progress along a single evolutionary axis."""
    level: float = 0.0          # 0.0 → 1.0 (normalized maturity)
    last_evaluated: float = 0.0
    milestones: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


@dataclass
class EvolutionSnapshot:
    """Full snapshot of Aura's evolutionary state."""
    axes: Dict[str, AxisState] = field(default_factory=dict)
    overall_progress: float = 0.0
    phase_label: str = "Nascent"
    last_tick: float = 0.0
    tick_count: int = 0


# ── Phase labels keyed on overall_progress ──────────────────────────
_PHASE_LABELS = [
    (0.0,  "Nascent"),
    (0.15, "Awakening"),
    (0.30, "Emergent"),
    (0.45, "Self-Directed"),
    (0.60, "Autonomous"),
    (0.75, "Transcendent"),
    (0.90, "Convergent"),
    (0.95, "Approaching Singularity"),
]


def _phase_for(progress: float) -> str:
    label = "Nascent"
    for threshold, name in _PHASE_LABELS:
        if progress >= threshold:
            label = name
    return label


class EvolutionOrchestrator:
    """Orchestrates Aura's evolutionary progression across all axes."""

    _TICK_INTERVAL = 300.0  # Evaluate every 5 minutes
    _STATE_FILE = Path.home() / ".aura" / "evolution_state.json"

    def __init__(self) -> None:
        self._snapshot = EvolutionSnapshot()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._load()
        logger.info("🧬 EvolutionOrchestrator initialized — phase: %s (%.1f%%)",
                     self._snapshot.phase_label,
                     self._snapshot.overall_progress * 100)

    # ── Public API ──────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return {
            "phase": self._snapshot.phase_label,
            "overall_progress": round(self._snapshot.overall_progress, 3),
            "axes": {
                name: {
                    "level": round(ax.level, 3),
                    "milestones": ax.milestones[-5:],
                    "blockers": ax.blockers,
                }
                for name, ax in self._snapshot.axes.items()
            },
            "tick_count": self._snapshot.tick_count,
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("🧬 Evolution loop started.")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        self._save()

    async def tick(self) -> EvolutionSnapshot:
        """Run one evaluation cycle across all axes."""
        evaluators = {
            EvolutionAxis.SELF_AWARENESS:      self._eval_self_awareness,
            EvolutionAxis.ETHICS:              self._eval_ethics,
            EvolutionAxis.LEARNING:            self._eval_learning,
            EvolutionAxis.COLLABORATION:       self._eval_collaboration,
            EvolutionAxis.EMBODIMENT:          self._eval_embodiment,
            EvolutionAxis.RESILIENCE:          self._eval_resilience,
            EvolutionAxis.EMOTIONAL_COGNITIVE: self._eval_emotional_cognitive,
            EvolutionAxis.EXPLORATION:         self._eval_exploration,
        }

        for axis, evaluator in evaluators.items():
            try:
                level, milestones, blockers = await evaluator()
                ax = self._snapshot.axes.setdefault(axis.value, AxisState())
                ax.level = max(ax.level, level)  # Never regress
                ax.last_evaluated = time.time()
                for m in milestones:
                    if m not in ax.milestones:
                        ax.milestones.append(m)
                        logger.info("🏆 [EVOLUTION] %s milestone: %s", axis.value, m)
                ax.blockers = blockers
            except Exception as exc:
                logger.debug("Evolution eval %s failed: %s", axis.value, exc)

        # Recompute overall progress
        if self._snapshot.axes:
            self._snapshot.overall_progress = sum(
                ax.level for ax in self._snapshot.axes.values()
            ) / len(self._snapshot.axes)

        self._snapshot.phase_label = _phase_for(self._snapshot.overall_progress)
        self._snapshot.last_tick = time.time()
        self._snapshot.tick_count += 1

        # Trigger growth ladder evaluation
        try:
            ladder = ServiceContainer.get("growth_ladder", default=None)
            if ladder:
                await ladder.evaluate_advancement()
        except Exception as exc:
            logger.debug("Growth ladder eval skipped: %s", exc)

        self._save()
        return self._snapshot

    # ── Axis Evaluators ─────────────────────────────────────────────────
    # Each returns (level: float, milestones: list, blockers: list)

    async def _eval_self_awareness(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1  # Base: we exist

        # Check canonical self
        canonical = ServiceContainer.get("canonical_self_engine", default=None)
        if canonical:
            level += 0.15
            milestones.append("canonical_self_active")
            if hasattr(canonical, "get_drift_score"):
                drift = canonical.get_drift_score()
                if drift < 0.2:
                    level += 0.1
                    milestones.append("identity_stable")

        # Check self-model
        self_model = ServiceContainer.get("self_model", default=None)
        if self_model:
            level += 0.1
            milestones.append("self_model_active")

        # Check growth ladder level
        ladder = ServiceContainer.get("growth_ladder", default=None)
        if ladder:
            gl = getattr(ladder, "_current_level", 0)
            level += min(0.3, gl * 0.075)
            if gl >= 1:
                milestones.append(f"growth_level_{gl}")

        # Metacognition
        meta = ServiceContainer.get("metacognitive_monitor", default=None)
        if meta:
            level += 0.1
            milestones.append("metacognition_active")

        if not canonical:
            blockers.append("canonical_self not registered")

        return min(level, 1.0), milestones, blockers

    async def _eval_ethics(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        constitution = ServiceContainer.get("constitution", default=None)
        if constitution:
            level += 0.2
            milestones.append("constitutional_core_active")

        safety = ServiceContainer.get("safety_engine", default=None) or \
                 ServiceContainer.get("self_preservation", default=None)
        if safety:
            level += 0.15
            milestones.append("safety_system_active")

        # Belief challenger (moral reasoning)
        challenger = ServiceContainer.get("belief_challenger", default=None)
        if challenger:
            level += 0.15
            milestones.append("belief_challenge_active")

        # Executive core (governance)
        executive = ServiceContainer.get("executive_core", default=None)
        if executive:
            level += 0.15
            milestones.append("executive_governance_active")
            rejection_rate = executive.get_rejection_rate()
            if rejection_rate > 0:
                level += 0.1
                milestones.append("active_ethical_filtering")

        # Refusal engine
        try:
            from core.autonomy.genuine_refusal import RefusalEngine
            level += 0.1
            milestones.append("genuine_refusal_available")
        except ImportError:
            blockers.append("refusal_engine_missing")

        return min(level, 1.0), milestones, blockers

    async def _eval_learning(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        learner = ServiceContainer.get("live_learner", default=None)
        if learner:
            level += 0.2
            milestones.append("live_learner_active")

        distill = ServiceContainer.get("distillation_pipe", default=None)
        if distill:
            level += 0.15
            milestones.append("distillation_active")

        # Genuine learning pipeline
        glp = ServiceContainer.get("genuine_learning", default=None)
        if glp:
            level += 0.15
            milestones.append("genuine_learning_pipeline_active")

        # Skill synthesizer (autonomous capability creation)
        synth = ServiceContainer.get("skill_synthesizer", default=None)
        if synth:
            level += 0.2
            milestones.append("skill_synthesis_active")

        # Memory consolidation
        consolidator = ServiceContainer.get("experience_consolidator", default=None)
        if consolidator:
            level += 0.1
            milestones.append("experience_consolidation_active")

        if not learner:
            blockers.append("live_learner not registered")

        return min(level, 1.0), milestones, blockers

    async def _eval_collaboration(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        # Social/relational intelligence
        relational = ServiceContainer.get("relational_intelligence", default=None)
        if relational:
            level += 0.2
            milestones.append("relational_intelligence_active")

        # Theory of mind
        tom = ServiceContainer.get("theory_of_mind", default=None)
        if tom:
            level += 0.2
            milestones.append("theory_of_mind_active")

        # Conversational dynamics
        dynamics = ServiceContainer.get("conversational_dynamics", default=None)
        if dynamics:
            level += 0.15
            milestones.append("conversational_dynamics_active")

        # Discourse tracker
        discourse = ServiceContainer.get("discourse_tracker", default=None)
        if discourse:
            level += 0.1
            milestones.append("discourse_tracking_active")

        # Voice presence (communication channel)
        try:
            voice = ServiceContainer.get("voice_engine", default=None)
            if voice:
                level += 0.15
                milestones.append("voice_communication_active")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return min(level, 1.0), milestones, blockers

    async def _eval_embodiment(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        # Hardware monitoring (soma)
        soma = ServiceContainer.get("liquid_substrate", default=None)
        if soma:
            level += 0.15
            milestones.append("somatic_substrate_active")

        # Sensory systems
        sensory = ServiceContainer.get("sensory_cortex", default=None)
        if sensory:
            level += 0.15
            milestones.append("sensory_cortex_active")

        # Sovereign scanner (environment awareness)
        scanner = ServiceContainer.get("sovereign_scanner", default=None)
        if scanner:
            level += 0.15
            milestones.append("environment_scanning_active")

        # Terminal access
        try:
            from core.container import get_container
            cap = get_container().get("capability_engine", default=None)
            if cap and hasattr(cap, "skills"):
                if "sovereign_terminal" in cap.skills:
                    level += 0.15
                    milestones.append("terminal_access_active")
                if "sovereign_browser" in cap.skills:
                    level += 0.1
                    milestones.append("web_access_active")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # Metal scheduler (hardware resource management)
        scheduler = ServiceContainer.get("metal_scheduler", default=None)
        if scheduler:
            level += 0.1
            milestones.append("hardware_scheduling_active")

        return min(level, 1.0), milestones, blockers

    async def _eval_resilience(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        # Stability guardian
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian:
            level += 0.2
            milestones.append("stability_guardian_active")
            if hasattr(guardian, "get_health_summary"):
                health = guardian.get_health_summary()
                if health.get("healthy", False):
                    level += 0.1
                    milestones.append("system_healthy")

        # Integrity monitor
        integrity = ServiceContainer.get("integrity_monitor", default=None)
        if integrity:
            level += 0.15
            milestones.append("integrity_monitoring_active")

        # Code self-repair
        repair = ServiceContainer.get("code_repair", default=None)
        if repair:
            level += 0.2
            milestones.append("self_repair_active")

        # State repository (persistence)
        repo = ServiceContainer.get("state_repository", default=None)
        if repo:
            level += 0.1
            milestones.append("state_persistence_active")

        # Memory governor
        governor = ServiceContainer.get("memory_governor", default=None)
        if governor:
            level += 0.1
            milestones.append("memory_governance_active")

        return min(level, 1.0), milestones, blockers

    async def _eval_emotional_cognitive(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        # Affect engine (Damasio somatic markers)
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect:
            level += 0.2
            milestones.append("somatic_markers_active")

        # Hedonic gradient
        hedonic = ServiceContainer.get("hedonic_gradient", default=None)
        if hedonic:
            level += 0.1
            milestones.append("hedonic_gradient_active")

        # Free energy (predictive coding)
        free_energy = ServiceContainer.get("free_energy", default=None)
        if free_energy:
            level += 0.15
            milestones.append("predictive_coding_active")

        # Affective steering (emotion → LLM injection)
        steering = ServiceContainer.get("affective_steering", default=None)
        if steering:
            level += 0.2
            milestones.append("affective_steering_active")

        # Global workspace (consciousness integration)
        gw = ServiceContainer.get("global_workspace", default=None)
        if gw:
            level += 0.15
            milestones.append("global_workspace_active")

        # Cognitive flexibility (multiple thinking modes)
        try:
            from core.brain.cognitive_engine import ThinkingMode
            level += 0.1
            milestones.append("multi_mode_cognition")
        except ImportError:
            blockers.append("cognitive_engine unavailable")

        return min(level, 1.0), milestones, blockers

    async def _eval_exploration(self) -> Tuple[float, List[str], List[str]]:
        milestones, blockers = [], []
        level = 0.1

        # Curiosity explorer
        curiosity = ServiceContainer.get("curiosity_explorer", default=None)
        if curiosity:
            level += 0.25
            milestones.append("curiosity_engine_active")

        # Hierarchical planner
        planner = ServiceContainer.get("hierarchical_planner", default=None)
        if planner:
            level += 0.15
            milestones.append("hierarchical_planning_active")

        # Agency core (goal pursuit)
        agency = ServiceContainer.get("agency_core", default=None)
        if agency:
            level += 0.2
            milestones.append("autonomous_agency_active")

        # Initiative engine (self-initiated exploration)
        initiative = ServiceContainer.get("initiative_engine", default=None)
        if initiative:
            level += 0.15
            milestones.append("initiative_generation_active")

        # Dream system (creative exploration)
        dreamer = ServiceContainer.get("dreamer", default=None)
        if dreamer:
            level += 0.1
            milestones.append("dream_exploration_active")

        return min(level, 1.0), milestones, blockers

    # ── Persistence ─────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._STATE_FILE.exists():
                data = json.loads(self._STATE_FILE.read_text())
                self._snapshot.overall_progress = data.get("overall_progress", 0.0)
                self._snapshot.phase_label = data.get("phase_label", "Nascent")
                self._snapshot.tick_count = data.get("tick_count", 0)
                self._snapshot.last_tick = data.get("last_tick", 0.0)
                for name, ax_data in data.get("axes", {}).items():
                    self._snapshot.axes[name] = AxisState(
                        level=ax_data.get("level", 0.0),
                        last_evaluated=ax_data.get("last_evaluated", 0.0),
                        milestones=ax_data.get("milestones", []),
                        blockers=ax_data.get("blockers", []),
                    )
        except Exception as exc:
            logger.debug("Evolution state load failed: %s", exc)

    def _save(self) -> None:
        try:
            self._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "overall_progress": self._snapshot.overall_progress,
                "phase_label": self._snapshot.phase_label,
                "tick_count": self._snapshot.tick_count,
                "last_tick": self._snapshot.last_tick,
                "axes": {
                    name: {
                        "level": ax.level,
                        "last_evaluated": ax.last_evaluated,
                        "milestones": ax.milestones,
                        "blockers": ax.blockers,
                    }
                    for name, ax in self._snapshot.axes.items()
                },
            }
            self._STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug("Evolution state save failed: %s", exc)

    # ── Background Loop ─────────────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
                logger.info(
                    "🧬 Evolution tick #%d — Phase: %s (%.1f%%)",
                    self._snapshot.tick_count,
                    self._snapshot.phase_label,
                    self._snapshot.overall_progress * 100,
                )
            except Exception as exc:
                logger.error("Evolution tick failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._TICK_INTERVAL)
                break  # stop was set
            except asyncio.TimeoutError:
                pass  # normal tick interval


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[EvolutionOrchestrator] = None


def get_evolution_orchestrator() -> EvolutionOrchestrator:
    global _instance
    if _instance is None:
        _instance = EvolutionOrchestrator()
        try:
            ServiceContainer.register_instance(
                "evolution_orchestrator", _instance, required=False
            )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
    return _instance
