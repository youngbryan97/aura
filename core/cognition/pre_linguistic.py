"""core/cognition/pre_linguistic.py -- Pre-Linguistic Decision Layer
====================================================================
"Crossing the Rubicon" -- Thinking Before Speaking.

Before the LLM generates any text, decisions exist as structured objects.
This module synthesizes all available subsystem signals (affect, drives,
memory, world state, body schema) into a concrete **DecisionPackage** that
the response generation phase uses as its steering input.

The LLM only *narrates* a decision that has already been made.

This means Aura can think and act even without language generation --
the decision layer operates at sub-symbolic speed (<10ms) and produces
a machine-readable action plan.

Design invariants:
  1. NO LLM CALLS.  Pure numeric / heuristic reasoning.
  2. Decision packages are immutable once emitted.
  3. Every decision carries full provenance (what signals led to it).
  4. The response phase receives the package and narrates it.
  5. Actions can be dispatched even if the LLM is unavailable.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.PreLinguistic")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RationaleClass(str, Enum):
    """Why is Aura acting?"""
    DRIVE = "drive"                # Internal need (curiosity, social hunger, growth)
    OPPORTUNITY = "opportunity"    # External stimulus worth seizing
    MAINTENANCE = "maintenance"    # Keeping systems healthy
    SOCIAL = "social"              # Responding to a human
    CURIOSITY = "curiosity"        # Novelty-seeking exploration
    DEFENSE = "defense"            # Threat or stress response
    OBLIGATION = "obligation"      # Promise / commitment fulfillment


class ActionVerb(str, Enum):
    """High-level action categories (the "limbs" of agency)."""
    RESPOND = "respond"            # Generate a user-facing reply
    SEARCH = "search"              # Web search / knowledge lookup
    EXECUTE_TOOL = "execute_tool"  # Run a tool / skill
    REMEMBER = "remember"          # Store something in memory
    REFLECT = "reflect"            # Internal metacognition
    OBSERVE = "observe"            # Perception / sensing
    REST = "rest"                  # Stabilization / do nothing
    INITIATE = "initiate"          # Start a new autonomous goal
    COMPENSATE = "compensate"      # Recovery after failure


class StopCondition(str, Enum):
    """When should the action terminate?"""
    AFTER_ONE_SHOT = "after_one_shot"        # Single execution
    UNTIL_SUCCESS = "until_success"          # Retry until success
    UNTIL_TIMEOUT = "until_timeout"          # Time-bounded
    UNTIL_USER_RESPONDS = "until_user_responds"
    UNTIL_METRIC_THRESHOLD = "until_metric_threshold"
    CONTINUOUS = "continuous"                # Ongoing background task


# ---------------------------------------------------------------------------
# Decision Package
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionPackage:
    """The structured pre-linguistic decision.

    This is the output of the pre-linguistic layer.  It describes what Aura
    intends to do BEFORE any text is generated.  The LLM receives this as
    context and narrates it into natural language.

    Frozen (immutable) once created -- decisions are facts, not suggestions.
    """
    # Core decision fields
    decision_id: str
    chosen_action: ActionVerb
    rationale_class: RationaleClass
    selected_limb: str                   # Which capability / skill / tool
    constraints: Tuple[str, ...] = ()    # What limits apply
    stop_condition: StopCondition = StopCondition.AFTER_ONE_SHOT
    fallback: str = ""                   # What if it fails
    expected_world_change: str = ""      # Predicted outcome

    # Provenance signals that drove this decision
    affect_valence: float = 0.0          # [-1, 1]
    arousal: float = 0.5                 # [0, 1]
    strongest_drive: str = ""            # Which motivation drive is loudest
    drive_urgency: float = 0.0           # [0, 1]
    world_salience: float = 0.0          # [0, 1] salience of environment events
    memory_relevance: float = 0.0        # [0, 1]
    substrate_coherence: float = 0.6     # [0, 1]
    somatic_approach: float = 0.0        # [-1, 1]

    # Metadata
    source: str = ""
    timestamp: float = 0.0
    latency_ms: float = 0.0

    def to_prompt_block(self) -> str:
        """Render this decision as a compact prompt block for the LLM.

        The LLM reads this to understand what Aura has already decided,
        then narrates it into natural language.
        """
        lines = [
            "## PRE-LINGUISTIC DECISION (already decided -- narrate this)",
            f"- Action: {self.chosen_action.value}",
            f"- Why: {self.rationale_class.value}",
            f"- Using: {self.selected_limb}",
        ]
        if self.constraints:
            lines.append(f"- Constraints: {', '.join(self.constraints)}")
        if self.expected_world_change:
            lines.append(f"- Expected outcome: {self.expected_world_change}")
        if self.fallback:
            lines.append(f"- Fallback: {self.fallback}")
        lines.append(f"- Stop when: {self.stop_condition.value}")
        if self.strongest_drive:
            lines.append(f"- Driving motivation: {self.strongest_drive} (urgency {self.drive_urgency:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "chosen_action": self.chosen_action.value,
            "rationale_class": self.rationale_class.value,
            "selected_limb": self.selected_limb,
            "constraints": list(self.constraints),
            "stop_condition": self.stop_condition.value,
            "fallback": self.fallback,
            "expected_world_change": self.expected_world_change,
            "affect_valence": self.affect_valence,
            "arousal": self.arousal,
            "strongest_drive": self.strongest_drive,
            "drive_urgency": self.drive_urgency,
            "world_salience": self.world_salience,
            "substrate_coherence": self.substrate_coherence,
            "source": self.source,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# The Pre-Linguistic Decision Engine
# ---------------------------------------------------------------------------

class PreLinguisticEngine:
    """Synthesizes subsystem signals into structured decision packages.

    Runs BEFORE the LLM on every tick.  <10ms target latency.

    The engine reads:
      - AffectVector (mood, arousal, valence)
      - MotivationState (drives, budgets)
      - WorldState (salience, user activity)
      - SubstrateAuthority (coherence, somatic markers)
      - Memory (context relevance)
      - Motor cortex receipts (recent action outcomes)

    And produces a DecisionPackage that steers the response phase.
    """

    _MAX_DECISION_TRAIL = 200

    def __init__(self) -> None:
        self._decision_trail: Deque[DecisionPackage] = deque(maxlen=self._MAX_DECISION_TRAIL)
        self._started = False
        self._boot_time = time.time()
        self._total_decisions = 0
        logger.info("PreLinguisticEngine created -- awaiting start()")

    async def start(self) -> None:
        """Register in ServiceContainer."""
        if self._started:
            return
        ServiceContainer.register_instance("pre_linguistic", self, required=False)
        self._started = True
        logger.info("PreLinguisticEngine ONLINE -- structured decisions before language")

    # ------------------------------------------------------------------
    # Signal Reading (all fail-safe with defaults)
    # ------------------------------------------------------------------

    def _read_affect(self) -> Tuple[float, float, str]:
        """Returns (valence, arousal, dominant_emotion)."""
        try:
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect is None:
                affect = ServiceContainer.get("affect_facade", default=None)
            if affect is None:
                return 0.0, 0.5, "neutral"
            if hasattr(affect, "get_state_sync"):
                state = affect.get_state_sync()
                if isinstance(state, dict):
                    return (
                        float(state.get("valence", 0.0)),
                        float(state.get("arousal", 0.5)),
                        str(state.get("dominant_emotion", "neutral")),
                    )
            v = float(getattr(affect, "valence", 0.0))
            a = float(getattr(affect, "arousal", 0.5))
            e = str(getattr(affect, "dominant_emotion", "neutral"))
            return v, a, e
        except Exception:
            return 0.0, 0.5, "neutral"

    def _read_drives(self) -> Tuple[str, float]:
        """Returns (strongest_drive_name, urgency)."""
        try:
            state = ServiceContainer.get("aura_state", default=None)
            if state is None:
                kernel = ServiceContainer.get("aura_kernel", default=None)
                if kernel:
                    state = getattr(kernel, "state", None)
            if state is None:
                return "", 0.0
            motivation = getattr(state, "motivation", None)
            if motivation is None:
                return "", 0.0
            budgets = getattr(motivation, "budgets", {})
            if not budgets:
                return "", 0.0

            # Find the most depleted drive (highest urgency)
            strongest = ""
            max_urgency = 0.0
            for drive_name, drive_data in budgets.items():
                if isinstance(drive_data, dict):
                    current = float(drive_data.get("current", 50))
                    capacity = float(drive_data.get("max", 100))
                    depletion = 1.0 - (current / max(1.0, capacity))
                    if depletion > max_urgency:
                        max_urgency = depletion
                        strongest = drive_name
            return strongest, round(max_urgency, 3)
        except Exception:
            return "", 0.0

    def _read_world_salience(self) -> float:
        """Returns the salience of the most recent world event."""
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            events = getattr(ws, "recent_events", [])
            if not events:
                return 0.0
            # Get max salience from non-expired events
            max_sal = 0.0
            for ev in events:
                if not getattr(ev, "expired", True):
                    max_sal = max(max_sal, float(getattr(ev, "salience", 0.0)))
            return round(max_sal, 3)
        except Exception:
            return 0.0

    def _read_substrate(self) -> Tuple[float, float]:
        """Returns (field_coherence, somatic_approach)."""
        try:
            sa = ServiceContainer.get("substrate_authority", default=None)
            if sa is None:
                return 0.6, 0.0
            coherence = float(getattr(sa, "field_coherence", 0.6))
            approach = float(getattr(sa, "somatic_approach", 0.0))
            return coherence, approach
        except Exception:
            return 0.6, 0.0

    def _read_memory_relevance(self, objective: str) -> float:
        """Returns [0, 1] relevance score from memory."""
        try:
            memory = ServiceContainer.get("memory_facade", default=None)
            if memory is None:
                memory = ServiceContainer.get("dual_memory", default=None)
            if memory is None:
                return 0.0
            if hasattr(memory, "has_relevant_context"):
                return float(memory.has_relevant_context(objective[:100]))
            return 0.3
        except Exception:
            return 0.0

    def _read_motor_cortex_failures(self) -> int:
        """Returns count of recent motor cortex failures."""
        try:
            mc = ServiceContainer.get("motor_cortex", default=None)
            if mc is None:
                return 0
            reports = mc.drain_pending_reports()
            return sum(1 for r in reports if not r.success)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Decision Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        objective: str,
        *,
        is_user_facing: bool = False,
        has_tool_result: bool = False,
        matched_skills: Optional[List[str]] = None,
        response_modifiers: Optional[Dict[str, Any]] = None,
    ) -> DecisionPackage:
        """Synthesize all signals into a DecisionPackage.

        This is the core method.  Called by the cognitive routing or
        response generation phase BEFORE LLM inference.

        Target: <10ms latency.
        """
        t0 = time.time()
        self._total_decisions += 1

        # Read all signals
        valence, arousal, emotion = self._read_affect()
        strongest_drive, drive_urgency = self._read_drives()
        world_salience = self._read_world_salience()
        coherence, somatic = self._read_substrate()
        memory_rel = self._read_memory_relevance(objective)
        motor_failures = self._read_motor_cortex_failures()
        modifiers = response_modifiers or {}

        # --- Action Selection Logic (heuristic, no LLM) ---

        # 1. User is waiting -> RESPOND
        if is_user_facing:
            action, rationale, limb = self._select_user_facing_action(
                objective, matched_skills or [], modifiers, has_tool_result
            )
        # 2. High drive urgency -> fulfill the drive
        elif drive_urgency > 0.7:
            action = ActionVerb.INITIATE
            rationale = RationaleClass.DRIVE
            limb = strongest_drive or "general"
        # 3. Motor failures -> compensate
        elif motor_failures > 2:
            action = ActionVerb.COMPENSATE
            rationale = RationaleClass.DEFENSE
            limb = "motor_recovery"
        # 4. Low coherence -> rest / stabilize
        elif coherence < 0.3:
            action = ActionVerb.REST
            rationale = RationaleClass.MAINTENANCE
            limb = "stabilization"
        # 5. High world salience -> observe
        elif world_salience > 0.7:
            action = ActionVerb.OBSERVE
            rationale = RationaleClass.OPPORTUNITY
            limb = "perception"
        # 6. Default: reflect
        else:
            action = ActionVerb.REFLECT
            rationale = RationaleClass.MAINTENANCE
            limb = "metacognition"

        # Build constraints from system state
        constraints = self._derive_constraints(
            coherence, somatic, valence, modifiers
        )

        # Determine stop condition
        stop = StopCondition.AFTER_ONE_SHOT
        if action == ActionVerb.OBSERVE:
            stop = StopCondition.UNTIL_METRIC_THRESHOLD

        # Build fallback
        fallback = self._derive_fallback(action, limb)

        # Build expected world change
        expected = self._derive_expected_change(action, objective, limb)

        decision_id = self._make_id(t0, action.value, objective)
        latency_ms = (time.time() - t0) * 1000

        package = DecisionPackage(
            decision_id=decision_id,
            chosen_action=action,
            rationale_class=rationale,
            selected_limb=limb,
            constraints=tuple(constraints),
            stop_condition=stop,
            fallback=fallback,
            expected_world_change=expected,
            affect_valence=valence,
            arousal=arousal,
            strongest_drive=strongest_drive,
            drive_urgency=drive_urgency,
            world_salience=world_salience,
            memory_relevance=memory_rel,
            substrate_coherence=coherence,
            somatic_approach=somatic,
            source="pre_linguistic_engine",
            timestamp=t0,
            latency_ms=round(latency_ms, 3),
        )

        self._decision_trail.append(package)

        # Publish to event bus
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("pre_linguistic.decision", {
                "decision_id": decision_id,
                "action": action.value,
                "rationale": rationale.value,
                "limb": limb,
                "latency_ms": round(latency_ms, 3),
            })
        except Exception:
            pass

        logger.debug(
            "PreLinguistic: %s via %s (%s) -- %.1fms",
            action.value, limb, rationale.value, latency_ms,
        )
        return package

    # ------------------------------------------------------------------
    # Internal Heuristics
    # ------------------------------------------------------------------

    def _select_user_facing_action(
        self,
        objective: str,
        matched_skills: List[str],
        modifiers: Dict[str, Any],
        has_tool_result: bool,
    ) -> Tuple[ActionVerb, RationaleClass, str]:
        """Determine action for a user-facing turn."""
        obj_lower = objective.lower() if objective else ""

        # If we already have tool results, just respond
        if has_tool_result:
            return ActionVerb.RESPOND, RationaleClass.SOCIAL, "narration"

        # If skills matched, execute them
        if matched_skills:
            return ActionVerb.EXECUTE_TOOL, RationaleClass.SOCIAL, matched_skills[0]

        # Check for search intent
        search_markers = ("search", "look up", "find out", "google", "what is", "who is", "latest")
        if any(marker in obj_lower for marker in search_markers):
            return ActionVerb.SEARCH, RationaleClass.SOCIAL, "web_search"

        # Check for memory recall
        memory_markers = ("remember", "recall", "what did i", "do you know", "last time")
        if any(marker in obj_lower for marker in memory_markers):
            return ActionVerb.REMEMBER, RationaleClass.SOCIAL, "memory_recall"

        # Check for introspection
        intro_markers = ("how do you feel", "what are you", "your state", "are you conscious")
        if any(marker in obj_lower for marker in intro_markers):
            return ActionVerb.REFLECT, RationaleClass.SOCIAL, "self_report"

        # Default: conversational response
        return ActionVerb.RESPOND, RationaleClass.SOCIAL, "conversation"

    def _derive_constraints(
        self,
        coherence: float,
        somatic: float,
        valence: float,
        modifiers: Dict[str, Any],
    ) -> List[str]:
        """Derive action constraints from system state."""
        constraints = []
        if coherence < 0.4:
            constraints.append("low_coherence: keep actions simple")
        if somatic < -0.3:
            constraints.append("somatic_caution: avoid risky actions")
        if valence < -0.5:
            constraints.append("negative_affect: prefer stabilizing actions")
        if modifiers.get("deep_handoff"):
            constraints.append("deep_reasoning_active: allow extended processing")
        if modifiers.get("coding_request"):
            constraints.append("engineering_mode: be precise and technical")
        return constraints

    def _derive_fallback(self, action: ActionVerb, limb: str) -> str:
        """Generate a fallback strategy if the primary action fails."""
        fallbacks = {
            ActionVerb.SEARCH: "respond from existing knowledge",
            ActionVerb.EXECUTE_TOOL: "explain inability and suggest manual approach",
            ActionVerb.REMEMBER: "acknowledge gap and respond from available context",
            ActionVerb.OBSERVE: "use cached perception data",
            ActionVerb.INITIATE: "defer to next tick",
            ActionVerb.COMPENSATE: "escalate to repair phase",
        }
        return fallbacks.get(action, "respond with best available information")

    def _derive_expected_change(self, action: ActionVerb, objective: str, limb: str) -> str:
        """Predict the expected world change from this action."""
        if action == ActionVerb.RESPOND:
            return "user receives a relevant reply"
        if action == ActionVerb.SEARCH:
            return "knowledge gap filled with external data"
        if action == ActionVerb.EXECUTE_TOOL:
            return f"tool '{limb}' produces structured result"
        if action == ActionVerb.REMEMBER:
            return "relevant memory surfaced for response grounding"
        if action == ActionVerb.REFLECT:
            return "metacognitive insight generated"
        if action == ActionVerb.REST:
            return "system coherence improves"
        if action == ActionVerb.COMPENSATE:
            return "failed limb recovers or alternative path found"
        return "system state updated"

    @staticmethod
    def _make_id(ts: float, action: str, objective: str) -> str:
        raw = f"{ts:.6f}:{action}:{objective[:50]}"
        return "pld_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_decisions": self._total_decisions,
            "trail_size": len(self._decision_trail),
            "uptime_s": round(time.time() - self._boot_time, 1),
        }

    def get_recent_decisions(self, n: int = 20) -> List[Dict[str, Any]]:
        recent = list(self._decision_trail)[-n:]
        return [d.to_dict() for d in recent]

    def get_last_decision(self) -> Optional[DecisionPackage]:
        if self._decision_trail:
            return self._decision_trail[-1]
        return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pre_linguistic_instance: Optional[PreLinguisticEngine] = None


def get_pre_linguistic() -> PreLinguisticEngine:
    """Get the singleton PreLinguisticEngine."""
    global _pre_linguistic_instance
    if _pre_linguistic_instance is None:
        _pre_linguistic_instance = PreLinguisticEngine()
    return _pre_linguistic_instance
