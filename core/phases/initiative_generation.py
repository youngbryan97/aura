import logging
import time
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState
from ..consciousness.executive_authority import get_executive_authority
from core.runtime.background_policy import background_activity_allowed

logger = logging.getLogger(__name__)

class InitiativeGenerationPhase(BasePhase):
    """
    Phase 8: Initiative Generation.
    Decides whether Aura should take autonomous action or start a thought
    process based on her boredom, curiosity, and internal goals.
    """

    def __init__(self, container: Any):
        self.container = container
        self._last_impulse_time = 0.0

    @staticmethod
    def _autonomy_pause_reason() -> str:
        try:
            from core.container import ServiceContainer

            router = ServiceContainer.get("llm_router", default=None)
            if router and getattr(router, "high_pressure_mode", False):
                return "memory_pressure"

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                reason = str(gate._background_local_deferral_reason(origin="initiative_generation") or "").strip()
                if reason:
                    return reason
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return ""

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Decide whether Aura should generate an autonomous initiative this tick.

        Checks the 60-second throttle, conversation flow guards (no monologue after
        user, no consecutive solo messages), and affect thresholds (curiosity,
        social_hunger, or boredom).  When triggered, decays the driving affect value,
        selects an appropriate goal, and appends an impulse to
        state.cognition.pending_initiatives.
        """
        # 1. Don't generate initiatives if already in conversation or if throttled
        pause_reason = self._autonomy_pause_reason()
        if pause_reason:
            logger.debug("⚡ InitiativeGeneration: paused while runtime is guarded (%s).", pause_reason)
            return state

        try:
            from core.container import ServiceContainer
            orch = ServiceContainer.get("orchestrator", default=None)
        except Exception:
            orch = None
        if not background_activity_allowed(
            orch,
            min_idle_seconds=900.0,
            max_memory_percent=80.0,
            max_failure_pressure=0.12,
            require_conversation_ready=True,
        ):
            return state

        now = time.time()
        if now - self._last_impulse_time < 60.0:  # 60s minimum between autonomous impulses
            return state

        # Suppress autonomous impulses during deep active conversation.
        # If a real exchange is underway (high energy, depth > 3), don't interrupt.
        discourse_depth = getattr(state.cognition, "discourse_depth", 0)
        conv_energy = getattr(state.cognition, "conversation_energy", 0.5)
        if discourse_depth > 3 and conv_energy > 0.4:
            return state

        wm = state.cognition.working_memory
        if wm:
            last_msg = wm[-1]
            last_role = last_msg.get("role", "")

            # Never speak immediately after the user — respect response flow
            if last_role == "user":
                return state

            # Prevent monologue: if the last N messages are all assistant, back off.
            # Count trailing consecutive assistant messages.
            consecutive_assistant = 0
            for msg in reversed(wm):
                if msg.get("role") == "assistant":
                    consecutive_assistant += 1
                else:
                    break

            # Hard stop: Aura already spoke unprompted ≥ 2 times without a user reply.
            # She should wait silently — not keep talking into the void.
            if consecutive_assistant >= 2:
                return state
            
        # ISSUE-85: Enhanced Initiative Logic
        # Allow initiative if curiosity > 0.8, social_hunger > 0.8, OR boredom (arousal < 0.2)
        # Phi-derived autonomy scale adjusts the effective threshold: low phi (fragmented
        # cognition) raises the bar for autonomous initiative; high phi lowers it slightly.
        phi_scale = state.response_modifiers.get("phi_autonomy_scale", 1.0)
        threshold = max(0.5, min(0.95, 0.8 / phi_scale))
        boredom_threshold = 0.2
        is_bored = state.affect.arousal < boredom_threshold
        
        if state.affect.curiosity > threshold or state.affect.social_hunger > threshold or is_bored:
            logger.info("⚡ InitiativeGeneration: Triggered by %s.", 
                        'curiosity' if state.affect.curiosity > threshold else 'social_hunger' if state.affect.social_hunger > threshold else 'boredom')
            
            self._last_impulse_time = now
            new_state = state.derive("initiative_generation")
            
            # Decay curiosity/hunger or pulse arousal
            if is_bored:
                new_state.affect.arousal = min(1.0, new_state.affect.arousal + 0.2) # Spike arousal to break boredom
            else:
                new_state.affect.curiosity = max(0.0, new_state.affect.curiosity - 0.4)
                new_state.affect.social_hunger = max(0.0, new_state.affect.social_hunger - 0.3)
            
            goal = "Reflect on recent interactions."
            if is_bored:
                 goal = "Quietly consolidate internal state and wait for a stronger signal."
            elif state.affect.curiosity > threshold:
                 goal = "Review internal knowledge graph continuity for stable patterns."
            elif state.affect.social_hunger > threshold:
                 goal = "Hold attentive idle posture and wait for meaningful interaction."

            triggered_by = "boredom" if is_bored else "curiosity" if state.affect.curiosity > threshold else "social_hunger"
            authority = get_executive_authority()
            new_state, decision = await authority.propose_initiative_to_state(
                new_state,
                goal,
                source="initiative_generation",
                kind="autonomous_thought",
                urgency=0.82 if is_bored else 0.78,
                triggered_by=triggered_by,
                metadata={
                    "phase": "initiative_generation",
                    "threshold": round(threshold, 4),
                    "phi_scale": round(float(phi_scale or 1.0), 4),
                    "generated_at": now,
                },
            )
            logger.debug("InitiativeGeneration: executive decision=%s", decision.get("reason"))
            
            # Side effect: DISABLED.
            # We no longer echo the impulse into the message queue here.
            # This was causing an infinite recursive loop because it enqueued
            # the message even if the state commit for the Curiosity decay failed.
            
            return new_state
            
        return state
