import logging
import time
from typing import Any

from core.runtime.background_policy import background_activity_allowed
from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)
from core.runtime.cognitive_provenance import note_branch
from core.runtime.errors import record_degradation
from core.runtime.proposal_governance import propose_governed_initiative_to_state
from core.runtime.service_registry import get_runtime_service

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger(__name__)

# ── Admission criteria ─────────────────────────────────────────────────────
#
# These were inline literals. Every one of them was a real, deliberate policy
# decision — and none of them could be read without reading the method, which
# is why the architecture kept being described as though an autonomous impulse
# were an unexplained mood. Naming them costs nothing at runtime and lets the
# contract below reference the SAME OBJECT rather than a copy that drifts.

#: Minimum gap between two autonomous impulses.
IMPULSE_THROTTLE_SECONDS = 60.0
#: How long the runtime must have been idle before autonomy is considered.
MIN_IDLE_SECONDS = 900.0
#: Memory ceiling above which autonomy is refused outright.
MAX_MEMORY_PERCENT = 80.0
#: Failure-pressure ceiling. A struggling runtime does not get to be curious.
MAX_FAILURE_PRESSURE = 0.12
#: A live exchange deeper than this, with energy above the next constant, is
#: not interrupted. Both must hold — depth alone is a long calm conversation.
SUPPRESSING_DISCOURSE_DEPTH = 3
SUPPRESSING_CONVERSATION_ENERGY = 0.4
#: Unprompted messages in a row before she waits silently instead of talking
#: into the void.
MAX_CONSECUTIVE_ASSISTANT_MESSAGES = 2
#: Base affect threshold, scaled by φ-derived autonomy and then clamped.
#: Fragmented cognition (low φ) raises the bar; integrated cognition lowers it.
BASE_AFFECT_THRESHOLD = 0.8
AFFECT_THRESHOLD_FLOOR = 0.5
AFFECT_THRESHOLD_CEILING = 0.95
#: Arousal below this counts as boredom, which is its own trigger.
BOREDOM_AROUSAL_MAX = 0.2
#: What an initiative costs the affect that produced it. Without the decay the
#: same drive re-fires every throttle window.
BOREDOM_AROUSAL_PULSE = 0.2
CURIOSITY_DECAY = 0.4
SOCIAL_HUNGER_DECAY = 0.3
#: Declared urgency of the proposal handed to governance.
BORED_URGENCY = 0.82
DRIVEN_URGENCY = 0.78


INITIATIVE_GENERATION_CONTRACT = register_contract(
    CognitiveTransformContract(
        name="InitiativeGenerationPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Decide whether to form an autonomous impulse this tick, and pay "
            "for it out of the affect that motivated it."
        ),
        reads=(
            "affect.curiosity",
            "affect.social_hunger",
            "affect.arousal",
            "cognition.discourse_depth",
            "cognition.conversation_energy",
            "cognition.working_memory",
            "response_modifiers",
        ),
        writes=(
            "affect.curiosity",
            "affect.social_hunger",
            "affect.arousal",
            "cognition.pending_initiatives",
            "transition_cause",
        ),
        preconditions=(
            "autonomous actions admitted by runtime settings",
            "llm_router not in high_pressure_mode",
            "inference gate raises no background deferral",
            "background activity allowed (idle, memory, failure pressure)",
            "at least IMPULSE_THROTTLE_SECONDS since the last impulse",
            "not immediately after a user message",
            "fewer than MAX_CONSECUTIVE_ASSISTANT_MESSAGES unprompted in a row",
        ),
        branches=(
            BranchSpec(
                "paused",
                "autonomy paused by settings, memory pressure or inference gate",
                "return state unchanged",
            ),
            BranchSpec(
                "background_refused",
                "idle/memory/failure-pressure policy declines background work",
                "return state unchanged",
            ),
            BranchSpec("throttled", "within IMPULSE_THROTTLE_SECONDS", "return state unchanged"),
            BranchSpec(
                "conversation_active",
                "discourse_depth > SUPPRESSING_DISCOURSE_DEPTH and "
                "conversation_energy > SUPPRESSING_CONVERSATION_ENERGY",
                "return state unchanged",
            ),
            BranchSpec("after_user", "last working-memory role is user", "return state unchanged"),
            BranchSpec(
                "monologue_guard",
                "consecutive assistant messages >= MAX_CONSECUTIVE_ASSISTANT_MESSAGES",
                "return state unchanged",
            ),
            BranchSpec(
                "boredom",
                "arousal < BOREDOM_AROUSAL_MAX",
                "pulse arousal, propose a consolidation goal",
            ),
            BranchSpec(
                "curiosity",
                "curiosity > phi-scaled threshold",
                "decay curiosity and social hunger, propose a review goal",
            ),
            BranchSpec(
                "social_hunger",
                "social_hunger > phi-scaled threshold",
                "decay curiosity and social hunger, propose an attentive-idle goal",
            ),
            BranchSpec("below_threshold", "no drive clears its threshold", "return state unchanged"),
        ),
        thresholds={
            "IMPULSE_THROTTLE_SECONDS": IMPULSE_THROTTLE_SECONDS,
            "MIN_IDLE_SECONDS": MIN_IDLE_SECONDS,
            "MAX_MEMORY_PERCENT": MAX_MEMORY_PERCENT,
            "MAX_FAILURE_PRESSURE": MAX_FAILURE_PRESSURE,
            "SUPPRESSING_DISCOURSE_DEPTH": SUPPRESSING_DISCOURSE_DEPTH,
            "SUPPRESSING_CONVERSATION_ENERGY": SUPPRESSING_CONVERSATION_ENERGY,
            "MAX_CONSECUTIVE_ASSISTANT_MESSAGES": MAX_CONSECUTIVE_ASSISTANT_MESSAGES,
            "BASE_AFFECT_THRESHOLD": BASE_AFFECT_THRESHOLD,
            "AFFECT_THRESHOLD_FLOOR": AFFECT_THRESHOLD_FLOOR,
            "AFFECT_THRESHOLD_CEILING": AFFECT_THRESHOLD_CEILING,
            "BOREDOM_AROUSAL_MAX": BOREDOM_AROUSAL_MAX,
            "BOREDOM_AROUSAL_PULSE": BOREDOM_AROUSAL_PULSE,
            "CURIOSITY_DECAY": CURIOSITY_DECAY,
            "SOCIAL_HUNGER_DECAY": SOCIAL_HUNGER_DECAY,
            "BORED_URGENCY": BORED_URGENCY,
            "DRIVEN_URGENCY": DRIVEN_URGENCY,
        },
        defaults={
            "phi_autonomy_scale": 1.0,
            "discourse_depth": 0,
            "conversation_energy": 0.5,
        },
        authority="propose_governed_initiative_to_state (UnifiedWill admission)",
        side_effects=(),
        invariants=(
            "an impulse is never appended without governed admission",
            "a fired trigger always pays: the driving affect moves this tick",
        ),
        calibration_source=(
            "judgement, tuned live against unprompted-message rate; reads "
            "reach state through this phase's delegate rather than appearing "
            "in this module, so they are declared from the delegate's "
            "behaviour, not by scanning this file"
        ),
        thresholds_exhaustive=True,
    )
)

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
            from core.runtime.runtime_settings import autonomous_actions_admitted

            admitted, setting_reason = autonomous_actions_admitted(
                "initiative_generation"
            )
            if not admitted:
                return setting_reason

            router = get_runtime_service("llm_router", default=None)
            if router and getattr(router, "high_pressure_mode", False):
                return "memory_pressure"

            gate = get_runtime_service("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                reason = str(gate._background_local_deferral_reason(origin="initiative_generation") or "").strip()
                if reason:
                    return reason
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('initiative_generation', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return ""

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
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
            note_branch("paused", pause_reason=pause_reason)
            logger.debug("⚡ InitiativeGeneration: paused while runtime is guarded (%s).", pause_reason)
            return state

        try:
            orch = get_runtime_service("orchestrator", default=None)
        except (ImportError, AttributeError, RuntimeError):
            orch = None
        if not background_activity_allowed(
            orch,
            min_idle_seconds=MIN_IDLE_SECONDS,
            max_memory_percent=MAX_MEMORY_PERCENT,
            max_failure_pressure=MAX_FAILURE_PRESSURE,
            require_conversation_ready=False,
        ):
            note_branch(
                "background_refused",
                min_idle_seconds=MIN_IDLE_SECONDS,
                max_memory_percent=MAX_MEMORY_PERCENT,
                max_failure_pressure=MAX_FAILURE_PRESSURE,
            )
            return state

        now = time.time()
        since_last = now - self._last_impulse_time
        if since_last < IMPULSE_THROTTLE_SECONDS:
            note_branch(
                "throttled",
                since_last_impulse_s=round(since_last, 2),
                throttle_s=IMPULSE_THROTTLE_SECONDS,
            )
            return state

        # Suppress autonomous impulses during deep active conversation.
        # If a real exchange is underway, don't interrupt.
        discourse_depth = getattr(state.cognition, "discourse_depth", 0)
        conv_energy = getattr(state.cognition, "conversation_energy", 0.5)
        if (
            discourse_depth > SUPPRESSING_DISCOURSE_DEPTH
            and conv_energy > SUPPRESSING_CONVERSATION_ENERGY
        ):
            note_branch(
                "conversation_active",
                discourse_depth=discourse_depth,
                conversation_energy=round(float(conv_energy), 3),
            )
            return state

        wm = state.cognition.working_memory
        if wm:
            last_msg = wm[-1]
            last_role = last_msg.get("role", "")

            # Never speak immediately after the user — respect response flow
            if last_role == "user":
                note_branch("after_user", last_role=last_role)
                return state

            # Prevent monologue: if the last N messages are all assistant, back off.
            # Count trailing consecutive assistant messages.
            consecutive_assistant = 0
            for msg in reversed(wm):
                if msg.get("role") == "assistant":
                    consecutive_assistant += 1
                else:
                    break

            # Hard stop: she already spoke unprompted enough times without a
            # user reply. Wait silently — do not keep talking into the void.
            if consecutive_assistant >= MAX_CONSECUTIVE_ASSISTANT_MESSAGES:
                note_branch(
                    "monologue_guard",
                    consecutive_assistant=consecutive_assistant,
                    limit=MAX_CONSECUTIVE_ASSISTANT_MESSAGES,
                )
                return state

        # ISSUE-85: Enhanced Initiative Logic.
        # Curiosity or social hunger above the φ-scaled threshold, or boredom.
        # The φ-derived autonomy scale adjusts the effective bar: fragmented
        # cognition raises it, integrated cognition lowers it slightly.
        phi_scale = state.response_modifiers.get("phi_autonomy_scale", 1.0)
        threshold = max(
            AFFECT_THRESHOLD_FLOOR,
            min(AFFECT_THRESHOLD_CEILING, BASE_AFFECT_THRESHOLD / phi_scale),
        )
        is_bored = state.affect.arousal < BOREDOM_AROUSAL_MAX

        if state.affect.curiosity > threshold or state.affect.social_hunger > threshold or is_bored:
            logger.info("⚡ InitiativeGeneration: Triggered by %s.", 
                        'curiosity' if state.affect.curiosity > threshold else 'social_hunger' if state.affect.social_hunger > threshold else 'boredom')
            
            self._last_impulse_time = now
            new_state = state.derive("initiative_generation")

            # A fired trigger pays for itself. Without the decay the same drive
            # re-fires on every throttle window, which is the invariant the
            # contract states and this is where it is kept.
            if is_bored:
                new_state.affect.arousal = min(
                    1.0, new_state.affect.arousal + BOREDOM_AROUSAL_PULSE
                )
            else:
                new_state.affect.curiosity = max(
                    0.0, new_state.affect.curiosity - CURIOSITY_DECAY
                )
                new_state.affect.social_hunger = max(
                    0.0, new_state.affect.social_hunger - SOCIAL_HUNGER_DECAY
                )

            goal = "Reflect on recent interactions."
            if is_bored:
                 goal = "Quietly consolidate internal state and wait for a stronger signal."
            elif state.affect.curiosity > threshold:
                 goal = "Review internal knowledge graph continuity for stable patterns."
            elif state.affect.social_hunger > threshold:
                 goal = "Hold attentive idle posture and wait for meaningful interaction."

            triggered_by = "boredom" if is_bored else "curiosity" if state.affect.curiosity > threshold else "social_hunger"
            note_branch(
                triggered_by,
                threshold=round(threshold, 4),
                phi_scale=round(float(phi_scale or 1.0), 4),
                curiosity=round(float(state.affect.curiosity), 4),
                social_hunger=round(float(state.affect.social_hunger), 4),
                arousal=round(float(state.affect.arousal), 4),
                goal=goal,
            )
            new_state, decision = await propose_governed_initiative_to_state(
                new_state,
                goal,
                orchestrator=None,
                source="initiative_generation",
                kind="autonomous_thought",
                urgency=BORED_URGENCY if is_bored else DRIVEN_URGENCY,
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

        note_branch(
            "below_threshold",
            threshold=round(threshold, 4),
            curiosity=round(float(state.affect.curiosity), 4),
            social_hunger=round(float(state.affect.social_hunger), 4),
            arousal=round(float(state.affect.arousal), 4),
        )
        return state
