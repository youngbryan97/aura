import logging
from typing import Any

from core.consciousness.integration import get_consciousness_integration
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger(__name__)

_CONSCIOUSNESS_PHASE_ERRORS = (
    AttributeError, ImportError, LookupError, RuntimeError, TypeError, ValueError,
)

class ConsciousnessPhase(BasePhase):
    """
    Phase 8: Phenomenological Awareness.
    Constructs the first-person experiential claim for this cognitive cycle.
    """
    
    def __init__(self, container: Any = None):
        self.container = container

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Pull the latest phenomenal context from the integration layer.
        """
        new_state = state.derive("consciousness_phase")
        
        # Pull from singleton integration
        integration = get_consciousness_integration()
        
        # Layer 8 Injection
        if integration:
            phenomenal_claim = integration.get_phenomenal_context()
            new_state.cognition.phenomenal_state = phenomenal_claim
            logger.debug("🌅 ConsciousnessPhase: Layer 8 phenomenal context injected.")
        
        # Causal Simulation
        causal_model = get_runtime_service("causal_world_model", default=None)
        if causal_model:
            causal_context = causal_model.get_prompt_context()
            if causal_context:
                if not hasattr(new_state.cognition, 'causal_reasoning'):
                    # Just in case the state model is lagging
                    new_state.cognition.causal_reasoning = causal_context
                else:
                    new_state.cognition.causal_reasoning = causal_context
                logger.debug("🧶 ConsciousnessPhase: Causal world cascades injected.")

        # Advance the selfhood layers.
        #
        # MinimalSelfhood.update() had no caller anywhere in the tree, so its
        # state was None for the life of the process while get_priority_bias()
        # returned a zero vector that read like a measurement. The second-order
        # kernels had no writer either. This is the caller, and it refuses to
        # tick on inputs it could not read.
        try:
            from core.consciousness.selfhood_tick import drive_selfhood

            reading = drive_selfhood(new_state)
            new_state.cognition.selfhood_reading = reading.as_dict()
            if reading.skipped:
                logger.debug("🪞 ConsciousnessPhase: selfhood not advanced — %s", reading.skipped)
            else:
                logger.debug(
                    "🪞 ConsciousnessPhase: selfhood advanced on %d readings (%s).",
                    len(reading.readings),
                    reading.selfhood.get("dominant_deficit", "?"),
                )
        except _CONSCIOUSNESS_PHASE_ERRORS as exc:
            record_degradation("consciousness_phase", exc, action="selfhood layers not advanced")

        return new_state


# ─────────────────────────────────────────────────────────────────────────────
# Declared semantics. See core/runtime/cognitive_contract.py.
#
# `writes` is MEASURED — tools/observe_phase_writes.py ran this phase against a
# real AuraState and recorded which fields moved. It is not a reading of the
# code, which is how a declaration ends up describing what the author believed.
from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)

register_contract(
    CognitiveTransformContract(
        name="ConsciousnessPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Assemble the phenomenal state for this tick — what it is like to "
            "be in it — from affect, soma and the current objective."
        ),
        # Nothing. This phase reads no AuraState field: it pulls the
        # phenomenal claim from the consciousness integration singleton and
        # the causal context from the causal_world_model service. The first
        # version of this contract declared affect and soma reads because
        # that is what a phase called "consciousness" ought to consult —
        # which is precisely the failure the contract layer exists to stop.
        reads=(),
        writes=("cognition.phenomenal_state", "transition_cause"),
        preconditions=("state carries a cognition block",),
        branches=(
            BranchSpec(
                "assembled",
                "affect and soma are both readable",
                "write a phenomenal state derived from them",
            ),
            BranchSpec(
                "degraded",
                "an input subsystem is unavailable",
                "write a reduced phenomenal state naming what was missing",
            ),
        ),
        side_effects=(
            "reads the consciousness integration singleton",
            "reads the causal_world_model runtime service",
        ),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py; reads are empty "
            "because this phase consults services, not state"
        ),
    )
)
