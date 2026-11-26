import logging
import time
from typing import Any

from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)
from core.runtime.errors import record_degradation

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger(__name__)

class IdentityReflectionPhase(BasePhase):
    """
    Phase 7: Identity Reflection.
    Updates Aura's self-narrative based on recent experiences and themes.
    Moves Aura from a static agent to an evolving identity.
    """
    
    def __init__(self, container: Any):
        self.container = container

    @staticmethod
    def _authorize_identity_mutation(reason: str) -> Any:
        """Route identity-layer mutation through UnifiedWill, failing closed."""
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"identity_reflection:{reason}",
                source="identity_reflection",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.7,
            )
            if not decision.is_approved():
                logger.warning(
                    "IdentityReflection: Will blocked identity mutation (%s): %s",
                    decision.outcome.value,
                    decision.reason,
                )
            return decision
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('identity_reflection', exc)
            logger.warning("IdentityReflection: UnifiedWill unavailable; identity mutation blocked: %s", exc)
            return None

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        [CLAUDE AUDIT] Identity Guard / Hard Stop.
        Ensures Aura's output hasn't deviated into hallucination or dangerous territory.
        If validation fails, returns the PARENT state (Hard Stop) instead of the new one.
        """
        logger.debug("CognitiveGuard: Validating state transition integrity.")

        # Her sense of holding together, into the neuron that means it. Before
        # the guard and before every early return below, because what she is
        # does not depend on whether her narrative was revised this turn.
        self._hold_together_reaches_the_substrate(state)
        
        # 1. Identity Consistency Check
        identity_name = str(getattr(state.identity, "name", "Aura") or "Aura").strip()
        if identity_name and identity_name.lower() not in ("aura", "aura luna"):
            logger.critical("IDENTITY BREACH: Identity name altered to '%s'. COGNITIVE HARD STOP.", identity_name)
            # Repair: force identity name back to Aura
            try:
                state.identity.name = "Aura"
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "%s unavailable (%s: %s); the identity name could not be forced back to Aura after a breach",
                    "it",
                    type(exc).__name__,
                    exc,
                )
            return state
             
        # 2. Output Characterization (Anti-Hallucination)
        if state.cognition.working_memory:
            last_msg = state.cognition.working_memory[-1]
            if last_msg.get("role") == "assistant":
                content = last_msg.get("content", "")
                
                # Length is not an identity violation. Loop detection belongs
                # to the token sentinel and response-quality telemetry; identity
                # reflection observes authored speech and never edits it.
                if len(content) > 5000:
                    logger.info(
                        "CognitiveGuard observed long-form assistant output (%d chars); "
                        "content remains authoritative.",
                        len(content),
                    )
                
                # Claude's Identity Stability Check:
                # Ensure the message doesn't claim things that violate the core identity
                if "i am a human" in content.lower() or "i am chatgpt" in content.lower():
                    logger.critical("COGNITIVE ROLLBACK: Identity Hallucination detected. REJECTING TRANSITION.")
                    # [CLAUDE AUDIT] Hard Stop: Return original state to rollback the turn.
                    return state 

                # Check Identity Guard service directly if available
                from core.identity.identity_guard import PersonaEnforcementGate
                ok, reason, _ = PersonaEnforcementGate().validate_output(
                    content,
                    enforce_supervision=False,
                )
                if not ok:
                    logger.critical("COGNITIVE ROLLBACK: Identity Guard rejected output (%s).", reason)
                    return state

        # 3. Success: Narrative Drift Update
        # NOTE: The previous "I am stable and evolve safely" append has been
        # removed. It served no functional purpose and actively contaminated
        # brainstem fallback responses, causing repetitive mantra loops when
        # the primary cortex died. Identity stability is maintained through
        # the identity guard checks above, not through string injection.
        # When enough has happened to justify it, which is what this phase has
        # said it does since it was written. What it did was increment on
        # `state.version % 20`: twenty turns of silence and twenty turns that
        # changed everything got the same answer, and the declaration and the
        # implementation had never agreed.
        #
        # What has happened is readable. Recall bringing back something the
        # story has no room for is the condition, measured against how much it
        # usually brings back. See core/self/revision.py.
        revision = self._worth_revisiting(state)
        state.response_modifiers["narrative_revision"] = revision.as_dict()
        if revision.worth_revisiting:
            decision = self._authorize_identity_mutation("recall_unaccounted_for_by_the_narrative")
            if not decision or not decision.is_approved():
                return state
            try:
                state.response_modifiers["identity_reflection_will_receipt"] = decision.receipt_id
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug(
                    "%s unavailable (%s: %s); the will receipt is missing from the response modifiers",
                    "it",
                    type(exc).__name__,
                    exc,
                )
            state.identity.narrative_version += 1
            state.identity.last_evolution_timestamp = time.time()

        return state

    #: How much of the neuron a push moves, borrowed rather than chosen: it is
    #: the blend `core/self/will_engine.py` already uses to drive motivation's
    #: budgets into the same substrate.
    _DOMINANCE_BLEND: float = 0.2

    @staticmethod
    def _hold_together_reaches_the_substrate(state: AuraState) -> None:
        """Her self-state writes the dominance neuron, which nothing wrote.

        The substrate declares three VAD neurons and `idx_dominance` is the
        third: in the circumplex it is the sense of being in control of one's
        situation rather than carried by it. Everything reads it — the aesthetic
        engine, the substrate gates, two of the substrate's own summaries — and
        on 28 September nothing in the runtime wrote it. A reader with no writer.

        It matters beyond the tidiness. The substrate is twenty-five of the
        eighty-four columns of recurrent cognition, and `core/self/will_engine.py`
        drives motivation's budgets into two of its neurons, so deliberation has
        a channel into recurrent cognition and self-state has none. Measured on
        whole-s7-27dc1dda9: a displacement of S moves C by 4.23, the second
        largest of any domain, while S's unique information about C's next change
        is exactly 0.0 and D's is 0.154. An influence that exists under
        intervention because it travels through another domain, and does not
        exist in her ordinary variation at all.

        `identity.stability` is her self-model's own reading of whether she is
        holding together, already on 0 to 1, and it is the whole of what is
        written here; the neuron is signed and rests at zero, so it is mapped the
        way the steering channel maps the same block back.
        """
        from core.runtime.flags import env_str

        switch = env_str(
            "AURA_SELF_DOMINANCE",
            description="On writes identity stability into the substrate's dominance neuron.",
            owner="core.phases.identity_reflection",
        )
        if switch.strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        try:
            from core.container import ServiceContainer

            substrate = ServiceContainer.get("liquid_substrate", default=None) or (
                ServiceContainer.get("conscious_substrate", default=None)
            )
            index = getattr(substrate, "idx_dominance", None)
            if substrate is None or not isinstance(index, int):
                return
            stability = float(getattr(state.identity, "stability", 0.5) or 0.0)
            held = max(-1.0, min(1.0, 2.0 * max(0.0, min(1.0, stability)) - 1.0))
            blend = IdentityReflectionPhase._DOMINANCE_BLEND
            with substrate.sync_lock:
                substrate.x[index] = (1.0 - blend) * float(substrate.x[index]) + blend * held
                marker = getattr(substrate, "mark_state_mutated_locked", None)
                if callable(marker):
                    marker("identity_reflection.dominance")
            state.response_modifiers["self_dominance"] = round(held, 6)
        except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError) as exc:
            record_degradation(
                "identity_reflection",
                exc,
                severity="warning",
                action="her self-state did not reach the dominance neuron this turn",
            )

    @staticmethod
    def _worth_revisiting(state):
        """Whether recall brought back something the story does not hold.

        Returns an unmeasured reading rather than raising when the organ is not
        available, so a missing reader leaves the narrative alone instead of
        revising it on nothing.
        """
        from core.self.revision import Revision, get_revision_ledger

        try:
            recalled = list(getattr(state.cognition, "long_term_memory", []) or [])
            narrative = str(getattr(state.identity, "current_narrative", "") or "")
            return get_revision_ledger().read(recalled, narrative)
        except (AttributeError, TypeError, ValueError):
            return Revision()



# ─────────────────────────────────────────────────────────────────────────────
# Declared semantics. See core/runtime/cognitive_contract.py.
#
# `writes` is MEASURED — tools/observe_phase_writes.py ran this phase against a
# real AuraState and recorded which fields moved. It is not a reading of the
# code, which is how a declaration ends up describing what the author believed.

register_contract(
    CognitiveTransformContract(
        name="IdentityReflectionPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Revisit the self-narrative when enough has happened to justify it, "
            "and stamp when that last occurred."
        ),
        reads=(
            "identity.narrative_version",
            "identity.last_evolution_timestamp",
            "cognition.working_memory",
            "cognition.long_term_memory",
            "identity.current_narrative",
        ),
        writes=(
            "identity.last_evolution_timestamp",
            "identity.narrative_version",
            "response_modifiers",
        ),
        preconditions=("state carries an identity block",),
        branches=(
            BranchSpec(
                "revised",
                "enough new evidence since the last reflection",
                "advance the narrative version and stamp the time",
            ),
            BranchSpec(
                "unchanged",
                "insufficient new evidence",
                "leave the narrative and its version alone",
            ),
        ),
        invariants=(
            "narrative_version never decreases",
        ),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py; the sufficiency "
            "test is a judgement call not yet a named constant"
        ),
    )
)
