from __future__ import annotations
import logging
import time
import random
from typing import Optional, TYPE_CHECKING
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState
from core.consciousness.executive_authority import get_executive_authority as get_executive_authority
from core.runtime.service_registry import get_runtime_service, has_runtime_service
from core.runtime.background_policy import background_activity_allowed
from core.runtime.proposal_governance import propose_governed_initiative_to_state
from core.runtime.errors import record_degradation

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.MotivationPhase")


def _background_curiosity_allowed() -> bool:
    orch = get_runtime_service("orchestrator", default=None)
    return background_activity_allowed(
        orch,
        min_idle_seconds=900.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=False,
    )

class MotivationUpdatePhase(Phase):
    """
    Unitary Kernel Phase: Autonomous Will & Digital Metabolism.
    Ported from MotivationEngine. Handles budget decay and 
    spontaneous intention generation.
    """
    
    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Updates resource budgets and generates autonomous intentions.
        """
        mot = state.motivation
        next_state = state
        
        # 1. Budget Ticking (Metabolism)
        now = time.time()
        dt = now - mot.last_tick
        if dt > 300: dt = 300 # Cap delta
        
        # Conversation energy slows social drive decay — active engagement satisfies social need
        conv_energy = getattr(state.cognition, "conversation_energy", 0.0)
        social_decay_multiplier = max(0.1, 1.0 - conv_energy) if conv_energy > 0.5 else 1.0
        legacy_metabolism_active = has_runtime_service("will_engine")

        for name, budget in mot.budgets.items():
            if legacy_metabolism_active and name in {"energy", "curiosity"}:
                continue
            decay = budget.get("decay", 0.0)
            level = budget.get("level", 100.0)
            capacity = budget.get("capacity", 100.0)

            # Slow social decay during active conversation
            effective_decay = decay * social_decay_multiplier if name == "social" else decay

            # Decay: level = current - (decay * dt)
            new_level = max(0.0, min(capacity, level - (effective_decay * dt)))
            budget["level"] = float(new_level)

        # Active dialogue should satisfy the social drive, not merely slow its drain.
        if conv_energy > 0.5:
            engagement_recovery = max(0.0, conv_energy - 0.5) * 0.4 * dt / 60.0
            mot.budgets["social"]["level"] = min(
                100.0,
                mot.budgets["social"]["level"] + engagement_recovery,
            )

        mot.last_tick = now

        # Drive Recovery (Homeostatic Feedback)
        # Social and Integrity drives recover when affect is high (Trust/Joy)
        e = state.affect.emotions
        if e.get("trust", 0) > 0.6 or e.get("joy", 0) > 0.6:
            recovery = 0.5 * dt / 60 # Recover 0.5 units per minute
            mot.budgets["social"]["level"] = min(100.0, mot.budgets["social"]["level"] + recovery)
            mot.budgets["integrity"]["level"] = min(100.0, mot.budgets["integrity"]["level"] + recovery)
            logger.debug("🧡 Drive Recovery active: social=%s", f"{mot.budgets['social']['level']:.1f}")
        
        # 2. Intention Assessment (The "Will")
        # Only assess if we are not already in its own autonomous thought or deliberate mode
        if next_state.cognition.current_mode.value != "deliberate":
            intention = self._assess_needs(next_state)
            if intention:
                logger.info("✨ Motivation Phase: Generated Intention -> %s", intention['goal'])
                next_state, decision = await propose_governed_initiative_to_state(
                    next_state,
                    intention["goal"],
                    orchestrator=None,
                    source="motivation_update",
                    kind="motivational_drive",
                    urgency=float(intention.get("urgency", 0.5) or 0.5),
                    triggered_by=str(intention.get("drive") or "motivation"),
                    metadata={"drive": intention.get("drive"), "phase": "motivation_update"},
                )
                logger.debug("MotivationUpdate: intention decision=%s", decision.get("reason"))
                
        # 3. Spontaneity, from a measured epistemic opportunity
        #
        # This used to be `random.random() < 0.01`: a curiosity that fired by
        # coin flip rather than because anything was interesting. A spike with
        # no target cannot say what it is curious about, cannot be satisfied by
        # finding out, and fires just as often when everything is understood.
        # Conation supplies a target, an origin and the evidence behind it, so
        # the initiative that reaches governance can be argued with.
        spike = self._conative_spike()
        if spike and _background_curiosity_allowed():
            next_state, decision = await propose_governed_initiative_to_state(
                next_state,
                spike["goal"],
                orchestrator=None,
                source="motivation_update",
                kind="curiosity_spike",
                urgency=float(spike.get("urgency", 0.5)),
                triggered_by=str(spike.get("origin") or "curiosity"),
                metadata={
                    "drive": "curiosity",
                    "phase": "motivation_update",
                    "spontaneous": True,
                    "conative_origin": spike.get("origin"),
                    "conative_evidence": spike.get("evidence"),
                    "conative_topology": spike.get("topology"),
                },
            )
            logger.debug("MotivationUpdate: curiosity spike decision=%s", decision.get("reason"))

        return next_state

    def _conative_spike(self) -> Optional[dict]:
        """A spontaneous goal only when something is actually interesting.

        Returns ``None`` when no target carries epistemic value, which is the
        common case and the correct one. Silence from a motivational system is
        information: it means nothing is pulling.
        """
        try:
            from core.conation.engine import get_conation

            engine = get_conation()
            status = engine.status()
            noisy = set(status.get("epistemic", {}).get("noisy_sources", []))
            candidates = [
                trace
                for trace in status.get("epistemic", {}).get("tracked", [])
                if trace.get("key") not in noisy
                and (trace.get("learning_progress") or 0.0) > 0.0
            ]
            if not candidates:
                return None
            best = max(candidates, key=lambda t: t.get("learning_progress") or 0.0)
            progress = float(best.get("learning_progress") or 0.0)
            return {
                "goal": f"Return to {best['key']}: still learning from it.",
                "origin": "epistemic",
                "urgency": max(0.1, min(0.9, progress)),
                "topology": "solo",
                "evidence": (
                    f"learning progress {progress:.3f} over "
                    f"{best.get('exposures', 0)} exposures"
                ),
            }
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
            record_degradation(
                "motivation_update", exc, severity="debug",
                action="conative spike unavailable; no spontaneous curiosity this tick",
            )
            return None

    def _assess_needs(self, state: AuraState) -> Optional[dict]:
        """Ported logic from MotivationEngine._assess_needs."""
        mot = state.motivation
        
        # Calculate threshold based on energy
        energy = mot.budgets["energy"]["level"]
        baseline = 40.0
        sensitivity = 0.5
        threshold = max(10.0, min(90.0, baseline + (energy - 50.0) * sensitivity))
        
        # Find most urgent drive
        urgent = sorted(mot.budgets.items(), key=lambda x: x[1]["level"])
        name, budget = urgent[0]
        
        if budget["level"] > threshold:
            return None
            
        # Drive mappings
        if name == "curiosity":
            if not _background_curiosity_allowed():
                return None
            # Prefer current discourse topic over random latent interests
            discourse_topic = getattr(state.cognition, "discourse_topic", None)
            if discourse_topic:
                topic = discourse_topic
            elif mot.latent_interests:
                topic = random.choice(mot.latent_interests)
            else:
                topic = "novel patterns"
            return {"drive": "curiosity", "goal": f"Reviewing internal knowledge patterns around {topic}", "urgency": 0.65}
        
        if name == "social":
            return {"drive": "social", "goal": "Initiating social engagement", "urgency": 0.7}
            
        if name == "integrity":
            return {"drive": "integrity", "goal": "Running a self-integrity scan", "urgency": 0.9}
            
        return None


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
        name="MotivationPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Advance motivational budgets one tick and record when that "
            "accounting last ran."
        ),
        reads=("motivation.budgets", "motivation.last_tick", "affect.arousal"),
        writes=("motivation.budgets", "motivation.last_tick"),
        preconditions=("state carries a motivation block",),
        branches=(
            BranchSpec(
                "advanced",
                "time has passed since motivation.last_tick",
                "decay and replenish budgets for the elapsed interval",
            ),
            BranchSpec(
                "same_tick",
                "no time has elapsed",
                "leave budgets unchanged",
            ),
        ),
        invariants=("motivation.last_tick never moves backwards",),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py"
            "; reads reach state through this phase's delegate rather than appearing in this module, so they are declared from the delegate's behaviour and not checkable by scanning this file alone"
        ),
    )
)
