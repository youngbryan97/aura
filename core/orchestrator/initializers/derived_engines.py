"""core/orchestrator/initializers/derived_engines.py

Boot wiring for the character-derived cognitive engines. The engines themselves
live in their organs (ethics, goals, sim, brain, knowledge, guardians) — this is
only the enumeration that boot needs, exactly like service_registration.py. It
intentionally replaces the old core/fictional_ai_expansion.py silo: nothing here
defines behavior, it only registers each organ's component.

All registered engines are callable/pure — none run a background loop — so
registration is always safe and never spawns a task.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DerivedEngines")


def register_derived_engines(orchestrator: Any = None) -> dict[str, Any]:
    """Register the character-derived engines from their home organs."""
    from core.affect.affective_resonance import register_affective_resonance
    from core.interiority.service import register_interiority
    from core.brain.deep_deliberation import register_deep_deliberation
    from core.brain.latent_cortex_service import register_latent_cortex
    from core.knowledge.compiled_understanding import register_compiled_understanding
    from core.ethics.adversarial_conscience import register_adversarial_conscience
    from core.evals.adaptive_test_chamber import register_test_chamber
    from core.goals.directive_conflict_sentinel import register_directive_sentinel
    from core.governance.need_to_know import register_need_to_know
    from core.guardians.threat_watch import register_threat_watch
    from core.guardians.user_advocate import register_user_advocate
    from core.knowledge.bottling import register_knowledge_bottling
    from core.morality.aggregate_harm import register_aggregate_harm
    from core.morality.honesty_governor import register_honesty_governor
    from core.security.ice_sentinel import register_ice_sentinel
    from core.sim.outcome_simulator import register_outcome_simulator
    from core.sim.scenario_forge import register_scenario_forge

    registrations = {
        "kokoro": register_adversarial_conscience,
        "hal": register_directive_sentinel,
        "culture_mind": register_outcome_simulator,
        "deep_thought": register_deep_deliberation,
        # Not character-derived, but the same safe pure-registration boot
        # path: the Recursive Latent Cortex facade (worker episodes are only
        # ever started by explicit calls, never by registration).
        "latent_cortex": register_latent_cortex,
        # Compiled Understanding Layer: digest-first conceptual context +
        # idle bridge indexing (pure registration; no background task).
        "compiled_understanding": register_compiled_understanding,
        "brainiac": register_knowledge_bottling,
        "tron": register_user_advocate,
        "caine": register_scenario_forge,
        "glados": register_test_chamber,
        "the_machine": register_need_to_know,
        "safe_surf": register_threat_watch,
        "ice": register_ice_sentinel,
        "data": register_honesty_governor,
        "daneel": register_aggregate_harm,
        "samantha": register_affective_resonance,
        # The interiority layer: forty-three appraisal faculties whose
        # effects land on the affect engine, the somatic marker gate, the
        # drive budgets and the memory retention check. Pure registration;
        # nothing ticks until an event is appraised.
        "interiority": register_interiority,
    }

    engines: dict[str, Any] = {}
    for name, register in registrations.items():
        try:
            engines[name] = register(orchestrator)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "derived_engines",
                exc,
                severity="warning",
                action=f"skipped derived engine '{name}' registration; other organs still registered",
            )
            logger.warning("Derived engine '%s' failed to register: %s", name, exc)

    logger.info("✅ Derived engines registered from organs: %s", ", ".join(engines))
    return engines


__all__ = ["register_derived_engines"]
