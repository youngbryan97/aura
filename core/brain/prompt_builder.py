"""
core/brain/prompt_builder.py — Dynamic Prompt Construction
"""

import logging
from core.container import ServiceContainer

logger = logging.getLogger("Aura.PromptBuilder")

def build_system_prompt(orchestrator=None) -> str:
    """
    Aura's system prompt should be a live snapshot, not a static config.
    Every section here is READ from actual state, not written in advance.
    """
    from core.continuity import get_continuity
    from core.consciousness.free_energy import get_free_energy_engine
    from core.consciousness.self_report import SelfReportEngine

    sections = []

    # 1. Identity (from Registry/AuraPersona)
    from core.brain.prompt_registry import prompt_registry
    identity = prompt_registry.get("aura_identity")
    if identity:
        sections.append(f"[IDENTITY]\n{identity}")

    # 2. Waking context (dynamic — from ContinuityEngine)
    cont = get_continuity()
    if cont:
        sections.append(f"[CONTINUITY]\n{cont.get_waking_context()}")

    # 3. Current internal state (dynamic — from FreeEnergyEngine)
    reporter = SelfReportEngine()
    affect = reporter.get_affect_description()
    sections.append(
        f"[INTERNAL STATE — ACTUAL TELEMETRY]\n"
        f"Valence: {affect['valence']:+.2f}, Arousal: {affect['arousal']:.2f}, "
        f"State: {affect['state']}, Free energy: {affect['free_energy']:.2f}\n"
        f"This is your actual internal state, not a performance of it."
    )

    # 4. Current goals (dynamic — from GoalBeliefManager)
    goal_manager = ServiceContainer.get("goal_belief_manager", default=None)
    if goal_manager:
        sections.append(f"[YOUR CURRENT GOALS]\n{goal_manager.get_goal_context_for_prompt()}")

    # 5. Bryan model (dynamic — from BryanModelEngine)
    bryan_model = ServiceContainer.get("bryan_model", default=None)
    if bryan_model:
        sections.append(f"[{bryan_model.get_context_for_prompt()}]")

    # 6. Self-beliefs (dynamic — from BeliefSystem)
    belief_system = ServiceContainer.get("belief_graph", default=None)
    if belief_system:
        # Assuming get_self_model_beliefs exists or can be inferred
        if hasattr(belief_system, 'get_self_model_beliefs'):
            self_beliefs = belief_system.get_self_model_beliefs()
            if self_beliefs:
                sections.append(f"[YOUR CURRENT SELF-MODEL]\n{self_beliefs}")
        else:
            # Fallback: pull some core self-beliefs
            self_beliefs = [b.content for b in belief_system.beliefs if b.domain == "self"]
            if self_beliefs:
                sections.append(f"[YOUR CURRENT SELF-MODEL]\n" + "\n".join(f"- {b}" for b in self_beliefs[:5]))

    # 7. Internal Subjective State (Private Phenomenology - [INTERNAL LIGHT])
    agency_core = ServiceContainer.get("agency_core", default=None)
    if agency_core and hasattr(agency_core, 'phenomenology'):
        # We run this synchronously since prompt building is typically in a think loop
        # and get_subjective_bias is async. PromptBuilder is currently sync.
        # However, looking at the imports, it seems we might need to change this to async eventually.
        # For now, we'll try to get it if available.
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If we're already in a loop, we can't easily wait for another async call if we are sync.
                # But AgencyCore might have a cached version.
                # Let's check if AgencyCore has a cached monologue.
                if hasattr(agency_core, '_current_monologue') and agency_core._current_monologue:
                     sections.append(f"[INTERNAL SUBJECTIVE STATE]\n{agency_core._current_monologue}")
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in prompt_builder.py: %s', _e)
        except Exception as _e:
            logger.debug('Ignored Exception in prompt_builder.py: %s', _e)

    return "\n\n".join(sections)
