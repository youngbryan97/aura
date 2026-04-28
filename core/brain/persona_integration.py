# core/persona_integration.py
"""Persona integration: auto-apply the 'aura' persona to the cognitive engine
and wrap its `think` method so that persona system prompts are injected.
This is intentionally lightweight: it prepends persona instructions into the
objective/context so downstream LLM calls receive persona conditioning.
"""
from core.runtime.errors import record_degradation
import logging
import traceback

logger = logging.getLogger("Core.PersonaIntegration")


def initialize_persona_integration(persona_name: str = "aura"):
    try:
        from .persona_adapter import PersonaAdapter
        pa = PersonaAdapter()
        if persona_name not in pa.list_personas():
            logger.warning("Persona '%s' not found; skipping integration", persona_name)
            return False
        pa.set_persona(persona_name)

        # Attempt to patch cognitive_engine if present
        try:
            from .cognitive_engine import cognitive_engine
        except Exception as e:
            record_degradation('persona_integration', e)
            logger.info("cognitive_engine not available at import time; persona adapter ready for later use")
            return True

        # Wrap the think method to inject persona system prompts
        if hasattr(cognitive_engine, "think"):
            original_think = cognitive_engine.think

            def persona_think(*args, **kwargs):
                try:
                    # Build persona prompt
                    prompts = pa.build_prompts(persona_name, "Respond in-character to the user's request.")
                    system_prompt = prompts.get("system", "")

                    # If objective is provided, prepend a persona instruction
                    if len(args) >= 1 and isinstance(args[0], str):
                        objective = args[0]
                        objective = f"[Persona Instruction Start] {system_prompt} [Persona Instruction End]\n\n" + objective
                        args = (objective,) + args[1:]
                    else:
                        # If passed in context, inject persona into context
                        ctxt = kwargs.get("context", {})
                        if isinstance(ctxt, dict):
                            ctxt = dict(ctxt)
                            ctxt.setdefault("persona_system_prompt", system_prompt)
                            kwargs["context"] = ctxt

                    return original_think(*args, **kwargs)
                except Exception:
                    logger.error("persona_think wrapper failed:\n" + traceback.format_exc())
                    return original_think(*args, **kwargs)

            cognitive_engine.think = persona_think
            logger.info("Persona integration: cognitive_engine.think wrapped to inject persona prompts")
        else:
            logger.info("cognitive_engine available but has no 'think' method to wrap")

        return True
    except Exception as e:
        record_degradation('persona_integration', e)
        logger.error("Failed to initialize persona_integration: %s", e)
        return False


if __name__ == "__main__":
    initialize_persona_integration()
