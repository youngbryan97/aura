from __future__ import annotations

from typing import Any

from core.container import ServiceContainer
from core.exceptions import ServiceNotFoundError
from core.runtime.errors import record_degradation


def optional_service(*names: Any, default: Any = None) -> Any:
    for name in names:
        if name in (None, ""):
            continue
        try:
            service = ServiceContainer.get(name, default=None)
        except (ImportError, AttributeError, RuntimeError):
            continue
        if service is not None:
            return service
    return default


def require_service(*names: Any) -> Any:
    last_error: Exception | None = None
    for name in names:
        if name in (None, ""):
            continue
        try:
            service = ServiceContainer.require(name)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('service_access', exc)
            last_error = exc
            continue
        if service is not None:
            return service
    if last_error is not None:
        raise last_error
    raise ServiceNotFoundError("No service names provided to require_service().")


def resolve_orchestrator(*, default: Any = None) -> Any:
    return optional_service("orchestrator", default=default)


def resolve_state_repository(orchestrator: Any = None, *, default: Any = None) -> Any:
    for attr_name in ("state_repository", "state_repo"):
        repo = getattr(orchestrator, attr_name, None) if orchestrator is not None else None
        if repo is not None:
            return repo
    return optional_service("state_repository", "state_repo", default=default)


def resolve_kernel_interface(orchestrator: Any = None, *, default: Any = None) -> Any:
    kernel_interface = getattr(orchestrator, "kernel_interface", None) if orchestrator is not None else None
    if kernel_interface is not None:
        return kernel_interface
    return optional_service("kernel_interface", default=default)


def resolve_goal_engine(*, default: Any = None) -> Any:
    return optional_service("goal_engine", default=default)


def resolve_task_engine(*, default: Any = None) -> Any:
    return optional_service("task_engine", default=default)


def resolve_source_body(*, default: Any = None) -> Any:
    """The source-body proprioception organ (boot-over-boot code deltas)."""
    return optional_service("source_body", default=default)


def resolve_canonical_self_engine(*, default: Any = None, autocreate: bool = True) -> Any:
    engine = optional_service("canonical_self_engine", default=None)
    if engine is not None:
        return engine
    if not autocreate:
        return default
    try:
        from core.self.canonical_self import get_canonical_self_engine

        return get_canonical_self_engine()
    except (ImportError, AttributeError, RuntimeError):
        return default


def resolve_canonical_self(*, default: Any = None, autocreate: bool = True) -> Any:
    current = optional_service("canonical_self", default=None)
    if current is not None:
        return current

    engine = resolve_canonical_self_engine(default=None, autocreate=autocreate)
    getter = getattr(engine, "get_self", None) if engine is not None else None
    if callable(getter):
        try:
            current = getter()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return default
        if current is not None:
            return current
    return default


def resolve_identity_model(*, default: Any = None) -> Any:
    canonical = resolve_canonical_self(default=None)
    if canonical is not None:
        return canonical
    return optional_service("self_model", "identity", default=default)


def resolve_identity_prompt_surface(orchestrator: Any = None, *, default: Any = None) -> Any:
    """The identity surface that can render a bounded system prompt.

    ``identity_system`` is checked alongside ``identity``: that is the name the
    cognitive context manager registered under, and the resolver not knowing it
    is why that call site looked the services up itself — and then raised
    "identity service has no bounded prompt surface" on every turn. Either
    name, and either getter, resolves here.
    """
    for service_name in ("identity", "identity_system"):
        identity = optional_service(service_name, default=None)
        if identity is not None and (
            hasattr(identity, "get_full_system_prompt")
            or hasattr(identity, "get_full_system_prompt_injection")
        ):
            return identity
    try:
        from core.identity import get_identity_system

        prompt_surface = get_identity_system(orchestrator)
    except (ImportError, AttributeError, RuntimeError):
        prompt_surface = None
    if prompt_surface is not None and (
        hasattr(prompt_surface, "get_full_system_prompt")
        or hasattr(prompt_surface, "get_full_system_prompt_injection")
    ):
        return prompt_surface
    fallback = resolve_identity_model(default=None)
    if fallback is not None and hasattr(fallback, "get_full_system_prompt"):
        return fallback
    return default


def resolve_identity_ego_surface(*, default: Any = None) -> Any:
    identity = optional_service("identity", default=None)
    if identity is not None and (hasattr(identity, "get_ego_prompt") or hasattr(identity, "get_self_awareness_prompt")):
        return identity
    model = optional_service("self_model", default=None)
    if model is not None and (hasattr(model, "get_ego_prompt") or hasattr(model, "get_self_awareness_prompt")):
        return model
    return default


def resolve_llm_router(*, kernel_interface: Any = None, default: Any = None) -> Any:
    router = optional_service("llm_router", default=None)
    if router is not None:
        return router

    ki = kernel_interface or resolve_kernel_interface(default=None)
    kernel = getattr(ki, "kernel", None) if ki is not None else None
    organs = getattr(kernel, "organs", {}) or {}
    llm_organ = organs.get("llm")
    if llm_organ is None:
        return default

    getter = getattr(llm_organ, "get_instance", None)
    if callable(getter):
        try:
            instance = getter()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            instance = None
        if instance is not None:
            return instance

    return getattr(llm_organ, "instance", default)


def resolve_dialogue_cognition(*, default: Any = None) -> Any:
    dialogue = optional_service("dialogue_cognition", default=None)
    if dialogue is not None:
        return dialogue
    try:
        from core.social.dialogue_cognition import get_dialogue_cognition

        return get_dialogue_cognition()
    except (ImportError, AttributeError, RuntimeError):
        return default


def resolve_social_imagination(*, default: Any = None) -> Any:
    imagination = optional_service("social_imagination", default=None)
    if imagination is not None:
        return imagination
    try:
        from core.social.social_imagination import get_social_imagination

        return get_social_imagination()
    except (ImportError, AttributeError, RuntimeError):
        return default


def resolve_conversational_dynamics(*, default: Any = None) -> Any:
    return optional_service("conversational_dynamics", default=default)


def resolve_memory_facade(*, default: Any = None) -> Any:
    return optional_service("memory_facade", "conversation_engine", default=default)


def resolve_liquid_substrate(*, default: Any = None) -> Any:
    return optional_service("liquid_substrate", "conscious_substrate", "liquid_state", default=default)


def resolve_conscious_substrate(*, default: Any = None) -> Any:
    return optional_service("conscious_substrate", "liquid_substrate", "liquid_state", default=default)


def resolve_affect_engine(*, default: Any = None) -> Any:
    return optional_service("affect_engine", "liquid_state", default=default)


def resolve_belief_graph(*, default: Any = None) -> Any:
    return optional_service("belief_graph", "knowledge_graph", default=default)


def resolve_self_prediction(*, default: Any = None) -> Any:
    return optional_service("self_prediction", default=default)


def resolve_motivation_engine(*, default: Any = None) -> Any:
    return optional_service("motivation_engine", "drive_engine", default=default)


def resolve_curiosity_engine(*, default: Any = None) -> Any:
    return optional_service("curiosity_engine", default=default)


def resolve_personality_engine(*, default: Any = None) -> Any:
    return optional_service("personality_engine", "personality", default=default)


def resolve_voice_engine(*, default: Any = None) -> Any:
    return optional_service("voice_engine", default=default)


def resolve_vector_memory_engine(*, default: Any = None) -> Any:
    return optional_service("vector_memory_engine", default=default)


def resolve_attention_schema(*, default: Any = None) -> Any:
    return optional_service("attention_schema", default=default)


def resolve_global_workspace(*, default: Any = None) -> Any:
    return optional_service("global_workspace", default=default)


def resolve_temporal_binding(*, default: Any = None) -> Any:
    return optional_service("temporal_binding", default=default)


def resolve_homeostatic_coupling(*, default: Any = None) -> Any:
    return optional_service("homeostatic_coupling", default=default)


def resolve_inquiry_engine(*, default: Any = None) -> Any:
    return optional_service("inquiry_engine", default=default)


def resolve_narrative_thread(*, default: Any = None) -> Any:
    return optional_service("narrative_thread", default=default)


def resolve_semantic_memory(*, default: Any = None) -> Any:
    return optional_service("semantic_memory", default=default)


def resolve_metabolic_monitor(*, default: Any = None) -> Any:
    return optional_service("metabolic_monitor", "metabolism", default=default)


def resolve_edi(*, default: Any = None) -> Any:
    return optional_service("edi", default=default)


def resolve_theory_of_mind(*, default: Any = None) -> Any:
    return optional_service("theory_of_mind", default=default)


def resolve_epistemic_state(*, default: Any = None) -> Any:
    return optional_service("epistemic_state", default=default)


def resolve_epistemic_humility(*, default: Any = None) -> Any:
    return optional_service("epistemic_humility", default=default)


def resolve_cognitive_engine(*, default: Any = None) -> Any:
    return optional_service("cognitive_engine", default=default)


# ── the remaining core seams (July critique: 'cross-wired through
#    ServiceContainer.get() rather than clean interfaces') ──────────────────
#
# These named resolvers are the front door for the load-bearing services.
# New code resolves through them; tests/test_service_access_ratchet.py caps
# raw ServiceContainer.get() usage so the cross-wiring only shrinks.

def resolve_will(*, default: Any = None) -> Any:
    will = optional_service("will", default=None)
    if will is not None:
        return will
    try:
        from core.will import get_will

        return get_will()
    except (ImportError, AttributeError, RuntimeError):
        return default


def resolve_inference_gate(*, default: Any = None) -> Any:
    return optional_service("inference_gate", default=default)


def resolve_skill_router(*, default: Any = None) -> Any:
    """The tool-execution seam (CapabilityEngine)."""
    return optional_service("skill_router", "capability_engine", default=default)


def resolve_mlx_client(*, default: Any = None) -> Any:
    client = optional_service("mlx_client", default=None)
    if client is not None:
        return client
    try:
        from core.brain.llm.mlx_client import get_mlx_client

        return get_mlx_client()
    except (ImportError, AttributeError, RuntimeError):
        return default


def resolve_weight_compounding(*, default: Any = None) -> Any:
    return optional_service("weight_compounding", default=default)


def resolve_selfplay_flywheel(*, default: Any = None) -> Any:
    return optional_service("selfplay_flywheel", default=default)


def resolve_practice_director(*, default: Any = None) -> Any:
    """The Practice Director (failure-directed curriculum), spine-resolved
    only — consumers must never self-create one, or hermetic tests would
    read (and write) the real machine's practice ledger."""
    return optional_service("practice_director", default=default)


def resolve_incident_narrator(*, default: Any = None) -> Any:
    narrator = optional_service("incident_narrator", default=None)
    if narrator is not None:
        return narrator
    try:
        from core.observability.incident_narrator import get_incident_narrator

        return get_incident_narrator()
    except (ImportError, AttributeError, RuntimeError):
        return default
