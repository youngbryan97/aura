from __future__ import annotations

from typing import Any

from core.container import ServiceContainer


def optional_service(*names: Any, default: Any = None) -> Any:
    for name in names:
        if name in (None, ""):
            continue
        try:
            service = ServiceContainer.get(name, default=None)
        except Exception:
            continue
        if service is not None:
            return service
    return default


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
        except Exception:
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
    except Exception:
        return default


def resolve_social_imagination(*, default: Any = None) -> Any:
    imagination = optional_service("social_imagination", default=None)
    if imagination is not None:
        return imagination
    try:
        from core.social.social_imagination import get_social_imagination

        return get_social_imagination()
    except Exception:
        return default


def resolve_conversational_dynamics(*, default: Any = None) -> Any:
    return optional_service("conversational_dynamics", default=default)


def resolve_memory_facade(*, default: Any = None) -> Any:
    return optional_service("memory_facade", "conversation_engine", default=default)


def resolve_liquid_substrate(*, default: Any = None) -> Any:
    return optional_service("liquid_substrate", "conscious_substrate", "liquid_state", default=default)
