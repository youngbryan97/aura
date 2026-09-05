"""Register Aura's self-knowing bridge services."""
from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service


def initialize_self_knowing(orchestrator: Any | None = None) -> dict[str, Any]:
    """Initialize bounded self-knowing services and register them as optional organs."""
    try:
        from core.consciousness.automatic_self_knowing import AutomaticSelfKnowingKernel
        from core.consciousness.phenomenal_knowing import get_phenomenal_knowing_kernel
        from core.consciousness.recursive_self_knowing import get_recursive_self_knowing_kernel

        phenomenal = get_runtime_service("phenomenal_knowing", default=None)
        if phenomenal is None:
            phenomenal = get_phenomenal_knowing_kernel()
            register_runtime_service("phenomenal_knowing", phenomenal, required=False)

        recursive = get_runtime_service("recursive_self_knowing", default=None)
        if recursive is None:
            recursive = get_recursive_self_knowing_kernel()
            register_runtime_service("recursive_self_knowing", recursive, required=False)

        automatic = get_runtime_service("automatic_self_knowing", default=None)
        if automatic is None:
            substrate = get_runtime_service("live_substrate", default=None) or get_runtime_service(
                "liquid_substrate", default=None
            )
            automatic = AutomaticSelfKnowingKernel(
                recursive_self_knowing=recursive,
                phenomenal_knowing=phenomenal,
                live_substrate=substrate,
            )
            register_runtime_service("automatic_self_knowing", automatic, required=False)

        if orchestrator is not None:
            setattr(orchestrator, "phenomenal_knowing", phenomenal)
            setattr(orchestrator, "recursive_self_knowing", recursive)
            setattr(orchestrator, "automatic_self_knowing", automatic)

        return {
            "phenomenal_knowing": phenomenal,
            "recursive_self_knowing": recursive,
            "automatic_self_knowing": automatic,
        }
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "self_knowing_initializer",
            exc,
            action="continued without self-knowing bridge registration",
        )
        return {}


__all__ = ["initialize_self_knowing"]
