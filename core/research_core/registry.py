"""ServiceContainer registration for the SelfImprovingResearchCore.

Aura owns the research core: registering it in the global
``ServiceContainer`` makes the rest of the runtime able to discover
it without explicit wiring.  Other modules (curriculum loop,
diagnostics bundle, runbooks) look it up via the standard service
name.
"""
from __future__ import annotations

from typing import Any, Optional

from core.research_core.core import ResearchCoreConfig, SelfImprovingResearchCore


def register_research_core(
    *,
    cfg: Optional[ResearchCoreConfig] = None,
    container: Optional[Any] = None,
) -> SelfImprovingResearchCore:
    """Construct + register the research core in the ServiceContainer.

    Idempotent: returns the existing instance if already registered.
    """
    if container is None:
        try:
            from core.container import ServiceContainer

            container = ServiceContainer
        except (ImportError, AttributeError, RuntimeError):
            container = None

    if container is not None:
        existing = None
        try:
            existing = container.get(SelfImprovingResearchCore.SERVICE_NAME, default=None)
        except (OSError, ConnectionError, TimeoutError):
            existing = None
        if existing is not None:
            return existing

    core = SelfImprovingResearchCore(cfg=cfg)

    if container is not None:
        try:
            container.register_instance(SelfImprovingResearchCore.SERVICE_NAME, core)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            # Container failures must not stop the core from running
            # — callers that hold a direct reference still work. Callers that
            # ask the container do not, so the failure is recorded.
            from core.runtime.errors import record_degradation

            record_degradation(
                "research_core",
                exc,
                severity="warning",
                action="kept the research core running unregistered in the container",
            )
    return core
