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
        except Exception:
            container = None

    if container is not None:
        existing = None
        try:
            existing = container.get(SelfImprovingResearchCore.SERVICE_NAME, default=None)
        except Exception:
            existing = None
        if existing is not None:
            return existing

    core = SelfImprovingResearchCore(cfg=cfg)

    if container is not None:
        try:
            container.register_instance(SelfImprovingResearchCore.SERVICE_NAME, core)
        except Exception:
            # Container failures must not stop the core from running
            # — callers that hold a direct reference still work.
            pass
    return core
