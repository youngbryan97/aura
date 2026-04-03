from __future__ import annotations

import logging
from typing import Any, Optional

from core.container import ServiceContainer
from core.consciousness.executive_closure import ExecutiveClosureEngine

from . import BasePhase
from ..state.aura_state import AuraState

logger = logging.getLogger("Aura.ExecutiveClosurePhase")


class ExecutiveClosurePhase(BasePhase):
    """Bind the distributed cognition stack into the live AuraState."""

    def __init__(self, container: Any = None):
        super().__init__(container=container)
        self._engine: Optional[ExecutiveClosureEngine] = None

    def _get_engine(self) -> ExecutiveClosureEngine:
        if self._engine is not None:
            return self._engine

        engine = ServiceContainer.get("executive_closure", default=None)
        if engine is None:
            engine = ExecutiveClosureEngine()
            try:
                ServiceContainer.register_instance("executive_closure", engine)
            except Exception as exc:
                logger.debug("ExecutiveClosurePhase: registration skipped: %s", exc)
        self._engine = engine
        return engine

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        engine = self._get_engine()
        new_state = state.derive("executive_closure", origin="system")
        return await engine.integrate(new_state)
