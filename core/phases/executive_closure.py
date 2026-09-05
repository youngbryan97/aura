from __future__ import annotations

import logging
from typing import Any

from core.consciousness.executive_closure import ExecutiveClosureEngine
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger("Aura.ExecutiveClosurePhase")


class ExecutiveClosurePhase(BasePhase):
    """Bind the distributed cognition stack into the live AuraState."""

    def __init__(self, container: Any = None):
        super().__init__(container=container)
        self._engine: ExecutiveClosureEngine | None = None

    def _get_engine(self) -> ExecutiveClosureEngine:
        if self._engine is not None:
            return self._engine

        engine = get_runtime_service("executive_closure", default=None)
        if engine is None:
            engine = ExecutiveClosureEngine()
            try:
                register_runtime_service("executive_closure", engine)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('executive_closure', exc)
                logger.debug("ExecutiveClosurePhase: registration skipped: %s", exc)
        self._engine = engine
        return engine

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        engine = self._get_engine()
        new_state = state.derive("executive_closure", origin="system")
        return await engine.integrate(new_state)
