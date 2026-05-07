from __future__ import annotations

import logging
import time
from typing import Any, Optional

from . import BasePhase
from ..state.aura_state import AuraState
from core.unity import get_unity_runtime

logger = logging.getLogger(__name__)


class UnityBindingPhase(BasePhase):
    """Bind the current rolling present into a durable UnityState."""

    def __init__(self, container: Any = None):
        super().__init__(container)

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        new_state = state.derive("unity_binding", origin="UnityBindingPhase")
        tick_id = f"unity_{new_state.version}_{int(time.time() * 1000)}"
        runtime = get_unity_runtime()
        runtime.apply_to_state(
            new_state,
            objective=objective or str(getattr(new_state.cognition, "current_objective", "") or ""),
            tick_id=tick_id,
            will_receipt_id=str(new_state.response_modifiers.get("will_receipt_id") or ""),
        )
        logger.debug("UnityBindingPhase: unity_score=%.3f level=%s", new_state.cognition.unity_state.unity_score if new_state.cognition.unity_state else -1.0, getattr(new_state.cognition.unity_state, "level", "unknown"))
        return new_state
