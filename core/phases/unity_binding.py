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


# ─────────────────────────────────────────────────────────────────────────────
# Declared semantics. See core/runtime/cognitive_contract.py.
#
# `writes` is MEASURED — tools/observe_phase_writes.py ran this phase against a
# real AuraState and recorded which fields moved. It is not a reading of the
# code, which is how a declaration ends up describing what the author believed.
from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)

register_contract(
    CognitiveTransformContract(
        name="UnityBindingPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Bind the tick's separate streams into one mind-moment and score "
            "how fragmented that binding was."
        ),
        reads=(
            "cognition.phenomenal_state",
            "cognition.modifiers",
            "affect.valence",
            "affect.arousal",
        ),
        writes=(
            # Written by the unity delegate at core/unity/runtime.py:732, on a
            # path tools/observe_phase_writes.py cannot exercise without a
            # model — so the measurement below reported six fields and the
            # receipt check found a seventh the first time the real branch
            # ran, several hundred times over. That is the check working, and
            # the omission is read out of the code exactly as the tool's own
            # documentation says an unexercised branch must be.
            "cognition.coherence_score",
            "cognition.fragmentation_score",
            "cognition.mind_moment",
            "cognition.phenomenal_state",
            "cognition.unity_state",
            "response_modifiers",
            "transition_cause",
        ),
        preconditions=("state carries a cognition block",),
        branches=(
            BranchSpec(
                "bound",
                "streams are present and mutually consistent",
                "emit a mind-moment with a low fragmentation score",
            ),
            BranchSpec(
                "fragmented",
                "streams disagree or are partially missing",
                "emit the moment and raise the fragmentation score",
            ),
        ),
        invariants=("cognition.mind_moment is written whenever unity_state is",),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py"
            "; reads reach state through this phase's delegate rather than appearing in this module, so they are declared from the delegate's behaviour and not checkable by scanning this file alone"
        ),
    )
)
