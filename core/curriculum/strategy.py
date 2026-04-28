"""Strategy controller — escalates approach when prior strategy stalls.

The controller picks one of:

    default         start here for an unfamiliar belief
    more_examples   the model needed more diverse priming
    decompose       break the task into sub-questions
    ask_user        ask the user for a clarifying example
    park            give up on this iteration; return to it later

After ``patience`` consecutive failures with the current strategy, the
controller escalates to the next.  A success resets back to default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Strategy(str, Enum):
    DEFAULT = "default"
    MORE_EXAMPLES = "more_examples"
    DECOMPOSE = "decompose"
    ASK_USER = "ask_user"
    PARK = "park"


_ESCALATION_ORDER = [
    Strategy.DEFAULT,
    Strategy.MORE_EXAMPLES,
    Strategy.DECOMPOSE,
    Strategy.ASK_USER,
    Strategy.PARK,
]


@dataclass
class _BeliefState:
    current: Strategy = Strategy.DEFAULT
    consecutive_failures: int = 0


class StrategyController:
    def __init__(self, *, patience: int = 1):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = int(patience)
        self._states: Dict[str, _BeliefState] = {}

    def _state(self, belief: str) -> _BeliefState:
        return self._states.setdefault(belief, _BeliefState())

    def current(self, belief: str) -> Strategy:
        return self._state(belief).current

    def record_outcome(self, belief: str, *, success: bool) -> Strategy:
        state = self._state(belief)
        if success:
            state.current = Strategy.DEFAULT
            state.consecutive_failures = 0
            return state.current
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.patience:
            idx = _ESCALATION_ORDER.index(state.current)
            if idx + 1 < len(_ESCALATION_ORDER):
                state.current = _ESCALATION_ORDER[idx + 1]
                state.consecutive_failures = 0
        return state.current

    def reset(self, belief: str) -> None:
        self._states.pop(belief, None)
