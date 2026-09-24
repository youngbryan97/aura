"""How much control a moment is worth, and where it is spent.

Cognitive control is allocated by its expected value: what is at stake, times
how likely control is to change the outcome, less what it costs (Shenhav,
Botvinick and Cohen, "The expected value of control", Neuron 79, 2013). The
first two multiply. People put effort into a reward only when their effort is
efficacious (Frömer, Lin, Dean Wolf, Inzlicht and Shenhav, "Expectations of
reward and efficacy guide cognitive control allocation", Nature Communications
12, 2021), and the control allocated reaches prefrontal cortex as
catecholaminergic gain (Aston-Jones and Cohen, "An integrative theory of locus
coeruleus-norepinephrine function", Annual Review of Neuroscience 28, 2005).

Her drives say what is at stake and her model of herself says whether steering
herself works, and nothing put the two together: the executive tier of her mesh
ran at the same gain whatever she wanted and however well she knew herself.

    at stake   the urgency of the most pressing goal or initiative she holds,
               as the arbiter decided it
    efficacy   her confidence in her model of herself, scored against
               predicting no change (core/consciousness/self_prediction.py).
               A regulator works to the degree it models what it regulates
               (Conant and Ashby, "Every good regulator of a system must be a
               model of that system", 1970), so control over her own cognition
               works to the degree she can predict it
    value      their product

The executive tier's gain is that value against her ordinary value, saturating:
2v / (v + ordinary). It is one at an ordinary moment, rises towards two as more
is worth controlling, and falls towards zero when nothing is at stake or nothing
she does to herself works. Her ordinary value is the running mean of her own,
as the substrate prices its work against its ordinary step. No weight is set
here: the form is the saturating one and the scale is hers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger("Consciousness.ControlAllocation")

__all__ = ["ControlAllocation", "allocate_control", "at_stake", "current_state", "efficacy"]


def _clip(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number)) if number == number else 0.0


def at_stake(items: Iterable[Any]) -> float:
    """The most pressing urgency among what she holds, decided over declared."""
    best = 0.0
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        stated = item.get("decided_urgency")
        if stated is None:
            stated = item.get("urgency")
        if stated is not None:
            best = max(best, _clip(stated))
    return best


def efficacy() -> float | None:
    """Her confidence in her model of herself, or None when there is no model yet."""
    try:
        from core.runtime.service_registry import get_runtime_service

        predictor = get_runtime_service("self_prediction", default=None)
        prediction = predictor.get_current_prediction() if predictor is not None else None
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("no model of herself to price control with: %s", exc)
        return None
    if prediction is None:
        return None
    return _clip(getattr(prediction, "confidence", None))


def current_state() -> Any:
    """Her state as the repository holds it, or None."""
    try:
        from core.container import ServiceContainer

        repository = ServiceContainer.get("state_repository", default=None)
    except (ImportError, RuntimeError) as exc:
        logger.debug("no repository to read her state from: %s", exc)
        return None
    return getattr(repository, "_current", None) if repository is not None else None


class ControlAllocation:
    """The executive tier's gain, from what control is worth now against usually."""

    def __init__(self) -> None:
        self.ordinary: float = 0.0
        self.steps: int = 0
        self.value: float = 0.0
        self.gain: float = 1.0

    def allocate(self, stake: float, works: float) -> float:
        self.value = _clip(stake) * _clip(works)
        self.steps += 1
        self.ordinary += (self.value - self.ordinary) / self.steps
        if self.ordinary <= 1e-12:
            self.gain = 1.0
        else:
            self.gain = 2.0 * self.value / (self.value + self.ordinary)
        return self.gain


def allocate_control(mesh: Any, allocation: ControlAllocation, state: Any = None) -> float | None:
    """Set the mesh's executive gain from her state. None when there is nothing to read."""
    state = state if state is not None else current_state()
    if mesh is None or state is None or not hasattr(mesh, "set_regional_modulation"):
        return None
    works = efficacy()
    if works is None:
        return None
    cognition = getattr(state, "cognition", None)
    holds = list(getattr(cognition, "active_goals", None) or []) + list(
        getattr(cognition, "pending_initiatives", None) or []
    )
    gain = allocation.allocate(at_stake(holds), works)
    mesh.set_regional_modulation({"executive": (gain, 1.0)})
    return gain
