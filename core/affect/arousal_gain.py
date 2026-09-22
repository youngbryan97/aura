"""Arousal widens the gap between what matters and what does not.

Arousal does not raise every signal together. Under arousal, what is already
prioritised is processed more and what is not is processed less: Mather,
Clewett, Sakaki and Harley, "Norepinephrine ignites local hotspots of neuronal
excitation: How arousal amplifies selectivity in perception and memory",
Behavioral and Brain Sciences 39 (2016), e200.

A reading here is a level in [0, 1] with a point where it is unremarkable, and
arousal moves the level away from that point in whichever direction it already
lies. The product term is the whole mechanism. With arousal at zero a level
comes back unchanged, and a level at its usual point is unchanged at any
arousal, so neither reading decides the result without the other.
"""

from __future__ import annotations

__all__ = ["gained"]


def _unit(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


def gained(level: float, arousal: float, *, usual: float) -> float:
    """``level`` moved away from ``usual`` in proportion to ``arousal``, in [0, 1].

    ``usual`` is the level's own unremarkable point, which the caller knows and
    this function does not. For a reading divided by itself plus its running
    mean it is one half, where the moment is exactly as large as usual.
    """
    now = _unit(level)
    return _unit(now + _unit(arousal) * (now - _unit(usual)))
