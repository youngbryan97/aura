"""Feeding the world model what happened, so its surprise means something.

`UnifiedWorldModel` carries a forward-dynamics model with a running surprise,
and the runtime reads that surprise in two places: affect grounding uses it as
prediction error, and the free-energy engine takes it as a signal. Nothing fed
it. The only caller of `observe` in the tree was the ontogeny organ, on its own
separate model, so the number both of those readers depend on was the surprise
of a model that had never seen anything.

A predictive model that is never shown the world does not have a low prediction
error, it has no prediction error, and the two are the same number.

The observation is the cycle's own situation as a vector: what she perceived,
what her body was doing, how she felt, what she was attending to, how much she
remembered and how many goals were open. The action is what she did about it.
Both are read from the state that just finished a turn, so the model learns the
transition the organism actually made rather than a summary written for it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.state.percepts import read_percept

__all__ = ["observation_of", "action_of", "observe_cycle"]

logger = logging.getLogger("Aura.WorldModel.Cycle")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _novelty(percepts: list[Any]) -> float:
    """How much of the recent stream is unlike the rest of it."""
    if not percepts:
        return 0.0
    tail = [read_percept(item).content for item in percepts[-8:]]
    return len(set(tail)) / float(len(tail))


def observation_of(state: Any) -> np.ndarray:
    """The situation, as the model sees it. Ordered, fixed width, no strings."""
    cognition = getattr(state, "cognition", None)
    world = getattr(state, "world", None)
    affect = getattr(state, "affect", None)
    soma = getattr(state, "soma", None)
    hardware = (getattr(soma, "hardware", {}) or {}) if soma else {}
    percepts = list(getattr(world, "recent_percepts", []) or []) if world else []
    newest = read_percept(percepts[-1]) if percepts else None
    scores = list(getattr(cognition, "memory_scores", []) or []) if cognition else []
    return np.array(
        [
            _num(len(percepts)),
            # What arrived, not only how much. A model that sees the count of
            # percepts and not their strength is being shown that something
            # happened and not what.
            _num(newest.salience) if newest is not None else 0.0,
            _num(_novelty(percepts)),
            # And how strongly what is in mind was recalled. A recollection
            # that answered the question is a different situation from one that
            # scraped in, and the ranking that says which is already computed.
            max((_num(score) for score in scores), default=0.0),
            _num(hardware.get("cpu_usage")) / 100.0,
            _num(hardware.get("temperature")) / 100.0,
            _num(getattr(affect, "valence", 0.0)) if affect else 0.0,
            _num(getattr(affect, "arousal", 0.5), 0.5) if affect else 0.5,
            _num(getattr(affect, "curiosity", 0.5), 0.5) if affect else 0.5,
            _num(getattr(cognition, "coherence_score", 1.0), 1.0) if cognition else 1.0,
            _num(getattr(cognition, "fragmentation_score", 0.0)) if cognition else 0.0,
            _num(getattr(cognition, "conversation_energy", 0.5), 0.5) if cognition else 0.5,
            _num(len(getattr(cognition, "working_memory", []) or [])) if cognition else 0.0,
            _num(len(getattr(cognition, "active_goals", []) or [])) if cognition else 0.0,
            _num(len(getattr(world, "facts", {}) or {})) if world else 0.0,
            _num(getattr(state, "phi", 0.0)),
            _num(getattr(state, "vitality", 1.0), 1.0),
        ],
        dtype=np.float64,
    )


def action_of(state: Any) -> np.ndarray:
    """What she did about it, as the model sees it.

    Zeros mean she did nothing this cycle, which is a real action and is
    exactly what a passive observation should look like to a forward model.
    """
    cognition = getattr(state, "cognition", None)
    world = getattr(state, "world", None)
    last = (getattr(world, "facts", {}) or {}).get("last_action", {}) if world else {}
    goals = getattr(cognition, "active_goals", []) or [] if cognition else []
    return np.array(
        [
            1.0 if last else 0.0,
            1.0 if isinstance(last, dict) and last.get("verified") else 0.0,
            1.0 if isinstance(last, dict) and last.get("actor") == "self" else 0.0,
            _num(len(goals)),
        ],
        dtype=np.float64,
    )


def observe_cycle(state: Any, world_model: Any = None) -> float | None:
    """Show the model this turn. Returns the surprise, or None if it could not.

    Total: a world model that will not take an observation is a degradation,
    never an exception into a cognitive phase.
    """
    try:
        if world_model is None:
            from core.container import ServiceContainer

            world_model = ServiceContainer.get("unified_world_model", default=None)
        if world_model is None:
            return None
        world_model.observe(observation_of(state), action_of(state), learn=True)
        surprise = world_model.surprise()
        return None if surprise is None else float(surprise)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        record_degradation(
            "world_model_cycle",
            exc,
            severity="warning",
            action="the world model did not see this cycle",
        )
        return None
