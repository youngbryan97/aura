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

import hashlib
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


#: How many coordinates the recalled content enters as. The observation is
#: padded to the model's width, so this costs nothing the model did not already
#: have room for.
CONTENT_WIDTH: int = 4

#: How many tokens of the recalled context enter the coordinate. Bounded so a
#: long recollection costs no more to read than a short one.
_RECALL_TOKENS: int = 64

#: The situation as the model sees it: seventeen readings plus the coordinate
#: of what she recalled. Declared here rather than counted in a test, so a
#: widening is a deliberate edit to a contract rather than a number that broke.
OBSERVATION_READINGS: int = 17
#: What she is trying to do, as a coordinate, and how hard her most depleted
#: drive is pulling. Both sit after the recalled coordinate rather than among
#: the readings: the model has learned a weight for every position before them,
#: and an insertion would move each of those onto a feature it never saw.
GOAL_WIDTH: int = CONTENT_WIDTH
PRESSURE_READINGS: int = 1
OBSERVATION_WIDTH: int = OBSERVATION_READINGS + CONTENT_WIDTH + GOAL_WIDTH + PRESSURE_READINGS


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
            # And what she actually remembered, as a coordinate. Everything
            # above about memory is metadata: how strongly the best recollection
            # scored, and how many things are in working memory. A world model
            # shown the strength of a recall and not its content cannot let
            # remembered context contribute to what it infers — which is the
            # thing recalled context is supposed to do. Measured before this,
            # memory reached the world model at 0.22 against a bar of 0.30, and
            # it was the only channel keeping active memory from a second route
            # out of the workspace.
            *_recalled(cognition),
            # What she is trying to do, and not only how many things. The goal
            # count above moves by one when an intention is added, among twenty
            # other inputs, and a displaced drive reached the world model at a
            # median of zero in run_023: the model was shown that she had
            # intentions and never what they were.
            *_intended(cognition),
            _pressure(state),
        ],
        dtype=np.float64,
    )


def _coordinate(text: str, width: int = CONTENT_WIDTH) -> list[float]:
    """A body of text as a coordinate: sharing most words means being close.

    Every token is projected onto every coordinate by a fixed function of the
    token, and the profile is the mean over tokens. Two contents sharing most
    of their tokens have nearly the same profile; two sharing none are
    near-orthogonal. A single hash of the whole string would not do: it has no
    magnitude, so two recollections differing by one word would sit as far
    apart as two with nothing in common, and everything downstream of this is a
    distance.

    Written here rather than imported. The subject-core schema computes the
    same shape for its own columns, and a production module reaching into the
    measurement package for it would put the instrument inside the thing being
    measured.
    """
    raw = str(text or "")
    if not raw:
        return [0.0] * width
    tokens = (raw.split() or [raw])[:_RECALL_TOKENS]
    totals = [0.0] * width
    for token in tokens:
        digest = hashlib.blake2b(
            token.encode("utf-8", "ignore"), digest_size=2 * width
        ).digest()
        for index in range(width):
            word = int.from_bytes(digest[2 * index : 2 * index + 2], "big")
            totals[index] += word / 32767.5 - 1.0
    return [value / float(len(tokens)) for value in totals]


def _recalled(cognition: Any) -> list[float]:
    """What is in mind, as a coordinate rather than as a count.

    The retrieved set first, because that is what recall put there; working
    memory when nothing was retrieved, because a turn with no recall still has
    a context.
    """
    if cognition is None:
        return [0.0] * CONTENT_WIDTH
    recalled = list(getattr(cognition, "long_term_memory", []) or [])
    text = " ".join(str(item) for item in recalled[-4:])
    if not text:
        working = list(getattr(cognition, "working_memory", []) or [])
        text = " ".join(
            str(item.get("content", "") if isinstance(item, dict) else item)
            for item in working[-4:]
        )
    return _coordinate(text)


def _intended(cognition: Any) -> list[float]:
    """What she is trying to do, as a coordinate rather than as a count.

    The newest three goals, by the text the goal engine writes for them, under
    the same projection as a recollection, so an intention and a memory that
    share their words sit near each other.
    """
    if cognition is None:
        return [0.0] * GOAL_WIDTH
    parts: list[str] = []
    for goal in list(getattr(cognition, "active_goals", []) or [])[-3:]:
        if isinstance(goal, dict):
            text = goal.get("goal") or goal.get("objective") or goal.get("description") or ""
        else:
            text = getattr(goal, "description", "") or getattr(goal, "title", "") or goal
        if text:
            parts.append(str(text))
    return _coordinate(" ".join(parts), GOAL_WIDTH)


def _pressure(state: Any) -> float:
    """How far her most depleted drive is from full, as a share of its capacity.

    Deliberation acts on the most pressing need, so of everything about
    motivation this is the number that decides what she does next. Zero when
    every drive is full or there are none to read.
    """
    motivation = getattr(state, "motivation", None)
    budgets = (getattr(motivation, "budgets", None) or {}) if motivation is not None else {}
    deepest = 0.0
    for entry in budgets.values():
        if not isinstance(entry, dict):
            continue
        capacity = _num(entry.get("capacity", 100.0), 100.0) or 100.0
        level = _num(entry.get("current", entry.get("level", capacity)), capacity)
        deepest = max(deepest, max(0.0, capacity - level) / capacity)
    return min(1.0, deepest)


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
