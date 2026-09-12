"""A measurement turn that takes the path the channels are actually on.

The hourly influence campaign asked the inference gate for a *background*
generation, and every ``apply_channel`` site sits behind a guard a background
call does not pass: the circumplex behind ``not is_background and
self._origin_is_user_facing(origin)``, and the live-mind and sampling-bias
sites inside the clean-user-surface contract in ``cognitive_engine``. So the
job rotated through nine channels its own generator could not move, and would
have gone on doing that however well its generations went.

This is the generator that can move them. It runs one turn through the
desktop quick-reply lane — the lane the UI talks through — with an origin that
classifies as foreground, which is what opens both guards. The advisory frames
it passes are produced by the real advisory passes over a real state, not
written here: a frame this module invented would make the sampling-bias
channels measure a value this module chose.

What it does NOT do is pretend to be a person. The origin is its own
(``desktop_influence_probe``), the turn carries no conversation history, and
the campaign that calls it is admitted only when nothing is in the foreground
— so the model time it takes is time nobody was waiting for. The lesion around
it is ContextVar-scoped, so a turn running beside it is never served by a
lesioned faculty.

Layering: this measures faculties it must not import at module scope. Every
import is at call time, inside the function that needs it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Verify.InfluenceTurnProbe")

__all__ = [
    "PROBE_ORIGIN",
    "TurnProbeUnavailableError",
    "advisory_frames",
    "run_probe_turn",
]

#: Foreground by classification — `is_foreground_objective_origin` accepts the
#: ``desktop_`` prefix — and used by nothing else, so a transcript can always
#: tell a measurement from a turn somebody took.
PROBE_ORIGIN = "desktop_influence_probe"


class TurnProbeUnavailableError(RuntimeError):
    """The engine or its lane is not there, so no turn can be run."""


def advisory_frames(objective: str) -> dict[str, Any]:
    """The frames the sampling-bias channels carry, made the way a turn makes them.

    Each pass is a plain function over an ``AuraState`` and a prompt. Running
    them here is what gives the three bias channels a value to be lesioned out
    of — without frames they are lesioned from nothing, which measures the
    absence rather than the faculty.
    """
    from core.brain.advisory_passes import (
        apply_bicameral_advisory,
        apply_imagination_workspace,
        apply_spiking_active_inference,
    )
    from core.state.aura_state import AuraState

    state = AuraState()
    context: dict[str, Any] = {}
    for step in (
        apply_spiking_active_inference,
        apply_imagination_workspace,
        apply_bicameral_advisory,
    ):
        try:
            context = step(state, objective, PROBE_ORIGIN, context, is_background=False) or context
        except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            # One advisory that will not run is one channel this turn cannot
            # move. Counted rather than hidden: the caller reports which
            # frames carried a bias.
            logger.debug("Advisory pass %s did not run: %s", getattr(step, "__name__", step), exc)
    return context


def frames_carrying_a_bias(frames: dict[str, Any]) -> tuple[str, ...]:
    """Which advisory frames actually hold a sampling bias."""
    return tuple(
        key
        for key in ("spiking_active_inference", "imagination_workspace", "bicameral_advisory")
        if isinstance(frames.get(key), dict) and frames[key].get("sampling_bias")
    )


def probe_context(frames: dict[str, Any], *, max_tokens: int = 96) -> dict[str, Any]:
    """The context that opens the clean-user-surface lane, and nothing more.

    ``desktop_quick_reply_contract`` is what ``_direct_desktop_quick_reply``
    requires before it will run, and it is the block that applies the four
    live-mind channels and the three sampling biases. The frames ride in under
    the keys the lane reads them from.
    """
    context: dict[str, Any] = {
        "desktop_quick_reply_contract": True,
        "origin": PROBE_ORIGIN,
        "max_tokens": int(max_tokens),
    }
    for key in ("spiking_active_inference", "imagination_workspace", "bicameral_advisory"):
        if key in frames:
            context[key] = frames[key]
    return context


async def run_probe_turn(prompt: str, *, timeout_s: float = 90.0, max_tokens: int = 96) -> str:
    """One turn down the lane the channels are applied on. Returns its text.

    Raises :class:`TurnProbeUnavailableError` when there is no engine to run it, and
    when the turn produces nothing — an arm that generated no text has not been
    measured, and recording it as a zero divergence is what made three weeks of
    this campaign worthless.
    """
    from core.brain.types import ThinkingMode
    from core.container import ServiceContainer

    engine = ServiceContainer.get("cognitive_engine", default=None)
    if engine is None or not hasattr(engine, "think"):
        raise TurnProbeUnavailableError("cognitive_engine_not_registered")

    frames = advisory_frames(prompt)
    thought = await engine.think(
        prompt,
        context=probe_context(frames, max_tokens=max_tokens),
        mode=ThinkingMode.FAST,
        origin=PROBE_ORIGIN,
        timeout_s=timeout_s,
    )
    text = str(getattr(thought, "content", "") or "")
    if not text.strip():
        raise TurnProbeUnavailableError("the probe turn produced no text")
    return text
