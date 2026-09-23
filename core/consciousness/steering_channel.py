"""Her state, from the substrate in this process to the steering hooks in the worker.

The affective steering hooks sit in the MLX worker's forward pass, and her
substrate lives here, in the parent. They meet in a shared array of sixteen
doubles the client allocates before the worker is started. The worker's sync
thread reads it twenty times a second as her substrate vector; the last slot is
the worker's own liveness flag.

Until 23 September the path was broken at both ends. Nothing in the parent
wrote the fifteen state slots, and the sync thread had no way to read a shared
array at all, so every tick it found no state and fell back to "neutral" moods
of 0.0. A hook reads each steered dimension as an activation from 0 to 1
centred on 0.5, so every attached worker steered towards low valence, arousal,
frustration, curiosity and energy, a weight of -0.76 on each, the same whatever
she felt, and its governor never ran, so alpha never followed her arousal. The
fusion certificate that opened the channel on user-facing turns was measured by
handing states to the hooks directly, so it certified a path live traffic did
not take.

`publish` writes her substrate into the array, on the scale the hooks read.
Valence and arousal are signed in the substrate and rest at 0 (valence is
clipped to -1..1 and arousal read out as (x + 1) / 2), so each is mapped from
-1..1 onto 0..1. Frustration, curiosity, energy and focus are declared as
activations from 0 to 1 and are only clipped. Every other neuron is carried
as it is; no hook reads it as an activation.

The client publishes before each generation, so a generation is steered by the
state she was in when it was asked for; the backward arrow drains at the same
point. Until the first publish the array reads 0.5, the midpoint of every
activation, which stands steering down rather than steering somewhere.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.consciousness.mood_weight import NEUTRAL_MOOD

logger = logging.getLogger("Aura.SteeringChannel")

__all__ = [
    "CHANNEL_LENGTH",
    "activation_state",
    "create_channel",
    "governor_inputs",
    "her_substrate",
    "publish",
    "steering_now",
]

#: Fifteen state slots and the worker's liveness flag.
CHANNEL_LENGTH = 16

#: Where arousal sits in the published state. The substrate's idx_arousal and the
#: steering library's arousal dimension both say 1; a test holds the two together.
AROUSAL_SLOT = 1

#: The service names the worker's sync thread looked her substrate up by when
#: it ran in-process, in its order.
_SUBSTRATE_SERVICES = ("liquid_substrate", "conscious_substrate", "liquid_state")


def create_channel(mp_context: Any) -> Any:
    """The shared array, reading neutral. Called in the parent, before the worker starts."""
    channel = mp_context.Array("d", CHANNEL_LENGTH, lock=False)
    for index in range(CHANNEL_LENGTH - 1):
        channel[index] = NEUTRAL_MOOD
    return channel


def activation_state(substrate: Any) -> np.ndarray | None:
    """The substrate's state with each steered dimension as an activation from 0 to 1.

    Read through the substrate's non-blocking snapshot, which the event loop may
    call. None when there is no substrate or no readable state.
    """
    if substrate is None:
        return None
    try:
        raw = substrate._state_snapshot_nowait()["x"]
        x = np.array(raw, dtype=np.float64).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.debug("No readable substrate state to steer with: %s", exc)
        return None
    if x.size == 0 or not np.isfinite(x).all():
        return None
    state = x.copy()
    for name in ("idx_valence", "idx_arousal"):
        index = getattr(substrate, name, None)
        if isinstance(index, int) and 0 <= index < x.size:
            state[index] = np.clip((x[index] + 1.0) / 2.0, 0.0, 1.0)
    for name in ("idx_frustration", "idx_curiosity", "idx_energy", "idx_focus"):
        index = getattr(substrate, name, None)
        if isinstance(index, int) and 0 <= index < x.size:
            state[index] = np.clip(x[index], 0.0, 1.0)
    return state


def her_substrate() -> Any:
    """Her substrate as the runtime registered it, or None."""
    try:
        from core.runtime.service_registry import get_runtime_service
    except ImportError:  # pragma: no cover - shipped together
        return None
    for name in _SUBSTRATE_SERVICES:
        substrate = get_runtime_service(name, default=None)
        if substrate is not None and hasattr(substrate, "_state_snapshot_nowait"):
            return substrate
    return None


def publish(channel: Any, substrate: Any = None) -> np.ndarray | None:
    """Write her state into the channel's state slots. Returns what was written, or None.

    With no substrate the channel keeps what it held, which is neutral until
    the first publish. The liveness slot is the worker's and is never written.
    """
    if channel is None:
        return None
    state = activation_state(substrate if substrate is not None else her_substrate())
    if state is None:
        return None
    count = min(len(channel) - 1, state.size)
    for index in range(count):
        channel[index] = float(state[index])
    return state[:count]


def steering_now() -> list[float] | None:
    """What a publish would write now: the state a generation asked for now is steered by."""
    state = activation_state(her_substrate())
    if state is None:
        return None
    return [float(value) for value in state[: CHANNEL_LENGTH - 1]]


def governor_inputs(substrate_x: Any, moods: dict[str, float] | None) -> tuple[float, float]:
    """Arousal and coherence for the steering governor, which sets alpha from them.

    In the worker the governor read both from a neurochemical system that exists
    only in the parent, found an empty mood vector, and took arousal as 0.0. Its
    sigmoid gave 0.0067 of the base alpha of 0.2, so every hook ran at 0.0013 of
    the stream, well under the 0.1 the fusion certificate found the least that
    moves the model's output. Arousal is read from the state the hooks steer by,
    where the published vector carries it as an activation, so the governor and
    the arousal hook read one number in either process. No organ publishes a
    coherence in either process (the mood vector has no such key), so it stays
    at the 1.0 the governor has always been given.
    """
    moods = moods or {}
    arousal = float(moods.get("arousal", 0.0))
    if substrate_x is not None and len(substrate_x) > AROUSAL_SLOT:
        arousal = float(substrate_x[AROUSAL_SLOT])
    return arousal, float(moods.get("coherence", 1.0))
