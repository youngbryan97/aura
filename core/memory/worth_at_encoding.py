"""What a turn was worth to her when a memory was made, kept with it, and how near this turn's worth lies to it.

Each turn she reads what it paid on nine channels against what she expected of
each (core/affect/what_it_was_worth.py). The errors drive her chemistry, and her
field and substrate learn by their sum. Her memories kept the feeling a moment
held and where her field stood, and nothing of what it paid, so nothing she had
learned from a turn's payoff could call up a memory.

On the seed 73 content run at 24d1a0191 a nudge towards feeling good moved
worth more than any other column of hers: every percept class read as a
payoff of 3.4 to 5.7, where at rest threats and errors read about -2.5 and
praise +3.9. Her field's input weights, which learn by the sign of that
error, carried the same split and lost it the same way. What she recalled followed
neither: affect against memory -0.65 over pairs of percepts. Reward at
encoding binds a memory to what the moment was worth (Adcock and others,
"Reward-motivated learning", Neuron 50, 2006), and in the context models of
recall the state a memory was made in is part of what cues it again (Talmi,
Lohnas and Daw, 2019); the motivational state is part of that state.

So each memory is stamped with the latest reading of her nine channel errors,
which are in units of each channel's own spread, and a turn's recall is weighed
again once the turn's worth is read, by how near the two readings lie in level
and in shape (`near`). Not measured from what the competing memories share, as
the feeling and field stamps are: what every candidate shares is the level, and
under a displacement the level is what moves.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["WORTH_KEY", "decode", "encode", "near", "neutral_worth_canary", "stamp", "worth_now"]

#: The metadata key a memory's worth travels under.
WORTH_KEY: str = "worth"


def worth_now() -> np.ndarray | None:
    """The latest measured turn's channel errors, in the ledger's channel order, or None."""
    try:
        from core.affect.what_it_was_worth import CHANNELS, get_worth_ledger

        reading = get_worth_ledger().read()
    except (ImportError, AttributeError, RuntimeError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("memory_worth", exc, action="read no worth for this memory or recall")
        return None
    return _measured_vector(reading, CHANNELS)


def _measured_vector(reading: Any, channels: tuple[str, ...]) -> np.ndarray | None:
    """Keep a measured neutral reading; zero error means the turn met its expectation."""
    if not getattr(reading, "measured", False):
        return None
    errors = getattr(reading, "error", None) or {}
    if not any(name in errors for name in channels):
        return None
    vector = np.asarray([float(errors.get(name, 0.0) or 0.0) for name in channels], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def neutral_worth_canary() -> bool:
    """A private ledger reaches neutral worth without losing its measured reading."""
    from core.affect.what_it_was_worth import CHANNELS, WorthLedger

    ledger = WorthLedger()
    if _measured_vector(ledger.read(), CHANNELS) is not None:
        return False
    for _ in range(4):
        ledger.note({name: 1.0 for name in CHANNELS})
    vector = _measured_vector(ledger.read(), CHANNELS)
    return bool(vector is not None and np.all(vector == 0.0)
                and np.array_equal(decode(encode(vector)), vector) and near(vector, vector) == 1.0)


def encode(vector: np.ndarray) -> str:
    """The errors as text a memory store keeps: four decimals, comma-separated."""
    return ",".join(f"{float(value):.4f}" for value in np.asarray(vector, dtype=np.float64).reshape(-1))


def decode(text: Any) -> np.ndarray | None:
    """The errors back out of a stamp; anything unreadable reads as no worth."""
    if not isinstance(text, str) or not text:
        return None
    try:
        vector = np.asarray([float(part) for part in text.split(",")], dtype=np.float64)
    except ValueError:
        return None
    if not vector.size or not np.all(np.isfinite(vector)):
        return None
    return vector


def stamp(metadata: dict[str, Any], vector: np.ndarray | None = None) -> dict[str, Any]:
    """Keep what the latest turn was worth with the memory being written.

    A stamp already present is left alone, as the feeling and field stamps are.
    """
    if not isinstance(metadata, dict) or metadata.get(WORTH_KEY):
        return metadata
    vector = worth_now() if vector is None else vector
    if vector is not None:
        metadata[WORTH_KEY] = encode(vector)
    return metadata


def near(now: np.ndarray | None, then: np.ndarray | None) -> float:
    """How near this turn's worth lies to the worth a memory was made at, from 0 to 1.

    One at the same reading. Each error is in units of its channel's spread,
    so two unrelated readings lie a squared distance of twice the channel count
    apart on average, and that is where nearness has fallen to 1/e.
    """
    if now is None or then is None or now.size != then.size or not now.size:
        return 0.0
    squared = float(np.sum((now - then) ** 2))
    return math.exp(-squared / (2.0 * now.size))
