"""Which frames of a roll a measurement reads.

A turn reads K after every phase, thirty-three times, and reading costs about
two fifths of the turn (profile of 27 October at 000cb4a8a: 2.78 s of 6.4 s over
sixteen turns). A carrier roll uses the frames at its two horizons and a
content presentation uses the end of one turn, so most of that reading went
nowhere. Under a plan the organism is stepped exactly as before and only the
named frames are read; the rest are `UnreadFrame`, which refuses to be read.

Frames are counted from the last restore, which is where every roll starts.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any


def lag_frames(lags: Iterable[int], frames_per_turn: int, turns: int) -> frozenset[int]:
    """The frames `lag_vector` reads at these lags, and the roll's last frame."""
    last = max(0, int(frames_per_turn) * int(turns) - 1)
    return frozenset({max(0, int(lag) - 1) for lag in lags} | {last})


@contextmanager
def reading_only(runtime: Any, frames: Iterable[int] | None) -> Iterator[None]:
    """Read only these frames since the last restore while inside; None reads all."""
    before = getattr(runtime, "read_plan", None)
    runtime.read_plan = None if frames is None else frozenset(int(index) for index in frames)
    try:
        yield
    finally:
        runtime.read_plan = before
