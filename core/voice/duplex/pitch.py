"""core/voice/duplex/pitch.py — Arriving at a note from underneath.

The synthesisers this lane runs take text, a voice and a speed. None of them
takes a pitch, so everything a singer does with pitch on purpose — starting a
line under the note and sliding up to it, the deviation that carries what the
exact value cannot — was out of reach however much of her state reached the
voice.

This is pitch control applied to the samples after synthesis, as a contour
rather than a single shift. The onset starts `cents` below where the voice
settles and glides up to it over `glide_ms` of what is heard, linearly in
cents, which is the unit pitch is heard in.

It is a time-varying resampling. Reading the source more slowly lowers the
pitch, so the k-th output sample reads the input at a position that advances by

    ratio(k) = 2 ** (-bend(k) / 1200)

where bend(k) falls from `cents` to zero across the glide and is zero after,
so everything past the onset is the synthesis exactly as it was. The contour is
exact and continuous and the cost is one interpolation over the onset, which
matters on a lane measured by time to first audio. Two consequences are stated
rather than hidden: the onset is lengthened by the source it had not yet read
when the glide ended — a few milliseconds for a bend of a semitone over a
tenth of a second — and formants move with the pitch during the glide, which is
below what a listener hears as a change of timbre at the depths these records
use.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["bend_into", "bend_length_added"]


def _onset_positions(rate: int, cents: float, glide_ms: float) -> np.ndarray | None:
    """Where each output sample of the bent onset reads the source.

    The glide is measured in output samples, because that is the time a
    listener hears the voice arrive over. Reading continues until the source
    has advanced by a whole glide, so the onset keeps every sample it had.
    """
    glide = int(round(rate * glide_ms / 1000.0))
    if glide < 2:
        return None
    slowest = 2.0 ** (-cents / 1200.0)
    # Enough output samples to read a glide of source at the slowest ratio.
    longest = int(math.ceil(glide / slowest)) + 2
    k = np.arange(longest, dtype=np.float64)
    bend = cents * np.clip(1.0 - k / glide, 0.0, 1.0)
    steps = np.power(2.0, -bend / 1200.0)
    positions = np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    return positions[positions < glide]


def bend_length_added(rate: int, cents: float, glide_ms: float) -> int:
    """How many samples the onset is lengthened by, for a contour this size."""
    try:
        depth = float(cents)
        length_ms = float(glide_ms)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0
    if depth <= 0.0 or length_ms <= 0.0 or rate <= 0:
        return 0
    positions = _onset_positions(rate, depth, length_ms)
    if positions is None:
        return 0
    return int(len(positions) - int(round(rate * length_ms / 1000.0)))


def bend_into(samples: np.ndarray, rate: int, cents: float, glide_ms: float) -> np.ndarray:
    """Start `cents` below the note and arrive at it over `glide_ms`.

    Returns the samples untouched when there is nothing to bend: no depth, no
    glide, audio that is not one channel, or a clip no longer than the glide
    it would be bent across.
    """
    audio = np.asarray(samples, dtype=np.float32)
    try:
        depth = float(cents)
        length_ms = float(glide_ms)
    except (TypeError, ValueError):
        return audio
    if not (depth > 0.0 and length_ms > 0.0 and rate > 0) or audio.ndim != 1:
        return audio
    glide = int(round(rate * length_ms / 1000.0))
    if glide < 2 or audio.size <= glide:
        return audio
    positions = _onset_positions(rate, depth, length_ms)
    if positions is None or positions.size == 0:
        return audio
    source = np.arange(glide + 1, dtype=np.float64)
    bent_onset = np.interp(positions, source, audio[: glide + 1].astype(np.float64))
    return np.concatenate((bent_onset.astype(np.float32), audio[glide:]))
