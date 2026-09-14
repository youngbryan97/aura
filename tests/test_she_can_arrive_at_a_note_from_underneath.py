"""Pitch control: an onset that starts under the note and glides up to it.

The synthesisers take text, a voice and a speed, and nothing about pitch. These
pin the contour the voice can now carry, checked on a pure tone where the
answer is known: the pitch at the start is where the bend says, the pitch
after the glide is exactly the synthesised pitch, and nothing past the glide
is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.voice.duplex.pitch import bend_into, bend_length_added

RATE = 24_000
TONE_HZ = 220.0


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return np.sin(2.0 * np.pi * TONE_HZ * t).astype(np.float32)


def _frequency(window: np.ndarray) -> float:
    """A pure tone's frequency from the times of its rising zero crossings.

    Each crossing is placed between samples by linear interpolation. Counting
    crossings instead resolves one crossing in a 30 ms window, which is 16 Hz
    and coarser than the 12 Hz a semitone moves a 220 Hz tone.
    """
    x = np.asarray(window, dtype=np.float64)
    rising = np.flatnonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))
    if rising.size < 2:
        return 0.0
    times = rising + x[rising] / (x[rising] - x[rising + 1])
    return (rising.size - 1) / ((times[-1] - times[0]) / RATE)


def test_nothing_to_bend_leaves_the_samples_as_synthesised() -> None:
    tone = _tone(0.5)
    assert np.array_equal(bend_into(tone, RATE, 0.0, 100.0), tone)
    assert np.array_equal(bend_into(tone, RATE, 100.0, 0.0), tone)
    assert np.array_equal(bend_into(tone[:100], RATE, 100.0, 100.0), tone[:100])


def test_the_onset_starts_below_the_note_by_the_bend() -> None:
    bent = bend_into(_tone(1.0), RATE, 100.0, 200.0)
    start = _frequency(bent[: int(RATE * 0.03)])
    expected = TONE_HZ * 2.0 ** (-100.0 / 1200.0)
    assert start == pytest.approx(expected, rel=0.03)
    assert start < TONE_HZ


def test_after_the_glide_the_pitch_is_exactly_what_was_synthesised() -> None:
    tone = _tone(1.0)
    bent = bend_into(tone, RATE, 100.0, 200.0)
    glide = int(RATE * 0.2)
    added = len(bent) - len(tone)
    assert np.array_equal(bent[glide + added :], tone[glide:])
    assert _frequency(bent[glide + added : glide + added + RATE // 4]) == pytest.approx(TONE_HZ, rel=0.01)


def test_the_onset_is_lengthened_by_the_slowdown_and_by_no_more() -> None:
    tone = _tone(1.0)
    bent = bend_into(tone, RATE, 100.0, 100.0)
    assert len(bent) - len(tone) == pytest.approx(bend_length_added(RATE, 100.0, 100.0), abs=2)
    # A semitone over a tenth of a second adds a few milliseconds.
    assert (len(bent) - len(tone)) / RATE * 1000.0 < 5.0


def test_a_deeper_bend_starts_lower() -> None:
    shallow = _frequency(bend_into(_tone(1.0), RATE, 50.0, 200.0)[: int(RATE * 0.03)])
    deep = _frequency(bend_into(_tone(1.0), RATE, 200.0, 200.0)[: int(RATE * 0.03)])
    assert deep < shallow < TONE_HZ


def test_unusable_arguments_leave_the_samples_alone() -> None:
    tone = _tone(0.3)
    assert np.array_equal(bend_into(tone, RATE, "deep", 100.0), tone)  # type: ignore[arg-type]
    assert np.array_equal(bend_into(np.stack([tone, tone]), RATE, 100.0, 50.0), np.stack([tone, tone]))
