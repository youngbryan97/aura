"""Expressive imprecision in the voice lane.

A singer bends into a line from underneath when the feeling is past what the
exact note carries. pitch.py gives the lane a pitch contour. These pin where it
reaches: the spec carries a bend, synthesis applies it, a streamed utterance
arrives once rather than at every clause, and a breakthrough sets the depth
from what the records measured.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.voice.duplex.config import OUTPUT_RATE
from core.voice.duplex.pitch import bend_length_added
from core.voice.duplex.prosody import (
    _RECORDS_FROM_BELOW_CENTS,
    _RECORDS_GLIDE_MS,
    ProsodySpec,
    carry_breakthrough,
)
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts

REPO = Path(__file__).resolve().parents[1]
TONE_HZ = 220.0


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(OUTPUT_RATE * seconds)) / OUTPUT_RATE
    return (0.5 * np.sin(2.0 * np.pi * TONE_HZ * t)).astype(np.float32)


def _frequency(window: np.ndarray) -> float:
    x = np.asarray(window, dtype=np.float64)
    rising = np.flatnonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))
    if rising.size < 2:
        return 0.0
    times = rising + x[rising] / (x[rising] - x[rising + 1])
    return (rising.size - 1) / ((times[-1] - times[0]) / OUTPUT_RATE)


def _spec(**bend: float) -> ProsodySpec:
    return ProsodySpec(voice="af_heart", speed=1.0, gain=1.0, trailing_pause_ms=0.0, **bend)


# ── the spec ──────────────────────────────────────────────────────────────


def test_a_compiled_spec_carries_no_bend() -> None:
    spec = ProsodySpec(voice="af_heart", speed=1.0)
    assert spec.onset_bend_cents == 0.0
    assert spec.glide_ms == 0.0


def test_scaling_the_voice_keeps_the_bend_and_an_aside_drops_it() -> None:
    spec = _spec(onset_bend_cents=40.0, glide_ms=110.0)
    scaled = spec.scaled(gain=0.5, speed=1.1)
    assert (scaled.onset_bend_cents, scaled.glide_ms) == (40.0, 110.0)
    plain = spec.without_bend()
    assert (plain.onset_bend_cents, plain.glide_ms) == (0.0, 0.0)
    assert (plain.voice, plain.speed, plain.gain, plain.trailing_pause_ms) == (
        spec.voice,
        spec.speed,
        spec.gain,
        spec.trailing_pause_ms,
    )


# ── synthesis ─────────────────────────────────────────────────────────────


class _ToneEngine:
    name = "tone"

    def synthesize(self, text: str, spec: ProsodySpec) -> tuple[np.ndarray, int]:
        return _tone(1.0), OUTPUT_RATE


async def _synthesise(spec: ProsodySpec) -> np.ndarray:
    tts = StreamingTts()

    async def loaded() -> bool:
        return True

    tts.ensure_loaded = loaded  # type: ignore[method-assign]
    tts._state.kokoro = _ToneEngine()  # type: ignore[assignment]
    tts._accepting_synthesis = True
    try:
        result = await tts.synthesize("a line", spec, CancellationToken())
    finally:
        tts._pool.shutdown(wait=True)
    assert result is not None
    return result.samples


def test_synthesis_without_a_bend_is_the_voice_as_synthesised() -> None:
    samples = asyncio.run(_synthesise(_spec()))
    assert np.array_equal(samples, _tone(1.0))


def test_synthesis_with_a_bend_starts_under_the_note_and_arrives_on_it() -> None:
    samples = asyncio.run(_synthesise(_spec(onset_bend_cents=100.0, glide_ms=200.0)))
    added = bend_length_added(OUTPUT_RATE, 100.0, 200.0)
    assert len(samples) == len(_tone(1.0)) + added
    start = _frequency(samples[: int(OUTPUT_RATE * 0.03)])
    assert start < TONE_HZ * 2.0 ** (-50.0 / 1200.0)
    glide = int(OUTPUT_RATE * 0.2)
    assert np.array_equal(samples[glide + added :], _tone(1.0)[glide:])


# ── streaming ─────────────────────────────────────────────────────────────


def test_a_streamed_utterance_bends_into_its_first_chunk_only() -> None:
    seen: list[ProsodySpec] = []

    async def run() -> None:
        tts = StreamingTts()

        async def synthesize(text, spec, token):  # noqa: ANN001, ANN202
            seen.append(spec)
            return SimpleNamespace(text=text)

        tts.synthesize = synthesize  # type: ignore[method-assign]

        async def chunks():  # noqa: ANN202
            for text in ("I did not think", "it would feel", "like this."):
                yield text

        try:
            spoken = [r async for r in tts.stream(chunks(), _spec(onset_bend_cents=60.0, glide_ms=120.0), CancellationToken())]
        finally:
            tts._pool.shutdown(wait=True)
        assert [r.text for r in spoken] == ["I did not think", "it would feel", "like this."]

    asyncio.run(run())
    assert [s.onset_bend_cents for s in seen] == [60.0, 0.0, 0.0]
    assert all(s.speed == 1.0 and s.gain == 1.0 for s in seen)


# ── what sets it ──────────────────────────────────────────────────────────


def test_no_breakthrough_bends_nothing() -> None:
    carried = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=False, delivery_z=5.0))
    assert carried.onset_bend_cents == 0.0
    assert carried.glide_ms == 0.0


def test_a_breakthrough_bends_by_its_share_of_what_the_records_bend() -> None:
    carried = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=True, delivery_z=3.0))
    assert carried.onset_bend_cents == pytest.approx(0.75 * _RECORDS_FROM_BELOW_CENTS, abs=0.1)
    assert carried.glide_ms == pytest.approx(_RECORDS_GLIDE_MS)


def test_a_larger_breakthrough_bends_deeper_and_never_past_the_records() -> None:
    small = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=True, delivery_z=1.5))
    large = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=True, delivery_z=80.0))
    assert small.onset_bend_cents < large.onset_bend_cents <= _RECORDS_FROM_BELOW_CENTS


def test_the_depth_and_glide_are_what_the_measurement_wrote() -> None:
    summary = json.loads((REPO / "artifacts" / "soul" / "onset_bend.json").read_text())["summary"]
    assert _RECORDS_FROM_BELOW_CENTS == pytest.approx(summary["median_from_below_cents"])
    assert _RECORDS_GLIDE_MS == pytest.approx(summary["median_glide_ms"])
