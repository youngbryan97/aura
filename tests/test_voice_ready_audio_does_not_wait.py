"""A synthesised clause can play while the mind is still forming the next."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.voice.duplex.prosody import ProsodySpec
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts, SynthesisResult


class Synthesizer:
    stream = StreamingTts.stream

    async def synthesize(self, text, spec, token):
        return SynthesisResult(np.ones(24, dtype=np.float32), 24000, text, 1.0, "test")


@pytest.mark.asyncio
async def test_ready_audio_plays_before_another_clause_exists():
    next_clause = asyncio.Event()

    async def clauses():
        yield "First sentence."
        await next_clause.wait()
        yield "Second sentence."

    stream = Synthesizer().stream(clauses(), ProsodySpec(voice="test"), CancellationToken())
    try:
        first = await asyncio.wait_for(anext(stream), timeout=.5)
        assert first.text == "First sentence."
        next_clause.set()
        assert (await asyncio.wait_for(anext(stream), timeout=.5)).text == "Second sentence."
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_closing_playback_cancels_a_blocked_clause_source():
    released = asyncio.Event()

    async def clauses():
        try:
            yield "First sentence."
            await asyncio.Event().wait()
        finally:
            released.set()

    stream = Synthesizer().stream(clauses(), ProsodySpec(voice="test"), CancellationToken())
    await anext(stream)
    await asyncio.sleep(0)
    await asyncio.wait_for(stream.aclose(), timeout=.5)
    assert released.is_set()
