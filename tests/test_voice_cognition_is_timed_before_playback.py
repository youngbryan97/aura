"""Speech playback and a superseding turn cannot change a cognition measurement."""
import asyncio
import time
from types import SimpleNamespace

import pytest

from core.voice.duplex.session import DuplexVoiceSession, TurnMetrics


@pytest.mark.asyncio
async def test_finished_cognition_is_measured_without_waiting_for_audio_or_touching_the_next_turn():
    session = object.__new__(DuplexVoiceSession)
    original = session._metrics = TurnMetrics()
    release = asyncio.Event()

    async def cognition():
        await release.wait()
        return 'A complete answer.'

    task = asyncio.create_task(cognition())
    session._measure_cognition(SimpleNamespace(task=task), time.perf_counter() - .1)
    next_turn = session._metrics = TurnMetrics()
    release.set()
    await task
    await asyncio.sleep(0)
    assert original.cognition_ms >= 100.
    assert original.reply_ready_at > 0
    assert original.first_audio_at == 0
    assert next_turn.cognition_ms == next_turn.reply_ready_at == 0
