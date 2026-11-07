"""Looking up a play request on this machine never holds her server.

LIVE 7 November 2026: "Find us a cooking video on YouTube ... let's watch it together" was looked up in the local media
library on the event loop, the walk of her media roots never returned, and every request to her server waited behind
it until her app was quit.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytestmark = pytest.mark.unit


def test_a_lookup_that_does_not_come_back_is_not_waited_on(monkeypatch):
    import core.media.playback as playback
    import interface.routes.chat_preflight as preflight

    release = threading.Event()

    def stuck(_message):
        release.wait(5.0)
        return None

    monkeypatch.setattr(playback, "resolve_play_request", stuck)
    monkeypatch.setattr(preflight, "MEDIA_LOOKUP_WAIT_S", 0.2)
    monkeypatch.setattr(preflight, "_MEDIA_LOOKUP_IN_FLIGHT", {})

    async def go():
        began = time.monotonic()
        first = await preflight._resolve_play_request_off_the_loop("let's watch a cooking video")
        waited = time.monotonic() - began
        # The loop is free while the lookup is stuck: something else runs beside it.
        ticks = 0
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks += 1
        began = time.monotonic()
        second = await preflight._resolve_play_request_off_the_loop("play something")
        return first, waited, ticks, second, time.monotonic() - began

    first, waited, ticks, second, second_waited = asyncio.run(go())
    release.set()
    assert first is None and waited < 1.0 and ticks == 3
    assert second is None and second_waited < 0.1  # no second lookup beside the stuck one


def test_a_lookup_that_comes_back_is_used(monkeypatch):
    import core.media.playback as playback
    import interface.routes.chat_preflight as preflight

    monkeypatch.setattr(playback, "resolve_play_request", lambda message: f"found for {message}")
    monkeypatch.setattr(preflight, "_MEDIA_LOOKUP_IN_FLIGHT", {})
    assert asyncio.run(preflight._resolve_play_request_off_the_loop("play x")) == "found for play x"
