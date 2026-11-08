"""Asked to watch something else while something plays, what plays is stopped first, and the request waits for it.

LIVE 8 November 2026, dress rehearsal: "Now asdfmovie. Pick whichever episode you want." came while a 44-minute video
was still playing; her remarks on it and her browsing for the next were one model's work, and the browsing failed.
"""

import asyncio

from core.skills import watching_together


def test_what_plays_is_stopped_and_the_request_waits_for_it(monkeypatch):
    monkeypatch.setattr(watching_together, "_PLAYING", {"person:bryan": {"title": "We Spent 8 Months 3D Printing"}})
    monkeypatch.setattr(watching_together, "_STOP_ASKED", {})

    async def go():
        async def the_watching_ends():
            await asyncio.sleep(0.2)
            assert watching_together._STOP_ASKED == {"person:bryan": "they asked to watch something else"}
            watching_together._PLAYING.clear()

        ending = asyncio.ensure_future(the_watching_ends())
        stopped = await watching_together.stop_for_the_next("Now asdfmovie. Pick whichever episode you want.")
        await ending
        return stopped

    assert asyncio.run(go()) == "We Spent 8 Months 3D Printing"
    assert watching_together._PLAYING == {}


def test_with_nothing_playing_nothing_is_stopped(monkeypatch):
    monkeypatch.setattr(watching_together, "_PLAYING", {})
    monkeypatch.setattr(watching_together, "_STOP_ASKED", {})
    assert asyncio.run(watching_together.stop_for_the_next("Now Pingu.")) == ""
    assert watching_together._STOP_ASKED == {}
