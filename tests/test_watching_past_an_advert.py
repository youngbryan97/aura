"""Watching past an advert: the video's length is the one its page states, and a skip control on screen is pressed.

LIVE 20 October 2026, the watching rehearsal: the video element held a six-second advert when its length was read,
so an eleven-minute video was taken for six seconds, every moment was stamped 0:06, and the watching ended two
minutes in. Bryan, watching an advert play in his browser: "one thing.... skipping ads".
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.skills.their_player import skip_on_screen as _skip_on_screen
from core.skills.watching_along import WHAT_THE_PAGE_SAYS

WINDOW = {"X": 864.0, "Y": 33.0, "Width": 864.0, "Height": 1022.0}


def test_a_skip_control_in_the_window_is_the_one_pressed():
    elements = [
        ("AXLink", "Skip navigation", (880.0, -877.0), (128.0, 40.0)),  # above the page, for keyboards
        ("AXButton", "Learn more", (900.0, 400.0), (120.0, 40.0)),
        ("AXButton", "Skip", (1600.0, 420.0), (100.0, 40.0)),
    ]
    assert _skip_on_screen(elements, WINDOW) == 2


def test_a_skip_control_off_screen_or_in_words_only_is_not():
    assert _skip_on_screen([("AXLink", "Skip navigation", (880.0, -877.0), (128.0, 40.0))], WINDOW) is None
    assert _skip_on_screen([("AXStaticText", "Skip", (1600.0, 420.0), (100.0, 40.0))], WINDOW) is None
    assert _skip_on_screen([("AXButton", "Don't skip this one", (1600.0, 420.0), (100.0, 40.0))], WINDOW) is None
    assert _skip_on_screen([("AXButton", "Skip", (100.0, 420.0), (100.0, 40.0))], WINDOW) is None  # another window


def _page(advert_s: float) -> str:
    stated = {"@context": "https://schema.org", "@type": "VideoObject",
              "name": "I built a glove that Explodes whatever it Punches", "duration": "PT11M13S"}
    return (f'<script type="application/ld+json">{json.dumps(stated)}</script>'
            f'<main><video width="640" height="360" data-advert="{advert_s}"></video><p>an advert plays first</p></main>')


def test_the_length_is_the_one_the_page_states_not_the_adverts():
    playwright = pytest.importorskip("playwright.async_api")

    async def look() -> dict:
        async with playwright.async_playwright() as running:
            try:
                browser = await running.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no browser here is a skip, not a failure
                pytest.skip(f"no headless browser here: {why}")
            try:
                page = await browser.new_page()
                await page.set_content(_page(6.0))
                return await page.evaluate(WHAT_THE_PAGE_SAYS)
            finally:
                await browser.close()

    said = asyncio.run(look())
    assert said["length"] == 673
    assert said["name"] == "I built a glove that Explodes whatever it Punches"
    assert "an advert plays first" in said["words"]


def test_a_remark_is_commentary_the_person_asked_for_not_a_second_answer(monkeypatch):
    """LIVE 2026-10-20: "EventBridge: withheld a second answer to a turn the route already answered: 'Okay, we're in'"."""
    from core.conversation import surface_delivery
    from core.skills import watching_along

    sent: list[dict] = []

    class _Orchestrator:
        async def emit_spontaneous_message(self, words, **kwargs):
            sent.append(kwargs)
            return {"ok": True}

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        staticmethod(lambda name, default=None: _Orchestrator() if name == "orchestrator" else default))
    assert asyncio.run(watching_along.say_it("He built the swing set out of a bed frame."))
    metadata = sent[0]["metadata"]
    assert metadata.get("narration") is True
    # What the bridge asks of it while the turn that asked for the watching is still open.
    assert surface_delivery.route_answer_supersedes("He built the swing set out of a bed frame.", unprompted=True,
                                                    answering=not bool(metadata.get("narration"))) is False


def _on_their_screen(monkeypatch, readings):
    from core.skills import watching_the_screen as screen
    from core.skills.their_player import PlayerReading

    window = {"kCGWindowNumber": 7, "kCGWindowName": "Voids - YouTube", "kCGWindowOwnerPID": 1}
    monkeypatch.setattr(screen, "_their_browser", lambda _url: ("com.apple.Safari", "Safari"))
    monkeypatch.setattr(screen, "_window_showing", lambda _owner, _title: window)
    queue = iter(readings)
    monkeypatch.setattr(screen, "read_their_player", lambda _window: PlayerReading(**next(queue), window_found=True))
    watched = screen.OnTheirScreen("https://www.youtube.com/watch?v=x", "Voids", 554.0)
    watched.began = 0.0
    return watched


def test_their_players_own_place_is_the_clock_and_an_advert_does_not_move_it(monkeypatch):
    """LIVE 2026-10-20, rehearsal 7: "Twenty seconds left" at 8:54 by the time since the window opened, with three and
    a half minutes of the video still to play after the adverts before it."""
    skip = object()
    watched = _on_their_screen(monkeypatch, [
        {"at": 4.0, "length": 15.0},                 # an advert: its length is not the video's
        {"at": 9.0, "length": 15.0, "skip": skip, "skip_words": "Skip"},
        {"at": 120.0, "length": 553.0},             # the video, as its player gives it
        {"at": 553.0, "length": 553.0},
    ])
    first, second, third, last = (asyncio.run(watched.state()) for _ in range(4))
    assert first["advert"] and second["advert"] and first["at"] == 0.0
    assert not third.get("advert") and third["at"] == 120.0 and not third["ended"]
    assert last["ended"]


def test_the_skip_their_player_offers_is_pressed(monkeypatch):
    from core.skills import watching_the_screen as screen

    pressed = []
    monkeypatch.setattr(screen, "press", lambda element: pressed.append(element) or True)
    skip = object()
    watched = _on_their_screen(monkeypatch, [{"at": 6.0, "length": 15.0, "skip": skip, "skip_words": "Skip"}])
    assert asyncio.run(watched.state())["advert"]
    assert asyncio.run(watched.tend()) == "Skip" and pressed == [skip]
    assert asyncio.run(watched.tend()) == ""  # once


def test_nothing_is_said_while_an_advert_plays_and_its_time_is_not_the_watchings(monkeypatch):
    import core.skills.watching_along as watching_along

    monkeypatch.setattr(watching_along, "AN_END_HOLDS_S", 0.0)
    monkeypatch.setattr(watching_along, "ADVERT_LOOK_S", 0.0)

    class _Player:
        def __init__(self):
            self.states, self.tended, self.at = 0, 0, 0.0

        async def says(self):
            return {"title": "Voids"}

        async def start(self):
            return None

        async def state(self):
            self.states += 1
            if self.states <= 4:
                return {"advert": True, "at": 0.0, "duration": 40.0, "ended": False}
            self.at += 10.0
            return {"at": self.at, "duration": 40.0, "ended": self.at > 40.0}

        async def tend(self):
            self.tended += 1
            return ""

        async def heard(self):
            return [{"words": "a pillow for side sleepers" if self.states <= 4 else f"the void at {int(self.at)}"}]

        def eyes(self):
            async def look(_when):
                return f"darkness at {int(self.at)}"
            return look

        async def done(self):
            return None

    asked, kept = [], []

    async def turn(_request, observation, _said):
        asked.append(observation)
        return "The void is almost empty, which is stranger than it sounds."

    player = _Player()
    asyncio.run(watching_along.watch_along(player, "watch this with me", turn=turn, say=lambda _w: asyncio.sleep(0, True),
                                           perceive=lambda *_: None, before=False, keep=kept.append,
                                           remember=lambda _w: asyncio.sleep(0), look_every_s=0.0, quiet_at_least_s=0.0))
    assert [row["kind"] for row in kept[1:5]] == ["advert"] * 4 and player.tended >= 4  # after the look before it plays
    assert not any("pillow" in str(row.get("read") or "") for row in kept if row.get("kind") == "moment")
    assert all("pillow" not in observation for observation in asked)


def test_what_plays_on_after_the_watching_is_paused(monkeypatch):
    """LIVE 2026-10-20, rehearsal 7: the site played its next video when hers ended, and her ear went on hearing it."""
    from core.skills import watching_the_screen as screen

    pressed = []
    monkeypatch.setattr(screen, "press", lambda element: pressed.append(element) or True)
    pause = object()
    watched = _on_their_screen(monkeypatch, [{"at": 3.0, "length": 400.0, "pause": pause}])
    watched.window = {"kCGWindowNumber": 7, "kCGWindowName": "Voids - YouTube", "kCGWindowOwnerPID": 1}
    asyncio.run(watched.done())
    assert pressed == [pause]


def test_while_it_plays_the_clock_runs_from_the_place_its_player_last_stated(monkeypatch):
    """LIVE 2026-10-20, YouTube in Safari: with its controls hidden the seek slider stayed at 0:02 for twenty seconds
    while the video played, offering Pause all along."""
    from core.skills import watching_the_screen as screen
    from core.skills.their_player import PlayerReading

    playing = object()
    watched = _on_their_screen(monkeypatch, [])
    moments = iter([100.0, 110.0, 120.0, 125.0, 140.0])
    monkeypatch.setattr(screen, "_now", lambda: next(moments))
    places = []
    for pause in (playing, playing, playing, None, None):
        watched.at = watched._place(PlayerReading(at=2.0, length=553.0, pause=pause), 553.0)
        places.append(watched.at)
    assert places == [2.0, 12.0, 22.0, 22.0, 22.0]  # playing, then paused where it was
