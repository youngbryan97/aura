"""Watched with a person where they watch: opened in their browser, its window looked at, its sound heard by her ear,
and over when the window stops showing it. Their browser, her eyes, her ear and her turns are stand-ins here."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ASKED = "Let's watch something together. Think out loud while we watch."


class _Ear:
    def __init__(self, bundle):
        self.bundle, self.listening, self.why_not = bundle, True, ""
        self.lines = [[{"at": 1.0, "words": "volcanoes reshape continents"}], [], [{"at": 9.0, "words": "the ocean floor is young"}]]

    async def start(self):
        return True

    def since(self):
        return self.lines.pop(0) if self.lines else []

    async def stop(self):
        self.stopped = True


def _screen(monkeypatch, showing):
    import core.senses.her_ear as her_ear
    import core.skills.watching_the_screen as screen

    monkeypatch.setattr(screen, "_their_browser", lambda _url: ("com.example.Browser", "Browser"))
    monkeypatch.setattr(screen, "_window_showing", lambda _owner, _title: next(showing))
    monkeypatch.setattr(her_ear, "HerEar", _Ear)

    class _Gateway:
        async def spawn_async(self, argv, **_kw):
            _Gateway.opened = argv
            return SimpleNamespace(wait=lambda: asyncio.sleep(0))

    monkeypatch.setattr("core.runtime.subprocess_gateway.get_subprocess_gateway", lambda: _Gateway())
    return screen, _Gateway


def test_it_is_opened_in_their_browser_heard_and_over_when_its_window_shows_something_else(monkeypatch):
    import core.skills.watching_along as watching_along
    from core.skills.watching_along import watch_along

    monkeypatch.setattr(watching_along, "AN_END_HOLDS_S", 0.0)

    window = {"kCGWindowNumber": 7, "kCGWindowName": "Why Volcanoes - Videos - Browser"}
    showing = iter([window, window, window, window, None, None])
    screen, gateway = _screen(monkeypatch, showing)
    said = []

    async def go():
        where = await screen.on_their_screen("https://videos.example/v/1", "Why Volcanoes - Videos", 300.0)
        assert where is not None and gateway.opened == ["/usr/bin/open", "-b", "com.example.Browser", "https://videos.example/v/1"]

        async def look(_when):
            return "a volcano erupting"

        async def turn(_request, observation, _said):
            said.append(observation)
            return f"remark {len(said)}"

        async def say(_words):
            return True

        return await watch_along(where, ASKED, look=look, turn=turn, say=say, perceive=lambda *_: None,
                                 remember=lambda _w: asyncio.sleep(0), look_every_s=0.0, quiet_at_least_s=0.0)

    watching = asyncio.run(go())
    assert watching.ended, "the window stopped showing it"
    heard = " ".join(m.read for m in watching.moments)
    assert "volcanoes reshape continents" in heard and "the ocean floor is young" in heard
    assert said[0].startswith("before it plays") and said[-1].startswith("it has ended")


def test_where_it_cannot_be_shown_on_their_screen_it_is_not(monkeypatch):
    screen, _gateway = _screen(monkeypatch, iter([None] * 100))
    monkeypatch.setattr(screen, "WINDOW_WAITED_S", 0.0)
    assert asyncio.run(screen.on_their_screen("https://videos.example/v/1", "Why Volcanoes", 300.0)) is None
    monkeypatch.setattr(screen, "_their_browser", lambda _url: ("", ""))
    assert asyncio.run(screen.on_their_screen("https://videos.example/v/1", "Why Volcanoes", 300.0)) is None


def test_a_window_is_the_one_whose_title_shares_most_of_what_was_opened(monkeypatch):
    import core.skills.watching_the_screen as screen

    windows = [{"kCGWindowOwnerName": "Browser", "kCGWindowLayer": 0, "kCGWindowName": "Inbox - Mail - Browser"},
               {"kCGWindowOwnerName": "Browser", "kCGWindowLayer": 0, "kCGWindowName": "Why Volcanoes Erupt - Videos - Browser"},
               {"kCGWindowOwnerName": "Other", "kCGWindowLayer": 0, "kCGWindowName": "Why Volcanoes Erupt - Videos"}]
    monkeypatch.setitem(__import__("sys").modules, "Quartz", SimpleNamespace(
        kCGWindowListOptionOnScreenOnly=1, kCGNullWindowID=0, CGWindowListCopyWindowInfo=lambda *_: windows))
    assert screen._window_showing("Browser", "Why Volcanoes Erupt - Videos")["kCGWindowName"].startswith("Why Volcanoes")
    assert screen._window_showing("Browser", "Something Else Entirely") is None


def test_it_opens_on_their_screen_under_its_own_receipt_whatever_step_began_it(monkeypatch):
    """LIVE 20 October 2026: the watching, begun from her browser's step, carried that step's network-call governance,
    under which nothing may be started on their desktop, and the film was never opened."""
    import core.skills.watching_the_screen as screen
    from core.governance_context import get_active_governance, governed_scope_sync

    window = {"kCGWindowNumber": 7, "kCGWindowName": "Why Volcanoes - Videos - Browser"}
    _screen_module, gateway = _screen(monkeypatch, iter([window] * 4))
    domains = []

    class _Gateway:
        async def spawn_async(self, argv, **_kw):
            token = get_active_governance()
            domains.append(token.domain if token else None)
            return SimpleNamespace(wait=lambda: asyncio.sleep(0))

    monkeypatch.setattr("core.runtime.subprocess_gateway.get_subprocess_gateway", lambda: _Gateway())

    async def go():
        with governed_scope_sync(SimpleNamespace(receipt_id="will-browse-1", domain="network_call", source="sovereign_browser")):
            return await screen.on_their_screen("https://videos.example/v/1", "Why Volcanoes - Videos", 300.0)

    assert asyncio.run(go()) is not None
    assert domains == ["environment_action"]


def test_her_eyes_are_given_a_picture_they_can_read_in_a_look():
    """LIVE 20 October 2026: the whole window at the screen's resolution was 6,869 image tokens, and every look ran
    out of time."""
    import io

    from PIL import Image

    from core.skills.watching_the_screen import PICTURE_LONGEST, _smaller

    taken = io.BytesIO()
    Image.new("RGB", (2880, 1800), (20, 40, 60)).save(taken, "JPEG")
    assert max(Image.open(io.BytesIO(_smaller(taken.getvalue()))).size) == PICTURE_LONGEST


def test_past_its_length_a_picture_that_has_stopped_changing_is_the_end(monkeypatch):
    """A page keeps its title when its video ends: LIVE 20 October 2026 a six-second film was watched for two minutes."""
    window = {"kCGWindowNumber": 7, "kCGWindowName": "A short film - Browser"}
    screen, _gateway = _screen(monkeypatch, iter([window] * 100))
    where = screen.OnTheirScreen("http://127.0.0.1/film", "A short film", 6.0)
    where.began = screen.time.monotonic() - 10.0
    where.pictures = ["a", "b"]
    assert asyncio.run(where.state())["ended"] is False, "still changing"
    where.pictures = ["b", "b"]
    assert asyncio.run(where.state())["ended"] is True
    where.began = screen.time.monotonic() - 2.0
    assert asyncio.run(where.state())["ended"] is False, "within its length a still picture is a still scene"


def test_her_reply_to_a_request_to_watch_is_her_own_turn_with_what_was_found(monkeypatch):
    """LIVE 20 October 2026 the browsing's receipt went out as her reply, and an empty reply sent the request through
    the browsing again."""
    import core.skills.watching_along as watching_along
    from core.skills.watching_together import her_answer_before_it_plays

    asked = []

    async def her_turn(request, observation, said, **_kw):
        asked.append((request, observation))
        return "Six seconds about colour? I'd guess a slow drift between two hues."

    monkeypatch.setattr(watching_along, "her_turn", her_turn)
    said = asyncio.run(her_answer_before_it_plays("Watch this with me", "A short film about colour (0:06)"))
    assert said.startswith("Six seconds about colour") and "A short film about colour (0:06)" in asked[0][1]

    async def nothing(*_a, **_kw):
        return ""

    monkeypatch.setattr(watching_along, "her_turn", nothing)
    assert asyncio.run(her_answer_before_it_plays("Watch this with me", "A short film")) == "A short film"
