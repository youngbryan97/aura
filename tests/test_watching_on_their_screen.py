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


def _a_window(frame: int, *, thumbnails_loaded: bool) -> bytes:
    """A page 1600x1000: a video at (200, 100)-(1000, 550) whose picture changes every frame, and thumbnails at the
    bottom that appear once."""
    import io

    from PIL import Image, ImageDraw

    page = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((200, 100, 999, 549), fill=((frame * 40) % 256, 30, 90))
    draw.ellipse((200 + frame * 30, 200, 300 + frame * 30, 300), fill="yellow")
    if thumbnails_loaded:
        draw.rectangle((100, 800, 1500, 950), fill="gray")
    out = io.BytesIO()
    page.save(out, format="PNG")
    return out.getvalue()


def test_her_eyes_are_given_where_it_plays_and_not_the_page_around_it():
    import io

    from PIL import Image

    from core.skills.watching_the_screen import _smaller, _WhereItPlays

    where = _WhereItPlays()
    assert where.took(_a_window(0, thumbnails_loaded=False)) is None
    seen = [where.took(_a_window(frame, thumbnails_loaded=frame >= 2)) for frame in range(1, 7)]
    left, top, right, bottom = seen[-1]
    # The video's box, as fractions of the window, within the two pixels of the grid her eyes' picture is measured
    # on that resampling spreads a change over; the thumbnails that loaded once are not in it.
    for found, video in zip((left, top, right, bottom), (200 / 1600, 100 / 1000, 1000 / 1600, 550 / 1000), strict=True):
        assert abs(found - video) <= 2 / 768
    shown = Image.open(io.BytesIO(_smaller(_a_window(6, thumbnails_loaded=True), within=seen[-1])))
    assert abs(shown.size[0] / shown.size[1] - 800 / 450) < 0.02 and max(shown.size) == 768


def test_a_window_that_changes_size_starts_where_it_plays_again():
    import io

    from PIL import Image

    from core.skills.watching_the_screen import _WhereItPlays

    where = _WhereItPlays()
    where.took(_a_window(0, thumbnails_loaded=False))
    assert where.took(_a_window(1, thumbnails_loaded=False)) is not None
    smaller = io.BytesIO()
    Image.open(io.BytesIO(_a_window(2, thumbnails_loaded=False))).resize((1200, 1000)).save(smaller, format="PNG")
    assert where.took(smaller.getvalue()) is None


def test_a_small_part_that_changes_is_a_control_and_not_where_it_plays():
    """LIVE 2026-10-20: on a film whose bars hold still, a button of the player's controls changed most."""
    import io

    from PIL import Image, ImageDraw

    from core.skills.watching_the_screen import _WhereItPlays

    def window(frame: int) -> bytes:
        page = Image.new("RGB", (1600, 1000), "white")
        ImageDraw.Draw(page).rectangle((1400, 900, 1450, 950), fill=(frame * 60 % 256, 0, 0))   # a blinking control
        out = io.BytesIO()
        page.save(out, format="PNG")
        return out.getvalue()

    where = _WhereItPlays()
    assert [where.took(window(frame)) for frame in range(4)] == [None] * 4


@pytest.mark.parametrize(("answer", "seen"), [
    ("Based on the image you provided, here's what is happening:\n\nYou are watching a short film titled “A short film "
     "about colour”.", "A short film titled “A short film about colour”."),
    ("In this frame, a volcano erupts at sea.", "A volcano erupts at sea."),
    ("A colour test pattern fills the window.", "A colour test pattern fills the window."),
    ("", ""),
])
def test_a_look_is_what_was_seen_not_a_reply_to_whoever_sent_the_picture(answer, seen):
    """LIVE 20 October 2026: every look of a watching opened "Based on the image you provided ... You are watching"."""
    from core.perception.her_eyes import plainly

    assert plainly(answer) == seen


def test_on_their_screen_a_video_is_never_said_to_be_past_its_length(monkeypatch):
    """LIVE 20 October 2026: "0:25 of a stated 6 seconds", and she wondered whether it looped. The time since its window
    opened is all that is read there, at most the video's length."""
    import time as clock

    window = {"kCGWindowNumber": 7, "kCGWindowName": "Why Volcanoes - Videos - Browser"}
    screen, _gateway = _screen(monkeypatch, iter([window] * 10))
    where = asyncio.run(screen.on_their_screen("https://videos.example/v/1", "Why Volcanoes - Videos", 6.0))
    where.began = clock.monotonic() - 25.0
    assert asyncio.run(where.state())["at"] == 6.0


def test_a_look_says_what_is_in_the_picture_not_that_it_is_one():
    """LIVE 20 October 2026: "This picture shows a **video player interface** — likely from", four to a remark."""
    from core.perception.her_eyes import plainly

    assert plainly("This image appears to be a starry night of pixels. The dots drift") == "A starry night of pixels."
    assert plainly("This picture shows a volcano at sea.") == "A volcano at sea."


def test_her_eyes_on_their_screen_look_through_the_vision_model_once(monkeypatch):
    """The look itself, run: a name it needed was missing once, and only a live look would have raised it."""
    import core.perception.her_eyes as her_eyes
    import core.skills.watching_the_screen as screen

    window = {"kCGWindowNumber": 7, "kCGWindowName": "Why Volcanoes - Videos - Browser"}
    _screen_module, _gateway = _screen(monkeypatch, iter([window] * 10))
    asked = []

    async def see(prompt, picture, *, max_tokens, timeout_s):
        asked.append(max_tokens)
        return "This picture shows a volcano at sea."

    monkeypatch.setattr(her_eyes, "look_with_her_eyes", see)
    monkeypatch.setattr("pathlib.Path.read_bytes", lambda _self: b"\x89PNG not really")
    monkeypatch.setattr(screen, "_smaller", lambda taken, within=None: taken)
    where = asyncio.run(screen.on_their_screen("https://videos.example/v/1", "Why Volcanoes - Videos", 60.0))
    where.where_it_plays = SimpleNamespace(took=lambda _picture: None)
    seen = asyncio.run(where.eyes()("playing"))
    assert seen == "A volcano at sea." and asked and asked[0] <= 70
