"""What a page draws is handed to her eyes on the same page, over the part it draws in.

LIVE 2026-10-02: a Flash game on webdesignmuseum.org is a 540 by 360 canvas in
Ruffle's shadow root with its PLAY button painted on it. The browser pursuit has
nothing there to read or press; the screen pursuit plays exactly that kind of
thing. Choosing the drawing hands her own page to the screen pursuit as its
surface: pictures her browser takes of it, keys and clicks her browser delivers
to it, and never the window in front, which that day was the person's Chrome.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

from core.skills import sovereign_browser_drawing as drawing
from core.skills.screen_pursuit_on_a_page import HER_OWN_PAGE, OnAPage, on_her_page

pytestmark = pytest.mark.unit


def test_the_band_is_read_from_the_page_or_refused():
    said = json.dumps({"left": 0.1, "top": 0.3, "right": 0.6, "bottom": 0.8})
    assert drawing._the_band(said) == (0.1, 0.3, 0.6, 0.8)
    assert drawing._the_band("") is None
    assert drawing._the_band(json.dumps({"left": 0.7, "top": 0.3, "right": 0.6, "bottom": 0.8})) is None


def test_a_move_on_the_drawing_is_a_hand_over():
    on = SimpleNamespace(selector=drawing.DRAWING)
    off = SimpleNamespace(selector="#start")
    assert drawing.chose_the_drawing([(off, ""), (on, "")])
    assert not drawing.chose_the_drawing([(off, "")])


def test_the_screen_pursuit_takes_a_measured_band():
    from core.skills.screen_pursuit import pursue_on_screen

    assert "drawn_at" in inspect.signature(pursue_on_screen).parameters
    source = inspect.getsource(pursue_on_screen)
    assert '{"where": drawn_at, "asked": drawn_at is not None}' in source


def test_the_observer_reports_what_a_page_draws():
    from core.capabilities import phantom_browser

    source = inspect.getsource(phantom_browser)
    assert "role: 'drawing'" in source and "walkDrawn(el.shadowRoot)" in source
    assert f"selector: '{drawing.DRAWING}'" in source


def test_what_the_page_draws_is_offered_ahead_of_its_links():
    """LIVE 2 Oct: a 598 by 399 game fell below forty links and could not be chosen."""
    from core.skills.sovereign_browser import SovereignBrowserSkill as S

    links = [{"role": "link", "name": f"Game {n}", "selector": f"#g{n}"} for n in range(60)]
    drawn = {"role": "drawing", "name": "what the page draws, 598 by 399", "selector": drawing.DRAWING}
    offered = S._controls_worth_offering([*links, drawn], "play the game")
    assert drawn in offered


# ── her own page as the surface ──────────────────────────────────────────────


class _Keys:
    def __init__(self, page):
        self.page = page

    async def press(self, key):
        self.page.pressed.append(key)
        if self.page.playing:
            self.page.score += 1


class _Mouse:
    def __init__(self, page):
        self.page = page

    async def click(self, x, y):
        self.page.clicked.append((round(x), round(y)))
        # The game's own PLAY, drawn at the middle of its lower part.
        if abs(x - 300) <= 20 and abs(y - 340) <= 20:
            self.page.playing = True


class _APageThatDraws:
    """A page with a 400 by 300 drawing at (100, 100) in an 800 by 600 viewport."""

    url = "https://example.test/a-game"
    viewport_size = {"width": 800, "height": 600}

    def __init__(self):
        self.pressed: list[str] = []
        self.clicked: list[tuple[int, int]] = []
        self.playing = False
        self.score = 0
        self.evaluated: list[str] = []
        self.keyboard = _Keys(self)
        self.mouse = _Mouse(self)

    async def evaluate(self, script):
        self.evaluated.append(script)
        if "scrollIntoView" in script:
            return json.dumps({"left": 0.125, "top": 1 / 6, "right": 0.625, "bottom": 2 / 3})
        return True

    async def title(self):
        return "A game"

    def reading(self, over):
        if not self.playing:
            layout = [{"text": "PLAY", "center_x": 0.5, "center_y": 0.8}]
        else:
            layout = [{"text": f"score {self.score}", "center_x": 0.2, "center_y": 0.1}]
        return layout


@pytest.fixture
def no_desktop(monkeypatch):
    """Anything that would touch the desktop fails the test."""
    from core.capabilities import host_automation, window_server

    def refuse(*_a, **_k):
        raise AssertionError("the desktop was touched while her own page was the surface")

    for name in ("front_owner", "window_of", "post_keys", "type_keys", "capture"):
        monkeypatch.setattr(window_server, name, refuse, raising=False)
    monkeypatch.setattr(host_automation, "get_host_automation", refuse)


@pytest.fixture
def a_page(monkeypatch, no_desktop):
    page = _APageThatDraws()

    async def look(page_, over=None, *, name, wait_for_stillness=True, still_within_s=None):
        layout = page_.reading(over)
        return {
            "ok": True,
            "text": "\n".join(region["text"] for region in layout),
            "layout": layout,
            "bounds": [100, 100, 400, 300],
            # As the real reader says it: a picture of only the part, so its
            # positions are shares of the drawing and are not cropped again.
            "read_within": "the part" if over is not None else "the page",
            "scoped_to": name,
            "in_front_then": name,
            "her_window_showing": True,
            "owner": name,
        }

    monkeypatch.setattr("core.perception.what_her_page_shows.look_at_a_page", look)
    from core.security import screen_capture_policy as policy

    async def may_look(*_a, **_k):
        return policy.ScreenCaptureAdmission(allowed=True)

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", may_look)
    return page


def test_keys_and_clicks_reach_the_page_and_nothing_else(a_page):
    on = OnAPage(page=a_page, name=drawing.HER_BROWSER)
    assert asyncio.run(on.press("left"))
    assert a_page.pressed == ["ArrowLeft"]
    assert any("focus()" in script for script in a_page.evaluated), "keys go to the drawing once it has focus"
    assert asyncio.run(on.press_many(["up", "space"])) == 2
    assert a_page.pressed[-2:] == ["ArrowUp", "Space"]
    assert asyncio.run(on.click(0.5, 0.8, [100, 100, 400, 300]))
    assert a_page.clicked == [(300, 340)]
    assert not asyncio.run(on.click(1.4, 0.5, [100, 100, 400, 300])), "never outside what it was given"
    assert asyncio.run(on.identity()) == {"url": a_page.url, "title": "A game", "error": ""}


def test_every_seam_answers_from_the_page_while_it_is_the_surface(a_page):
    from core.skills import screen_pursuit, screen_pursuit_looking, screen_pursuit_surface

    async def run():
        held = HER_OWN_PAGE.set(OnAPage(page=a_page, name=drawing.HER_BROWSER))
        try:
            seen = await screen_pursuit.read_screen(drawing.HER_BROWSER, (0.1, 0.1, 0.9, 0.9))
            assert seen["layout"][0]["text"] == "PLAY"
            assert await screen_pursuit_surface.press("down", expect_app="Google Chrome")
            assert await screen_pursuit_surface.press_many(["up"], expect_app="Google Chrome") == 1
            assert await screen_pursuit_surface.click_normalized(0.5, 0.8, bounds=[100, 100, 400, 300])
            assert await screen_pursuit_surface._ensure_frontmost("Google Chrome")
            assert await screen_pursuit._frontmost() == drawing.HER_BROWSER
            assert await screen_pursuit._whats_on_top(drawing.HER_BROWSER) == ""
            assert (await screen_pursuit.current_page_identity())["url"] == a_page.url
            assert await screen_pursuit_looking._bring_the_thing_back_to_the_front("x")
            assert not await screen_pursuit_surface._ensure_page("https://elsewhere.test/")
        finally:
            HER_OWN_PAGE.reset(held)
        assert on_her_page() is None

    asyncio.run(run())
    assert a_page.pressed == ["ArrowDown", "ArrowUp"]


def test_without_a_page_nothing_is_played(monkeypatch):
    called: list[Any] = []

    async def _pursue(**kwargs):
        called.append(kwargs)
        return {}

    monkeypatch.setattr("core.skills.screen_pursuit.pursue_on_screen", _pursue)
    step = asyncio.run(drawing.played_on_the_drawing(SimpleNamespace(), "play", {"url": "u"}))
    assert called == [] and step["error"] and step["landed"] == 0


def test_the_hand_over_plays_her_page_with_the_band_it_measured(monkeypatch, a_page):
    seen: dict[str, Any] = {}

    async def _pursue(**kwargs):
        seen.update(kwargs)
        seen["surface"] = on_her_page()
        return {"moves": [{"key": "left"}], "outcome": "reached", "completed": True}

    monkeypatch.setattr("core.skills.screen_pursuit.pursue_on_screen", _pursue)
    step = asyncio.run(
        drawing.played_on_the_drawing(SimpleNamespace(page=a_page), "play the game", {"url": a_page.url})
    )
    assert seen["drawn_at"] == pytest.approx((0.125, 1 / 6, 0.625, 2 / 3))
    assert seen["target_app"] == drawing.HER_BROWSER and seen["expect_page"] == a_page.url
    assert seen["surface"] is not None and seen["surface"].page is a_page
    assert on_her_page() is None, "the page is the surface for the hand-over and no longer"
    assert step["landed"] == 1 and step["completed"]


class _Store:
    def __init__(self):
        self.episodes = []

    def record(self, episode):
        self.episodes.append(episode)
        return f"ep_{len(self.episodes)}"

    def resolve(self, episode_id, outcome):
        self.episodes.append((episode_id, outcome))

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        self.episodes.append((action, outcome, success))


@pytest.mark.asyncio
async def test_a_game_on_her_page_is_started_by_its_drawn_play_and_then_played(a_page, monkeypatch):
    """The whole loop on her own page: keys do nothing, PLAY is clicked on the page, then she plays."""
    from screen_pursuit_support import patch_pursuit

    patch_pursuit(monkeypatch, "_how_long_to_wait", lambda: 0.05, raising=False)

    async def names_nothing(objective, evidence):
        return ""

    from core.skills import screen_pursuit as sp

    held = HER_OWN_PAGE.set(OnAPage(page=a_page, name=drawing.HER_BROWSER))
    try:
        await sp.pursue_on_screen(
            goal="play the game",
            target_app=drawing.HER_BROWSER,
            success_when="score 5",
            think=names_nothing,
            max_cycles=40,
            max_seconds=40.0,
            narrate=False,
            lived=False,
            spine=_Store(),
            graph=_Store(),
            drawn_at=(0.125, 1 / 6, 0.625, 2 / 3),
        )
    finally:
        HER_OWN_PAGE.reset(held)
    assert a_page.clicked and a_page.clicked[0] == (300, 340), "PLAY is clicked where the page shows it"
    assert a_page.playing and a_page.score > 0, "the started game was played with keys sent to the page"


def test_what_the_play_came_to_reaches_her_next_look_at_the_page():
    """A canvas's text never changes; the game's own last words have to be carried."""
    said = drawing.what_the_play_came_to(
        34, {"outcome": "out_of_cycles", "last_seen": "GAME  OVER\nSCORE 120"}
    )
    assert said == "played what the page draws (34 move(s), ended: out_of_cycles); the last thing it showed: GAME OVER SCORE 120"
    from core.skills.sovereign_browser_what_it_did import what_the_move_did

    page = {"url": "u", "text": "A game\nPlay Game", "elements": []}
    told = what_the_move_did(page, page, done=[said])
    assert "the last thing it showed: GAME OVER SCORE 120" in told


def test_the_last_reading_is_kept_in_the_pursuits_result():
    from core.skills.screen_pursuit_steps import _pursue_on_screen_result

    receipt = SimpleNamespace(completed=False, outcome="out_of_cycles", to_dict=lambda: {"completed": False, "outcome": "out_of_cycles"})
    result = _pursue_on_screen_result(
        {"value": False}, {"last": "", "count": 0}, [], [], {"choice": "", "waits": 0, "because": ""},
        receipt, {"count": 0, "because": ""}, {"value": False, "because": ""}, "",
        {"where": None, "last_seen": "YOU WIN"},
    )
    assert result["last_seen"] == "YOU WIN"
