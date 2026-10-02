"""What a page draws is handed to her eyes, in the same window, over the part it draws in.

LIVE 2026-10-02: a Flash game on webdesignmuseum.org is a 540 by 360 canvas in
Ruffle's shadow root with its PLAY button painted on it. The browser pursuit has
nothing there to read or press; the screen pursuit plays exactly that kind of
thing. The observer reports the drawing as one thing on the page, and choosing it
hands the window to the screen pursuit with the band the page says it draws in.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

from core.skills import sovereign_browser_drawing as drawing

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


def test_no_window_she_cannot_prove_is_hers_is_ever_played(monkeypatch):
    """LIVE 2 Oct the front window was the person's own Chrome."""
    called: list[Any] = []

    async def _pursue(**kwargs):
        called.append(kwargs)
        return {}

    monkeypatch.setattr("core.skills.screen_pursuit.pursue_on_screen", _pursue)
    monkeypatch.setattr("core.capabilities.window_server.front_owner", lambda: "Google Chrome")
    step = asyncio.run(
        drawing.played_on_the_drawing(SimpleNamespace(page=object()), "play", {"url": "u"})
    )
    assert called == []
    assert step["error"] and step["landed"] == 0


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
