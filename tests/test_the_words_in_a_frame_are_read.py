"""A page's words include the ones it shows in a frame.

LIVE 2026-10-02 23:40, on the OEJTS results page: "A detailed description of
this personality type is below", and the description is a text file shown in an
iframe. The observer read only the main document, so she pressed "more" and
"less" and scrolled for fifteen minutes, saying "I need to see the INTP
description text below", while it was on her screen.

The fixture is served at https addresses through route interception, as in
test_browser_observation_sees_real_controls.py: BrowserAuthority refuses any
other scheme, and no network is touched.
"""

from __future__ import annotations

import pytest

PAGE = "https://example.com/aura-frame-fixture"
DESCRIPTION = "People of this type are marked by an unquenchable thirst for knowledge."
WIDGET = "A second site's comment box, drawn in the page."

FIXTURE = """
<html><body style="margin:0">
  <p>When combined, that makes your type INTP. A detailed description is below.</p>
  <iframe src="https://example.org/aura-frame-widget" style="position:absolute;
          top:600px; left:0; width:300px; height:120px"></iframe>
  <iframe src="https://example.com/aura-frame-description.txt" style="position:absolute;
          top:100px; left:0; width:600px; height:400px"></iframe>
  <iframe src="https://example.com/aura-frame-hidden.txt" style="display:none"></iframe>
  <iframe src="http://example.net/aura-frame-refused" style="position:absolute;
          top:800px; left:0; width:300px; height:120px"></iframe>
</body></html>
"""

ROUTES = {
    "**/aura-frame-fixture": ("text/html", FIXTURE),
    "**/aura-frame-description.txt": ("text/plain", DESCRIPTION),
    "**/aura-frame-widget": ("text/html", f"<html><body>{WIDGET}</body></html>"),
    "**/aura-frame-hidden.txt": ("text/plain", "words no one can see"),
    "**/aura-frame-refused": ("text/html", "<html><body>served without https</body></html>"),
}


def _serves(kind: str, body: str):
    async def _fulfil(route) -> None:
        await route.fulfill(status=200, content_type=kind, body=body)

    return _fulfil


@pytest.fixture
async def browser():
    from core.capabilities.phantom_browser import PhantomBrowser

    instance = PhantomBrowser(visible=False, browser_type="chromium", principal="owner")
    if not await instance.ensure_ready():
        await instance.close()
        pytest.skip("no browser engine available in this environment")
    try:
        for pattern, (kind, body) in ROUTES.items():
            await instance.page.route(pattern, _serves(kind, body))
        await instance.page.goto(PAGE, wait_until="load")
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_the_words_in_a_frame_are_part_of_the_page(browser):
    text = (await browser.observe(principal="owner"))["text"]
    assert DESCRIPTION in text
    assert "(Shown in a frame on this page, from example.com/aura-frame-description.txt:)" in text


@pytest.mark.asyncio
async def test_frames_read_top_to_bottom_after_the_page_itself(browser):
    text = (await browser.observe(principal="owner"))["text"]
    assert text.index("A detailed description is below") < text.index(DESCRIPTION)
    assert text.index(DESCRIPTION) < text.index(WIDGET), "the higher frame comes first"


@pytest.mark.asyncio
async def test_a_frame_no_one_can_see_is_not_read(browser):
    assert "words no one can see" not in (await browser.observe(principal="owner"))["text"]


@pytest.mark.asyncio
async def test_a_frame_is_read_only_under_its_own_addresss_verdict(browser):
    """The page's verdict does not cover a frame from an address it would refuse."""
    assert "served without https" not in (await browser.observe(principal="owner"))["text"]


def _frame(where: str, said: str, *, y: float, height: float = 100.0) -> dict:
    return {"x": 0.0, "y": y, "width": 300.0, "height": height, "where": where, "said": said}


@pytest.mark.unit
def test_a_cut_carries_how_long_the_words_were():
    from core.capabilities.phantom_browser import PhantomBrowser, _the_words_to_carry

    room = PhantomBrowser.PAGE_TEXT_ROOM
    carried = _the_words_to_carry(
        {"text": "x" * room, "viewport_height": 800},
        [_frame("a.example/b", "y" * 50, y=2000)],
        room,
    )
    assert len(carried["text"]) == room
    assert carried["text_chars"] > room


@pytest.mark.unit
def test_a_frames_words_go_where_the_frame_is():
    from core.capabilities.phantom_browser import _the_words_to_carry

    carried = _the_words_to_carry(
        {
            "text": "before the frame. after the frame.",
            "frames_at": [{"at": 17, "left": 0, "top": 50}],
            "viewport_height": 800,
        },
        [_frame("a.example/f.txt", "INSIDE", y=50)],
        4000,
    )["text"]
    assert carried.index("before the frame.") < carried.index("INSIDE") < carried.index("after")


@pytest.mark.unit
def test_the_words_carried_start_where_she_is_looking():
    """No scroll changed what she read: the first 4000 characters, every time."""
    from core.capabilities.phantom_browser import _the_words_to_carry

    text = "".join(f"line {n:05d}\n" for n in range(2000))
    looking = text.index("line 01500")
    carried = _the_words_to_carry(
        {"text": text, "seen_at": looking, "viewport_height": 800}, [], 4000
    )
    assert carried["text"].startswith("line 01500")
    assert carried["text_from"] == looking
    assert carried["text_chars"] == len(text.strip())


@pytest.mark.unit
def test_a_frame_in_view_is_carried_even_when_its_top_is_above_her():
    from core.capabilities.phantom_browser import _the_words_to_carry

    text = "a" * 6000 + " tail words"
    carried = _the_words_to_carry(
        {
            "text": text,
            "seen_at": 6001,
            "frames_at": [{"at": 6000, "left": 0, "top": -200}],
            "viewport_height": 800,
        },
        [_frame("a.example/d.txt", "THE DESCRIPTION", y=-200, height=600)],
        4000,
    )["text"]
    assert "THE DESCRIPTION" in carried


@pytest.mark.unit
def test_a_decision_is_told_the_observer_cut_the_page():
    """A cut made before the decision was drawn is said, like one made there."""
    from core.skills.sovereign_browser import SovereignBrowserSkill

    rendered = SovereignBrowserSkill._render_observation(
        {"url": PAGE, "title": "t", "text": "the start", "text_chars": 9 + 500, "elements": []}
    )
    assert "goes on for 500 more characters" in rendered


@pytest.mark.unit
def test_a_decision_is_told_what_lies_above_where_she_looks():
    from core.skills.sovereign_browser import SovereignBrowserSkill

    rendered = SovereignBrowserSkill._render_observation(
        {"url": PAGE, "title": "t", "text": "the middle", "text_from": 300,
         "text_chars": 300 + 10, "elements": []}
    )
    assert "300 characters above this point" in rendered
    assert "goes on for" not in rendered
