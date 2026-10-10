"""A thing she is told to drag is carried to a place: a press on it, carried with the button held, let go there.

LIVE 2026-10-09 a game's rules said "drag and drop the piece to the
highlighted area". She had clicks only, and "drag" alone had sent her into
letting go of shots at the title screen, whose START she had read "sitarit".
"""
from __future__ import annotations

import pytest

from core.agency.acts_on_two_places import CARRY, MATCH, USE, two_places_of
from core.agency.playing_by_shots import sends_by_letting_go
from core.agency.putting_things_in_place import a_carry_of, speaks_of_carrying, what_is_carried
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

pytestmark = pytest.mark.unit

BOARD = (a_click_on("add a part"), a_click_on("Pipe"), a_click_on("the shape at 50% across, 55% down"), a_click_on("attach here!"),
         a_click_on("SCORE"), a_click_on("next"))
#: The piece is drawn; "Pipe" is written over it. What is drawn is carried; what is written is clicked.
PIECE = "the shape at 50% across, 55% down"


@pytest.mark.parametrize("words", [
    "Click Add a Part and select a game piece. Drag and drop the piece to the highlighted area.",
    "When you have your pick, click and drag it into place.",
    "Put the cup on the shelf.",
    "drop it in the bin",
])
def test_words_that_speak_of_carrying_a_thing_somewhere(words):
    assert speaks_of_carrying(words)


@pytest.mark.parametrize("words", [
    "Click the green Drop button to set your Jawbreaker in motion.",
    "Use the arrow keys to move.",
    "Press space to jump.",
])
def test_words_that_do_not(words):
    assert not speaks_of_carrying(words)


def test_dragging_into_place_is_not_sending_by_letting_go():
    assert not sends_by_letting_go("When you have your pick, click and drag it into place.")
    assert sends_by_letting_go("Drag the mouse in the direction you would like. Release the mouse button")


def test_carries_are_offered_only_where_carrying_was_spoken_of():
    here = WhatWorksHere()
    here.looked_at(BOARD)
    here.looked_at(BOARD)
    here.asked_for_by("Your goal: get the jawbreaker into the bucket", BOARD)
    assert not here.carries(here.on_screen)
    here.asked_for_by("", BOARD, counsel="When you have your pick, click and drag it into place.")
    carries = here.carries(here.on_screen)
    assert carries and set(carries) <= set(here.available())


def test_the_place_marked_as_the_one_comes_first_and_a_carry_that_did_nothing_is_not_offered_again():
    here = WhatWorksHere()
    here.looked_at(BOARD)
    here.looked_at(BOARD)
    here.asked_for_by("Drag and drop the piece to the highlighted area.", BOARD)
    carries = here.carries(here.on_screen)
    assert carries[0] == a_carry_of("add a part", "attach here!") or what_is_carried(carries[0])[1] == "attach here!"
    # Neither a readout nor a way on is carried; a way on is no thing to carry.
    assert not any(what_is_carried(move)[0] in ("SCORE", "next") for move in carries)
    assert not any(what_is_carried(move)[1] == "SCORE" for move in carries)
    tried = a_carry_of(PIECE, "attach here!")
    assert tried in carries and not any(what_is_carried(move)[0] in ("Pipe", "add a part") for move in carries)
    here.tried(tried, changed=False)
    assert tried not in here.carries(here.on_screen)


def test_written_tabs_are_clicked_not_carried_and_carrying_rests_when_it_keeps_doing_nothing():
    # LIVE 2026-10-10 she carried a device library's tabs ("HANGERS", "ROLLERS") onto one spot thirty-five times.
    from core.agency.putting_things_in_place import RESTS_AFTER

    library = (a_click_on("LAUNCHERS"), a_click_on("HANGERS"), a_click_on("ROLLERS"),
               a_click_on("the shape at 20% across, 40% down"), a_click_on("the shape at 30% across, 40% down"),
               a_click_on("the shape at 40% across, 40% down"), a_click_on("the one that stands out at 25% across, 70% down"))
    here = WhatWorksHere()
    here.looked_at(library)
    here.looked_at(library)
    here.asked_for_by("Place the device at + to the end of another device's arrow to make a connection. Drag it.", library)
    carries = here.carries(here.on_screen)
    assert carries and not any(what_is_carried(m)[0] in ("LAUNCHERS", "HANGERS", "ROLLERS") for m in carries)
    for move in carries[:RESTS_AFTER]:
        here.tried(move, changed=False)
    assert here.carries(here.on_screen) == ()                                    # rests: clicks are what is left
    opened = (*library, a_click_on("the shape at 60% across, 40% down"))
    here.looked_at(opened)
    here.looked_at(opened)
    assert here.carries(here.on_screen)                                          # a changed screen is carried on afresh
    words = WhatWorksHere()
    words.looked_at(library)
    words.looked_at(library)
    words.asked_for_by("Drag the words into the gaps in the sentence.", library)
    assert any(what_is_carried(m)[0] == "HANGERS" for m in words.carries(words.on_screen))


def test_a_move_on_two_places_is_read_once_for_every_act():
    assert two_places_of('use "key" on "door"').act == USE
    assert two_places_of('match "card 1" with "card 4"').act == MATCH
    carry = two_places_of(a_carry_of("Pipe", "attach here!"))
    assert carry.act == CARRY and not carry.by_clicks
    assert carry.said() == 'Carrying "Pipe" to "attach here!"'
    assert two_places_of('click "Pipe"') is None


_A_PIECE_AND_ITS_PLACE = """<!doctype html><body style="margin:0">
<canvas id="c" width="400" height="300" style="display:block"></canvas>
<script>
const c = document.getElementById('c'), g = c.getContext('2d');
const w = window.__world = {piece: {x: 60, y: 60}, held: false, placed: false, clicks: 0};
function draw() {
  g.fillStyle = '#fff'; g.fillRect(0, 0, 400, 300);
  g.strokeStyle = '#c00'; g.lineWidth = 4; g.strokeRect(280, 180, 60, 60);
  g.fillStyle = '#06c'; g.fillRect(w.piece.x - 20, w.piece.y - 20, 40, 40);
}
function at(e) { const r = c.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; }
c.addEventListener('mousedown', e => { const p = at(e); w.held = Math.abs(p.x - w.piece.x) < 20 && Math.abs(p.y - w.piece.y) < 20; });
c.addEventListener('mousemove', e => { if (w.held) { const p = at(e); w.piece = p; draw(); } });
c.addEventListener('mouseup', e => { if (w.held) { const p = at(e); w.placed = p.x > 280 && p.x < 340 && p.y > 180 && p.y < 240; } w.held = false; });
c.addEventListener('click', () => { w.clicks += 1; });
draw();
</script></body>"""


@pytest.mark.asyncio
async def test_on_her_page_a_carry_takes_the_piece_to_its_place():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.capabilities.phantom_browser import _chromium_graphics_arguments
    from core.skills.screen_pursuit_on_a_page import OnAPage

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True, args=_chromium_graphics_arguments())
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        try:
            page = await browser.new_page(viewport={"width": 600, "height": 400})
            await page.set_content(_A_PIECE_AND_ITS_PLACE)
            box = await page.locator("canvas").bounding_box()
            bounds = (box["x"], box["y"], box["width"], box["height"])
            hand = OnAPage(page, "a piece and its place")
            # A click on the piece does not carry it anywhere.
            assert await hand.click(60 / 400, 60 / 300, bounds)
            assert not await page.evaluate("__world.placed")
            assert await hand.carry((60 / 400, 60 / 300), (310 / 400, 210 / 300), bounds)
            world = await page.evaluate("__world")
        finally:
            await browser.close()
    assert world["placed"], world
    assert abs(world["piece"]["x"] - 310) < 3 and abs(world["piece"]["y"] - 210) < 3


def test_a_label_that_tells_the_place_to_do_something_is_pressed_not_carried():
    """LIVE 2026-10-10 she carried "SKIP INSTRUCTIONS" to "DELETE X"; a device is carried, a bin is carried to last."""
    from core.agency.putting_things_in_place import PuttingInPlace, what_is_carried

    putting = PuttingInPlace()
    putting.told_of_carrying("Drag the device to the end of another device's arrow to make a connection.")
    putting.pictures = {"anvil"}                                                   # a device her eyes named
    offered = putting.carries(['click "SKIP INSTRUCTIONS"', 'click "DELETE X"', 'click "TEST TRAP"', 'click "anvil"',
                               'click "the shape at the top"'])
    carried = [what_is_carried(move) for move in offered]
    assert all(thing not in ("SKIP INSTRUCTIONS", "DELETE X", "TEST TRAP") for thing, _place in carried)
    assert {thing for thing, _place in carried} == {"anvil", "the shape at the top"}
    assert carried[-1][1] == "DELETE X" and all(place != "TEST TRAP" for _thing, place in carried)


@pytest.mark.parametrize(("words", "place"), [
    ("DRAG THE DEVICE AT THE END OF ANOTHER DEVICE'S ARROW TO MAKE A CONNECTION", "the end of another device's arrow"),
    ("Drag and drop the piece to the highlighted area", "the highlighted area"),
    ("Put the toys in the toy box before mom comes home!", "the toy box"),
    ("Use the mouse to drag the cannon", ""),
    ("Click and drag with the left mouse button", ""),
])
def test_the_place_words_say_things_are_carried_to_is_read(words, place):
    from core.perception.where_the_words_point import the_place_named

    assert the_place_named(words) == place


def test_the_place_the_words_name_is_carried_to_first_and_is_not_carried():
    """LIVE 2026-10-10 told to put a device at the end of another's arrow, she carried shapes to shapes."""
    from core.agency.putting_things_in_place import PuttingInPlace, what_is_carried
    from core.perception.where_the_words_point import place_from

    putting = PuttingInPlace()
    putting.told_of_carrying("Drag the device at the end of another device's arrow to make a connection.")
    assert putting.place_named == "the end of another device's arrow"
    offered = putting.carries(['click "anvil"', 'click "the shape at the top"', 'click "the end of another device\'s arrow"'])
    first = what_is_carried(offered[0])
    assert first[1] == "the end of another device's arrow"
    assert all(what_is_carried(m)[0] != "the end of another device's arrow" for m in offered)
    assert place_from('{"bbox_2d": [100, 200, 300, 400]}') == (0.2, 0.3) and place_from('{"bbox_2d": null}') is None
