"""A guide to the place she is in says how it is worked, from every source, kept up to date, and never plays for her.

LIVE 2026-10-09: she clicked a game's "PAUSED" and "CHARGING..." to see what they did, read its charge meter as her
health, played a game the screen said was worked with WASD by the mouse, and pulled and let go two hundred times in a
game that said "click again to launch". Each was written in what the game showed or in the program it ran.
Nothing here is a game: a made-up program, a made-up page, words any place might say.
"""
from __future__ import annotations

import re

import pytest

from core.agency.mechanics_she_knows import (
    MECHANICS,
    changes_told,
    mechanics_in,
    what_a_label_measures,
)
from core.cognition.a_guide_to_a_place import COUNSEL, PAGE, PROGRAM, SCREEN, TOLD, Guide
from core.perception.reading_a_program import read_program

pytestmark = pytest.mark.unit

A_PROGRAM = b"""
<html><body><canvas id="stage"></canvas><button id="pauseBtn"></button><button id="muteButton"></button>
<script>
var help = "Use the arrow keys to move and press space to jump over the pits.";
var story = "Chapter two: the castle falls and the hero learns the secret of the tower.";
var answers = "The answer is 42 and the password is swordfish.";
document.addEventListener('keydown', function (e) {
  if (e.keyCode === 37) { hero.x -= 4; }
  if (e.keyCode === 39) { hero.x += 4; }
  if (e.key === ' ') { jump(); }
  if (e.keyCode === 192) { cheatMode(); skipLevel(); }
  if (e.keyCode === 80) { pause(); }
});
var gravity = 0.4; var score = 0; var lives = 3;
</script></body></html>
"""


def test_a_program_says_its_keys_its_controls_and_its_instructions_and_nothing_that_would_spoil_it():
    read = read_program(A_PROGRAM, "page.html")
    assert {"left", "right", "space", "p"} <= set(read.keys) and "`" not in read.keys
    assert read.keys["space"] == "jump" and read.keys["p"] == "pause"
    assert "pause" in read.buttons and "mute" in read.buttons
    assert any("arrow keys to move" in w for w in read.words)
    assert not any("answer" in w or "password" in w or "secret" in w for w in read.words)


def test_words_outrank_code_and_the_screen_outranks_what_was_read_before():
    guide = Guide(place="a place")
    guide.take_in_program(read_program(A_PROGRAM))
    assert set(guide.keys_for_play()) >= {"left", "right", "space"} and "p" not in guide.keys_for_play()
    assert any("pause" in c for c in guide.not_for_play)
    guide.take_in(TOLD, "Use WASD to move. Click to throw.")
    assert set(guide.keys_for_play()) == {"w", "a", "s", "d"}          # the place's words, not its code
    assert guide.pointer_named()
    assert guide.controls["w"].source == TOLD


@pytest.mark.parametrize("label, read", [("SCORE: 0", True), ("CHARGING...", True), ("GLIDE METER", True),
                                         ("TIME 01:50", True), ("LEVEL 2", True), ("Play Again", False),
                                         ("LEVEL SELECT", False), ("Next", False), ("PAUSED", False)])
def test_what_is_on_the_screen_to_be_read_is_told_from_what_is_to_be_pressed(label, read):
    assert Guide().is_to_read(label) is read


def test_a_meter_filled_by_holding_is_not_what_she_has_left():
    from types import SimpleNamespace

    from core.agency.what_she_has_left import what_a_bar_measures

    bar = SimpleNamespace(where=lambda wide, tall: (0.1, 0.9, 0.4, 0.95))
    assert what_a_bar_measures(bar, [{"text": "CHARGING...", "center_x": 0.15, "center_y": 0.86}], 320, 240)[1] == "neither"
    assert what_a_bar_measures(bar, [{"text": "FUEL", "center_x": 0.15, "center_y": 0.86}], 320, 240)[1] == "down is bad"
    assert what_a_label_measures("HEALTH") == "health"


def test_what_the_place_says_as_it_goes_changes_the_guide_and_each_change_is_kept_and_said():
    guide = Guide(place="a place", began=0.0)
    guide.take_in(SCREEN, "Use the arrow keys to move. Level 1. Collect the stars.", at=1.0)
    assert guide.stage == "level 1" and guide.in_play("collecting")
    news = guide.take_in(SCREEN, "Level 2. You can now double jump! Watch out for the new lasers.", at=30.0)
    assert any("level 2" in n for n in news)
    assert guide.in_play("jumping") and any("jumping came into play" in what for _when, what in guide.changes)
    news = guide.take_in(SCREEN, "Your shield wore off.", at=40.0)
    assert not guide.in_play("power-ups") and news
    assert guide.take_in(SCREEN, "Level 2. You can now double jump!", at=41.0) == []      # read again: no news
    assert "Gone from play: power-ups" in guide.for_thinking()


def test_a_power_for_a_while_is_in_play_for_that_while():
    import time

    guide = Guide()
    guide.take_in(SCREEN, "You got a shield! You are invincible for 10 seconds.", at=time.monotonic())
    assert guide.in_play("power-ups")
    guide.mechanics["power-ups"].until = time.monotonic() - 1
    assert not guide.in_play("power-ups")


def test_each_source_is_kept_and_said_and_nothing_tells_her_what_to_do():
    guide = Guide(place="a place")
    guide.take_in(PAGE, "Use the arrow keys to guide the lander safely onto the landing pad. Watch your fuel.")
    guide.take_in_program(read_program(A_PROGRAM))
    guide.take_in_counsel("Ease off the thrust before you touch down.")
    said, thinking = guide.says(), guide.for_thinking()
    assert said.startswith("From its page") and "its program" in said and "what I looked up" in said
    assert "thrust" in thinking and "fuel" in thinking
    assert guide.sources >= {PAGE, PROGRAM, COUNSEL}
    assert not re.search(r"\byou must\b|\bpress now\b", said + thinking, re.I)


def test_a_guide_is_kept_between_sittings_as_plain_data():
    guide = Guide(place="a place")
    guide.take_in(TOLD, "Press SPACE to jump. Avoid the spikes. Collect coins.")
    again = Guide.from_memory(guide.as_memory())
    assert again.keys_for_play() == guide.keys_for_play() and again.things == guide.things
    assert again.things.get("spikes") == "avoid" and again.things.get("coins") == "meet"


def test_every_way_of_playing_has_the_mechanics_that_say_what_it_means_and_every_mechanic_names_real_ways():
    from core.agency.ways_of_playing import WAYS

    names = {way.name for way in WAYS}
    for mechanic in MECHANICS:
        assert set(mechanic.ways) <= names, (mechanic.name, set(mechanic.ways) - names)
        re.compile(mechanic.words)
        re.compile(mechanic.program or "x")
        assert mechanic.what and mechanic.means and mechanic.play
    played = {way for m in MECHANICS for way in m.ways}
    acting = names - {"play as told", "a legend", "take stock", "counters", "bars", "pause", "carried", "a guide",
                      "checked against the guide", "read the program", "make"}
    assert acting <= played | {"make"}, acting - played


@pytest.mark.parametrize("said, has", [
    ("Use the arrow keys to guide the lander safely onto the landing platform! Watch your fuel.", {"thrust", "fuel", "steering"}),
    ("Fill in the required fields and press Submit. Please sign in to continue.", {"forms", "signing in"}),
    ("Search for a flight, then pick a seat from the drop-down and press Buy now to pay.", {"search", "choosing from options"}),
    ("Repeat the arrows in the same order as the book shows them.", {"following a sequence"}),
    ("Drag each part onto the machine so the gumball rolls from the chute to the bucket.", {"dragging"}),
])
def test_the_mechanics_she_knows_are_known_by_what_any_place_says(said, has):
    assert has <= set(mechanics_in(said)), sorted(mechanics_in(said))


def test_what_is_the_persons_to_do_is_known_as_a_boundary_and_checked_against():
    from core.cognition.checking_the_debate import DebateCheck

    guide = Guide(place="a shop")
    guide.take_in(SCREEN, "Your basket. Place your order. Sign in to check out.")
    check = DebateCheck()
    buy, basket = 'click "Place your order"', 'click "Your basket"'
    assert check.facts(buy, guide)[7] == 1.0 and check.facts(basket, guide)[7] == 0.0
    weighed = check.weigh({buy: 1.0, basket: 0.9}, guide)
    assert weighed[buy] < weighed[basket]


def test_the_check_scales_what_she_valued_by_what_the_guide_says_and_learns_from_what_acts_did():
    from core.cognition.checking_the_debate import FACTS, DebateCheck

    guide = Guide(place="a game")
    guide.take_in(TOLD, "Use the arrow keys to move. Space to jump.")
    guide.take_in(SCREEN, "SCORE: 0")
    check = DebateCheck()
    valued = {'click "SCORE: 0"': 1.0, "left": 0.6, 'click "Play"': 0.5}
    weighed = check.weigh(valued, guide)
    assert weighed['click "SCORE: 0"'] < weighed["left"]                   # a count is read, not pressed
    assert weighed['click "Play"'] > 0.5                                     # a way on is a way on
    before = check.weights[FACTS.index("a control it names")]
    for _ in range(20):
        check.weigh({"left": 1.0}, guide)
        check.learned("left", answered=False)
    assert check.weights[FACTS.index("a control it names")] < before        # a named key that never answered


def test_a_word_said_where_play_confirmed_a_mechanic_in_enough_places_becomes_a_sign_of_it_anywhere():
    from core.agency.mechanics_she_knows import LearnedSigns
    from core.cognition.how_a_change_is_promoted import WHAT_A_TIER_WANTS, a_ledger_of_its_own

    signs = LearnedSigns()
    wanted = max(2, WHAT_A_TIER_WANTS.get("word", 2))
    with a_ledger_of_its_own():
        for place in range(wanted - 1):
            assert signs.confirmed("steering", ["skate"], f"place {place}") == []
        assert signs.state[("steering", "skate")] == "shadow"
        assert signs.confirmed("steering", ["skate"], "one place more") == ["skate"]
    assert "skate" in signs.active()["steering"]


def test_the_programs_a_page_runs_are_told_from_its_furniture():
    from core.skills.sovereign_browser_the_program import programs_of

    loaded = [["https://example.org/game/thing.swf", "fetch", 1_000_000],
              ["https://example.org/js/main.js", "script", 40_000],
              ["https://www.googletagmanager.com/gtag/js?id=1", "script", 90_000],
              ["https://example.org/css/site.css", "link", 3000],
              ["https://example.org/img/logo.png", "img", 3000]]
    assert programs_of(loaded) == ["https://example.org/game/thing.swf", "https://example.org/js/main.js"]


def test_a_screen_taken_into_the_guide_offers_its_controls_and_not_what_is_to_be_read():
    from core.agency.what_i_can_do_here import a_click_on
    from core.skills.screen_pursuit_looking import _as_the_guide_reads_it

    paced: dict = {}
    clickable = (a_click_on("SCORE: 0"), a_click_on("CHARGING..."), a_click_on("Play"), a_click_on("LEVEL SELECT"))
    kept = _as_the_guide_reads_it(clickable, "SCORE: 0 CHARGING... Play LEVEL SELECT", paced, narrate=False)
    assert kept == (a_click_on("Play"), a_click_on("LEVEL SELECT"))
    assert isinstance(paced.get("guide"), Guide)


def test_a_site_has_a_guide_of_its_own_that_grows_as_she_goes_through_it():
    from core.skills.sovereign_browser_guide import how_this_place_works

    page = {"url": "https://shop.example/basket", "text": "Your basket has 2 items. Fill in the required fields."}
    elements = [{"role": "button", "name": "Place your order"}, {"role": "link", "name": "Sign in"}]
    said = how_this_place_works(page, elements)
    assert "forms" in said and "signing in" in said
    page2 = {"url": "https://shop.example/search", "text": "Search results for lamps."}
    assert "search" in how_this_place_works(page2, []) and "signing in" in how_this_place_works(page2, [])


def test_changes_told_are_read_from_any_place_words():
    told = changes_told("You can now double jump! Your shield wore off. Welcome to level 3.")
    assert ("added", "jumping") in {(how, name) for how, name, _s in told}
    assert ("taken", "power-ups") in {(how, name) for how, name, _s in told}
