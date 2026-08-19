"""Somebody else was opening the page before she could play.

A goal that names a place is not doing anything until she is there, and
getting there is part of the task. Naming a URL makes it an open. Naming
only the thing makes it a search — and search markup is written by whoever
wants to be found, which is why the browser controller returns vetted
results instead of opening one. Choosing among them is a decision, so it
goes through the ordinary deliberation, and arriving is checked by reading
the page back rather than assumed from the click.
"""
from __future__ import annotations

import pytest

from core.agency.reach_place import host_of, named_url, reach
from core.runtime.watched_goal import read_watched_goal


class _Receipt:
    def __init__(self, result):
        self.result = result


class _Browser:
    def __init__(self, results=None, lands_on=None):
        self._results = results if results is not None else []
        self._lands_on = lands_on
        self.opened = []
        self.searched = []
        self.page = {"url": "https://start.example/", "title": "start"}

    async def search_results(self, query, count=5):
        self.searched.append(query)
        return list(self._results)

    async def open_url(self, url, new_tab=True):
        self.opened.append(url)
        landed = self._lands_on if self._lands_on is not None else url
        self.page = {"url": landed, "title": host_of(landed)}
        return True

    async def current_page(self):
        return dict(self.page)


def _thinks(reply):
    async def think(objective, evidence):
        think.seen = list(evidence)
        return reply

    think.seen = None
    return think


RESULTS = [
    {"url": "https://play2048.co/", "title": "2048 — Play the Free Online Game"},
    {"url": "https://en.wikipedia.org/wiki/2048_(video_game)", "title": "2048 (video game) - Wikipedia"},
]


def test_a_url_in_the_request_is_the_place():
    assert named_url("open https://play2048.co/ and play") == "https://play2048.co/"
    assert named_url("play 2048") == ""


def test_a_request_that_says_it_is_already_open_names_no_place():
    """"2048 is open in Chrome" is a statement about the world."""
    assert read_watched_goal("2048 is open in Chrome. Keep playing until you get 128").where == ""


def test_a_request_that_names_the_thing_gives_her_something_to_find():
    assert read_watched_goal("play 2048 in Chrome until you get 128").where == "2048"
    assert read_watched_goal("go to wordle and keep guessing until it says solved").where == "wordle"


def test_a_pronoun_is_not_a_place():
    assert read_watched_goal("keep playing it until you get 128").where == ""


@pytest.mark.asyncio
async def test_a_named_url_is_opened_and_arrival_is_read_back():
    browser = _Browser()
    got = await reach("https://play2048.co/", browser=browser, lived=False)
    assert got.arrived
    assert browser.opened == ["https://play2048.co/"]
    assert browser.searched == []
    assert "Opened" in got.narrate()


@pytest.mark.asyncio
async def test_a_navigation_that_landed_elsewhere_is_not_an_arrival():
    """A click that reported success and went somewhere else is the failure."""
    browser = _Browser(lands_on="https://somewhere.else/")
    got = await reach("https://play2048.co/", browser=browser, lived=False)
    assert not got.arrived
    assert "not play2048.co" in got.reason


@pytest.mark.asyncio
async def test_a_named_thing_is_searched_for_and_the_destination_is_decided():
    browser = _Browser(results=RESULTS)
    think = _thinks("play2048.co is the game itself, wikipedia is an article about it")
    got = await reach("2048", browser=browser, think=think, lived=False)
    assert browser.searched == ["2048"]
    assert got.arrived
    assert got.url == "https://play2048.co/"
    assert "play2048.co" in " ".join(think.seen)


@pytest.mark.asyncio
async def test_the_choice_of_destination_is_between_real_results_only():
    browser = _Browser(results=RESULTS)
    think = _thinks("en.wikipedia.org")
    got = await reach("2048", browser=browser, think=think, lived=False)
    assert got.url == "https://en.wikipedia.org/wiki/2048_(video_game)"
    assert set(got.considered) == {"play2048.co", "en.wikipedia.org"}


@pytest.mark.asyncio
async def test_a_search_that_returns_nothing_openable_is_said_plainly():
    browser = _Browser(results=[])
    got = await reach("something that does not exist", browser=browser, think=_thinks("x"), lived=False)
    assert not got.arrived
    assert "returned nothing" in got.reason
    assert browser.opened == []


@pytest.mark.asyncio
async def test_nowhere_named_is_not_an_error():
    got = await reach("", browser=_Browser(), lived=False)
    assert not got.arrived
    assert got.reason == "nowhere was named"


@pytest.mark.asyncio
async def test_a_pursuit_that_cannot_get_there_does_not_start_playing(monkeypatch):
    from core.skills import screen_pursuit as sp

    pressed = []

    async def press(key, *, expect_app=""):
        pressed.append(key)
        return True

    async def read(app_name=""):
        return {"ok": True, "text": "board", "layout": [], "bounds": []}

    async def nowhere(wanted, **kw):
        from core.agency.reach_place import Reached

        return Reached(wanted=wanted, arrived=False, reason="the search returned nothing")

    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "read_screen", read)
    import core.agency.reach_place as rp

    monkeypatch.setattr(rp, "reach", nowhere)
    result = await sp.pursue_on_screen(
        goal="play the game",
        success_when="128",
        open_page="a game that does not exist",
        max_cycles=2,
        max_seconds=5.0,
        narrate=False,
        lived=False,
    )
    assert result["outcome"] == "could_not_get_there"
    assert pressed == [], "she started pressing keys at whatever was in front of her"


@pytest.mark.parametrize(
    "request_text,expected",
    [
        ("Go find a 2048 game online and play it until you get a 128 tile.", "2048 game"),
        ("play 2048 in Chrome until you get 128", "2048"),
        ("go to wordle and keep guessing until it says solved", "wordle"),
        ("look up a typing test and keep going until it says 60 wpm", "typing test"),
        ("pull up the crossword and keep at it until it says complete", "crossword"),
    ],
)
def test_the_place_is_read_however_a_person_asks_for_it(request_text, expected):
    """LIVE: "go find a 2048 game online" named no place, so she never moved.

    The narrow pattern read "play 2048" and missed the same request with more
    of the work spelled out.
    """
    goal = read_watched_goal(request_text)
    assert goal is not None, request_text
    assert goal.where == expected


@pytest.mark.parametrize(
    "request_text",
    [
        "2048 is open in Chrome. Keep playing it until you get a 128 tile.",
        "the wizard is already open, step through it until it says Finished",
        "keep refreshing the build page until it says passed",
    ],
)
def test_nothing_to_go_to_stays_nothing(request_text):
    goal = read_watched_goal(request_text)
    assert goal is not None
    assert goal.where == ""


def test_where_a_thing_lives_is_not_part_of_its_name():
    """"online" says where to look, not what to look for."""
    assert read_watched_goal("find a chess puzzle online and keep solving until it says mate").where == (
        "chess puzzle"
    )


@pytest.mark.asyncio
async def test_the_destination_decision_knows_what_she_means_to_do_there():
    """LIVE: she searched for a game, opened the encyclopedia article, and
    landed somewhere with nothing to play.

    Both results were about 2048. Only one of them was somewhere the task
    could be done, and nothing in the decision said which task.
    """
    browser = _Browser(results=RESULTS)
    think = _thinks("play2048.co")
    await reach(
        "2048 game",
        browser=browser,
        think=think,
        lived=False,
        purpose="play it until you get a 128 tile",
    )
    evidence = " ".join(think.seen)
    assert "play it until you get a 128 tile" in evidence
    assert "get to where this can be done" in think.seen[0] or any(
        "has to be possible" in line for line in think.seen
    )


@pytest.mark.asyncio
async def test_without_a_purpose_it_still_decides_on_the_name_alone():
    browser = _Browser(results=RESULTS)
    think = _thinks("play2048.co")
    got = await reach("2048 game", browser=browser, think=think, lived=False)
    assert got.arrived


def test_without_language_the_option_that_describes_the_task_wins():
    """LIVE: with the resident model down, every result looked identical and
    the encyclopedia article won on ordering.

    The only way to tell a place from a page about the place, without asking
    anything, is the words each option carries about itself.
    """
    from core.agency.deliberate_action import ActionOption, choose_without_language

    options = [
        ActionOption(name="en.wikipedia.org", detail="2048 (video game) - Wikipedia"),
        ActionOption(name="play2048.co", detail="2048 - Play the Free Online Game"),
        ActionOption(name="crazygames.com", detail="2048 Games - Play on CrazyGames"),
    ]
    playing, why = choose_without_language(
        options, wanted="get to where this can be done: play 2048 until you get a 128 tile"
    )
    assert playing.name == "play2048.co"
    assert "describes what I am trying to do" in why

    reading, _why = choose_without_language(
        options, wanted="read the wikipedia article about the 2048 video game"
    )
    assert reading.name == "en.wikipedia.org"


def test_a_record_of_something_working_still_outranks_a_description():
    from core.agency.deliberate_action import ActionOption, choose_without_language

    options = [ActionOption(name="up", detail="press up"), ActionOption(name="down", detail="press down")]
    chosen, _why = choose_without_language(
        options, recalled=["down worked before: tiles merged"] * 4, wanted="press up"
    )
    assert chosen.name == "down"


@pytest.mark.asyncio
async def test_being_there_already_is_not_a_reason_to_search_again():
    """A retried pursuit searched a second time and left the game."""
    browser = _Browser(results=RESULTS)
    browser.page = {"url": "https://play2048.co/", "title": "2048"}
    got = await reach("https://play2048.co/", browser=browser, lived=False)
    assert got.arrived
    assert got.because == "already there"
    assert browser.searched == [] and browser.opened == []


@pytest.mark.asyncio
async def test_finding_where_to_go_does_not_put_a_search_page_in_front():
    """LIVE: she searched, and the run ended could_not_get_there with the
    browser sitting on DuckDuckGo.

    Opening the search page made it the page in front, so opening the
    destination replaced the wrong thing and the arrival check — which reads
    the front tab — found a search engine.
    """
    browser = _Browser(results=RESULTS)
    got = await reach("2048 game", browser=browser, think=_thinks("play2048.co"), lived=False)
    assert got.arrived
    assert browser.opened == ["https://play2048.co/"], "something other than the destination was opened"
