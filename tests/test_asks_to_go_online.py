"""Asking her to go somewhere online is asking for her browser, as asking for a deck is asking for a file: the turn's
effect ceiling lets her reach it. LIVE 20 October 2026, the watching demo's dress rehearsal: "Go to YouTube and find a
random video from Allen Pan" ran under the self-service ceiling, and her browser was withheld."""
from __future__ import annotations

import pytest

from core.intent.asks_to_go_online import asks_to_go_online

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("said", [
    "Let's watch something together. Go to YouTube and find a random video from Allen Pan, under 15 minutes.",
    "Can you open reddit.com and read me the top post?",
    "Find a Salad Fingers episode on YouTube.",
    "Pull up Wikipedia's page on the Bronze Age collapse.",
    "Look it up online for me.",
    "Watch this with me: https://www.youtube.com/watch?v=RGQEwYsNdI4",
    # Bryan, 20 October 2026, the demo's second video:
    "Go to Allen Pan's YouTube channel, look through his videos, and pick the one that interests you most.",
    "Head over to the Kurzgesagt channel on YouTube.",
])
def test_going_somewhere_online(said):
    assert asks_to_go_online(said)


@pytest.mark.parametrize("said", [
    "What do you think about volcanoes?",
    "Write me a short poem about the sea.",
    "How are you feeling today?",
    "What is 87952 times 587923?",
    "I need to go to the store later.",
])
def test_not_going_anywhere(said):
    assert not asks_to_go_online(said)


def test_the_ceiling_reaches_her_browser_when_she_is_asked_to_go_online():
    from core.phases.response_contract import requested_effect_ceiling
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    ceiling, scopes = requested_effect_ceiling("Go to YouTube and find a random video from Allen Pan.")
    assert SKILL_EFFECT_SCOPES["sovereign_browser"] in scopes and ceiling == "external_io"
    ceiling, scopes = requested_effect_ceiling("What do you think about volcanoes?")
    assert SKILL_EFFECT_SCOPES["sovereign_browser"] not in scopes


def test_a_site_named_by_name_is_its_front_page_for_a_request_and_for_her_browser():
    """LIVE 2026-10-20, the dress rehearsal: "Go to YouTube and find a random video from Allen Pan" named no address,
    and her browser's call, a goal with no mode, defaulted to a search with no query and was refused."""
    from core.conversation.page_interaction import front_page_named
    from core.skills.sovereign_browser import BrowserInput

    asked = "Let's watch something together. Go to YouTube and find a random video from Allen Pan, under 15 minutes."
    assert front_page_named(asked) == "https://www.youtube.com"
    call = BrowserInput(goal=asked, max_steps=40)
    assert call.mode == "pursue" and call.url == "https://www.youtube.com"
    assert BrowserInput(mode="search", query="allen pan lightsaber").mode == "search"   # a stated mode stands
    assert front_page_named("What do you think about volcanoes?") == ""


_FIRST = ("Let's watch this together: https://www.youtube.com/watch?v=ujkD4SxPKOI — before it starts, tell me what you "
          "think it'll be about, think out loud while we watch, and tell me what you thought after.")


@pytest.mark.parametrize("said", [
    # Bryan, 22 October 2026: after two YouTube videos, this means a third on YouTube, as it would to anybody.
    "Now Salad Fingers. Pick whichever episode you want. Same again: tell me what you think it'll be about before it "
    "starts, think out loud while we watch, and tell me what you thought after.",
    "Now Salad Fingers, whichever episode you want.",
    "How about a Salad Fingers episode?",
    "Now let's do Allen Pan. Go to his channel, look through his videos, and pick the one that interests you most.",
    "Next episode please",
])
def test_a_request_that_names_no_place_goes_on_where_the_conversation_was(monkeypatch, said):
    import core.conversation.page_interaction as pages

    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [_FIRST, "what was that?"])
    monkeypatch.setattr(pages, "_asked_to_use_it", lambda body: "youtube.com" in body)  # the first was work on a page
    assert pages.page_interaction_target(said) == "https://www.youtube.com"


@pytest.mark.parametrize("said", [
    "did you like that video?",
    "Now, what did you think of that video?",
    "So what did you think of all three?",
    "watch me juggle",
    "run the tests",
    "read the label on this",
])
def test_talk_about_what_was_watched_goes_nowhere(monkeypatch, said):
    import core.conversation.page_interaction as pages

    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [_FIRST])
    monkeypatch.setattr(pages, "_asked_to_use_it", lambda body: "youtube.com" in body)
    assert pages.page_interaction_target(said) == ""


def test_a_place_carries_only_to_the_same_kind_of_thing():
    from core.intent.asks_to_go_online import the_place_it_continues

    assert the_place_it_continues("Now a Salad Fingers episode.", [_FIRST])[0] == "https://www.youtube.com"
    assert the_place_it_continues("Now play a game of chess.", [_FIRST]) == ("", "")
    # The latest request that named a place is the one gone on with.
    assert the_place_it_continues("Another episode.", [_FIRST, "Play a game on https://lichess.org"]) == ("", "")
    assert the_place_it_continues("A video on vimeo.com", [_FIRST]) == ("", "")


def test_the_ceiling_reaches_her_browser_when_she_goes_on_with_it(monkeypatch):
    import core.conversation.page_interaction as pages
    from core.phases.response_contract import requested_effect_ceiling

    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [_FIRST])
    monkeypatch.setattr(pages, "_asked_to_use_it", lambda body: "youtube.com" in body)
    ceiling, scopes = requested_effect_ceiling("Now Salad Fingers, whichever episode you want.")
    assert ceiling == "external_io" and "external_io" in scopes


def test_an_episode_is_something_to_watch_and_reading_stays_reading():
    from core.skills.whether_a_page_serves import what_the_task_needs

    assert what_the_task_needs("Now Salad Fingers, whichever episode you want.") == "run"
    assert what_the_task_needs("read about the history of video games") == "read"


def test_picking_is_an_instruction():
    from core.conversation.request_mood import RequestMood, assess_request_mood

    assert assess_request_mood("Pick whichever episode you want.").mood is RequestMood.DIRECTIVE
    assert assess_request_mood("Choose the one that interests you most.").mood is RequestMood.DIRECTIVE
