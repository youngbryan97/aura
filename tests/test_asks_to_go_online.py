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
@pytest.mark.model
def test_a_request_that_names_no_place_goes_on_where_the_conversation_was(monkeypatch, said):
    """Read by the inference model (core/epistemics/where_a_request_is_done.py); whether the turn asks her to act at all
    is her reading of the turn, stood in for here."""
    from huggingface_hub import try_to_load_from_cache

    import core.conversation.page_interaction as pages
    from core.epistemics import what_her_record_refutes
    from core.epistemics import where_a_request_is_done as where

    if not isinstance(try_to_load_from_cache(what_her_record_refutes.READER, "config.json"), str):
        pytest.skip("the inference model is not in the local cache")
    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [_FIRST, "what was that?"])
    monkeypatch.setattr(pages, "_asked_to_use_it", lambda body: "youtube.com" in body)  # the first was work on a page
    monkeypatch.setattr(pages, "_asked_to_act", lambda body: True)
    monkeypatch.setattr(where, "installed_program_names", lambda: ["en-croissant", "Chess", "Notes"])
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


def _reading_where(monkeypatch, *, there: bool, program: str = ""):
    """The conversation's place with a stand-in for the reader of where a request is done."""
    import core.conversation.page_interaction as pages
    from core.epistemics import where_a_request_is_done as where

    asked: list[tuple[str, str, str]] = []

    def continues(earlier, request, place):
        asked.append((earlier, request, place))
        return there

    monkeypatch.setattr(where, "continues_there", continues)
    monkeypatch.setattr(where, "program_it_names", lambda text, names=None: program)
    monkeypatch.setattr(pages, "_asked_to_act", lambda body: True)
    return pages, asked


def test_a_place_carries_where_the_reader_says_the_request_is_done_there(monkeypatch):
    pages, asked = _reading_where(monkeypatch, there=True)
    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [_FIRST])
    assert pages.the_conversations_place("Now a Salad Fingers episode.") == "https://www.youtube.com"
    assert asked == [(_FIRST, "Now a Salad Fingers episode.", "https://www.youtube.com")]

    pages, asked = _reading_where(monkeypatch, there=False)
    assert pages.the_conversations_place("Now play a game of chess.") == ""
    # A request naming its own place, or a program on this machine, is never read against the conversation's place.
    assert pages.the_conversations_place("A video on vimeo.com") == ""
    pages, asked = _reading_where(monkeypatch, there=True, program="en-croissant")
    assert pages.the_conversations_place("Open En Croissant and play a game of chess against Stockfish") == ""
    assert asked == []


def test_a_pasted_resume_lends_no_page_to_the_requests_after_it(monkeypatch):
    """LIVE 23 November 2026: "Open En Croissant and play a game of chess against Stockfish" went to gmail.com, then
    linkedin.com, the addresses in a résumé pasted earlier, because both held a word from one list. The résumé here is
    invented."""
    resume = ("Here is my resume. Jane Q. Doe, software engineer. jane.doe@gmail.com | linkedin.com/in/janeqdoe. Built "
              "game servers at a games studio. Find me jobs that fit this.")
    pages, asked = _reading_where(monkeypatch, there=False)
    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [resume])
    assert pages.the_conversations_place("Play a game of chess against the computer.") == ""
    assert asked and asked[0][2] == "https://linkedin.com"


@pytest.mark.model
@pytest.mark.parametrize(("earlier", "asked", "place", "there"), [
    ("Here is my resume. Jane Q. Doe - Software Engineer. jane.doe@gmail.com | linkedin.com/in/janeqdoe | "
     "github.com/janeqdoe. Experience: built multiplayer game servers at a games studio. Find me jobs that fit this.",
     "Open En Croissant and play a game of chess against Stockfish", "linkedin.com", False),
    ("Go to Allen Pan's YouTube channel and pick a video under ten minutes.",
     "Now Salad Fingers. Pick whichever episode you want.", "youtube.com", True),
    ("Let's watch this together: https://www.youtube.com/watch?v=8KkKuTCFvzI", "Play a game of chess against the computer.",
     "youtube.com", False),
    ("Take the personality test on 16personalities.com as yourself.", "Now do the next page.", "16personalities.com",
     True),
    ("Play a game of chess on lichess.org.", "Now play another one.", "lichess.org", True),
    ("Go to gmail.com and check my mail.", "Open En Croissant and play chess.", "gmail.com", False),
    ("Watch the next Allen Pan video on YouTube.", "How about another one?", "youtube.com", True),
    ("Check the weather on weather.com.", "Open Calculator.", "weather.com", False),
])
def test_whether_a_request_is_done_where_the_conversation_was_is_read_by_meaning(earlier, asked, place, there):
    from huggingface_hub import try_to_load_from_cache

    from core.epistemics import what_her_record_refutes
    from core.epistemics.where_a_request_is_done import continues_there

    if not isinstance(try_to_load_from_cache(what_her_record_refutes.READER, "config.json"), str):
        pytest.skip("the inference model is not in the local cache")
    assert continues_there(earlier, asked, place) is there


@pytest.mark.model
@pytest.mark.parametrize(("asked", "name", "named"), [
    ("Open En Croissant and play a game of chess against Stockfish", "en-croissant", True),
    ("open en croissant and play chess", "en-croissant", True),
    ("Play a game of chess against the computer.", "Chess", False),
    ("Open Chess and play a game.", "Chess", True),
    ("Take some notes on what we said.", "Notes", False),
    ("Open Notes and start a new note.", "Notes", True),
    ("I love a good croissant in the morning.", "en-croissant", False),
])
def test_a_program_is_named_by_meaning_in_any_spelling_of_its_name(asked, name, named):
    from huggingface_hub import try_to_load_from_cache

    from core.epistemics import what_her_record_refutes
    from core.epistemics.where_a_request_is_done import program_it_names

    if not isinstance(try_to_load_from_cache(what_her_record_refutes.READER, "config.json"), str):
        pytest.skip("the inference model is not in the local cache")
    assert (program_it_names(asked, [name]) == name) is named


def test_without_the_reader_a_program_is_named_only_as_written_as_a_name(monkeypatch):
    from core.epistemics import what_her_record_refutes
    from core.epistemics.where_a_request_is_done import continues_there, program_it_names

    monkeypatch.setattr(what_her_record_refutes, "_reader", lambda: None)
    assert program_it_names("Open En Croissant and play", ["en-croissant"]) == "en-croissant"
    assert program_it_names("play a game of chess", ["Chess"]) == ""
    assert program_it_names("Open Chess", ["Chess"]) == "Chess"
    # And no page is taken to be gone on with on a guess.
    assert continues_there("Watch it on youtube.com", "Now another one.", "https://www.youtube.com") is False


def test_the_ceiling_reaches_her_browser_when_she_goes_on_with_it(monkeypatch):
    from core.phases.response_contract import requested_effect_ceiling

    pages, _asked = _reading_where(monkeypatch, there=True)
    monkeypatch.setattr(pages, "_their_earlier_requests", lambda _text: [_FIRST])
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


def test_the_place_a_conversation_goes_on_at_is_that_conversations(monkeypatch):
    """LIVE 25 November 2026: the third message of a new conversation went on at a page a message of another had named."""
    from core.conversation import page_interaction as pages
    from core.conversation import persistence
    from core.conversation.session_scope import conversation_session_scope

    class Kept:
        def get_recent_history(self, limit=100, **_who):
            return [{"role": "user", "session_id": "other", "content": "Look at en.wikipedia.org for me"},
                    {"role": "user", "session_id": "this", "content": "What's a first telescope?"},
                    {"role": "user", "session_id": "this", "content": "Would a dobsonian be too heavy for her?"}]

    monkeypatch.setattr(persistence, "get_persistence", Kept)
    with conversation_session_scope("this"):
        assert pages._their_earlier_requests("Would a dobsonian be too heavy for her?") == ["What's a first telescope?"]
    with conversation_session_scope("other"):
        assert pages._their_earlier_requests("And then?") == ["Look at en.wikipedia.org for me"]
    assert len(pages._their_earlier_requests("And then?")) == 3, "where it is not known which conversation it is, they are all read"
