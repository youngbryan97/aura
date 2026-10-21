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
