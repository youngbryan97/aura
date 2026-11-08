"""A link to something she has watched is shown to her with when she watched it.

LIVE 8 November 2026, rehearsal 10: asked to pick a video from a channel, she chose one she had watched six hours
before ("This lamp looks like a pair of scissors") and spoke of it as new: "I want to see how she justifies it".
"""
from __future__ import annotations

import time

import pytest

from core.conversation import our_shared_past
from core.conversation.our_shared_past import the_same_thing, when_we_watched
from core.skills.sovereign_browser import SovereignBrowserSkill

LAMP = "https://www.youtube.com/watch?v=r8fCHfCYDM8"


@pytest.mark.parametrize(("link", "watched", "same"), [
    ("https://youtube.com/watch?v=r8fCHfCYDM8&pp=ygUN", LAMP, True),  # where it was found from does not matter
    ("https://www.youtube.com/watch?v=OTHER", LAMP, False),  # which video does
    ("https://www.youtube.com/watch?v=9C_HReR_McQ", "https://www.youtube.com/watch?v=9C_HReR_McQ&list=RD9C&start_radio=1", True),
    ("https://www.youtube.com/watch", LAMP, False),
    ("https://videos.test/films/the-bread/", "https://videos.test/films/the-bread", True),
    ("https://videos.test/films/other", "https://videos.test/films/the-bread", False),
    ("https://elsewhere.test/watch?v=r8fCHfCYDM8", LAMP, False),
])
def test_two_addresses_name_one_thing_by_site_path_and_the_fields_both_give(link, watched, same):
    assert the_same_thing(link, watched) is same


def test_when_she_watched_it_is_said_in_her_words(monkeypatch):
    now = time.time()
    monkeypatch.setattr(our_shared_past, "_watchings", lambda _now: [(LAMP, now - 6 * 3600, "This lamp")])
    said = when_we_watched("https://youtube.com/watch?v=r8fCHfCYDM8", now=now)
    assert said.startswith("I watched this ")
    assert when_we_watched("https://www.youtube.com/watch?v=OTHER", now=now) == ""


def test_the_page_she_reads_marks_the_link_she_has_watched(monkeypatch):
    now = time.time()
    monkeypatch.setattr(our_shared_past, "_watchings", lambda _now: [(LAMP, now - 60, "This lamp")])
    page = {"url": "https://www.youtube.com/results?search_query=Simone+Giertz", "title": "results", "elements": [
        {"role": "link", "name": "This lamp looks like a pair of scissors 9 minutes, 11 seconds", "selector": "#a",
         "href": LAMP + "&pp=ygUN"},
        {"role": "link", "name": "I made a truck 14 minutes", "selector": "#b", "href": "https://www.youtube.com/watch?v=TRUCK"},
    ]}
    shown = SovereignBrowserSkill._render_observation(page, "pick one under ten minutes")
    lamp = next(line for line in shown.splitlines() if "This lamp" in line)
    truck = next(line for line in shown.splitlines() if "truck" in line)
    assert "I watched this today at" in lamp
    assert "I watched this" not in truck


def test_her_browser_records_where_each_link_leads():
    import inspect

    from core.capabilities import phantom_browser

    assert "entry.href = there.href" in inspect.getsource(phantom_browser)
