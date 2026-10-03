"""An index page has to offer the link the errand names.

LIVE 2026-09-28. Asked to take the Open Extended Jungian Type Scales on
openpsychometrics.org, she arrived at the site root — an index of forty-odd
tests with no form on it — and the controls she was offered were ranked by what
they DO and then by document order. Links rank last by role, the budget is forty
built for questionnaires, and the link she had been sent for sat past it. The
page she could see held no test and no way to one.

Two other things the observer already knew and never said: that the list had
been cut at all, and that the page continued below the fold.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit

_GOAL = (
    "Take the Open Extended Jungian Type Scales personality test on "
    "openpsychometrics.org"
)


def _an_index_page() -> dict:
    """A directory: many links, the wanted one far down, as the live page is."""
    links = [
        {"role": "link", "name": name, "selector": f"a:nth-of-type({n + 1})"}
        for n, name in enumerate(
            [f"Filler Quiz {n}" for n in range(60)]
        )
    ]
    links[55] = {
        "role": "link",
        "name": "Open Extended Jungian Type Scales",
        "selector": "a:nth-of-type(56)",
    }
    return {
        "url": "https://openpsychometrics.org",
        "title": "Open-Source Psychometrics Project",
        "text": "This website provides a collection of interactive personality tests.",
        "elements": links,
        "scroll_y": 0,
        "scroll_height": 4000,
        "viewport_height": 800,
    }


def test_the_link_the_goal_names_is_offered():
    offered = SovereignBrowserSkill._controls_worth_offering(
        _an_index_page()["elements"], _GOAL
    )
    names = [str(element.get("name")) for element in offered]
    assert "Open Extended Jungian Type Scales" in names, (
        "the way in was not among the controls she was shown"
    )


def test_what_is_offered_keeps_the_order_the_page_gave_it():
    """The ranking chooses what makes the list; the page orders it.

    LIVE 2026-10-03 06:14: listed by rank, the games titled "Cartoon Network:
    ..." came first, and asked to count to the fifth game on the page she
    picked the fifth of her list instead.
    """
    elements = _an_index_page()["elements"]
    offered = SovereignBrowserSkill._controls_worth_offering(elements, _GOAL)
    places = [elements.index(element) for element in offered]
    assert places == sorted(places)
    assert "Open Extended Jungian Type Scales" in [str(e.get("name")) for e in offered]


def test_without_a_goal_the_old_order_is_unchanged():
    """Ranking by role and document order is still what an errandless list is."""
    offered = SovereignBrowserSkill._controls_worth_offering(
        _an_index_page()["elements"]
    )
    assert str(offered[0].get("name")) == "Filler Quiz 0"


def test_a_form_still_wins_its_place_over_links_when_there_is_no_room():
    """The goal's words break ties; they do not turn a survey into a nav bar."""
    budget = SovereignBrowserSkill.PURSUE_CONTROL_BUDGET
    elements = [
        {"role": "link", "name": f"Elsewhere {n}", "selector": f"a:nth-of-type({n})"}
        for n in range(budget)
    ] + [{"role": "radio", "name": "I agree", "group": "q1", "selector": "#q1a"}]
    offered = SovereignBrowserSkill._controls_worth_offering(elements, "answer q1")
    assert any(str(element.get("role")) == "radio" for element in offered)


def test_the_page_says_when_the_list_was_cut_and_when_it_continues():
    rendered = SovereignBrowserSkill._render_observation(_an_index_page(), _GOAL)
    assert "not listed" in rendered, "a cut list must say it was cut"
    assert "continues" in rendered, "a page with more below must say so"


def test_a_whole_short_page_claims_neither():
    page = {
        "url": "https://example.test",
        "title": "short",
        "text": "all of it",
        "elements": [{"role": "button", "name": "Next", "selector": "#next"}],
        "scroll_y": 0,
        "scroll_height": 700,
        "viewport_height": 800,
    }
    rendered = SovereignBrowserSkill._render_observation(page, "press next")
    assert "not listed" not in rendered
    assert "continues" not in rendered


def test_every_index_is_resolved_against_the_same_ranking():
    """`[3]` must name one control on screen and the same one in the click."""
    import inspect

    from core.skills import sovereign_browser, sovereign_browser_understanding

    for module in (sovereign_browser, sovereign_browser_understanding):
        source = inspect.getsource(module)
        for call in (
            "_render_observation(observation)",
            "_decision_is_usable(raw, observation)",
            "_controls_worth_offering(list(observation.get(\"elements\") or []))",
        ):
            assert call not in source, (
                f"{call} resolves its indices without the goal the other sites "
                "rank by, so one list is ordered two ways"
            )
