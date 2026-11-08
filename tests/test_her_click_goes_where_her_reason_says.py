"""A click whose number names one control and whose reason names another goes where her reason says.

LIVE 8 November 2026, dress rehearsal: "I'm clicking into the bathroom lock-in because it's the one video that's about a
person" chose the index of "Popular videos"; "I'm clicking the brain tumor video" chose an index off the list.
"""

from core.skills.sovereign_browser import SovereignBrowserSkill

ELEMENTS = [
    {"name": "Popular videos", "selector": "#popular"},
    {"name": "I Got Locked In My Bathroom For Three Days", "selector": "#bathroom"},
    {"name": "Things I've built that aren't shitty robots", "selector": "#robots"},
    {"name": "My brain tumor", "selector": "#tumor"},
    {"name": "4:22 Now playing", "selector": "#thumb"},
]


def _decide(index, why):
    return {"actions": [{"index": index, "type": "click"}], "why": why}


def test_a_number_that_names_another_control_goes_to_the_one_she_gave_her_reason_for():
    meant = SovereignBrowserSkill._the_one_she_means(
        _decide(0, "I'm clicking into the bathroom lock-in, the one where she's locked in her bathroom."), ELEMENTS)
    assert meant["actions"][0]["index"] == 1


def test_a_number_off_the_list_goes_to_the_one_she_named():
    meant = SovereignBrowserSkill._the_one_she_means(
        _decide(41, "I'm clicking the brain tumor video; a brain tumor reframes the channel."), ELEMENTS)
    assert meant["actions"][0]["index"] == 3


def test_a_number_her_reason_agrees_with_stays():
    decided = _decide(2, "The robots she built that aren't shitty, over the bathroom one.")
    assert SovereignBrowserSkill._the_one_she_means(decided, ELEMENTS)["actions"][0]["index"] == 2


def test_a_reason_that_names_nothing_leaves_the_number():
    decided = _decide(4, "That one looks good to me.")
    assert SovereignBrowserSkill._the_one_she_means(decided, ELEMENTS)["actions"][0]["index"] == 4


def test_two_controls_named_as_well_as_each_other_leave_the_number():
    elements = [{"name": "Robot arm build"}, {"name": "Robot arm repair"}, {"name": "Settings"}]
    decided = _decide(2, "The robot arm one.")
    assert SovereignBrowserSkill._the_one_she_means(decided, elements)["actions"][0]["index"] == 2


def test_a_click_on_a_picture_that_names_nothing_stays_where_it_is():
    """LIVE 8 November 2026: a click on the thumbnail of the video she described was taken to its description."""
    elements = [{"name": "4:05\nNow playing"}, {"name": "Joel Haver portrays two knights observing a mysterious steed"}]
    decided = _decide(0, "Two knights, a cryptic steed, foreboding visions - the one that does something with tone.")
    assert SovereignBrowserSkill._the_one_she_means(decided, elements)["actions"][0]["index"] == 0
