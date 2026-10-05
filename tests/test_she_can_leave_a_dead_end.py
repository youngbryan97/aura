"""A page that leads nowhere is left the way a person leaves it: an address, or the words looked up.

LIVE 2026-10-05: a museum's game page answered her browser with a block page.
She said the way on was to find the game elsewhere, and every action she could
choose was a control on the page in front of her. And on the list before it,
she worked out "number 10" counting from 0 and clicked control [10], the first
game, because nothing on the line said which game was number 10.
"""
from __future__ import annotations

from core.skills.sovereign_browser import SovereignBrowserSkill, _moves_from_the_decision
from core.skills.sovereign_browser_going import where_to_go


def test_an_address_is_gone_to_and_words_are_looked_up():
    assert where_to_go("https://archive.org/details/scooby-snapshot_flash") == "https://archive.org/details/scooby-snapshot_flash"
    assert where_to_go("archive.org/search?query=scooby") == "https://archive.org/search?query=scooby"
    assert where_to_go("www.example.com") == "https://www.example.com"
    assert where_to_go("Scooby Snapshot flash game internet archive") == (
        "https://html.duckduckgo.com/html/?q=Scooby+Snapshot+flash+game+internet+archive"
    )


def test_going_somewhere_needs_no_control_on_the_page():
    moves = _moves_from_the_decision({"actions": [{"type": "go", "value": "archive.org"}]}, [])
    assert [(action.type, action.value) for action, _said in moves] == [("go", "archive.org")]


def test_a_place_among_alike_items_is_given_both_ways_of_counting():
    observation = {"url": "u", "title": "t", "elements": [
        {"role": "link", "name": f"Game {n}", "selector": f"#g{n}", "alike": [n + 1, 6]} for n in range(6)
    ]}
    shown = SovereignBrowserSkill._render_observation(observation, "")
    assert "Game 0 (1 of 6 alike, number 0 counting from 0)" in shown
    assert "Game 5 (6 of 6 alike, number 5 counting from 0)" in shown
