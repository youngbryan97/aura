"""A setting the goal names is set to its number on a window of labelled controls, by the label nearest in meaning,
before the way on is taken (core/skills/screen_pursuit_a_setting.py).

Bryan, 27 November 2026: she plays the other player at the strength where her results vary. A game's setup shows the
other player's strength as a number beside a label, below the fold of its form.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import core.skills.screen_pursuit_a_setting as setting

pytestmark = pytest.mark.unit


def _form(depth: str) -> dict:
    """A setup form as a window's reading: a choice of who plays, a field beside a mode, and a row of choices."""
    regions = [("Human", 0.65, 0.20), ("Engine", 0.85, 0.20), ("Time Settings", 0.75, 0.26),
               ("Depth", 0.62, 0.40), (depth, 0.85, 0.40), ("Number of threads", 0.62, 0.48),
               ("1", 0.78, 0.48), ("2", 0.83, 0.48), ("4", 0.88, 0.48), ("8", 0.93, 0.48), ("Start game", 0.75, 0.70)]
    return {"layout": [{"text": t, "center_x": x, "center_y": y} for t, x, y in regions], "bounds": [0, 0, 1000, 800]}


def test_what_a_goal_sets_and_what_it_asks_done_are_read_apart() -> None:
    goal = "Play a game against the program, with the other player's strength set to 8."
    assert setting.settings_the_goal_sets(goal) == [("the other player's strength", "8")]
    assert setting.without_its_settings(goal) == "Play a game against the program."
    assert setting.settings_the_goal_sets("Play a game against the program.") == []


def test_each_number_goes_with_the_label_beside_it() -> None:
    beside = setting.numbers_beside_labels(_form("24"))
    assert [n["text"] for n in beside["Depth"]] == ["24"]
    assert [n["text"] for n in beside["Number of threads"]] == ["1", "2", "4", "8"]
    assert set(beside) == {"Depth", "Number of threads"}


class _Hand:
    def __init__(self, shown: dict) -> None:
        self.shown, self.done = shown, []

    async def scroll(self, dx=0, dy=0, *, at=None):
        self.done.append(("scroll", dy))

    async def hotkey(self, *keys, expect_app=""):
        self.done.append(("keys", keys))

    async def type_text(self, text, use_clipboard=True, *, expect_app=""):
        self.done.append(("type", text))
        self.shown["depth"] = text


def _set(monkeypatch, label_for, start: str = "24"):
    shown = {"depth": start}
    hand = _Hand(shown)
    clicks = []

    async def click(x, y, *, expect_app="", bounds=None):
        clicks.append((x, y))
        return True

    async def reading(read_screen, target_app, *, from_a_window=True):
        return _form(shown["depth"])

    async def chosen(target_app, name, labels):
        return labels.index(label_for) if label_for in labels else None

    import core.capabilities.host_automation as host_automation
    import core.skills.screen_pursuit as pursuit
    import core.skills.screen_pursuit_a_way_on as way_on
    import core.skills.screen_pursuit_surface as surface

    monkeypatch.setattr(host_automation, "get_host_automation", lambda: hand)
    monkeypatch.setattr(surface, "click_normalized", click)
    monkeypatch.setattr(way_on, "_a_reading", reading)
    monkeypatch.setattr(setting, "_the_label_for", chosen)
    monkeypatch.setattr(setting, "SHOWS_IN_S", 0.0)
    monkeypatch.setattr(pursuit, "_tell", lambda said: None)
    keep: dict = {}
    goal = "Play a game, with the other player's strength set to 8."
    done = asyncio.run(setting.set_what_the_goal_sets("A Board App", goal, keep, _form(start), None, True))
    return done, keep, hand, clicks


def test_the_field_beside_the_label_that_means_the_setting_is_typed_over_and_read_back(monkeypatch) -> None:
    done, keep, hand, clicks = _set(monkeypatch, "Depth")
    assert done and keep["settings_set"] == {"the other player's strength": "8"}
    assert clicks == [(0.85, 0.40)]
    assert ("keys", ("command", "a")) in hand.done and ("type", "8") in hand.done
    # Set once is not set again in the same pursuit.
    again = asyncio.run(setting.set_what_the_goal_sets("A Board App", "Play a game, with the other player's strength "
                                                       "set to 8.", keep, _form("8"), None, True))
    assert not again


def test_a_setting_not_on_the_form_is_looked_down_for_once_and_scrolled_back(monkeypatch) -> None:
    done, keep, hand, _clicks = _set(monkeypatch, "nothing here")
    assert not done
    downs = [d for kind, *rest in hand.done if kind == "scroll" for d in rest if d < 0]
    ups = [d for kind, *rest in hand.done if kind == "scroll" for d in rest if d > 0]
    assert len(downs) == setting.SCROLLS_AT_MOST and sum(ups) == -sum(downs)


def test_what_her_practice_changed_is_put_back_for_the_next_game_she_is_asked_to_play() -> None:
    from core.runtime.what_she_learned import recall

    name = "the other player's strength"
    marked = setting.FOR_HER_PRACTICE.set(True)
    try:
        setting._noted("A Board App", name, "24", "8")
        # A second practice game does not replace what the setting held before practice changed it.
        setting._noted("A Board App", name, "8", "4")
    finally:
        setting.FOR_HER_PRACTICE.reset(marked)
    assert recall(setting.PUT_BACK)["A Board App"] == {name: "24"}
    asked = setting.with_what_she_puts_back("Play a game against the program.", "A Board App")
    assert setting.settings_the_goal_sets(asked) == [(name, "24")]
    practice_goal = f"Play a game, with {name} set to 4."
    assert setting.with_what_she_puts_back(practice_goal, "A Board App") == practice_goal
    assert setting.with_what_she_puts_back("Play a game.", "Another App") == "Play a game."
    # Set back and read back: nothing is left to put back.
    setting._noted("A Board App", name, "4", "24")
    assert recall(setting.PUT_BACK)["A Board App"] == {}


def test_a_setting_the_person_asked_for_is_not_put_back_later() -> None:
    from core.runtime.what_she_learned import recall

    setting._noted("A Board App", "the other player's strength", "24", "10")
    assert not (recall(setting.PUT_BACK) or {}).get("A Board App")
