"""A name her eyes give a thing is taken once a second look gives it too, or where it is written.

LIVE 8 November 2026, rehearsal 9, a cartoon: its blue moose was "Lumpy" in five looks and "Pebbles" in one, a pink
character was "Marmalade" once, and her remarks named Marmalade and Pebbles.
"""
from __future__ import annotations

import inspect
from collections import Counter

from core.skills.watching_the_screen import OnTheirScreen, _a_name_given_again

SHOWN = "Happy Tree Friends - We're Scrooged (Ep #58)"


def test_a_name_given_once_is_a_character_and_one_given_again_is_kept():
    given: Counter[str] = Counter()
    first = _a_name_given_again("In this picture, we see **Marmalade** — a pink character — in a Santa hat.", SHOWN, given)
    assert "Marmalade" not in first and "we see a character" in first
    again = [_a_name_given_again(look, SHOWN, given) for look in (
        "Lumpy is standing; **Lumpy** (a blue moose) wears a tuxedo.",
        "From the show, the character **Lumpy** (a blue moose with antlers).",
        "we see the character **Pebbles** (a blue creature with a top hat).",
    )]
    assert again[0] == "A character is standing; a character (a blue moose) wears a tuxedo."
    assert "**Lumpy**" in again[1]
    assert again[2] == "we see a character (a blue creature with a top hat)."


def test_a_name_written_in_the_title_or_on_the_picture_is_kept_from_the_first():
    given: Counter[str] = Counter()
    said = _a_name_given_again("A **Happy** moment, and **Splendid** flies in.", f"{SHOWN} Splendid", given)
    assert said == "A **Happy** moment, and **Splendid** flies in."


def test_a_heading_her_eyes_bold_is_not_a_name():
    said = _a_name_given_again("**Setting**: a shop. **The characters**: two.", SHOWN, Counter())
    assert said == "**Setting**: a shop. **The characters**: two."


def test_each_look_on_their_screen_goes_through_it_with_its_own_count():
    source = inspect.getsource(OnTheirScreen)
    assert "self.names_given: Counter[str] = Counter()" in source
    assert "_a_name_given_again(seen," in source
