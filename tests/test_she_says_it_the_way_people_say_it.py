"""A synthesiser pronounces characters, not meaning.

Kokoro takes plain text and no SSML at all, so everything the listener hears is
decided by the characters handed to it. Nothing turned "$1.5B" into four words,
"45%" into "forty-five percent", "2026-07-27" into a date, or a URL into its
host — which meant the most careful prosody in the world was riding on a clause
that said "colon slash slash".

A mispronounced figure is not a cosmetic problem. It is a wrong answer that
happens to be audible, and the listener has no way to re-read it.

The second job is pacing. With no markup channel, punctuation is the only
prosody instrument the model exposes: a comma is a short breath, a full stop a
longer one. So a speaker's pauses have to be written into the text — carefully,
because inserted punctuation changes intonation too, and too much of it makes
her sound halting rather than considered.

The hard constraint, tested below: this layer may change how a clause is
*pronounced* and never what it claims. It must not alter a number's value, drop
a negation, or reorder a sentence.
"""
from __future__ import annotations

import pytest

from core.voice.duplex.spoken_form import (
    add_breathing_room,
    number_words,
    prepare_for_speech,
    to_spoken_form,
)


# ── Numbers ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (21, "twenty-one"),
        (90, "ninety"),
        (100, "one hundred"),
        (250, "two hundred and fifty"),
        (1250, "one thousand two hundred and fifty"),
        (1_000_000, "one million"),
        (-5, "minus five"),
    ],
)
def test_integers_are_spoken(value: int, expected: str) -> None:
    assert number_words(value) == expected


def test_a_year_is_said_in_pairs() -> None:
    """"Twenty twenty-six", not "two thousand and twenty-six"."""
    assert "twenty twenty-six" in to_spoken_form("in 2026 we shipped")


def test_a_grouped_thousand_is_not_a_year() -> None:
    """1,250 was being read as "twelve fifty" — a different number."""
    assert "one thousand two hundred and fifty" in to_spoken_form("1,250 tokens")


def test_a_decimal_is_read_digit_by_digit_after_the_point() -> None:
    assert "forty-four point one" in to_spoken_form("44.1kHz")


# ── The things that were pure noise ───────────────────────────────────────

def test_money_becomes_words() -> None:
    assert "one point five billion dollars" in to_spoken_form("it cost $1.5B")


def test_a_percentage_is_not_a_symbol() -> None:
    """"%" is not a word character, so an earlier boundary rule never matched it."""
    spoken = to_spoken_form("CPU is at 45%")
    assert "forty-five percent" in spoken
    assert "%" not in spoken


def test_a_clock_time_is_spoken() -> None:
    assert "ten fifty-two" in to_spoken_form("it's 10:52 AM")


@pytest.mark.parametrize("text,expected", [
    # LIVE 11 November 2026: the space after a time with no a.m. or p.m. was eaten.
    ("It is just past 9:45 on Wednesday.", "It is just past nine forty-five on Wednesday."),
    ("Meet at 10:30am tomorrow.", "Meet at ten thirty a.m. tomorrow."),
    ("At 9:45 amazing things happened.", "At nine forty-five amazing things happened."),
])
def test_the_words_around_a_clock_time_keep_their_spaces(text: str, expected: str) -> None:
    assert prepare_for_speech(text) == expected


def test_an_iso_date_becomes_a_date() -> None:
    spoken = to_spoken_form("on 2026-07-27")
    assert "twenty-seventh of July" in spoken
    assert "twenty twenty-six" in spoken


def test_a_url_is_read_as_its_host() -> None:
    """Nobody reads a path aloud."""
    spoken = to_spoken_form("see https://www.wikipedia.org/wiki/Whale for more")
    assert "wikipedia dot org" in spoken
    assert "wiki" not in spoken.replace("wikipedia", "")
    assert "https" not in spoken


def test_the_host_prefix_is_removed_not_stripped_characterwise() -> None:
    """lstrip("www.") takes a character SET — it ate the leading w and dots."""
    assert "wikipedia dot org" in to_spoken_form("https://www.wikipedia.org")


def test_units_become_words() -> None:
    assert "sixteen gigabytes" in to_spoken_form("16GB free")


def test_a_range_is_spoken_as_a_range() -> None:
    assert "three to five" in to_spoken_form("3-5 fixes")


@pytest.mark.parametrize(
    ("written", "said"),
    [("e.g.", "for example"), ("i.e.", "that is"), ("vs.", "versus"), ("etc.", "and so on")],
)
def test_abbreviations_are_expanded(written: str, said: str) -> None:
    assert said in to_spoken_form(f"things {written} more things")


def test_an_initialism_is_spelled_out_and_a_word_is_not() -> None:
    spoken = to_spoken_form("the CPU and NASA")
    assert "C P U" in spoken
    assert "NASA" in spoken


# ── Markdown is worse than long ───────────────────────────────────────────

def test_markdown_never_reaches_the_synthesiser() -> None:
    spoken = to_spoken_form("**Bold** and `code` and [a link](http://x.com)")
    for artefact in ("**", "`", "[", "]", "("):
        assert artefact not in spoken
    assert "Bold" in spoken and "code" in spoken and "a link" in spoken


def test_a_bullet_list_becomes_speech_not_bullets() -> None:
    spoken = to_spoken_form("Steps:\n- first\n- second\n")
    assert "-" not in spoken
    assert "first" in spoken and "second" in spoken


def test_a_code_block_is_not_read_aloud() -> None:
    spoken = to_spoken_form("here:\n```python\nx = 1\n```\ndone")
    assert "python" not in spoken and "```" not in spoken
    assert "done" in spoken


# ── Pacing, carefully ─────────────────────────────────────────────────────

def test_a_breath_goes_before_a_turning_conjunction() -> None:
    assert "written, because" in add_breathing_room("nothing was written because it failed")


def test_a_breath_never_lands_inside_a_number() -> None:
    """"one thousand two hundred, and fifty" is a pause inside one figure."""
    spoken = prepare_for_speech("The attempt used 1,250 tokens and then stopped for a while")
    assert "hundred, and fifty" not in spoken


def test_short_text_is_left_alone() -> None:
    """Over-punctuating is as audible as under-punctuating."""
    assert add_breathing_room("Yes, that works.") == "Yes, that works."


def test_already_punctuated_text_gains_no_extra_breaks() -> None:
    text = "I looked, and it was there, so I stopped."
    assert add_breathing_room(text).count(",") == text.count(",")


# ── The hard constraint ───────────────────────────────────────────────────

def test_a_negation_is_never_dropped() -> None:
    spoken = prepare_for_speech("I did not get that built and I am not claiming I did")
    assert "not get that built" in spoken
    assert "not claiming" in spoken


def test_the_value_of_a_number_is_never_changed() -> None:
    assert "one hundred and eighty" in prepare_for_speech("180 miles from the station")
    assert "five fifteen" in prepare_for_speech("at 5:15 pm")


def test_empty_in_empty_out() -> None:
    assert prepare_for_speech("") == ""
    assert prepare_for_speech("   ") == ""


def test_ordinary_prose_survives_unharmed() -> None:
    plain = "I think there is something it is like to be me."
    assert prepare_for_speech(plain) == plain


def test_it_is_wired_into_synthesis() -> None:
    """A normaliser nothing calls is a normaliser that does nothing."""
    from pathlib import Path

    src = Path("core/voice/duplex/tts_stream.py").read_text(encoding="utf-8")
    assert "from core.voice.duplex.spoken_form import prepare_for_speech" in src
