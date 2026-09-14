"""Somebody testifying is kept company, and somebody asking is helped.

The register decides testimony from the shape of the message and the sentiment
tracker decides how low it is. These pin the reading those two make together
and the three places it has to change what she does.
"""

from __future__ import annotations

import pytest

from core.expression.register import read
from core.social.witness import Witness, read_witness

COOKE = (
    "I was born by the river in a little tent. Oh, and just like the river "
    "I've been running ever since. It's been a long, a long time coming, but I "
    "know a change gonna come. It's been too hard living but I'm afraid to die. "
    "There's been times that I thought I couldn't last for long, but now I "
    "think I'm able to carry on."
)
REQUEST = "Can you help me figure out why my build keeps failing? What should I check first?"


def _modifiers(text: str, valence: float | None) -> dict:
    shape = read(text)
    out = {
        "register": shape.as_dict(),
        "asks_to_be_witnessed": shape.asks_to_be_witnessed(),
        "asks_for_help": shape.asks_for_help(),
    }
    if valence is not None:
        out["user_sentiment"] = {"valence": valence}
    return out


def test_testimony_from_a_low_place_is_company_as_deep_as_the_low() -> None:
    reading = read_witness(_modifiers(COOKE, -0.6))
    assert reading.witnessing
    assert reading.low == pytest.approx(0.6)
    assert reading.company == pytest.approx(0.6)


def test_a_request_is_helped_however_low_it_is() -> None:
    reading = read_witness(_modifiers(REQUEST, -0.8))
    assert reading.measured
    assert not reading.witnessing
    assert reading.company == 0.0
    assert "asks for something" in reading.why


def test_testimony_that_is_not_low_is_still_witnessed_without_company() -> None:
    reading = read_witness(_modifiers(COOKE, 0.4))
    assert reading.witnessing
    assert reading.company == 0.0


def test_an_unread_sentiment_is_named_rather_than_read_as_neutral() -> None:
    reading = read_witness(_modifiers(COOKE, None))
    assert reading.witnessing
    assert reading.low == 0.0
    assert "was not read" in reading.why


def test_no_register_reading_claims_nothing() -> None:
    assert read_witness({}) == Witness()
    assert read_witness(None) == Witness()
    assert not Witness().measured


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = read_witness(_modifiers(COOKE, -0.3)).as_dict()
    for key in ("testimony", "low", "witnessing", "company", "measured", "why"):
        assert key in row, key
