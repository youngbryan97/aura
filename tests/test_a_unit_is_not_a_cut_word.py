"""A measurement ending a sentence is not a truncated tail.

LIVE, 2026-08-20. Asked to read an endpoint and report the temperature, the
repair pass produced

    "The temperature reported by the API is 11.6°C."

which was correct, complete, and fetched from the document she had been asked
to read. The reliability gate refused it as ``truncated_tail`` and the person
got "I couldn't get to an answer I'd stand behind on that one."

The rule is right in general — a reply ending in a one- or two-letter word is
usually cut mid-word — and a unit is the exception. Which words are units is
not a vocabulary question: they are identified by the number they attach to.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import _has_truncated_tail


@pytest.mark.parametrize(
    "reply",
    [
        "The temperature reported by the API is 11.6°C.",
        "The temperature reported by the API is 11.6C.",
        "The signal came back at a frequency of exactly 60 Hz.",
        "Total mass measured across all of the samples was 3.5 kg.",
        "The distance between those two points came to about 42 km.",
        "The reading held steady overnight at roughly 21.4 °C.",
        "Peak throughput across the run settled at 120 MB.",
    ],
)
def test_a_measurement_completes_a_sentence(reply: str) -> None:
    assert _has_truncated_tail(reply) is False, reply


@pytest.mark.parametrize(
    "reply",
    [
        "I looked at the data and then decided to go to th",
        "The value we computed from the whole table is ab",
        "After checking every row of the spreadsheet the total is un",
    ],
)
def test_a_word_cut_in_half_is_still_caught(reply: str) -> None:
    assert _has_truncated_tail(reply) is True, reply


def test_the_unit_test_is_structural_not_a_vocabulary() -> None:
    """A unit nobody listed still works, because the number identifies it."""
    from core.conversation.response_reliability import _terminal_word_is_a_unit

    body = "The reading came back as 44 qx"
    assert _terminal_word_is_a_unit(body, len(body) - 2) is True

    prose = "The reading came back as something we do not"
    assert _terminal_word_is_a_unit(prose, len(prose) - 2) is False
