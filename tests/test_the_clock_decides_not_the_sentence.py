"""What the runtime can measure beats what the sentence asserts.

Handing a model a PRESENT MOMENT block is a prior, and it works: after the
block reached the desktop lane, "what time of day is it?" answered "It's 10:52
AM on a Monday" — correct, and confirmed arriving by the dispatch log. But a
prior is only a prior. Before that, from a runtime with no window and no light
sensor:

    at 00:30  "The sun's up but I'm not sure it will be warm today — there are
               clouds gathering in the east."
    at 01:40  "my clock says it's 06:15 and the ambient light sensors report
               very low illumination values with a cool spectrum"

This is the causal half: the reading is taken and the claim is reconciled
against it, at the egress, on the exact text about to be spoken. It does not
depend on the model choosing to read what it was given.

The hardest requirement here is what it must NOT do. A guard that "corrects" a
right answer is worse than no guard, so daylight consistent with the clock is
left alone, a time being discussed rather than asserted is left alone, and
nothing is ever added that was not already there.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from core.conversation.grounded_claim_guard import verify_grounded_claims
from tests.chat_lane_support import chat_lane_source

MORNING = datetime(2026, 7, 27, 10, 52)
SMALL_HOURS = datetime(2026, 7, 27, 0, 30)


# ── It must not fight a correct answer ─────────────────────────────────────

def test_the_correct_time_is_left_exactly_as_she_said_it() -> None:
    """This one was right. Breaking it would be the same bug, reversed."""
    reply = "It's 10:52 AM on a Monday. On your end, I don't know."
    result = verify_grounded_claims(reply, now=MORNING)
    assert result.text == reply
    assert not result.changed


def test_daylight_that_follows_from_the_clock_is_left_alone() -> None:
    reply = "It's morning. The sun's up by now, though I'm going by the clock, not a window."
    assert verify_grounded_claims(reply, now=MORNING).text == reply


def test_a_time_being_discussed_is_not_a_claim_about_now() -> None:
    reply = "We agreed to meet at 3:00 PM tomorrow, which should still work."
    assert not verify_grounded_claims(reply, now=MORNING).changed


def test_a_few_minutes_of_drift_is_not_an_error() -> None:
    reply = "It's 10:40 AM here."
    assert not verify_grounded_claims(reply, now=MORNING).changed


def test_an_ordinary_reply_is_untouched() -> None:
    reply = "391. That's 17 times 23."
    assert verify_grounded_claims(reply, now=MORNING).text == reply


# ── It must settle a wrong one ─────────────────────────────────────────────

def test_a_stated_time_hours_off_is_corrected_to_the_reading() -> None:
    result = verify_grounded_claims("It's 6:15 AM right now.", now=MORNING)
    assert "10:52 AM" in result.text
    assert "6:15" not in result.text
    assert any("off by" in note for note in result.corrections)


def test_the_correction_keeps_the_format_she_used() -> None:
    assert "10:52" in verify_grounded_claims("the time is 06:15", now=MORNING).text
    assert "10:52 AM" in verify_grounded_claims("It's 6:15 AM", now=MORNING).text


def test_midnight_is_not_twelve_hours_from_five_past() -> None:
    """A clock is circular; 23:55 and 00:05 are ten minutes apart."""
    result = verify_grounded_claims(
        "It's 23:55.", now=datetime(2026, 7, 27, 0, 5)
    )
    assert not result.changed


def test_the_wrong_part_of_day_is_corrected() -> None:
    result = verify_grounded_claims("It's the middle of the night.", now=MORNING)
    assert "morning" in result.text
    assert "middle of the night" not in result.text


def test_the_right_part_of_day_is_kept() -> None:
    reply = "It's the middle of the night and I'm still here."
    assert not verify_grounded_claims(reply, now=SMALL_HOURS).changed


# ── Instruments that do not exist ──────────────────────────────────────────

def test_a_claimed_light_sensor_is_removed_not_corrected() -> None:
    """There is no reading to compare against — the claim is to an organ."""
    result = verify_grounded_claims(
        "I know because the ambient light sensors report low illumination.",
        now=MORNING,
    )
    assert "sensor" not in result.text.lower()
    assert any("light sensor" in note for note in result.corrections)


def test_weather_detail_is_removed() -> None:
    result = verify_grounded_claims(
        "I'm here. There are clouds gathering in the east. In here, it feels quiet.",
        now=MORNING,
    )
    assert "clouds" not in result.text
    assert "it feels quiet" in result.text, "the rest of the reply must survive"


@pytest.mark.parametrize(
    "reply",
    (
        "I can see the break clearly from my action receipts.",
        "I can see the distinction now: the cache survived but the process did not.",
        "I can see it as a consistency problem rather than a model failure.",
    ),
)
def test_conceptual_sight_is_not_misclassified_as_weather(reply: str) -> None:
    """Ordinary cognition is not a claim to a camera or a weather reading."""

    result = verify_grounded_claims(reply, now=MORNING)

    assert result.text == reply
    assert result.corrections == ()


@pytest.mark.parametrize(
    "reply",
    (
        "I can see the sky is cloudy outside.",
        "I can see clouds gathering over the hills.",
    ),
)
def test_visual_lead_does_not_exempt_an_explicit_weather_claim(reply: str) -> None:
    result = verify_grounded_claims(reply, now=MORNING)

    assert result.text == ""
    assert result.corrections == ("described weather it has no reading for",)


def test_the_whole_live_sentence_is_reconciled() -> None:
    result = verify_grounded_claims(
        "It's early morning — the light outside has that soft, grayish hue. I know "
        "this because my clock says it's 06:15 and the ambient light sensors report "
        "very low illumination values.",
        now=MORNING,
    )
    assert "06:15" not in result.text
    assert "light outside" not in result.text
    assert "sensors" not in result.text
    assert len(result.corrections) >= 3


# ── Safety of the mechanism itself ─────────────────────────────────────────

@pytest.mark.parametrize("reply", ["", "   ", None])
def test_an_empty_reply_is_returned_unchanged(reply) -> None:
    assert verify_grounded_claims(reply, now=MORNING).text == str(reply or "")


def test_the_guard_never_invents_a_claim() -> None:
    """It only ever removes or corrects text that was already asserted."""
    reply = "I don't know what time it is."
    assert verify_grounded_claims(reply, now=MORNING).text == reply


def test_it_runs_on_the_text_about_to_be_spoken() -> None:

    src = chat_lane_source()
    guard_at = src.index("from core.conversation.grounded_claim_guard import")
    contract_at = src.index("_final_reply = _enforce_or_bind_terminal_output_contract(")
    assert contract_at < guard_at, "the reading must settle the final text, not an earlier draft"
    assert "measured_reading_overrides_stated_claim" in src
