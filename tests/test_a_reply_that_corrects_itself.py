"""The reply withdrew its own opening and delivered it anyway.

LIVE, 2026-09-08. Asked how much daylight 45°N loses between the solstice and
the equinox, Aura opened with "Roughly **105 minutes**", worked through the
hour-angle formula, wrote "My previous 105 was a mental slip", and gave 210.
All three sentences reached the screen, with the withdrawn one at the top.
A reader who reads the first line and stops — most readers, and every audience
watching a demonstration — takes away a number she had already withdrawn.
"""

from __future__ import annotations

from core.conversation.a_reply_that_corrects_itself import (
    the_reply_corrects_its_own_headline,
)

_THE_LIVE_REPLY = """Roughly **105 minutes**, or about 1 hour and 45 minutes.

Here is the working: at 45 degrees north the sun's declination moves from
+23.5 to 0, so daylight runs about 15.5 hours at the solstice and 12 at the
equinox.

My previous "105" was a mental slip (halving the difference incorrectly). The
correct rough figure is **210 minutes** (3.5 hours)."""


def test_the_withdrawn_opening_is_marked_as_withdrawn():
    corrected = the_reply_corrects_its_own_headline(_THE_LIVE_REPLY)
    assert corrected is not None
    assert corrected.superseded == "105"
    assert corrected.corrected == "210"
    assert corrected.text.startswith("~~Roughly **105 minutes**")
    assert "~~" in corrected.text.split("\n")[0]


def test_nothing_of_hers_is_removed_or_rewritten():
    """Her words, and only a mark on them. Rewriting the value would leave
    'or about 1 hour and 45 minutes' restating the withdrawn one, and putting
    that right means recomputing her answer."""
    corrected = the_reply_corrects_its_own_headline(_THE_LIVE_REPLY)
    assert corrected is not None
    without_marks = corrected.text.replace("~~", "")
    assert without_marks.strip() == _THE_LIVE_REPLY.strip()


def test_a_reply_that_retracts_nothing_is_untouched():
    for reply in (
        "The answer is 210 minutes, and the working below agrees.",
        "Roughly 105 minutes.\n\nThe working: 15.5 hours less 12 hours.",
        "",
        "   ",
    ):
        assert the_reply_corrects_its_own_headline(reply) is None


def test_a_correction_of_something_never_said_up_front_is_left_alone():
    """The opening has to be what was withdrawn. A correction about a value
    from the middle of the working is the working doing its job."""
    assert the_reply_corrects_its_own_headline(
        "Here is the plan.\n\nCorrection: the 5 above should have been 9."
    ) is None


def test_a_correction_to_the_same_value_changes_nothing():
    assert the_reply_corrects_its_own_headline(
        "It is 210 minutes.\n\nThe correct figure is 210 minutes."
    ) is None


def test_the_look_back_stops_at_the_marker_s_own_sentence():
    """A fixed look-back read '45 degrees north' out of the working above the
    retraction and called that the withdrawn value."""
    corrected = the_reply_corrects_its_own_headline(_THE_LIVE_REPLY)
    assert corrected is not None
    assert corrected.superseded != "45"


def test_a_retraction_with_no_replacement_is_left_alone():
    assert the_reply_corrects_its_own_headline(
        "The total is 105.\n\nThat figure above was wrong; I will redo it."
    ) is None


def test_the_headline_ends_at_the_first_blank_line():
    """A fixed 400-character window swallowed the start of the working, so a
    value from the working counted as a value from the headline."""
    reply = (
        "It comes to 105.\n\n"
        + "Working: " + ("a" * 600) + " 77 units.\n\n"
        + 'My previous "77" was a slip. The correct figure is 88.'
    )
    # 77 is not in the opening claim, so nothing is struck.
    assert the_reply_corrects_its_own_headline(reply) is None
