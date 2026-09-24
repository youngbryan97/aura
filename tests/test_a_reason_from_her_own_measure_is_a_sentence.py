"""A move chosen by a property she invented is explained as hers, in a sentence.

Live 2026-09-23: "Left — it scores better on how small it is between
neighbours, on average than right." The invented measure's name is a recipe,
and read into a frame meant for a noun it made no sentence.
"""

from __future__ import annotations

from core.agency.saying_what_a_move_does import why_this_one

INVENTED = "how small it is between neighbours, on average"


def test_an_invented_measure_is_named_as_hers():
    said = why_this_one({INVENTED: 0.8}, {INVENTED: 0.2}, {INVENTED: 0.4}, runner_up_name="right")
    assert said == f"by a measure I worked out myself ({INVENTED}), it comes out ahead of right"


def test_without_a_runner_up_it_still_reads():
    said = why_this_one({INVENTED: 0.8}, {INVENTED: 0.2}, {INVENTED: 0.4})
    assert said == f"by a measure I worked out myself ({INVENTED}), it comes out ahead"


def test_a_name_with_braces_does_not_break_the_sentence():
    odd = "how {big} it is"
    said = why_this_one({odd: 0.8}, {odd: 0.2}, {odd: 0.4}, runner_up_name="up")
    assert said.endswith("it comes out ahead of up")
    assert "{big}" in said


def test_her_written_terms_keep_their_own_words():
    said = why_this_one({"room": 0.8}, {"room": 0.2}, {"room": 1.0}, runner_up_name="up")
    assert said == "it leaves more room than up would"
