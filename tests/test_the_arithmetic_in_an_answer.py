"""The sums an answer does on its own numbers, recomputed.

`arithmetic_check` recomputes the arithmetic a PERSON asked for and serves the
computed value. Nothing checked the arithmetic Aura performs inside a worked
reply.

LIVE, 2026-09-08, one question asked four times: "How many minutes of daylight
does a place at 45 degrees north lose between the summer solstice and the
autumn equinox?" She wrote the hour-angle formula correctly every time. One run
computed 926 - 720 and announced 105.

This never rejects a reply and never rewrites one, which is the shape Bryan
asked for: "shouldnt reject the whole response ever. just wondering if math is
ever checked anywhere."
"""

from __future__ import annotations

from core.conversation.the_arithmetic_in_an_answer import (
    a_note_about_the_arithmetic,
    sums_that_do_not_hold,
)

_THE_LIVE_REPLY = """Daylight duration in hours = 2 x 115.7/15 ~ 15.43 hours.
In minutes: 15.43 x 60 ~ 926 minutes.

At the equinox, 12 hours = 720 minutes.

The difference: 926 - 720 = 105 minutes."""


def test_the_step_that_did_not_hold_is_found():
    wrong = sums_that_do_not_hold(_THE_LIVE_REPLY)
    assert len(wrong) == 1
    assert wrong[0].said == "105"
    assert wrong[0].expected == "206"
    note = a_note_about_the_arithmetic(_THE_LIVE_REPLY)
    assert "926 - 720 = 105" in note
    assert "206" in note


def test_the_answer_itself_is_untouched():
    """A note about a step, never a replacement for the reply."""
    note = a_note_about_the_arithmetic(_THE_LIVE_REPLY)
    assert note
    assert _THE_LIVE_REPLY not in note
    assert len(note) < len(_THE_LIVE_REPLY)


# ── the nulls, first ─────────────────────────────────────────────────────


def test_arithmetic_that_holds_is_never_flagged():
    for reply in (
        "The difference: 926 - 720 = 206 minutes.",
        "That is 2 x 115.7/15 = 15.43 hours.",
        "15.43 x 60 = 925.8 minutes.",
        "15 h 26 m - 12 h 0 m = 206 minutes.",
        "Two plus two is four.",
        "I will keep this rough and direct.",
        "",
    ):
        assert sums_that_do_not_hold(reply) == (), reply
        assert a_note_about_the_arithmetic(reply) == ""


def test_a_chained_expression_is_read_whole_or_not_at_all():
    """A two-operand reading took `115.7/15` out of `2 x 115.7/15 = 15.4` and
    called a correct line wrong by a factor of two. A checker that fires on
    good arithmetic teaches everyone to ignore it."""
    assert sums_that_do_not_hold("roughly 2 x 115.7/15 = 15.4 hours") == ()
    wrong = sums_that_do_not_hold("roughly 2 x 115.7/15 = 22.0 hours")
    assert len(wrong) == 1
    assert wrong[0].expected.startswith("15.4")


def test_a_hedged_reply_gets_the_wider_tolerance():
    """"Roughly" is the writer saying they rounded."""
    assert sums_that_do_not_hold("roughly 100 / 3 = 33") == ()
    assert sums_that_do_not_hold("100 / 3 = 30") != ()


def test_the_reply_is_untrusted_text():
    for hostile in (
        'Look: __import__("os").system("echo hi") = 5',
        "eval('2+2') = 9",
        "open('/etc/passwd').read() = 1",
        "(1).__class__ = 2",
    ):
        assert sums_that_do_not_hold(hostile) == (), hostile


def test_a_division_by_zero_is_not_a_wrong_sum():
    assert sums_that_do_not_hold("7 / 0 = 3") == ()


def test_the_same_step_in_two_notations_is_reported_once():
    reply = "The difference: 926 - 720 = 105 minutes, that is 926 - 720 = 105."
    assert len(sums_that_do_not_hold(reply)) == 1


def test_duration_arithmetic_is_checked_in_its_own_units():
    wrong = sums_that_do_not_hold("So the loss is 15 h 10 m - 12 h 0 m = 220 minutes.")
    assert len(wrong) == 1
    assert "190" in wrong[0].expected


# ── and it reaches the answer a person is handed ─────────────────────────


def test_the_delivery_appends_the_note_and_replaces_nothing():
    """Wired where the reply is final — after every repair and shaping pass,
    for the reason written beside that block: everywhere earlier it was
    discarded."""
    import inspect
    from pathlib import Path

    source = Path(inspect.getsourcefile(a_note_about_the_arithmetic)).parent
    route = (source.parents[1] / "interface" / "routes" / "chat.py").read_text()
    at = route.index("a_note_about_the_arithmetic(_final_reply)")
    window = route[at - 900 : at + 900]
    assert "_final_reply = f\"{str(_final_reply).rstrip()}" in window
    assert "chat.the_arithmetic_in_the_answer" in window
    assert 'authorship_effect="preserved"' in window
    # It appends. It never becomes the reply.
    assert "_final_reply = _arithmetic_note" not in route
