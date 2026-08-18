"""Arithmetic written with units is still arithmetic.

Live 2026-08-18. Asked how many seconds are in 3 years 7 months 12 days, she
reasoned it perfectly — 1095 + 210 + 12 = 1317 days, 86400 seconds in a day —
and then wrote:

    1317 days * 86400 seconds/day = 113,923,200 seconds

The product is 113,788,800. Every step of the method was right and the
arithmetic was wrong, so nothing that reads the METHOD could catch it, and the
question itself contained no sum for the requested-arithmetic verifier to
check.

`SymbolicBridge.check_arithmetic_claims` exists for exactly this and did not
fire. It was not broken: written bare, the same equation WAS caught. Its
pattern allowed only whitespace around operators, so a unit word between the
number and the operator ended the match — and a unit between the number and
the operator is how shown work is written. The check worked and had simply
never seen the sentence a person writes.
"""

from __future__ import annotations

import pytest

from core.reasoning.symbolic_bridge import SymbolicBridge

#: Verbatim from the live reply.
LIVE_LINE = "1317 days * 86400 seconds/day = 113,923,200 seconds"


@pytest.fixture()
def bridge() -> SymbolicBridge:
    return SymbolicBridge()


def test_the_equation_that_reached_the_user_is_caught(bridge):
    errors = bridge.check_arithmetic_claims(LIVE_LINE)
    assert len(errors) == 1
    assert errors[0]["stated"] == pytest.approx(113_923_200)
    assert errors[0]["correct"] == pytest.approx(113_788_800)


def test_correct_working_with_units_stays_silent(bridge):
    """Over-flagging her own correct reasoning is the same defect inverted."""
    assert bridge.check_arithmetic_claims(
        "1317 days * 86400 seconds/day = 113,788,800 seconds"
    ) == []


@pytest.mark.parametrize(
    "line",
    [
        "12 apples + 5 apples = 18 apples",
        "60 minutes * 24 hours = 1441 minutes",
        "1,000 m / 4 s = 251 m/s",
    ],
)
def test_units_do_not_hide_an_error(bridge, line):
    assert bridge.check_arithmetic_claims(line), line


def test_bare_arithmetic_is_still_caught(bridge):
    """The form that already worked must keep working."""
    assert [e["claim"] for e in bridge.check_arithmetic_claims("2 + 2 = 5")] == ["2 + 2 = 5"]


def test_min_and_max_survive_unit_stripping(bridge):
    """They are the one alphabetic construct the evaluator honours.

    Stripping letters as units would reduce them to their separators, so
    expressions using them are left alone.
    """
    assert bridge.check_arithmetic_claims("min(3, 7) = 3") == []
    assert bridge.check_arithmetic_claims("max(3, 7) = 7") == []
    assert bridge.check_arithmetic_claims("min(3, 7) = 7")


def test_an_equation_named_as_false_is_not_corrected(bridge):
    """Discussing a wrong equation is not asserting one."""
    assert bridge.check_arithmetic_claims("That is wrong: 2 + 2 = 5") == []


def test_x_between_numbers_is_multiplication_but_not_inside_a_unit(bridge):
    """The old normalizer replaced every "x", including letters in units."""
    assert bridge.check_arithmetic_claims("6 x 7 = 43")
    assert bridge.check_arithmetic_claims("6 x 7 = 42") == []


def test_the_repair_corrects_only_the_number(bridge):
    fixed, repairs = bridge.repair_arithmetic_claims(LIVE_LINE)
    assert "113788800" in fixed
    assert fixed.startswith("1317 days * 86400 seconds/day =")
    assert len(repairs) == 1
    # Running it again changes nothing, as the docstring promises.
    again, more = bridge.repair_arithmetic_claims(fixed)
    assert again == fixed and not more


def test_the_whole_live_reply_yields_exactly_one_error(bridge):
    """Multi-line working must not produce fragment false positives.

    "1095 + 210 + 12 = 1317" is correct as a whole and false as its own
    suffix; reporting the suffix would be a confident wrong correction.
    """
    reply = (
        "3 years = 3 * 365 = 1095 days\n"
        "7 * 30 = 210 days\n"
        "Total days: 1095 + 210 + 12 = 1317 days\n"
        f"{LIVE_LINE}\n"
    )
    errors = bridge.check_arithmetic_claims(reply)
    assert len(errors) == 1, [e["claim"] for e in errors]
    assert errors[0]["correct"] == pytest.approx(113_788_800)
