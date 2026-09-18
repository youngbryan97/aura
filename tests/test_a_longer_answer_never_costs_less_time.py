"""A budget's estimate rises with the budget, and reading is not deadline-side.

Both defects were measured on the live reserve, 512 readings, 2026-09-17.

"Comparable length" is a *set* that shrinks as the budget grows, so its
slow percentile moves as readings drop out of it. The answer moved the
wrong way: 101 tokens came back at 218.4 seconds and 102 tokens at 35.2,
with 26 such steps below 2,048. Anything sized on it then said a longer
answer was cheaper than a shorter one.

And the reading budget divided that estimate the other way round. The
slow tail exists so a deadline does not cancel a slow generation; used to
buy reading, the same pessimism afforded 126,798 characters of prompt for
a 457-token answer, against the turn that showed the defect reading
44,192. The budget could not fire.
"""

from __future__ import annotations

import pytest

from core.brain.llm import thinking_reserve


@pytest.fixture(autouse=True)
def _clean_reserve():
    thinking_reserve.forget()
    yield
    thinking_reserve.forget()


def _mixed_window() -> None:
    # Short runs that decode fast, long runs that decode slowly: the shape
    # that made the comparable-length set move the estimate the wrong way.
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=40, elapsed_s=1.0)
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=40, elapsed_s=80.0)
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=600, elapsed_s=24.0)


def test_the_estimate_never_falls_as_the_budget_grows():
    _mixed_window()
    previous = 0.0
    for wanted in range(1, 1200):
        seconds = thinking_reserve.seconds_to_decode(wanted)
        if seconds <= 0.0:
            continue
        assert seconds >= previous - 1e-9, (
            f"{wanted} tokens costs {seconds:.1f}s, less than a shorter budget's "
            f"{previous:.1f}s"
        )
        previous = seconds


def test_the_typical_tail_never_falls_either():
    _mixed_window()
    previous = 0.0
    for wanted in range(1, 1200):
        seconds = thinking_reserve.seconds_to_decode(wanted, typical=True)
        if seconds <= 0.0:
            continue
        assert seconds >= previous - 1e-9
        previous = seconds


def test_reading_is_budgeted_on_the_typical_rate_not_the_slow_one():
    _mixed_window()
    slow = thinking_reserve.seconds_to_decode(600)
    typical = thinking_reserve.seconds_to_decode(600, typical=True)
    assert typical < slow, "the reading budget must not inherit the deadline's pessimism"


def test_the_reading_budget_uses_it():
    from core.brain.llm.context_budget import budget_for_answer

    _mixed_window()
    afford = budget_for_answer(600)
    from core.brain.llm.thinking_reserve import chars_readable_in

    assert afford == chars_readable_in(
        thinking_reserve.seconds_to_decode(600, typical=True)
    )


def test_unmeasured_stays_unmeasured():
    # Zero means unmeasured, not instant. Lifting it to a shorter budget's
    # estimate would invent evidence for a budget that has none.
    for _ in range(40):
        thinking_reserve.record_decode_rate(generated_tokens=20, elapsed_s=0.5)
    assert thinking_reserve.seconds_to_decode(896) == 0.0
    assert thinking_reserve.seconds_to_decode(896, typical=True) == 0.0
