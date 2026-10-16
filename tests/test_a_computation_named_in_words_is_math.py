"""A request that names a computation in words is classed as math, so a program can check its answer.

Live chat, 16 October: "What is 15% of 340?" read as a factual question and "Convert 1203321 from base 4 to
base 10." as nothing at all, so neither reached the amplifier or the idle program check.
"""

from __future__ import annotations

import pytest

from core.brain.reasoning_task_kinds import classify_task_type


@pytest.mark.parametrize(
    "request_",
    [
        "What is 15% of 340?",
        "What's 12.5 percent of 80?",
        "Convert 1203321 from base 4 to base 10.",
        "What is 777 in base 8?",
        "What is the square root of 1764?",
        "What is 7350 divided by 15?",
        "What is 48 multiplied by 26?",
        "What's 17 times 23?",
        "What is the average of 4, 8, 15, 16, 23 and 42?",
    ],
)
def test_a_computation_named_in_words_is_math(request_):
    assert classify_task_type(request_) == "math"


@pytest.mark.parametrize(
    "request_",
    [
        "What is the capital of Australia?",
        "Who won the 2019 World Series?",
        "What was the population of Ohio in 2010 and 2020?",
        "How was your afternoon?",
        "What percentage of your attention is on me right now, roughly?",
        "I had three times as much fun as last time!",
    ],
)
def test_a_request_that_names_no_computation_is_not_math(request_):
    assert classify_task_type(request_) != "math"
