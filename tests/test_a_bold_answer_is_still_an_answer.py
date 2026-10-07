"""The final number in a reply is read as written, emphasis, grouping and minus included.

LIVE 2026-10-06, the G05 pilot: her correct answers "**2,104,802,751,450**."
and "**476 583**" were graded 4 and 9. The bold kept the final number from
looking like a number, so the last plain number in her working table won.
"""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.experiment_tasks import Task, answer_tokens
from core.brain.llm.latent_cortex.experiments import extract_final_numeric_claim
from core.learning.semantic_program_ordinary_baseline import parse_integral_numeric_claim

_HER_TABLE = (
    "| **Final** | consolidated + kiln trim | 2,104,802,751,446 + 4 | **2,104,802,751,450** |\n\n"
    "### The worksheet yields **2,104,802,751,450**."
)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (_HER_TABLE, 2104802751450),
        ("| **Final** | 476 592 − 9 | **476 583** |\n\nThe worksheet yields **476 583**.", 476583),
        ("So the answer is `42`.", 42),
        ("The total falls to −60.", -60),
        ("Not 5; it is 7.", 7),
        ("Add 12 and 345.", 345),
        ("Step 3 gives 12, and the total is 1,204.", 1204),
        # G05 2026-10-07: her open-channel answers, read as 20, 18 and 8.
        ("25,758,139 ÷ 29 = 888,211 remainder 20\n\n$$\\boxed{888211}$$", 888211),
        ("Consolidated measure + Kiln trim = −6993 + 18\n\n$$\\boxed{-6975}$$", -6975),
        ("consolidated measure − lab trim = 56 570 − 8\n\n$$\\boxed{56562}$$", 56562),
        ("The result is \\(\\boxed{56\\,562}\\).", 56562),
        ("so $x = 1{,}204$", 1204),
        ("\\[ \\fbox{42} \\]", 42),
    ],
)
def test_the_final_number_is_read_as_written(reply: str, expected: int) -> None:
    assert parse_integral_numeric_claim(reply) == expected


def test_a_number_split_across_a_line_is_not_joined() -> None:
    assert answer_tokens("I count 12\n345 items") == ["I", "count", "12", "345", "items"]


def test_the_verifier_and_the_extractor_read_the_same_answer() -> None:
    task = Task(family="arithmetic", depth=1, seed=0, prompt="", answer="476583")
    reply = "The worksheet yields **476 583**."
    assert task.verify(reply)
    assert extract_final_numeric_claim(reply) == "476583"
