"""L09's ordinary-use grader reads her prose and a bare kept answer by one rule (tools/l09_ordinary_use.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "l09_ordinary_use.py"
_spec = importlib.util.spec_from_file_location("l09_ordinary_use", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

NUMBER, LIST, YES_NO, WORDS = ({"kind": k} for k in ("number", "number_list", "yes_no", "word_list"))
SORTED = "basil ember lantern orchard quarry velvet"


@pytest.mark.parametrize(("text", "fixed", "question", "right"), [
    ("17 times 23 is 391.", {**NUMBER, "answer": "391"}, "17 times 23", True),
    ("It is 392.", {**NUMBER, "answer": "391"}, "", False),
    ("In base 10 it is 6,393.", {**NUMBER, "answer": "6393"}, "", True),
    ("Work...\n\n\\boxed{6393}", {**NUMBER, "answer": "6393"}, "", True),
    ("The prime factors are 337, 569 and 34211.", {**LIST, "answer": "337, 569, 34211"}, "6560061883", True),
    ("The prime factors are 337 and 569.", {**LIST, "answer": "337, 569, 34211"}, "6560061883", False),
    ("6560061883 = 337 x 569 x 34211", {**LIST, "answer": "337, 569, 34211"},
     "What are the prime factors of 6560061883?", True),
    ("The prime factors are 3, 7, 13 and 1000003.", {**LIST, "answer": "3, 7, 1000003"}, "21000063", False),
    ("3, 7, 1000003", {**LIST, "answer": "3, 7, 1000003"}, "21000063", True),
    ("No, 829348951 is not prime.", {**YES_NO, "answer": "no"}, "", True),
    ("Yes, 829348951 is a prime number.", {**YES_NO, "answer": "no"}, "", False),
    ("basil, ember, lantern, orchard, quarry, velvet", {**WORDS, "answer": SORTED}, "", True),
    ("ember, basil, lantern, orchard, quarry, velvet", {**WORDS, "answer": SORTED}, "", False),
])
def test_one_rule_for_prose_and_for_a_bare_answer(text, fixed, question, right):
    assert tool.states(text, fixed, question) is right


def test_the_second_rule_reads_the_paragraph_that_states_a_new_number():
    # The first attempt's r192: the factors stated, then a closing sentence naming only the number asked about.
    reply = ("The prime factors of 6,469,693,231 are:\n\n**331 × 571 × 34,231**\n\n"
             "All three are prime numbers, and their product equals 6,469,693,231.")
    fixed = {**LIST, "answer": "331, 571, 34231"}
    question = "What are the prime factors of 6469693231?"
    assert tool.states(reply, fixed, question) is False
    assert tool.states(reply, fixed, question, rule=2) is True
    # A wrong list is still wrong, and a yes or no is read as before.
    assert tool.states("They are **331 × 571**.\n\nThat is all.", fixed, question, rule=2) is False
    assert tool.states("No, 829348951 is not prime.", {**YES_NO, "answer": "no"}, "", rule=2) is True
