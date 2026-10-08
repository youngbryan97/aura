"""Procedures written from solved examples are kept only when every known answer agrees."""

from __future__ import annotations

import pytest

from core.learning.procedures_from_solved_examples import (
    ProcedureBook,
    SolvedExample,
    answers_agree,
    extract_program,
    induce,
    split_examples,
)
from core.sandbox.untrusted_python import available_boundary

needs_boundary = pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")

REVERSE_WRONG = "```python\ndef solve(problem):\n    return problem.split(':')[1].strip()\n```"
REVERSE_RIGHT = ("```python\ndef solve(problem):\n    words = problem.split(':')[1].split()\n"
                 "    return ' '.join(reversed(words))\n```")
COUNT = "```python\ndef solve(problem):\n    return str(len(problem.split(':')[1].split()))\n```"


WORDS = ["apple", "river", "stone", "cloud", "lamp", "fern", "anchor", "violin", "maple", "copper", "harbor"]
COLOURS = ["red", "green", "blue", "cyan", "gold", "violet", "amber", "teal"]


def reversal(index: int) -> SolvedExample:
    words = [WORDS[(index * 3 + k) % len(WORDS)] for k in range(2 + index % 3)]
    return SolvedExample(f"rev{index}", "Reverse the order of these words: " + " ".join(words),
                         " ".join(reversed(words)))


def counting(index: int) -> SolvedExample:
    words = [COLOURS[(index * 5 + k) % len(COLOURS)] for k in range(1 + index % 4)]
    return SolvedExample(f"cnt{index}", "How many colour names appear in this list: " + " ".join(words),
                         str(len(words)))


def test_answers_agree_on_form_not_on_meaning() -> None:
    assert answers_agree(" (A) ", "(a)")
    assert answers_agree("4.00", "4")
    assert answers_agree("1,000", "1000")
    assert not answers_agree("(A)", "(B)")
    assert not answers_agree("4.01", "4")


def test_the_last_block_that_defines_solve_is_taken() -> None:
    text = "```python\nx = 1\n```\nthen\n```python\ndef solve(p):\n    return p\n```\n```\nprint(2)\n```"
    assert extract_program(text) == "def solve(p):\n    return p\n"
    assert extract_program("no code") is None


def test_the_split_is_disjoint_complete_and_fixed() -> None:
    examples = [reversal(i) for i in range(11)]
    pool, sealed = split_examples(examples)
    assert {e.key for e in pool} | {e.key for e in sealed} == {e.key for e in examples}
    assert not {e.key for e in pool} & {e.key for e in sealed}
    assert split_examples(list(reversed(examples))) == (pool, sealed)


@needs_boundary
def test_a_wrong_function_is_revised_from_its_own_counterexamples_and_sealed_problems_stay_unseen() -> None:
    examples = [reversal(i) for i in range(8)]
    _pool, sealed = split_examples(examples)
    requests_seen: list[str] = []

    def propose(requests: list[str]) -> list[str]:
        requests_seen.extend(requests)
        return [REVERSE_RIGHT if "It is wrong on" in r else REVERSE_WRONG for r in requests]

    families = induce({"reversal": examples}, propose, shots=2, lines=1, rounds=3)
    family = families["reversal"]
    assert [c.admitted for c in family.candidates] == [False, True]
    assert "It is wrong on" in requests_seen[1]
    for example in sealed:
        assert all(example.problem not in r for r in requests_seen)


@needs_boundary
def test_a_function_that_memorises_what_it_was_shown_fails_the_sealed_half() -> None:
    examples = [reversal(i) for i in range(8)]
    pool, _sealed = split_examples(examples)
    table = {e.problem: e.answer for e in pool}
    memoriser = "```python\nTABLE = " + repr(table) + "\ndef solve(problem):\n    return TABLE.get(problem, '')\n```"
    families = induce({"reversal": examples}, lambda requests: [memoriser] * len(requests),
                      shots=2, lines=1, rounds=2)
    candidates = families["reversal"].candidates
    assert candidates and not any(c.admitted for c in candidates)
    assert all(c.pool_agreed == len(c.pool) for c in candidates)
    assert all(c.sealed_agreed < len(c.sealed) for c in candidates)


@needs_boundary
def test_a_problem_goes_only_to_its_own_kind_and_an_unfamiliar_one_is_left_alone() -> None:
    families = induce(
        {"reversal": [reversal(i) for i in range(24)], "counting": [counting(i) for i in range(24)]},
        lambda requests: [COUNT if "colour" in r else REVERSE_RIGHT for r in requests],
        shots=2, lines=1, rounds=1)
    book = ProcedureBook.from_families(families)
    assert set(book.procedures) == {"reversal", "counting"}
    new_reversal, new_count = reversal(101), counting(101)
    assert book.answer(new_reversal.problem)["answer"] == new_reversal.answer
    assert book.answer(new_count.problem)["answer"] == new_count.answer
    unfamiliar = book.answer("Name the capital of the country that borders Chile to the east.")
    assert unfamiliar["answer"] is None and unfamiliar["declined"] == "no_kind_near_enough"
    restored = ProcedureBook.from_json(book.to_json())
    assert restored.answer(new_reversal.problem)["answer"] == new_reversal.answer


@needs_boundary
def test_a_kind_with_no_kept_procedure_declines_rather_than_borrowing_another() -> None:
    families = induce(
        {"reversal": [reversal(i) for i in range(24)], "counting": [counting(i) for i in range(24)]},
        lambda requests: ["no code" if "colour" in r else REVERSE_RIGHT for r in requests],
        shots=2, lines=1, rounds=1)
    book = ProcedureBook.from_families(families)
    receipt = book.answer(counting(101).problem)
    assert receipt["family"] == "counting" and receipt["declined"] == "kind_has_no_kept_procedure"
