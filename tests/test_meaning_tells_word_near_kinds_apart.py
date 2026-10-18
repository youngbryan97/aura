"""When the words leave several kinds near, her meaning may pick the one the request is typical of.

Live, 17 October: "What are the prime factors of 21000063?" was as near by its words to
the kind that lists a number's prime factors as to the kind that computes the number from
equations first, which keeps no procedure, so the kept answer was declined.
"""
from __future__ import annotations

import asyncio

import numpy as np

from core.learning.kinds_by_meaning import KindMeaning
from core.learning.procedures_from_solved_examples import ProcedureBook
from core.phases import kept_procedures_in_a_turn as turn

PLAIN, COMPOSED = "list_prime_factors", "list_prime_factors_composed"


def _book(plain_bar: float, composed_bar: float) -> ProcedureBook:
    book = ProcedureBook({}, {})
    book.meanings = {
        PLAIN: KindMeaning(centroid=[1.0, 0.0], nearest_required=plain_bar),
        COMPOSED: KindMeaning(centroid=[0.0, 1.0], nearest_required=composed_bar),
    }
    book.procedures = {PLAIN: ["def solve(p):\n    return '3, 7, 1000003'\n"], COMPOSED: []}
    book.agreed = {PLAIN: 12}
    return book


def test_exactly_one_typical_kind_decides():
    book = _book(0.8, 0.8)
    assert book.typical_among(np.array([0.95, 0.31]), [PLAIN, COMPOSED]) == PLAIN


def test_two_typical_kinds_or_none_decide_nothing():
    assert _book(0.2, 0.2).typical_among(np.array([0.7, 0.7]), [PLAIN, COMPOSED]) is None
    assert _book(0.99, 0.99).typical_among(np.array([0.7, 0.7]), [PLAIN, COMPOSED]) is None


def _wire(monkeypatch, book: ProcedureBook, vector) -> None:
    import core.learning.procedures_from_solved_examples as procedures

    monkeypatch.setattr(procedures, "kept_procedure_book", lambda: book)
    monkeypatch.setattr(book, "answer", lambda problem: {"answer": None, "near_enough": [PLAIN, COMPOSED],
                                                         "declined": "a_kind_near_enough_keeps_no_procedure"})
    monkeypatch.setattr(book, "_run_kind", lambda kind, problem, receipt: {**receipt, "answer": "3, 7, 1000003",
                                                                          "agreed_known": 12, "confidence": 0.93})
    monkeypatch.setattr(turn, "her_query_vector", lambda text, task: vector)


def test_the_plain_kind_answers_when_its_meaning_alone_is_typical(monkeypatch):
    book = _book(0.8, 0.8)
    _wire(monkeypatch, book, np.array([0.95, 0.31]))
    kept = asyncio.run(turn.kept_answer_among_the_word_near("What are the prime factors of 21000063?"))
    assert kept is not None and kept.answer == "3, 7, 1000003"
    assert kept.receipt.cognitive_operations[0]["form"] == "words_then_meaning"


def test_the_kind_without_a_procedure_declines(monkeypatch):
    book = _book(0.8, 0.8)
    _wire(monkeypatch, book, np.array([0.31, 0.95]))
    assert asyncio.run(turn.kept_answer_among_the_word_near("Let p = 3*7. What are the prime factors of p?")) is None


def test_no_embedder_held_no_choice(monkeypatch):
    book = _book(0.8, 0.8)
    _wire(monkeypatch, book, None)
    assert asyncio.run(turn.kept_answer_among_the_word_near("What are the prime factors of 21000063?")) is None


def test_a_served_form_is_not_evidence_only():
    stands, how = turn.after_the_draft("I think it is 3 and 7.", {"form": "words_then_meaning", "answer": "3, 7, 1000003"})
    assert not stands and how == "computed_answer_served"
