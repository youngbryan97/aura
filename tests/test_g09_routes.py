"""G09's kept-arm routes: by words, by meaning, and by words then meaning."""

from __future__ import annotations

import pytest

from tools.g09_routes import kept_receipt


class _Book:
    procedures = {"gcd": ["code"], "lcm": ["code"]}

    def answer(self, request):
        return {"answer": "6" if "highest common divisor" in request else None}

    def route_by_meaning(self, vector):
        return ("lcm", 0.8, "") if vector == "lcm" else (None, 0.1, "no_kind_meant_clearly")

    def _run_kind(self, kind, request, receipt):
        return {**receipt, "answer": "12"}


def _query(text):
    return "lcm" if "least common multiple" in text else "nothing"


@pytest.mark.parametrize(
    ("route", "request_", "answer"),
    [
        ("words", "Calculate the highest common divisor of 12 and 18.", "6"),
        ("words", "What is the least common multiple of 4 and 6?", None),
        ("meaning", "What is the least common multiple of 4 and 6?", "12"),
        ("meaning", "How was your day?", None),
        ("words_then_meaning", "Calculate the highest common divisor of 12 and 18.", "6"),
        ("words_then_meaning", "What is the least common multiple of 4 and 6?", "12"),
    ],
)
def test_each_route_answers_what_it_reaches(route, request_, answer):
    receipt = kept_receipt(_Book(), request_, route, query=_query)
    assert (receipt or {}).get("answer") == answer


def test_an_unknown_route_is_refused():
    with pytest.raises(ValueError):
        kept_receipt(_Book(), "x", "by_vibes", query=_query)
