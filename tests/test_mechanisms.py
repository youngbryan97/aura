"""The shared exact mechanisms give exact answers and say when an answer is not settled."""

from __future__ import annotations

import datetime as dt
from fractions import Fraction

import pytest

from core.reasoning import mechanisms as m


def test_constraints_prove_one_solution_find_two_and_find_none() -> None:
    houses = (1, 2, 3)
    variables = {"red": houses, "green": houses, "blue": houses}
    rules = [(("red", "green"), lambda red, green: green == red + 1), (("blue",), lambda blue: blue == 1)]
    assert m.solve_constraints(variables, rules, all_different=[("red", "green", "blue")]) == [
        {"red": 2, "green": 3, "blue": 1}]
    assert m.only_solution(variables, rules, all_different=[("red", "green", "blue")])["green"] == 3
    loose = m.solve_constraints(variables, rules[:1], all_different=[("red", "green", "blue")])
    assert len(loose) == 2
    assert m.solve_constraints(variables, rules + [(("blue",), lambda blue: blue == 3)],
                               all_different=[("red", "green", "blue")]) == []
    with pytest.raises(ValueError):
        m.only_solution(variables, rules[:1], all_different=[("red", "green", "blue")])


def test_constraints_over_three_variables_are_checked() -> None:
    variables = {name: range(1, 6) for name in "abc"}
    found = m.solve_constraints(variables, [(("a", "b", "c"), lambda a, b, c: a + b + c == 12 and a < b < c)],
                                limit=10)
    assert sorted((s["a"], s["b"], s["c"]) for s in found) == [(3, 4, 5)]


def test_orders_settle_contradict_or_leave_open() -> None:
    assert m.orders("abc", [("a", "b"), ("b", "c")]) == [["a", "b", "c"]]
    assert len(m.orders("abc", [("a", "b")])) == 2
    assert m.orders("ab", [("a", "b"), ("b", "a")]) == []
    pairs = m.precedences_from_sequences([["old", "red", "steel"], ["red", "wool"]])
    assert ("old", "steel") in pairs and ("red", "wool") in pairs and ("steel", "old") not in pairs


def test_exact_arithmetic_and_logic() -> None:
    assert m.exact("1/3 + 1/6") == Fraction(1, 2)
    assert m.exact("2 ** 10 - 0.1") == Fraction(10239, 10)
    assert m.holds("max(-3, 8, 5, -2) - min(-3, 8, 5, -2) <= 11")
    assert m.holds("-5 - (1 / -5) > -7 and not (False or 3 * -2 > 0)")
    assert m.holds("1 < 2 < 3") and not m.holds("1 < 3 < 2")
    assert m.exact("x * 2 + y", {"x": Fraction(3, 2), "y": 1}) == 4
    for unsafe in ("__import__('os')", "(1).__class__", "open('f')", "[1, 2]"):
        with pytest.raises(ValueError):
            m.exact(unsafe)
    assert m.decimal_text(Fraction(1, 8), 2) == "0.13"
    assert m.decimal_text(Fraction(-1, 8), 2) == "-0.13"
    assert m.decimal_text(Fraction(5, 2), 0) == "3"


def test_clock_times_dates_and_intervals() -> None:
    assert [m.minutes(t) for t in ("9:30", "17:00", "9am", "3:45 PM", "12:00 a.m.", "12pm")] == [
        570, 1020, 540, 945, 0, 720]
    assert m.clock(945, twelve_hour=True) == "3:45 PM" and m.clock(1500) == "01:00"
    assert m.clock(570, pad_hour=False) == "9:30"
    with pytest.raises(ValueError):
        m.minutes("25:00")
    assert m.add_to_date(dt.date(2024, 1, 31), months=1) == dt.date(2024, 2, 29)
    assert m.add_to_date(dt.date(2024, 2, 29), years=1) == dt.date(2025, 2, 28)
    assert m.add_to_date(dt.date(2024, 12, 31), days=1) == dt.date(2025, 1, 1)
    assert m.weekday_of(dt.date(2026, 10, 8)) == "Thursday"
    assert m.merge_intervals([(1, 3), (2, 5), (7, 8), (5, 6)]) == [(1, 6), (7, 8)]
    busy = [(540, 570), (600, 660), (650, 700)]
    assert m.free_intervals(busy, 540, 1020, length=30) == [(570, 600), (700, 1020)]
    assert m.free_intervals(busy, 540, 1020, length=60) == [(700, 1020)]


def test_the_reference_names_every_mechanism() -> None:
    text = m.reference()
    for name in m.__all__:
        if name != "reference":
            assert f"mechanisms.{name}" in text
