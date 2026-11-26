"""A condition that cannot come out both ways where it stands is no repair (Master Todo F15 and F16, 26 November 2026).

Her mend of ``core.runtime.subprocess_gateway._coerce_argv`` reached origin as
``if any(not part for part in coerced) and not argv: raise ValueError("argv entries must not be empty")``. Three lines
above, ``if not isinstance(argv, (list, tuple)) or not argv: raise`` has already left for an empty ``argv``, so the new
condition is never true and the refusal it guards is dead: a deletion written as a narrowing. Her synthesis had offered
it beside ``X and not X`` and ``not coerced and coerced``, which its values check passed because no recorded value read
their parts. A condition is now read as logic too, over its own parts and what the exits above it make true there.
"""
from __future__ import annotations

import ast

import pytest

from core.self_modification.condition_synthesis import Example, can_come_out, conditions
from core.self_modification.semantic_repair import holds_at, places_at

pytestmark = pytest.mark.unit

_GATEWAY = '''
def _coerce_argv(argv):
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError("argv must be a non-empty list or tuple")
    coerced = [str(part) for part in argv]
    if any(not part for part in coerced):
        raise ValueError("argv entries must not be empty")
    return coerced
'''


def _function(source: str) -> ast.AST:
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef))


def test_what_an_exit_above_leaves_true_is_said_where_the_condition_stands():
    line = next(n for n, text in enumerate(_GATEWAY.splitlines(), 1) if "any(not part" in text)
    assert holds_at(_function(_GATEWAY), line) == ("not (not isinstance(argv, (list, tuple)) or not argv)",)
    (place,) = [p for p in places_at(_GATEWAY, "gateway.py", [line]) if p.kind == "cond"]
    assert place.holds == ("not (not isinstance(argv, (list, tuple)) or not argv)",)


def test_the_condition_her_mend_landed_cannot_be_true_there():
    holds = ("not (not isinstance(argv, (list, tuple)) or not argv)",)
    landed = "any(not part for part in coerced) and not argv"
    assert can_come_out(landed, True) and can_come_out(landed, False), "on its own it reads both ways"
    assert not can_come_out(landed, True, holds), "where it stands it never refuses"
    assert can_come_out("any(not part for part in coerced)", True, holds), "the check that was there still can"


def test_a_contradiction_is_never_offered_whatever_was_recorded():
    for dead in ("x and not x", "not coerced and coerced", "any(not p for p in c) and not any(not p for p in c)"):
        assert not can_come_out(dead, True), dead
    assert not can_come_out("x or not x", False)


def test_the_synthesis_offers_no_condition_that_deletes_the_refusal():
    # The runs she had: her fault's, where the branch was taken and had not to be, and passing runs that never reached
    # it taken. Every condition that decides as they needed is false for all of them.
    original = "any(not part for part in coerced)"
    runs = [Example({"argv is None": False, "len(argv)": 11, "coerced is None": False, "len(coerced)": 11},
                    want=False, decided=True, parts={original: True}),
            Example({"argv is None": False, "len(argv)": 2, "coerced is None": False, "len(coerced)": 2},
                    want=False, decided=False, parts={original: False})]
    holds = ("not (not isinstance(argv, (list, tuple)) or not argv)",)
    for offered in conditions(runs, original=original, names=["any", "part", "coerced"], holds=holds):
        assert can_come_out(offered, True, holds), offered
    assert "any(not part for part in coerced) and not any(not part for part in coerced)" not in conditions(
        runs, original=original, names=["any", "part", "coerced"])


def test_an_exit_whose_names_change_before_the_condition_says_nothing_there():
    source = '''
def f(items):
    if not items:
        return []
    items = list(filter(None, items))
    if items and items[0]:
        return items
'''
    line = next(n for n, text in enumerate(source.splitlines(), 1) if "items and items[0]" in text)
    assert holds_at(_function(source), line) == ()


def test_the_branch_a_condition_is_inside_holds_there_and_its_else_the_negation():
    source = '''
def f(a, b):
    if a > 1:
        if b:
            return 1
    else:
        if b:
            return 2
'''
    lines = [n for n, text in enumerate(source.splitlines(), 1) if text.strip() == "if b:"]
    assert holds_at(_function(source), lines[0]) == ("a > 1",)
    assert holds_at(_function(source), lines[1]) == ("not (a > 1)",)
