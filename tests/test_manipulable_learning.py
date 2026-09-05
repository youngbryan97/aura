"""Improving how she improves, without a tower of meta-learners.

Aura adapts. What she cannot do is change the kind of adaptation available to
her, because the learning mechanism is code and the things it learns are data.
Improving a policy is a search; improving the thing that improves policies is
somebody writing a new file.

The way out is not MetaImprover then MetaMetaImprover, but a representation
general enough that the task, the policy and the mechanism are all objects of
the same sort. The claim is falsifiable: one search function, applied at two
levels, with no level-specific code.
"""

from __future__ import annotations

import ast
import inspect
import random

import pytest

from core.learning.manipulable_learning import (
    Program,
    as_mechanism,
    improve_the_improver,
    levels_share_a_search,
    mutate,
    search,
)

VOCABULARY = ("probe", "commit", "wait", "retry", "split")
WORTH = {"probe": 0.4, "commit": 0.3, "split": 0.25, "wait": -0.1, "retry": -0.05}


def task_score(program: Program) -> float:
    return sum(WORTH.get(move, 0.0) for move in program.moves) - 0.03 * len(program.moves)


POLICY = Program("policy", moves=("wait",), level=0)
MECHANISM = Program(
    "hill_climb", moves=("mutate",), params={"breadth": 2.0, "rounds": 1.0}, level=1
)


# ── the claim ────────────────────────────────────────────────────────────


def test_the_same_function_improves_a_policy():
    result = search(POLICY, task_score, VOCABULARY)
    assert result.improved
    assert result.score > task_score(POLICY)


def test_the_same_function_improves_the_mechanism():
    result = improve_the_improver(MECHANISM, POLICY, task_score, VOCABULARY, budget=600)
    assert result.improved
    assert result.best.params["breadth"] != MECHANISM.params["breadth"]


def test_the_improved_mechanism_finds_a_better_policy():
    """A mechanism is scored by what it produces, which is what it is for."""
    before = as_mechanism(MECHANISM, VOCABULARY)(POLICY, task_score)
    improved = improve_the_improver(
        MECHANISM, POLICY, task_score, VOCABULARY, budget=600
    )
    after = as_mechanism(improved.best, VOCABULARY)(POLICY, task_score)
    assert after.score > before.score


def test_it_is_literally_the_same_function_object():
    assert levels_share_a_search(search, search) is True
    assert levels_share_a_search(search, as_mechanism) is False


def test_the_search_never_reads_what_level_it_is_at():
    """If it had to branch on the level, the tower would have started."""
    tree = ast.parse(inspect.getsource(search))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "level"
        for node in ast.walk(tree)
    )


def test_the_mutation_never_reads_what_level_it_is_at():
    tree = ast.parse(inspect.getsource(mutate))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "level"
        for node in ast.walk(tree)
    )


# ── the step size ────────────────────────────────────────────────────────


def test_a_parameter_step_scales_with_the_parameter():
    """A fixed step is wrong in an unknown scale: on a breadth of 2 that
    rounds to an integer, every mutation landed back on 2 and the meta search
    found nothing while reporting that it had looked."""
    rng = random.Random(1)
    small = Program("p", moves=("m",), params={"x": 1.0})
    large = Program("p", moves=("m",), params={"x": 1000.0})
    small_moves = [
        abs(mutate(small, (), random.Random(s)).params["x"] - 1.0) for s in range(40)
    ]
    large_moves = [
        abs(mutate(large, (), random.Random(s)).params["x"] - 1000.0) for s in range(40)
    ]
    assert max(large_moves) > max(small_moves) * 10


def test_a_parameter_at_zero_still_moves():
    rng = random.Random(2)
    at_zero = Program("p", moves=("m",), params={"x": 0.0})
    moved = [mutate(at_zero, (), random.Random(s)).params["x"] for s in range(40)]
    assert any(abs(value) > 0.1 for value in moved)


# ── the budget ───────────────────────────────────────────────────────────


def test_the_meta_search_cannot_spend_unboundedly_at_the_level_below():
    """A meta search that runs the level below forever is not a search."""
    calls = {"n": 0}

    def counted(program):
        calls["n"] += 1
        return task_score(program)

    improve_the_improver(MECHANISM, POLICY, counted, VOCABULARY, budget=50)
    assert calls["n"] <= 50


def test_a_search_stops_when_a_round_finds_nothing():
    flat = Program("flat", moves=("wait",))
    result = search(flat, lambda p: 1.0, VOCABULARY, rounds=20)
    assert result.considered < 20 * 8


# ── the shared shape ─────────────────────────────────────────────────────


def test_a_policy_and_a_mechanism_are_the_same_sort_of_object():
    assert isinstance(POLICY, Program) and isinstance(MECHANISM, Program)
    assert set(POLICY.to_dict()) == set(MECHANISM.to_dict())


def test_one_mutation_operator_serves_both():
    rng = random.Random(3)
    assert isinstance(mutate(POLICY, VOCABULARY, rng), Program)
    assert isinstance(mutate(MECHANISM, ("breadth", "rounds"), rng), Program)


def test_a_mechanism_with_no_improvement_says_so():
    result = search(POLICY, lambda p: 0.0, VOCABULARY)
    assert result.improved is False
    assert "no candidate beat" in result.because
