"""What a cycle can plan and what a cycle can run are the same set.

The planner could emit ``developmental`` and ``asked_the_forge``. The
execution dispatch had a branch for neither. Both were appended to
``attempted_actions`` and then nothing happened for them, so the ledger
recorded an attempt nobody made and the native half of the improvement loop
stopped one step short of running.

The targeted RSI suite was green throughout. One test asserted the planner
contains the developmental action; none asserted that a cycle then takes it.
A part exists, a test proves the part exists, another part names it, and the
chain still does not close — which is why this file checks the two sides
against each other rather than each against itself.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.learning.recursive_self_improvement import (
    WHAT_A_CYCLE_CAN_DO,
    ImprovementPlan,
    RecursiveSelfImprovementLoop,
)

_SOURCE = pathlib.Path("core/learning/recursive_self_improvement.py")


def _function(name: str) -> ast.AST:
    tree = ast.parse(_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is not in {_SOURCE}")


def _names_in(name: str) -> set[str]:
    """The declared action names this function mentions."""
    node = _function(name)
    said = {
        one.value
        for one in ast.walk(node)
        if isinstance(one, ast.Constant) and isinstance(one.value, str)
    }
    return said & set(WHAT_A_CYCLE_CAN_DO)


def test_the_planner_emits_only_declared_actions():
    assert _names_in("_make_plan") <= set(WHAT_A_CYCLE_CAN_DO)


def test_every_action_the_planner_emits_can_be_run():
    """The defect, stated as the invariant it violated."""
    planned = _names_in("_make_plan")
    runnable = _names_in("_run_cycle_locked")
    assert not (planned - runnable), (
        f"the planner can emit {sorted(planned - runnable)} and the executor "
        "has no branch for them, so they would be recorded as attempted and "
        "silently do nothing"
    )


def test_the_executor_runs_only_declared_actions():
    assert _names_in("_run_cycle_locked") <= set(WHAT_A_CYCLE_CAN_DO)


def test_the_vocabulary_has_no_duplicates():
    assert len(WHAT_A_CYCLE_CAN_DO) == len(set(WHAT_A_CYCLE_CAN_DO))


def test_an_unknown_action_is_recorded_as_unrun_rather_than_attempted():
    """The fallback: a gap in the dispatch must show up in the ledger.

    ``attempted`` named the action either way. Without a result saying no
    implementation ran, a future divergence looks exactly like a success.
    """
    node = _function("_run_cycle_locked")
    source = ast.unparse(node)
    assert "no implementation for" in source


def test_the_forge_probe_decides_rather_than_asks():
    """A planner that acts while planning makes the plan a record of the past.

    This module says so itself, two methods above the one that was doing it.
    """
    decide = ast.unparse(_function("_worth_asking_the_forge"))
    assert "create_task" not in decide, "the planner is still starting the forge"
    ask = ast.unparse(_function("_ask_the_forge"))
    assert "create_task" in ask, "nothing starts the forge any more"


def test_the_plan_carries_the_developmental_decision_not_its_name():
    """Asking again draws again, so the cycle would run a different action."""
    plan = ImprovementPlan(objective="o", actions=[], rationale=[], depth=0)
    assert plan.developmental_decision is None
    assert plan.forge_gaps == ()
    probe = ast.unparse(_function("_what_she_could_change_about_her_own_terms"))
    assert "return decided if" in probe, "the probe still returns a name"


@pytest.mark.asyncio
async def test_a_plan_with_no_developmental_choice_says_so():
    loop = RecursiveSelfImprovementLoop.__new__(RecursiveSelfImprovementLoop)
    plan = ImprovementPlan(objective="o", actions=["developmental"], rationale=[], depth=0)
    said = await loop._run_developmental(plan)
    assert said["ok"] is False
    assert "no developmental choice" in said["reason"]
