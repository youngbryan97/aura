"""The cognition order is a compiled, sealed plan, not a thing you read the code for.

Three peer architectures make the same complaint from three directions. In
Generative Agents a reviewer can enumerate the causal order from one function.
Soar's decision cycle is a state machine anyone can walk. LangGraph compiles
its graph before it runs and refuses to start when the topology does not
resolve.

Aura had the pieces and not the join: `pipeline_blueprint` holds the order and
which phases a priority tick suppresses, `cognitive_contract` holds what each
phase reads and writes, and `pass_manager` records what actually ran. Nothing
put them together, so a field two phases both write was something you found by
watching it happen.
"""

from __future__ import annotations

import pytest

from core.runtime.the_shape_of_one_turn import (
    LAST_IN_THE_ORDER,
    THE_MODES,
    APhaseInThePlan,
    TheShapeOfOneTurn,
    compile_the_cognition,
    declare_write_mode,
    the_declared_and_the_realised,
    the_seal,
    write_mode_for,
)


def test_every_mode_compiles_and_holds():
    for mode in THE_MODES:
        plan = compile_the_cognition(mode)
        assert plan.phases, f"{mode} compiled to no phases"
        assert plan.holds, f"{mode} does not hold: {plan.refusals}"
        assert plan.seal, f"{mode} produced no seal"


def test_a_priority_tick_runs_fewer_phases_than_a_background_one():
    """The frequency is part of the plan, which is what "modes" means here."""

    assert len(compile_the_cognition("foreground").runs) < len(
        compile_the_cognition("background").runs
    )


def test_the_seal_is_stable_and_answers_to_the_plan():
    first = compile_the_cognition("foreground")
    again = compile_the_cognition("foreground")
    assert first.seal == again.seal, "two compiles of one commit disagree"
    assert first.seal != compile_the_cognition("background").seal, (
        "two different plans share a seal"
    )
    # And it answers to the content rather than to the object.
    moved = TheShapeOfOneTurn(
        mode=first.mode, phases=tuple(reversed(first.phases)),
    )
    assert the_seal(moved) != first.seal, "reordering the phases left the seal alone"


def test_a_plan_that_does_not_resolve_is_refused_rather_than_run():
    """The whole point of compiling: the refusal comes before the running."""

    twice = TheShapeOfOneTurn(
        mode="foreground",
        phases=(
            APhaseInThePlan(0, "a", "APhase", runs=True, writes=("x",), undeclared=False),
            APhaseInThePlan(1, "a", "APhase", runs=True, writes=("x",), undeclared=False),
        ),
        refusals=("a appears twice in the order",),
    )
    assert not twice.holds
    assert compile_the_cognition("no such mode").refusals


def test_several_writers_of_one_field_are_settled_by_the_order():
    """Not a race while the phases run one after another — and said so.

    LangGraph declares a write mode per key and rejects unspecified concurrent
    writes. In a sequential pipeline the order IS the mode, and what matters is
    that this is written down rather than assumed: the day two of these run
    together it stops being true.
    """

    plan = compile_the_cognition("background")
    shared = [one for one in plan.remarks if "the order settles it" in one]
    assert shared, "no field has several writers, so this test measures nothing"
    for said in shared:
        assert "is the value" in said, said


def test_a_field_can_declare_a_mode_other_than_the_order():
    assert write_mode_for("nothing.declared.this") == LAST_IN_THE_ORDER
    declare_write_mode("affect.arousal", "highest")
    try:
        assert write_mode_for("affect.arousal") == "highest"
        plan = compile_the_cognition("background")
        assert any("combined by highest" in one for one in plan.remarks), plan.remarks
    finally:
        declare_write_mode("affect.arousal", LAST_IN_THE_ORDER)
    with pytest.raises(ValueError):
        declare_write_mode("affect.arousal", "whatever seems best")


def test_what_was_declared_is_compared_against_what_ran():
    """The realised order has been recorded the whole time; nothing compared it."""

    seen = the_declared_and_the_realised("foreground")
    assert seen["seal"] == compile_the_cognition("foreground").seal
    assert set(seen) >= {"declared", "realised", "ran_outside_the_plan", "refusals"}
    assert seen["declared"], "the plan declares nothing to run"


def test_the_shape_reaches_the_health_report():
    """A plan nobody can read is the thing this replaces."""

    from core.runtime.health_contract import _runtime_integrity_block  # noqa: PLC2701

    block = _runtime_integrity_block().get("the_shape_of_one_turn") or {}
    assert "foreground" in block and "background" in block, block
    assert block["foreground"]["seal"], block["foreground"]
    assert block["foreground"]["holds"] is True, block["foreground"]["refusals"]


def test_a_plan_reports_the_mode_it_was_compiled_for():
    """It did not.

    A loop variable inside the multi-writer check was also called ``mode``, so
    every compiled plan reported its own mode as whichever write mode the last
    shared field happened to declare — both foreground and background said
    "last in the order". The seal covers the mode, so the seals were digests
    of a plan mislabelling itself.
    """
    from core.runtime.the_shape_of_one_turn import THE_MODES, compile_the_cognition

    for mode in THE_MODES:
        assert compile_the_cognition(mode).mode == mode


def test_the_three_modes_have_three_different_seals():
    """Same phases, different frequency, different plan."""
    from core.runtime.the_shape_of_one_turn import THE_MODES, compile_the_cognition

    seals = {mode: compile_the_cognition(mode).seal for mode in THE_MODES}
    assert len(set(seals.values())) == len(THE_MODES), seals
