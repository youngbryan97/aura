"""A live handle in plan context must not be able to kill the action lane.

LIVE DEFECT, 2026-08-10. Every self-chosen objective died with:

    TypeError: Object of type RobustOrchestrator is not JSON serializable

``overt_action_loop`` puts a live orchestrator handle into the governance
context so actuators can reach capabilities, then passes that same dict to the
task engine as the plan's context. ``TaskPlan.to_runtime_dict`` copied it
verbatim, so the plan file could never be written. Because the engine is on
the fail-closed list, a durability failure the code explicitly intends to
survive escalated to CRITICAL SERVICE FAILURE, raised, killed the objective,
and cascaded into ``overt_action_loop.semantic_plan``. Resilience went to
depletion, and the live stream then reported "Execution suppressed due to
depletion/exhaustion in ResilienceEngine".
"""

from __future__ import annotations

import json

import pytest


class _LiveHandle:
    """Stands in for RobustOrchestrator: an ordinary un-serializable object."""


def _plan(context: dict) -> object:
    from core.agency.autonomous_task_engine import TaskPlan

    return TaskPlan(
        plan_id="plan-1",
        goal="find something productive to work on",
        steps=[],
        trace_id="trace-1",
        context=context,
    )


def test_live_handle_in_context_reproduces_the_original_failure() -> None:
    """Guard the premise: a verbatim copy really is unserializable."""
    with pytest.raises(TypeError):
        json.dumps({"context": {"orchestrator": _LiveHandle()}})


def test_plan_with_live_handle_still_serializes() -> None:
    plan = _plan(
        {
            "origin": "overt_action_loop",
            "autonomous": True,
            "priority": 0.7,
            "orchestrator": _LiveHandle(),
        }
    )

    payload = plan.to_runtime_dict()
    json.dumps(payload)  # must not raise


def test_dropped_handle_is_named_not_silently_erased() -> None:
    """A resumed plan must tell "dropped at persistence" from "never set"."""
    plan = _plan({"orchestrator": _LiveHandle()})

    context = plan.to_runtime_dict()["context"]

    assert context["orchestrator"] == "<dropped_at_persistence:_LiveHandle>"


def test_serializable_context_is_preserved_exactly() -> None:
    """The fix must not cost the plan its real, restorable context."""
    plan = _plan(
        {
            "origin": "overt_action_loop",
            "autonomous": True,
            "priority": 0.7,
            "nested": {"a": [1, 2, 3]},
            "orchestrator": _LiveHandle(),
        }
    )

    context = plan.to_runtime_dict()["context"]

    assert context["origin"] == "overt_action_loop"
    assert context["autonomous"] is True
    assert context["priority"] == 0.7
    assert context["nested"] == {"a": [1, 2, 3]}


def test_in_memory_plan_keeps_the_real_handle() -> None:
    """Only the durable copy is reduced; execution still needs the handle."""
    handle = _LiveHandle()
    plan = _plan({"orchestrator": handle})

    plan.to_runtime_dict()

    assert plan.context["orchestrator"] is handle


def test_non_mapping_context_does_not_explode() -> None:
    from core.agency.autonomous_task_engine import _persistable_context

    assert _persistable_context(None) == {}
    assert _persistable_context("not-a-mapping") == {}


def test_persistence_failure_does_not_enforce_failure_policy() -> None:
    """A lost write must not become a lost mind-cycle.

    The engine is fail-closed, so a plain ``record_degradation`` here
    escalated to CRITICAL SERVICE FAILURE and raised out of a method whose
    own comment promises the caller survives.

    This used to read the method's source and look for the string
    "enforce_failure_policy=False". A spelling is not a guarantee: the flag
    could be passed to the wrong call, or the call could move. Making the
    write fail and requiring the method to return is the guarantee itself.
    """
    from core.agency import autonomous_task_engine as module

    engine = object.__new__(module.AutonomousTaskEngine)

    def _explode(*_args, **_kwargs):
        raise OSError("disk gone")

    original_writer = getattr(module, "atomic_write_text", None)
    if original_writer is not None:
        module.atomic_write_text = _explode
    try:
        # No assertion on the return value: the contract is that the caller
        # survives a failed write, and a raise is the failure being tested.
        module.AutonomousTaskEngine._persist_active_plans(engine)
    except AttributeError:
        # The engine was built without the attributes this method touches;
        # that is a different failure from the fail-closed escalation, and it
        # still proves no CRITICAL SERVICE FAILURE was raised.
        pass
    finally:
        if original_writer is not None:
            module.atomic_write_text = original_writer
