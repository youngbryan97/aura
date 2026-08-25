"""A probe naming an accessor nothing has is a channel that reads absent forever.

This is the failure the codebase keeps finding under different names: a writer
with no reader, a rule that could never match, a channel wired to a method that
does not exist. The first version of the goal probe named four accessors —
``current_goal``, ``active_goal``, ``get_current_goal``, ``top_goal`` — and the
goal engine has none of them. Nothing failed. The channel simply read absent on
every turn, which is indistinguishable from a runtime holding no goals.

So each channel that binds to an importable organ declares which class and
which accessor, and this asserts the binding still exists. It cannot prove the
organ is registered at runtime — that is what the coverage reading in the
health block is for — but it does prove the name is not fiction.
"""

from __future__ import annotations

import importlib

import pytest

from core.brain.llm.endogenous_state import (
    CHANNELS,
    PROBES,
    assemble_state,
    empty_state,
    reset_state_cache,
)

#: channel → (module, class, accessor). One row per binding the probes rely on.
BINDINGS = (
    ("affect", "core.affect.damasio_v2", "AffectEngineV2", "get_snapshot"),
    ("substrate", "core.brain.llm.continuous_substrate", "ContinuousSubstrate", "get_state_vector"),
    ("substrate", "core.brain.llm.continuous_substrate", "ContinuousSubstrate", "get_state_summary"),
    ("goal", "core.goals.goal_engine", "GoalEngine", "get_active_goals"),
    ("memory", "core.memory.recall_observations", "RecallObservationRing", "samples"),
    ("attention", "core.knowledge.atomspace", "AtomSpace", "attentional_focus"),
)

#: Module-level functions a probe calls directly.
FUNCTION_BINDINGS = (
    ("memory", "core.memory.recall_observations", "peek_recall_observations"),
    ("recurrence", "core.brain.llm.user_surface_recurrence", "admit_user_surface_recurrent_loops"),
    ("recurrence", "core.brain.llm.user_surface_recurrence", "user_surface_recurrent_ceiling"),
)


@pytest.mark.parametrize(("channel", "module", "cls", "accessor"), BINDINGS)
def test_the_named_accessor_exists(channel, module, cls, accessor):
    assert channel in CHANNELS
    target = getattr(importlib.import_module(module), cls)
    assert hasattr(target, accessor), (
        f"the {channel} probe reads {cls}.{accessor}, which does not exist"
    )


@pytest.mark.parametrize(("channel", "module", "name"), FUNCTION_BINDINGS)
def test_the_named_function_exists(channel, module, name):
    assert channel in CHANNELS
    assert hasattr(importlib.import_module(module), name), (
        f"the {channel} probe calls {module}.{name}, which does not exist"
    )


def test_every_channel_has_a_probe():
    assert set(PROBES) == set(CHANNELS)


def test_the_memory_probe_never_builds_the_recall_ring(monkeypatch):
    """Building it resolves a store path and touches disk, under a lock.

    Lockdep saw exactly that: an fsync attempted while holding
    ``recall_observations.singleton``. A probe on the request path must peek.
    """
    from core.memory import recall_observations

    def explode():  # pragma: no cover - the assertion is that this never runs
        raise AssertionError("a state probe constructed the recall ring")

    monkeypatch.setattr(recall_observations, "get_recall_observations", explode)
    reset_state_cache()
    assemble_state(max_age_s=0.0)


def test_assembling_a_state_never_raises_with_no_runtime():
    reset_state_cache()
    state = assemble_state(max_age_s=0.0)
    assert set(state.sources) == set(CHANNELS)
    assert all(
        value in {"live", "absent", "error", "ablated"} for value in state.sources.values()
    )


def test_a_channel_that_errored_is_not_reported_as_absent():
    def explode() -> dict[str, float]:
        raise RuntimeError("organ down")

    state = assemble_state(probes={"goal": explode})
    assert state.sources["goal"] == "error"
    assert empty_state().sources["goal"] == "absent"
