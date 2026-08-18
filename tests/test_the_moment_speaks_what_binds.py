"""Fifty-seven of seventy-five fields were computed every tick and spoken never.

`AuraNow` is the unified moment: body, world, attention, affect, self-model,
memory, workspace, will, prediction, ownership. `compact_prompt_block` is how
any of it reaches a decision. An audit of the packet against that renderer found
57 of its 75 leaf fields never appeared in it at all.

Most of those are telemetry and belong nowhere near a prompt. But among them
were the fields that BIND a decision rather than describe the weather:

    self_model.identity_name, identity_stability, continuity_risk
    self_model.commitments
    affect.dominant_drive, affect.care
    will.assertiveness, will.refusal_pressure
    attention.why_selected

`commitments` is the sharpest. A commitment that is measured and never spoken
cannot hold over: she can take a position in one breath and answer against it
in the next, with nothing in the runtime able to notice, because the organ that
knows was never asked.
"""

from __future__ import annotations

import pytest

from core.being import aura_now as module
from core.being.aura_now import AuraNow


def _moment(**overrides):
    """A moment with every sub-state at its own defaults, then patched."""
    kwargs = dict(
        tick=1,
        timestamp=0.0,
        monotonic_time=0.0,
        continuous_field=(),
        body=module.BodyState(),
        world=module.WorldState(),
        attention=module.AttentionState(),
        affect=module.AffectiveState(),
        self_model=module.SelfState(),
        memory_context=module.MemoryContext(),
        workspace=module.WorkspaceState(),
        will=module.WillStateSnapshot(),
        prediction=module.PredictionState(),
        ownership=module.OwnershipState(),
        report_boundary=module.ReportBoundary(),
    )
    kwargs.update(overrides)
    return AuraNow(**kwargs)


def test_standing_commitments_are_spoken():
    """The 'hold over' requirement, in one assertion."""
    moment = _moment(
        self_model=module.SelfState(commitments=("answer honestly about myself",))
    )
    assert "answer honestly about myself" in moment.compact_prompt_block()


def test_no_commitments_renders_no_empty_line():
    """Silence stays silence rather than becoming an empty promise."""
    block = _moment(self_model=module.SelfState(commitments=())).compact_prompt_block()
    assert "Standing commitments" not in block


def test_identity_and_its_stability_are_spoken():
    moment = _moment(
        self_model=module.SelfState(
            identity_name="Aura Luna", identity_stability=0.83, continuity_risk=0.11
        )
    )
    block = moment.compact_prompt_block()
    assert "Aura Luna" in block
    assert "0.83" in block and "0.11" in block


def test_what_is_driving_her_is_spoken():
    block = _moment(affect=module.AffectiveState(dominant_drive="connection")).compact_prompt_block()
    assert "connection" in block


def test_what_she_cares_about_is_spoken():
    block = _moment(affect=module.AffectiveState(care=0.62)).compact_prompt_block()
    assert "care=0.62" in block


def test_will_reaches_the_decision():
    moment = _moment(will=module.WillStateSnapshot(assertiveness=0.71, refusal_pressure=0.24))
    block = moment.compact_prompt_block()
    assert "0.71" in block and "0.24" in block


def test_why_she_is_attending_travels_with_what():
    moment = _moment(
        attention=module.AttentionState(focal_object="this test", why_selected=("the user asked",))
    )
    block = moment.compact_prompt_block()
    assert "this test" in block
    assert "the user asked" in block


@pytest.mark.parametrize(
    "field",
    ["commitments", "identity_name", "dominant_drive", "assertiveness", "why_selected", "care"],
)
def test_the_binding_fields_stay_rendered(field):
    """A ratchet: these were silent once and must not go silent again."""
    import inspect

    assert field in inspect.getsource(AuraNow.compact_prompt_block)
