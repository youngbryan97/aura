"""The brainstem's memory floor has to follow the model bound to the lane.

LIVE 2026-09-18, every thirty seconds for a whole uptime:

    Deferring background local endpoint Brainstem
    (desktop_background_headroom:Brainstem:67.9%/20.6GB(need <100.0% and >=22.0GB))

The lane was configured, registered, reported by the router, and could never
load. 22.0 was calibrated when the lane held a 9B at about 6GB; the host holds
around 20GB available with the Cortex resident, so the floor asked for more
than the machine ever has.

The comment beside that constant already recorded the same shape happening
once before — a 48% / 34GB-free gate that could never admit, so mind_tick
never completed a tick, the launcher read that as death and respawned a
second 32B. A constant cannot follow the model it is protecting, and this is
the second time it did not.
"""
from __future__ import annotations

import math

import pytest

from core.brain import llm_background_deferral as deferral


def test_the_floor_is_above_what_the_model_costs_to_load():
    """It can never admit something bigger than the room it demands.

    A 4GB floor once admitted a 9B beside the Cortex and the emergency
    reclaimer killed the Cortex. A floor derived from the checkpoint cannot
    make that mistake, because the footprint is what it is derived FROM.
    """
    from core.brain.llm.how_big_is_the_checkpoint import _projected_model_footprint_gb
    from core.brain.llm.model_registry import get_brainstem_path

    projected = float(_projected_model_footprint_gb(get_brainstem_path()))
    floor = deferral._floor_for_the_model_on_this_lane()

    assert math.isfinite(floor) and floor > 0.0
    assert floor > projected, (floor, projected)


def test_the_floor_moves_when_the_lane_model_does(monkeypatch):
    """Two models, two floors. A constant would give the same answer twice."""

    def _floor_for(projected_gb: float) -> float:
        monkeypatch.setattr(
            "core.brain.llm.how_big_is_the_checkpoint._projected_model_footprint_gb",
            lambda _path: projected_gb,
        )
        return deferral._floor_for_the_model_on_this_lane()

    small = _floor_for(6.0)
    large = _floor_for(11.0)

    assert large > small, (small, large)


def test_an_unreadable_checkpoint_does_not_admit_an_unknown_size(monkeypatch):
    """The old constant is the fallback, not zero.

    Failing open here would admit a model whose footprint nobody could read,
    which is the one case where holding the conservative number is right.
    """

    def _raise(_path):
        raise OSError("no artifact")

    monkeypatch.setattr(
        "core.brain.llm.how_big_is_the_checkpoint._projected_model_footprint_gb",
        _raise,
    )
    assert (
        deferral._floor_for_the_model_on_this_lane()
        == deferral._BRAINSTEM_FLOOR_IF_UNREADABLE_GB
    )


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_a_nonsense_footprint_holds_the_conservative_floor(monkeypatch, bad):
    """NaN compares false against everything, which is fail-open admission."""
    monkeypatch.setattr(
        "core.brain.llm.how_big_is_the_checkpoint._projected_model_footprint_gb",
        lambda _path: bad,
    )
    assert (
        deferral._floor_for_the_model_on_this_lane()
        == deferral._BRAINSTEM_FLOOR_IF_UNREADABLE_GB
    )


def test_the_gate_reads_the_derived_floor_rather_than_a_literal():
    """The constant must not be back in the comparison by another route."""
    from tests.source_contract import function_with_its_helpers

    source = function_with_its_helpers(
        deferral, "_desktop_background_endpoint_deferral_reason", depth=2
    )
    assert "_floor_for_the_model_on_this_lane()" in source
    assert "AURA_BACKGROUND_BRAINSTEM_MIN_AVAILABLE_GB" in source, (
        "the override stays, so this can be tuned without a code change"
    )
