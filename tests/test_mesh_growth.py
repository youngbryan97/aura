"""The mesh arrives by growing, and the growing has to beat cutting at random.

Huttenlocher counted synapses in human cortex across the lifespan: density
overshoots the adult level and is then cut back. These check that the
trajectory runs, that it lands where the anatomy says, and that pruning by what
a synapse carried is worth more than pruning the same number by coin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.connectome.mesh_growth import HUTTENLOCHER, TIER_PEAK, grow_mesh

RECORD = Path("artifacts/connectome/mesh_growth.json")


def test_the_measurement_behind_the_trajectory_is_named():
    assert "Huttenlocher" in HUTTENLOCHER["source"]
    assert "human" in HUTTENLOCHER["recorded_in"]
    assert HUTTENLOCHER["peak_over_adult"] > 1.0


def test_the_bands_peak_in_the_order_cortex_does():
    """Sensory areas finish while prefrontal is still overshooting."""
    assert TIER_PEAK["sensory"] < TIER_PEAK["association"] < TIER_PEAK["executive"]


def test_the_order_is_spent_rather_than_declared():
    """A table nothing reads is a claim with no mechanism behind it.

    The first version of this module cut every band at the end of the run and
    left TIER_PEAK unread by anything but the test above.
    """
    _, result = grow_mesh(ticks=400, seed=7)
    assert result.pruned_at, "no band records when it was cut"
    assert (
        result.pruned_at["sensory"]
        < result.pruned_at["association"]
        < result.pruned_at["executive"]
    )
    assert result.pruned_at["executive"] < 400


def test_growth_overshoots_and_comes_back():
    mesh, result = grow_mesh(ticks=200, seed=7)
    for band, peak in result.peak_density.items():
        target = result.target_density[band]
        assert peak > target, f"{band} never overshot"
        assert peak / target == pytest.approx(HUTTENLOCHER["peak_over_adult"], abs=0.05)


def test_every_band_lands_on_the_density_its_anatomy_specifies():
    _, result = grow_mesh(ticks=200, seed=7)
    for band, adult in result.adult_density.items():
        assert adult == pytest.approx(result.target_density[band], abs=0.002)


def test_growth_makes_synapses_and_pruning_removes_them():
    _, result = grow_mesh(ticks=200, seed=7)
    assert result.grown > 1000
    assert result.pruned > 1000
    assert abs(result.grown - result.pruned) < result.grown * 0.05


def test_the_grown_mesh_still_obeys_dale():
    mesh, _ = grow_mesh(ticks=200, seed=7)
    for column in mesh.columns[:8]:
        sent = column.W[:, column.inh_mask]
        assert np.all(sent[sent != 0] < 0)
        sent = column.W[:, ~column.inh_mask]
        assert np.all(sent[sent != 0] > 0)


def test_pruning_by_traffic_is_scored_against_a_null():
    """A result with no null is an assumption."""
    _, result = grow_mesh(ticks=200, seed=7)
    assert result.carried_at_random > 0.0
    assert result.carried_by_use > 0.0


def test_reachability_cannot_tell_the_two_prunings_apart():
    """It is computed from matrices neither pruning touches, and says so."""
    _, result = grow_mesh(ticks=200, seed=7)
    assert result.reach_by_use == result.reach_at_random


def test_the_recorded_run_beat_the_null_on_most_seeds():
    assert RECORD.exists(), "the growth comparison has never been recorded"
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["of"] >= 5
    assert record["beat_the_null"] * 2 > record["of"]
    assert record["median_ratio"] > 1.0
    assert record["landed_on_anatomy"] is True


def test_the_recorded_run_names_its_source():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert "Huttenlocher" in record["source"]["source"]
