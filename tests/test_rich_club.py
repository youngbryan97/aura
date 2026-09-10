"""The hubs cortex has, and whether her long-range wiring has them.

Van den Heuvel and Sporns measured that the human connectome's best-connected
regions are wired to each other past what their degrees require. Her long-range
wiring had the opposite: measured against degree-preserving rewiring of her own
graph, the normalised coefficient FELL as the cutoff rose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.connectome.rich_club import (
    HUMAN_RICH_CLUB,
    hub_columns,
    measure_rich_club,
    rich_club_coefficient,
)

RECORD = Path("artifacts/connectome/rich_club.json")


def test_the_measurement_behind_the_wiring_is_named():
    assert "van den Heuvel" in HUMAN_RICH_CLUB["source"]
    assert "human" in HUMAN_RICH_CLUB["recorded_in"]
    assert HUMAN_RICH_CLUB["falsified_by"].strip()


def test_one_node_in_seven_belongs_to_the_club():
    assert HUMAN_RICH_CLUB["hub_fraction"] == pytest.approx(12 / 82)


def test_a_ring_has_no_club():
    """Every node has the same degree, so there is nothing for a club to be."""
    size = 20
    adjacency = np.zeros((size, size), dtype=bool)
    for index in range(size):
        adjacency[index, (index + 1) % size] = True
        adjacency[(index + 1) % size, index] = True
    assert measure_rich_club(adjacency, nulls=5).peak <= 1.0


def test_a_graph_built_with_a_club_has_one():
    rng = np.random.default_rng(1)
    size = 64
    adjacency = rng.random((size, size)) < 0.05
    adjacency = adjacency | adjacency.T
    hubs = np.zeros(size, dtype=bool)
    hubs[:10] = True
    block = rng.random((10, 10)) < 0.8
    adjacency[np.ix_(hubs, hubs)] |= block | block.T
    np.fill_diagonal(adjacency, False)
    assert measure_rich_club(adjacency, nulls=10).has_a_club


def test_the_coefficient_is_a_density():
    adjacency = np.ones((6, 6), dtype=bool)
    np.fill_diagonal(adjacency, False)
    assert rich_club_coefficient(adjacency, 1) == pytest.approx(1.0)


def test_a_cutoff_past_every_degree_returns_nothing():
    adjacency = np.zeros((6, 6), dtype=bool)
    assert rich_club_coefficient(adjacency, 99) == 0.0


def test_hubs_are_drawn_from_the_bands_cortex_draws_them_from():
    names = ["sensory"] * 16 + ["association"] * 32 + ["executive"] * 16
    hubs = hub_columns(64, 12 / 82, names, np.random.default_rng(3))
    assert hubs.sum() == 9
    assert not any(hubs[index] for index in range(16)), "a primary sensory hub"


def test_hubs_fall_back_to_every_column_when_the_bands_are_unknown():
    hubs = hub_columns(10, 0.5, ["?"] * 10, np.random.default_rng(3))
    assert hubs.sum() == 5


def test_the_mesh_now_has_a_club():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    assert int(mesh._hubs.sum()) == 9
    assert measure_rich_club(mesh._inter_W, nulls=20).has_a_club


def test_turning_the_club_off_takes_it_away():
    """The knob has to be able to remove what it added, or it proves nothing."""
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig(hub_fraction=0.0, hub_coupling=1.0))
    assert int(mesh._hubs.sum()) <= 1


def test_the_recorded_sweep_goes_past_the_human_setting():
    """Cortex is a floor. A sweep that stops at it cannot show what is above."""
    if not RECORD.exists():
        pytest.skip("the rich-club sweep has not been recorded yet")
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["sweep"], "an empty sweep"
    assert max(row["coupling"] for row in record["sweep"]) > 1.0
    assert any(row["clubs"] * 2 > row["of"] for row in record["sweep"])
