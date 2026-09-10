"""The wiring law H01 measured, and the test that chose between two of them.

Three numbers from a cubic millimetre of human temporal cortex. Two laws that
could have produced the first of them. The third settles which.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.connectome.island import (
    MAX_CONTACTS,
    REFUTED_BELOW,
    candidate_laws,
    connected_pairs,
    contacts_per_pair,
    expected_heaviest_pair,
    probability_of_the_observed_maximum,
    surviving_law,
    wire_island,
)
from core.connectome.types import H01_REFERENCE


def test_both_laws_reproduce_the_single_contact_fraction():
    """It is what each was fitted to, so neither is predicting anything here."""
    measured = float(H01_REFERENCE.get("single_contact_fraction"))
    for law in candidate_laws():
        assert law.p(1) == pytest.approx(measured, abs=1e-6)


def test_the_one_parameter_law_predicts_the_four_or_more_fraction_and_misses():
    """Fitted to one number, wrong about the second by half. That is a result."""
    plain = candidate_laws()[0]
    measured = float(H01_REFERENCE.get("four_or_more_contact_fraction"))
    predicted = plain.p_at_least(4)
    assert predicted > measured
    assert predicted / measured == pytest.approx(1.58, abs=0.05)


def test_the_two_parameter_law_predicts_nothing():
    """Two parameters against two numbers is a fit; it hits both exactly."""
    damped = candidate_laws()[1]
    assert damped.p_at_least(4) == pytest.approx(
        float(H01_REFERENCE.get("four_or_more_contact_fraction")), abs=1e-8
    )
    assert len(damped.fitted_to) == 2


def test_the_observed_maximum_refutes_the_cutoff():
    """Neither law was fitted to it, and only one survives it."""
    plain, damped = candidate_laws()
    assert probability_of_the_observed_maximum(plain) > 0.3
    assert probability_of_the_observed_maximum(damped) < 1e-6
    assert probability_of_the_observed_maximum(damped) < REFUTED_BELOW


def test_the_surviving_law_is_the_one_fitted_to_less():
    law = surviving_law()
    assert law.name == "power law"
    assert law.fitted_to == ("single_contact_fraction",)


def test_the_volume_holds_about_as_many_pairs_as_synapses():
    """Mean multiplicity is just over one, so the two counts nearly agree."""
    law = surviving_law()
    pairs = connected_pairs(law)
    assert 1.0 < law.mean() < 1.1
    assert pairs == pytest.approx(float(H01_REFERENCE.get("synapses")) / law.mean())


def test_the_heaviest_pair_is_expected_about_once():
    """H01 saw one. A law expecting none of them did not produce that volume."""
    assert 0.1 < expected_heaviest_pair(surviving_law()) < 10.0


def test_contacts_are_whole_numbers_within_the_observed_support():
    law = surviving_law()
    drawn = contacts_per_pair(law, 20_000, np.random.default_rng(3))
    assert drawn.min() >= 1
    assert drawn.max() <= MAX_CONTACTS
    assert np.mean(drawn == 1) == pytest.approx(law.p(1), abs=0.01)


def test_an_island_has_a_tail_a_gaussian_does_not():
    rng = np.random.default_rng(11)
    island = wire_island(256, 0.128775, rng)
    strengths = np.abs(island[island != 0])
    assert strengths.size > 0
    # Every strength is a whole number of contacts at the contact strength.
    contacts = strengths / 0.1
    assert np.allclose(contacts, np.round(contacts))
    assert strengths.max() / np.median(strengths) >= 4.0


def test_an_island_keeps_the_density_it_was_asked_for():
    rng = np.random.default_rng(5)
    island = wire_island(512, 0.1, rng)
    density = np.count_nonzero(island) / (512 * 511)
    assert density == pytest.approx(0.1, abs=0.01)


def test_an_island_has_no_self_connections():
    island = wire_island(64, 0.9, np.random.default_rng(1))
    assert not np.any(np.diag(island))


def test_an_island_needs_units():
    with pytest.raises(ValueError, match="island_needs_units"):
        wire_island(0, 0.1, np.random.default_rng(1))


def test_the_mesh_wires_every_column_from_the_measurement():
    """Intra-column wiring is local wiring, which is what H01 measured."""
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    assert all(column.human_island for column in mesh.columns)
    strengths = np.concatenate(
        [np.abs(column.W[column.W != 0]).ravel() for column in mesh.columns]
    )
    assert float(np.median(strengths)) == pytest.approx(0.1, abs=1e-6)
    assert float(strengths.max()) > 0.4


def test_a_partial_island_wires_only_what_it_names():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig(human_island_columns=8))
    assert sum(1 for column in mesh.columns if column.human_island) == 8


def test_no_island_leaves_the_mesh_as_it_was():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig(human_island_columns=0))
    assert not any(column.human_island for column in mesh.columns)


# ── Regions with different computational matter ────────────────────────────


def test_each_tier_carries_its_own_layers_composition():
    """One figure averaged over four layers is not what any of them measured."""
    from core.connectome.cortical_constants import derived_tier_constants

    tiers = derived_tier_constants()
    assert tiers["sensory"]["inhibitory_fraction"] == pytest.approx(0.2000, abs=1e-4)
    assert tiers["association"]["inhibitory_fraction"] == pytest.approx(0.2200, abs=1e-4)
    assert tiers["executive"]["inhibitory_fraction"] == pytest.approx(0.1725, abs=1e-4)
    fractions = {entry["inhibitory_fraction"] for entry in tiers.values()}
    assert len(fractions) == 3, "three bands with one number between them is not three bands"


def test_tier_densities_come_from_the_same_matrix():
    from core.connectome.cortical_constants import derived_tier_constants
    from core.connectome.microcircuit import CORTICAL_CONN_PROBS

    tiers = derived_tier_constants()
    inside_layer_four = [CORTICAL_CONN_PROBS[row][column] for row in (2, 3) for column in (2, 3)]
    assert tiers["sensory"]["intra_column_density"] == pytest.approx(
        sum(inside_layer_four) / 4, abs=1e-6
    )


def test_the_mesh_wires_its_tiers_differently():
    from core.consciousness.neural_mesh import CorticalTier, MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    observed = {}
    for tier in CorticalTier:
        columns = [column for column in mesh.columns if column.tier is tier]
        observed[tier.name.lower()] = (
            float(np.mean([column.inh_mask.mean() for column in columns])),
            float(
                np.mean(
                    [
                        np.count_nonzero(column.W) / (column.n * (column.n - 1))
                        for column in columns
                    ]
                )
            ),
        )
    assert observed["association"][0] > observed["sensory"][0] > observed["executive"][0]
    assert observed["association"][1] > observed["sensory"][1] > observed["executive"][1]


# ── One stream per structure ───────────────────────────────────────────────


def test_local_wiring_cannot_move_the_long_range_graph():
    """The confound that made the tier change look like a reachability failure.

    Everything drew from one stream in construction order, so changing how a
    column was wired changed how many numbers came out before the long-range
    matrices were built. The tier change appeared to cost three columns and
    three executive targets, and had caused none of it.
    """
    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    graphs = []
    for tier_table in (None, {}):
        mesh_module._TIER_CONSTANTS = tier_table
        graphs.append(np.array(NeuralMesh(MeshConfig())._inter_W, copy=True))
    mesh_module._TIER_CONSTANTS = None
    assert np.array_equal(graphs[0] != 0, graphs[1] != 0)


def test_the_streams_are_reproducible():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    first, second = NeuralMesh(MeshConfig()), NeuralMesh(MeshConfig())
    assert np.array_equal(first._inter_W, second._inter_W)
    assert np.array_equal(first.columns[0].W, second.columns[0].W)


# ── No column is tissue the mesh cannot use ────────────────────────────────


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 12345])
def test_every_column_has_a_way_in_and_a_way_out(seed, monkeypatch):
    """A column with no edges is not a quiet column. There are none in cortex."""
    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    monkeypatch.setattr(mesh_module, "_MESH_SEED", seed)
    weights = NeuralMesh(MeshConfig())._inter_W
    present = np.abs(weights) > 0
    assert not np.any(~present.any(axis=1)), "a column with nothing leaving it"
    assert not np.any(~present.any(axis=0)), "a column with nothing arriving"


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 12345])
def test_a_sensory_signal_reaches_every_executive_column(seed, monkeypatch):
    from core.connectome.neural import build_mesh_layer, signal_can_cross
    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    monkeypatch.setattr(mesh_module, "_MESH_SEED", seed)
    report = signal_can_cross(build_mesh_layer(NeuralMesh(MeshConfig())))
    assert report["executive_reached_from_sensory"] == report["executive_columns"]
    assert report["components"] == 1
    assert report["isolated_columns"] == 0


# ── Dale's law, on the axis a synapse actually has ─────────────────────────


def test_a_cell_sends_one_sign():
    """`recurrent = W @ x` makes a cell's output its COLUMN, not its row.

    The mesh negated the row, which is a cell's input, so every inhibitory cell
    received nothing but inhibition and sent whichever sign the draw gave it.
    Measured on column 0 before the fix: 71.7% of what inhibitory cells sent
    was negative and 64.4% of what excitatory cells sent was negative too.
    """
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    for column in NeuralMesh(MeshConfig()).columns[:8]:
        outgoing = column.W[:, column.inh_mask]
        assert np.all(outgoing[outgoing != 0] < 0)
        outgoing = column.W[:, ~column.inh_mask]
        assert np.all(outgoing[outgoing != 0] > 0)


def test_an_inhibitory_synapse_is_stronger_than_an_excitatory_one():
    """Potjans and Diesmann's g, and the reason the mesh does not saturate."""
    from core.consciousness.neural_mesh import MeshConfig

    assert MeshConfig().relative_inhibitory_strength == pytest.approx(4.0)


def test_dale_survives_plasticity():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    width = mesh.cfg.sensory_end * mesh.cfg.neurons_per_column
    rng = np.random.default_rng(2)
    for _ in range(120):
        mesh.inject_sensory(rng.standard_normal(width).astype(np.float32) * 0.3)
        mesh._tick_inner()
    column = mesh.columns[0]
    outgoing = column.W[:, column.inh_mask]
    assert np.all(outgoing[outgoing != 0] <= 0)
    outgoing = column.W[:, ~column.inh_mask]
    assert np.all(outgoing[outgoing != 0] >= 0)


# ── Two plasticity rules, not one ──────────────────────────────────────────


def test_the_inhibitory_rule_has_its_own_window_and_depression():
    from core.consciousness.neural_mesh import MeshConfig

    config = MeshConfig()
    assert config.inhibitory_stdp_window == pytest.approx(0.020)
    assert config.stdp_window == pytest.approx(0.0168)
    # alpha = 2 * rho * tau, with rho 5 Hz and tau 20 ms.
    assert config.inhibitory_stdp_depression == pytest.approx(0.2)


def test_plasticity_does_not_grow_synapses():
    """It changes synapses that exist. Density is measured at construction."""
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    before = [int(np.count_nonzero(column.W)) for column in mesh.columns]
    width = mesh.cfg.sensory_end * mesh.cfg.neurons_per_column
    rng = np.random.default_rng(4)
    for _ in range(200):
        mesh.inject_sensory(rng.standard_normal(width).astype(np.float32) * 0.3)
        mesh._tick_inner()
    after = [int(np.count_nonzero(column.W)) for column in mesh.columns]
    assert all(later <= earlier for earlier, later in zip(before, after, strict=True))


# ── A unit rests at its drive ──────────────────────────────────────────────


def test_the_mesh_does_not_run_to_its_ceiling():
    """`dx = (-decay*x + drive)*dt` rests at drive/decay, which is forty times
    the drive. Every unit ran to its clip and stayed there."""
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    width = mesh.cfg.sensory_end * mesh.cfg.neurons_per_column
    drive = np.random.default_rng(3).standard_normal(width).astype(np.float32)
    saturated = []
    for _ in range(600):
        mesh.inject_sensory(drive)
        mesh._tick_inner()
        state = np.concatenate([column.x for column in mesh.columns])
        saturated.append(float(np.mean(np.abs(state) > 0.99)))
    assert max(saturated[-100:]) < 0.05, "the mesh is pinned against its clip"


def test_a_driven_column_settles_rather_than_drifting():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    width = mesh.cfg.sensory_end * mesh.cfg.neurons_per_column
    drive = np.random.default_rng(3).standard_normal(width).astype(np.float32)
    trace = []
    for _ in range(600):
        mesh.inject_sensory(drive)
        mesh._tick_inner()
        sensory = np.concatenate(
            [column.x for column in mesh.columns[: mesh.cfg.sensory_end]]
        )
        trace.append(float(np.mean(np.abs(sensory))))
    early, late = np.mean(trace[100:200]), np.mean(trace[500:600])
    assert late > 0.05, "a driven tier that carries nothing"
    assert abs(late - early) < 0.1, "a driven tier that never settles"


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 12345, 2026, 7777])
def test_reachability_is_guaranteed_not_just_degree(seed, monkeypatch):
    """Every column having edges at both ends does not make it reachable.

    Found when the rich club changed the long-range graph: on one seed a
    sensory signal reached 15 of 16 executive columns while every column held
    edges in both directions.
    """
    from core.connectome.neural import build_mesh_layer, signal_can_cross
    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    monkeypatch.setattr(mesh_module, "_MESH_SEED", seed)
    report = signal_can_cross(build_mesh_layer(NeuralMesh(MeshConfig())))
    assert report["executive_reached_from_sensory"] == report["executive_columns"]


# ── One convention, and three builders that did not follow it ──────────────


def test_the_receiver_is_the_row_everywhere():
    """`recurrent = W @ x` makes W[i, j] the weight from j into i.

    Three builders wrote `weights[source, target]` and one reader read them
    that way too, so each agreed with itself and none agreed with the tick that
    actually moves activity.
    """
    from core.consciousness.neural_mesh import (
        CorticalTier,
        MeshConfig,
        NeuralMesh,
    )

    mesh = NeuralMesh(MeshConfig())
    tiers = [column.tier for column in mesh.columns]

    # Top-down feedback runs from a higher band into a lower one. Read on the
    # wrong axis this matrix put its drive on the executive columns that sent
    # it, so the columns it was written for received nothing at all.
    present = np.abs(mesh._feedback_W) > 0
    for target, source in zip(*np.nonzero(present), strict=True):
        assert tiers[source].value >= tiers[target].value, (
            "a feedback edge runs upward, so the matrix is on the wrong axis"
        )
    assert present.any(), "there is no feedback pathway at all"

    # And a sensory column has to be able to SEND, which is its column of the
    # inter-column matrix.
    inter = np.abs(mesh._inter_W) > 0
    sensory = [
        index
        for index, tier in enumerate(tiers)
        if tier is CorticalTier.SENSORY
    ]
    assert inter[:, sensory].any(), "no sensory column sends anything"


def test_the_feedback_pathway_changes_the_state():
    """It delivered nothing, and an ablation of it was a no-op that passed."""
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    config = MeshConfig(
        total_neurons=512,
        columns=16,
        neurons_per_column=32,
        sensory_end=4,
        association_end=10,
    )
    with_feedback = NeuralMesh(config)
    without = NeuralMesh(config)
    without.set_recurrent_feedback_enabled(False)
    for _ in range(5):
        with_feedback._tick_inner()
        without._tick_inner()
    difference = float(
        np.linalg.norm(with_feedback.get_field_state() - without.get_field_state())
    )
    assert difference > 1e-6, "cutting the top-down pathway changed nothing"
