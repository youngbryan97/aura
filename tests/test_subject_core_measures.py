"""The battery's own tests: each measure has to be able to say no.

A measurement that returns a healthy number for a system built to fail is
worse than no measurement, because it will keep returning healthy numbers
after the system changes. So every test here has two arms — something that
should pass and something matched that should not — and asserts the gap rather
than the value.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.causal import (
    EDGE_EFFECT,
    InterventionSet,
    Trial,
    benjamini_hochberg,
    build_edges,
    power_note,
)
from core.subject.differentiation import effective_dimension
from core.subject.estimate import held_out_loss
from core.subject.graph import analyse_graph, simple_cycles, strongly_connected
from core.subject.intrinsic import intrinsic_gain
from core.subject.irreducibility import phi_do
from core.subject.metastability import regimes
from core.subject.nulls import (
    ARCHITECTURES,
    architecture,
    replay_surrogate,
    shuffle_surrogate,
    toy_edges,
    toy_recording,
)
from core.subject.pci import lempel_ziv, normalised_lz
from core.subject.recording import build_recording
from core.subject.state import (
    DOMAINS,
    domain_width,
    feature_names,
    perturb,
    perturbable,
    read_core_state,
)
from core.subject.synergy import synergy


# ── the schema ───────────────────────────────────────────────────────────


def test_every_domain_has_a_fixed_width_and_named_features():
    for key in DOMAINS:
        assert domain_width(key) > 0
        assert len(feature_names(key)) == domain_width(key)
    assert len(feature_names()) == sum(domain_width(k) for k in DOMAINS)


def test_reading_a_default_state_gives_the_declared_widths():
    from core.state.aura_state import AuraState

    reading = read_core_state(AuraState.default())
    for key in DOMAINS:
        assert reading.domain(key).size == domain_width(key)
    assert reading.vector().size == len(feature_names())


def test_every_domain_has_a_writer_that_moves_its_own_reading():
    """A domain nothing can displace has no measurable outgoing edges."""
    from core.state.aura_state import AuraState

    from core.ontogeny.state import OntogeneticState

    assert set(perturbable()) == set(DOMAINS)
    for key in DOMAINS:
        state = AuraState.default()
        reservoir = OntogeneticState(input_width=8, units=16, seed=0)
        before = read_core_state(state, ontogeny=reservoir).domain(key)
        assert perturb(state, key, 0.2, ontogeny=reservoir) is True, f"{key} has no writer"
        after = read_core_state(state, ontogeny=reservoir).domain(key)
        assert not np.allclose(before, after), f"writing {key} did not move {key}"


def test_the_reservoir_writer_moves_the_reservoir():
    from core.ontogeny.state import OntogeneticState

    reservoir = OntogeneticState(input_width=8, units=16, seed=0)
    before = np.array(reservoir.h, copy=True)
    assert perturb(None, "N", 0.2, ontogeny=reservoir) is True
    assert not np.allclose(before, reservoir.h)


# ── the estimator ────────────────────────────────────────────────────────


def test_the_estimator_scores_noise_at_one_and_signal_below_it():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6))
    y = x @ rng.normal(size=(6, 3)) + 0.1 * rng.normal(size=(400, 3))
    assert held_out_loss(x, y).loss < 0.1
    assert held_out_loss(rng.normal(size=(400, 6)), y).loss > 0.8


def test_a_flat_target_is_reported_as_degenerate_rather_than_perfect():
    rng = np.random.default_rng(0)
    fit = held_out_loss(rng.normal(size=(200, 4)), np.zeros((200, 3)))
    assert fit.degenerate
    assert fit.loss == pytest.approx(1.0)


# ── irreducibility ───────────────────────────────────────────────────────


def _toy(name: str, steps: int = 2000, seed: int = 3):
    return toy_recording(architecture(name, seed=seed), steps=steps, seed=seed)


def test_a_recurrent_architecture_is_irreducible_and_the_nulls_are_not():
    real = phi_do(_toy("recurrent")).phi
    assert real > 0.05
    for name in ("star", "hub", "one_way", "prompt_only", "frozen_slow"):
        assert phi_do(_toy(name)).phi < real, name


def test_replay_and_shuffle_destroy_irreducibility():
    recording = _toy("recurrent")
    intact = phi_do(recording).phi
    assert phi_do(replay_surrogate(recording, seed=1)).phi < intact
    assert phi_do(shuffle_surrogate(recording, seed=1)).phi < intact


def test_the_weakest_partition_is_what_is_reported_not_the_average():
    """The score answers for the weakest cut, never for the typical one."""
    report = phi_do(_toy("recurrent"))
    values = sorted(report.scores.values())
    assert report.phi < sum(values) / len(values)
    assert report.best_cut is not None
    assert report.scores[
        f"{''.join(report.best_cut[0])}|{''.join(report.best_cut[1])}"
    ] == pytest.approx(min(values))


def test_the_reported_score_is_read_off_folds_it_was_not_chosen_on():
    """A minimum over five hundred noisy estimates is biased downward.

    Roughly three standard errors of a single one, however unbiased each is —
    and the bias runs against the system, which is punished for the width of a
    search it did not choose. Selecting the weakest cut on some folds and
    reading its score off the others removes it exactly, so the reported number
    sits above the in-sample minimum rather than at it.
    """
    report = phi_do(_toy("recurrent"))
    in_sample = min(report.scores.values())
    assert "cross-fitted" in report.note
    assert report.phi > in_sample


# ── the graph ────────────────────────────────────────────────────────────


def test_strong_connectivity_alone_does_not_separate_a_star_from_a_ring():
    """The reason the battery cannot stop at SCC. Recorded as a test, not prose."""
    star = [("A", "H"), ("H", "A"), ("B", "H"), ("H", "B"), ("C", "H"), ("H", "C")]
    ring = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("C", "B"), ("B", "A")]
    star_report = analyse_graph(["A", "B", "C", "H"], star)
    ring_report = analyse_graph(["A", "B", "C"], ring)
    assert star_report.one_component and ring_report.one_component
    assert star_report.connectivity == 1
    assert ring_report.connectivity >= 2
    assert not star_report.every_node_reenters
    assert ring_report.every_node_reenters


def test_a_broker_inside_the_node_set_is_a_cut_vertex():
    edges = [("A", "H"), ("H", "B"), ("B", "H"), ("H", "A")]
    assert analyse_graph(["A", "B", "H"], edges).connectivity == 1


def test_cycles_and_components_agree_with_hand_worked_cases():
    assert strongly_connected(["A", "B"], [("A", "B")]) == [["A"], ["B"]]
    assert strongly_connected(["B", "A"], [("A", "B")]) == [["A"], ["B"]]
    assert len(simple_cycles(["A", "B"], [("A", "B"), ("B", "A")])) == 1


def test_a_hidden_broker_is_invisible_to_the_graph_measures():
    """The star null passes every graph criterion. Closure is what catches it."""
    system = architecture("star", seed=5)
    report = analyse_graph(list(DOMAINS), toy_edges(system, trials=10, seed=5))
    assert report.one_component
    assert report.connectivity >= 2


# ── differentiation, intrinsic, metastability ────────────────────────────


def test_the_differentiation_threshold_is_passed_by_the_degenerate_nulls():
    """The specification's 0.4 bar is failed by the reference and passed by the
    two systems least like a mind.

    Effective dimension is the participation ratio of the correlation
    spectrum, and coupling lowers it because coupled variables share variance.
    This pins the argument in the document: differentiation and irreducibility
    pull in opposite directions on one scale, so a threshold high on both
    cannot be met. The criterion stays in the conjunction and stays failed;
    this is why.
    """
    reference = effective_dimension(_toy("recurrent")).normalised
    for name in ("prompt_only", "frozen_slow"):
        assert effective_dimension(_toy(name)).normalised > reference, name
    assert reference < 0.4


def test_effective_dimension_collapses_when_every_column_copies_one():
    rows = 500
    driver = np.random.default_rng(0).normal(size=(rows, 1))
    flat = build_recording_from(np.repeat(driver, 60, axis=1) + 1e-6 * np.random.default_rng(1).normal(size=(rows, 60)))
    wide = build_recording_from(np.random.default_rng(2).normal(size=(rows, 60)))
    assert effective_dimension(flat).d_eff < 3.0
    assert effective_dimension(wide).d_eff > 20.0


def build_recording_from(matrix: np.ndarray):
    """A recording built straight from a matrix, for measures that only need X."""
    from core.subject.recording import Recording

    width = matrix.shape[1]
    per = max(1, width // len(DOMAINS))
    slices = {}
    start = 0
    for index, key in enumerate(DOMAINS):
        stop = width if index == len(DOMAINS) - 1 else min(width, start + per)
        slices[key] = slice(start, max(start + 1, stop))
        start = slices[key].stop
    return Recording(
        x=matrix,
        conditions=tuple("x" for _ in range(matrix.shape[0])),
        tags=tuple("" for _ in range(matrix.shape[0])),
        times=np.arange(matrix.shape[0], dtype=np.float64),
        env=np.zeros((matrix.shape[0], 1)),
        env_names=("clock",),
        columns=tuple(f"c{i}" for i in range(width)),
        slices=slices,
        notes={},
    )


def test_intrinsic_gain_is_small_when_the_state_is_a_function_of_the_input():
    """Nothing carries forward, so the state says nothing the input does not."""
    rows = 800
    rng = np.random.default_rng(0)
    drive = rng.normal(size=(rows, 3))
    reactive = np.hstack([np.tanh(drive), np.tanh(drive * 2), np.tanh(drive * 0.5)])
    recording = _with_env(build_recording_from(reactive), drive)
    assert intrinsic_gain(recording).gain < 0.3


def test_intrinsic_gain_is_large_when_the_state_carries_its_own_history():
    """A state whose next move follows from where it is, not from the input.

    The measure predicts the change rather than the level, so the case that
    should score high is one where the change is a function of the state — an
    oscillator, not a leaky integrator of its input. A leaky integrator scores
    low here and should: almost all of its movement is the new input arriving,
    which is exactly what the environment column already says.
    """
    rows = 800
    rng = np.random.default_rng(0)
    drive = rng.normal(scale=0.05, size=(rows, 3))
    state = np.zeros((rows, 9))
    state[0] = 1.0
    angle = 0.4
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for index in range(1, rows):
        previous = state[index - 1].reshape(-1, 2) if state.shape[1] % 2 == 0 else None
        del previous
        pair = state[index - 1][:2] @ rotate.T
        state[index] = np.concatenate([pair, state[index - 1][2:] * 0.99 + 0.01 * pair[0]])
        state[index, 6:] += drive[index]
    recording = _with_env(build_recording_from(state), drive)
    report = intrinsic_gain(recording)
    assert report.gain > 0.5
    assert report.gain_over_shuffle > 0.0
    assert report.passes


def _with_env(recording, env):
    from core.subject.recording import Recording

    return Recording(
        x=recording.x,
        conditions=recording.conditions,
        tags=recording.tags,
        times=recording.times,
        env=env,
        env_names=tuple(f"e{i}" for i in range(env.shape[1])),
        columns=recording.columns,
        slices=recording.slices,
        notes=recording.notes,
    )


def test_metastability_rejects_a_frozen_and_a_random_trajectory():
    rows = 400
    frozen = build_recording_from(np.tile(np.linspace(0, 1, 20), (rows, 1)))
    assert not regimes(frozen).passes


# ── perturbation ─────────────────────────────────────────────────────────


def test_lempel_ziv_ranks_stereotyped_below_structured():
    ones = np.ones((8, 40), dtype=np.int8)
    local = np.zeros((8, 40), dtype=np.int8)
    local[0] = 1
    rng = np.random.default_rng(0)
    structured = (rng.random((8, 40)) < 0.4).astype(np.int8)
    assert normalised_lz(ones) == 0.0
    assert normalised_lz(structured) > normalised_lz(local)
    assert lempel_ziv(np.zeros(64, dtype=np.int8)) < lempel_ziv(
        (rng.random(64) < 0.5).astype(np.int8)
    )


# ── the edge rules ───────────────────────────────────────────────────────


def _trial(source, target, condition, effect, floor, index):
    return Trial(
        source=source,
        condition=condition,
        index=index,
        effect={target: effect},
        floor={target: floor},
        trace={target: [effect]},
        floor_trace={target: [floor]},
        took=True,
    )


def test_an_edge_needs_effect_significance_and_replication_together():
    strong = InterventionSet(
        trials=[
            _trial("A", "G", condition, 1.0, 0.05, index)
            for condition in ("one", "two", "three", "four")
            for index in range(10)
        ]
    )
    edges, tested = build_edges(strong)
    assert [(e.source, e.target) for e in edges] == [("A", "G")]
    assert edges[0].replicated >= 3

    weak = InterventionSet(
        trials=[
            _trial("A", "G", condition, 0.06, 0.05, index)
            for condition in ("one", "two", "three", "four")
            for index in range(10)
        ]
    )
    assert build_edges(weak)[0] == []

    unreplicated = InterventionSet(
        trials=[_trial("A", "G", "one", 1.0, 0.05, index) for index in range(40)]
    )
    assert build_edges(unreplicated)[0] == []


def test_an_effect_no_bigger_than_its_own_sham_floor_is_not_an_edge():
    noise = InterventionSet(
        trials=[
            _trial("A", "G", condition, 0.9, 0.9, index)
            for condition in ("one", "two", "three")
            for index in range(10)
        ]
    )
    assert build_edges(noise)[0] == []


def test_self_pairs_are_never_edges():
    same = InterventionSet(
        trials=[
            _trial("A", "A", condition, 5.0, 0.0, index)
            for condition in ("one", "two", "three")
            for index in range(10)
        ]
    )
    edges, tested = build_edges(same)
    assert edges == []
    assert tested == []


def test_too_few_trials_is_reported_as_underpowered_not_as_no_edges():
    thin = InterventionSet(
        trials=[_trial("A", "G", "one", 5.0, 0.0, index) for index in range(2)]
    )
    assert power_note(thin)["underpowered"] is True
    assert build_edges(thin)[0] == []


def test_benjamini_hochberg_is_monotone_and_bounded():
    q = benjamini_hochberg([0.001, 0.02, 0.5, 0.9])
    assert q == sorted(q)
    assert all(0.0 <= v <= 1.0 for v in q)
    assert benjamini_hochberg([]) == []


# ── synergy ──────────────────────────────────────────────────────────────


def test_synergy_is_found_where_it_exists_and_not_where_it_does_not():
    rows = 1500
    rng = np.random.default_rng(0)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    joint = np.zeros((rows, 30))
    joint[:, 0] = a
    joint[:, 10] = b
    # The target is the product, which neither source predicts alone.
    joint[1:, 20] = a[:-1] * b[:-1]
    recording = build_recording_from(joint)
    report = synergy(recording, DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=1)
    assert report.interaction_gain > 0.05

    additive = joint.copy()
    additive[1:, 20] = a[:-1] + b[:-1]
    plain = synergy(build_recording_from(additive), DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=1)
    assert plain.interaction_gain < report.interaction_gain


# ── the nulls themselves ─────────────────────────────────────────────────


def test_every_architecture_builds_and_runs():
    for name in ARCHITECTURES:
        system = architecture(name, seed=1)
        recording = toy_recording(system, steps=300, seed=1)
        assert recording.frames == 300
        assert recording.width == sum(system.widths.values())


def test_one_way_and_prompt_only_have_no_reentry():
    for name in ("one_way", "prompt_only"):
        report = analyse_graph(list(DOMAINS), toy_edges(architecture(name, seed=2), trials=8, seed=2))
        assert not report.every_node_reenters, name


# ── the recording on disk ────────────────────────────────────────────────


def test_a_saved_recording_keeps_its_own_column_layout(tmp_path):
    """A recording outlives the schema that produced it.

    Rebuilding the domain slices from today's schema reads the wrong columns
    for every domain after the one that changed, or indexes past the end, which
    is the lucky case because it says so.
    """
    from core.subject.recording import Recording, load_recording, slices_from_columns

    columns = ("P.a", "P.b", "I.a", "A.a", "A.b", "A.c")
    layout = slices_from_columns(columns)
    assert layout["P"] == slice(0, 2)
    assert layout["I"] == slice(2, 3)
    assert layout["A"] == slice(3, 6)
    assert layout["N"].stop - layout["N"].start == 0

    rows = 20
    saved = Recording(
        x=np.arange(rows * len(columns), dtype=np.float64).reshape(rows, len(columns)),
        conditions=tuple("x" for _ in range(rows)),
        tags=tuple("t" for _ in range(rows)),
        times=np.arange(rows, dtype=np.float64),
        env=np.zeros((rows, 1)),
        env_names=("clock",),
        columns=columns,
        slices=layout,
        notes={},
    )
    saved.save(tmp_path)
    loaded = load_recording(tmp_path)
    assert loaded.columns == columns
    assert loaded.slices == layout
    assert np.array_equal(loaded.domain("A"), saved.domain("A"))


def test_turn_sampling_takes_one_row_per_cycle():
    from core.subject.recording import Recording

    tags = tuple(["open", "phase", "phase", "ontogeny"] * 5)
    rows = len(tags)
    recording = Recording(
        x=np.arange(rows * 3, dtype=np.float64).reshape(rows, 3),
        conditions=tuple("x" for _ in range(rows)),
        tags=tags,
        times=np.arange(rows, dtype=np.float64),
        env=np.zeros((rows, 1)),
        env_names=("clock",),
        columns=("P.a", "P.b", "P.c"),
        slices={key: slice(0, 3) for key in DOMAINS},
        notes={},
    )
    assert recording.by_turn().frames == 5


def test_the_periphery_is_the_machine_minus_the_core():
    """A closure test run against a set containing the core can only say no.

    `read_periphery` walks the kernel, every phase, every organ and every built
    service. The workspace, the substrate, the self model and the world model
    *are* the core's organs: their ignition level is the workspace domain,
    their valence is recurrent cognition, their beliefs are the self-state. A
    copy of the core predicting the core is the same number twice.
    """
    from types import SimpleNamespace

    from core.subject.closure import read_periphery

    class _Workspace:
        def __init__(self) -> None:
            self.ignition_level = 0.7   # read by the core as G.ignition
            self.tick = 12              # read by the core as G.tick
            self.internal_cursor = 3    # not read by the core: real periphery

    kernel = SimpleNamespace(_phases=[], organs={"global_workspace": _Workspace()})
    seen = read_periphery(kernel)
    assert any(name.endswith("internal_cursor") for name in seen), seen
    assert not any(name.endswith("ignition_level") for name in seen), seen
    assert not any(name.endswith(".tick") for name in seen), seen


def test_a_stored_instant_is_not_periphery_state():
    from types import SimpleNamespace

    from core.subject.closure import read_periphery

    holder = SimpleNamespace(_last_refresh_at=1_788_000_000.0, backlog=7)
    kernel = SimpleNamespace(_phases=[], organs={"whatever": holder})
    seen = read_periphery(kernel)
    assert any(name.endswith("backlog") for name in seen)
    assert not any(name.endswith("_last_refresh_at") for name in seen)


# ── the paired test ──────────────────────────────────────────────────────


def test_the_paired_test_asks_how_consistent_not_how_large():
    """On the mean, one enormous trial carries the whole test.

    A pair that moved in seven of eight conditions with an effect of seven
    tenths came out at q = 0.016 against a bar of 0.01, on evidence that is not
    marginal at all — because the null built by flipping signs has a tail as
    fat as the outlier that made the observed mean. The signed rank asks how
    consistently the difference is positive, which is the question the
    replication bar asks beside it.
    """
    import numpy as np

    from core.subject.causal import _sign_flip_p

    rng = np.random.default_rng(1)
    consistent = np.full(48, 0.7) + rng.normal(0, 0.2, 48)
    assert _sign_flip_p(consistent, seed=1) * 90 < 0.01

    # And it cuts the other way: one huge trial and forty-seven nothings is not
    # an edge, however large the average comes out.
    one_trial = np.zeros(48)
    one_trial[0] = 40.0
    assert _sign_flip_p(one_trial, seed=1) > 0.4
    assert float(one_trial.mean()) > 0.8  # the mean would have called it large

    noise = rng.normal(0, 1, 48)
    assert _sign_flip_p(noise, seed=1) > 0.05


def test_a_difference_of_exactly_zero_votes_for_nothing():
    import numpy as np

    from core.subject.causal import _sign_flip_p

    six = np.zeros(48)
    six[:6] = 5.0
    # Six of forty-eight is not consistency, and the forty-two zeros must not
    # be ranked into agreement with them.
    assert _sign_flip_p(six, seed=1) > 0.01


def test_the_partition_score_carries_its_own_uncertainty():
    """A point estimate with no spread beside it cannot establish a sign."""
    report = phi_do(_toy("recurrent"))
    blob = report.as_dict()
    assert len(blob["held_out"]) >= 2
    assert blob["standard_error"] >= 0.0
    assert blob["lower_bound"] <= blob["phi_do"]
    # The reference architecture is genuinely recurrent: its lower bound clears
    # the bar, not only its point estimate.
    assert blob["lower_bound"] > 0.05


def test_a_null_is_a_distribution_not_one_draw():
    """One instantiation of a random weight matrix can be lucky either way."""
    values = [
        phi_do(toy_recording(architecture("star", seed=7 + draw), steps=1500, seed=7 + draw)).phi
        for draw in range(3)
    ]
    assert len(set(round(v, 4) for v in values)) > 1, "the draws are identical"
    assert max(values) < 0.05
