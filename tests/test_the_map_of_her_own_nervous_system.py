"""The connectome package, checked against the things it claims.

Every test here is either a property that must hold for the code to mean what
it says, or a case where the right answer is known independently — a branching
process with a set ratio, a graph whose cut vertex is obvious, a benchmark run
on data built so that structure does predict activity and again on data built so
that it does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.connectome.activity import ActivityRecorder, ActivityTrace, ObservedEdges, RecorderConfig
from core.connectome.beyond import (
    apply_rewiring,
    compile_delays,
    propose_rewiring,
    whorl_census,
)
from core.connectome.celltypes import adjusted_rand_index, refine_types
from core.connectome.criticality import (
    branching_ratio_mr,
    extract_avalanches,
    naive_branching_ratio,
)
from core.connectome.development import compare_pruning, prune_at_random, prune_by_use
from core.connectome.dimorphism import compare_individuals
from core.connectome.gating import Gate, GateSet, routing_change
from core.connectome.lesion import measure_effect
from core.connectome.microcircuit import (
    CORTICAL_CONN_PROBS,
    CORTICAL_SIZES,
    POPULATIONS,
    assign_layers,
    compare_to_cortex,
    connection_probabilities,
    trophic_levels,
)
from core.connectome.neuromodulation import (
    Evidence,
    Modulator,
    ReceptorField,
    fit_interventional,
    fit_observational,
)
from core.connectome.proofreading import EditLedger, focused_queue, repair_observed_splits
from core.connectome.segmentation import expected_run_length, score_against_observation
from core.connectome.synaptology import (
    compartment_profile,
    ei_report,
    measure_multiplicity,
    strong_connections,
)
from core.connectome.topology import (
    DiGraphView,
    degree_preserving_rewire,
    power_law_fit,
    reciprocity,
)
from core.connectome.types import (
    CORTICAL_EI_RATIO,
    FLY_MALE_CNS_REFERENCE,
    H01_REFERENCE,
    CellClass,
    Compartment,
    Connection,
    ConnectomeSnapshot,
    EdgeKind,
    Neuropil,
    Unit,
    stable_id,
)
from core.connectome.volume import ReconstructionConfig, VolumeReconstructor, classify_external

# ---------------------------------------------------------------------------
# Fixtures: a tiny source tree, and a hand-built graph
# ---------------------------------------------------------------------------

_TINY_MODULE = '''
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def is_allowed(value):
    """A predicate. Every exit is a boolean, so this is a gate."""
    if value is None:
        return False
    return value > 0


def refuse_everything(value):
    """Two of three exits refuse."""
    if value is None:
        return None
    if not value:
        return None
    return value


def announce(value):
    """Logs and returns nothing. Glial."""
    logger.info("value %s", value)
    logger.debug("again %s", value)


def build(value):
    """Calls a gate from a guard, then produces something."""
    if not is_allowed(value):
        return None
    announce(value)
    doubled = value * 2
    return doubled


def act(path):
    """Touches the world."""
    return subprocess.run(["true"], check=False)


def sense(path):
    """Reads the world."""
    return os.listdir(path)


class Holder:
    def __init__(self):
        self.value = 0

    def set_value(self, value):
        """No productive exit and it mutates state. Modulatory."""
        self.value = value

    def compute(self):
        return build(self.value)
'''


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    package = tmp_path / "core" / "tiny"
    package.mkdir(parents=True)
    (tmp_path / "core" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(_TINY_MODULE)
    return tmp_path


@pytest.fixture
def tiny_snapshot(tiny_repo: Path) -> ConnectomeSnapshot:
    reconstructor = VolumeReconstructor(tiny_repo, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    return reconstructor.build()


def _unit(uid: str, *, cell_class: CellClass = CellClass.EXCITATORY, region: str = "r") -> Unit:
    return Unit(uid=uid, name=uid, neuropil=f"{region}.m", region=region, cell_class=cell_class)


def _graph_snapshot(edges: list[tuple[str, str, int]]) -> ConnectomeSnapshot:
    units: dict[str, Unit] = {}
    connections: dict[tuple[str, str, str], Connection] = {}
    for pre, post, contacts in edges:
        units.setdefault(pre, _unit(pre))
        units.setdefault(post, _unit(post))
        connections[(pre, post, str(EdgeKind.DRIVE))] = Connection(
            pre=pre, post=post, contacts=contacts, sign=1, kind=EdgeKind.DRIVE
        )
    return ConnectomeSnapshot(
        version=1,
        units=units,
        connections=connections,
        neuropils={"r.m": Neuropil(name="r.m", parent="r")},
        source="test",
    )


# ---------------------------------------------------------------------------
# Types and references
# ---------------------------------------------------------------------------


def test_published_reference_values_are_the_published_ones():
    assert H01_REFERENCE.get("single_contact_fraction") == pytest.approx(0.965)
    assert H01_REFERENCE.get("four_or_more_contact_fraction") == pytest.approx(0.00092)
    assert H01_REFERENCE.get("cells") == pytest.approx(57_000)
    assert FLY_MALE_CNS_REFERENCE.get("neurons") == pytest.approx(166_000)
    assert FLY_MALE_CNS_REFERENCE.get("sex_specific_types") == pytest.approx(262)


def test_cortical_ratio_is_derived_from_the_population_table():
    excitatory = CORTICAL_SIZES[0] + CORTICAL_SIZES[2] + CORTICAL_SIZES[4] + CORTICAL_SIZES[6]
    inhibitory = CORTICAL_SIZES[1] + CORTICAL_SIZES[3] + CORTICAL_SIZES[5] + CORTICAL_SIZES[7]
    assert sum(CORTICAL_SIZES) == 77_169
    assert CORTICAL_EI_RATIO == pytest.approx(excitatory / inhibitory)
    assert CORTICAL_EI_RATIO == pytest.approx(4.035, abs=0.001)


def test_the_cortical_matrix_is_square_and_probabilities():
    assert len(CORTICAL_CONN_PROBS) == len(POPULATIONS) == 8
    for row in CORTICAL_CONN_PROBS:
        assert len(row) == 8
        assert all(0.0 <= value <= 1.0 for value in row)
    # The densest cortical connection is inhibition onto layer 5 excitatory cells.
    flat = [(value, POPULATIONS[c]) for r, row in enumerate(CORTICAL_CONN_PROBS) for c, value in enumerate(row)]
    assert max(flat)[0] == pytest.approx(0.3726)


def test_identity_is_content_addressed_and_stable():
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("a", "c")


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_cell_class_is_measured_from_exits_not_from_the_name(tiny_snapshot):
    by_name = {unit.name.rsplit(":", 1)[1]: unit for unit in tiny_snapshot.units.values()}
    assert by_name["is_allowed"].cell_class is CellClass.INHIBITORY
    assert by_name["refuse_everything"].cell_class is CellClass.INHIBITORY
    assert by_name["announce"].cell_class is CellClass.GLIAL
    assert by_name["Holder.set_value"].cell_class is CellClass.MODULATORY
    # One guard and one real return is a producer with a guard, not a gate.
    assert by_name["build"].cell_class is CellClass.EXCITATORY
    assert by_name["build"].suppression == pytest.approx(0.5)


def test_a_guard_that_calls_a_gate_lands_on_the_initial_segment(tiny_snapshot):
    by_name = {unit.name.rsplit(":", 1)[1]: unit for unit in tiny_snapshot.units.values()}
    gate = by_name["is_allowed"].uid
    caller = by_name["build"].uid
    ret = tiny_snapshot.connections.get((gate, caller, str(EdgeKind.RETURN)))
    assert ret is not None
    assert Compartment.AXON_INITIAL_SEGMENT in ret.compartments
    assert ret.sign == -1


def test_drive_and_return_edges_are_separate(tiny_snapshot):
    drive = tiny_snapshot.edges(EdgeKind.DRIVE)
    returns = tiny_snapshot.edges(EdgeKind.RETURN)
    assert drive and returns
    assert all(conn.kind is EdgeKind.DRIVE for conn in drive)
    assert all(conn.kind is EdgeKind.RETURN for conn in returns)


def test_world_touching_calls_are_attributed_to_the_cell_that_makes_them(tiny_snapshot):
    by_name = {unit.name.rsplit(":", 1)[1]: unit for unit in tiny_snapshot.units.values()}
    assert by_name["act"].attrs.get("efferent", 0) >= 1
    assert by_name["sense"].attrs.get("afferent", 0) >= 1
    assert by_name["build"].attrs.get("efferent", 0) == 0


def test_external_classification_reads_the_module_before_the_name():
    assert classify_external("subprocess", "anything") == "efferent"
    assert classify_external("os", "system") == "efferent"
    assert classify_external("os", "listdir") == "afferent"
    assert classify_external("math", "sqrt") == "compute"
    assert classify_external("json", "dumps") == "compute"


def test_reconstruction_is_deterministic(tiny_repo):
    digests = []
    for _ in range(2):
        reconstructor = VolumeReconstructor(tiny_repo, ReconstructionConfig(roots=("core",)))
        reconstructor.scan()
        digests.append(reconstructor.build().digest())
    assert digests[0] == digests[1]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_the_recorder_sees_cells_fire_and_labels_the_condition(tmp_path):
    recorder = ActivityRecorder(
        Path(__file__).resolve().parents[1],
        RecorderConfig(frame_seconds=0.01, max_wall_seconds=20.0),
    )
    if not recorder.start("first"):
        pytest.skip("sys.monitoring slot unavailable in this process")
    try:
        for _ in range(200):
            classify_external("os", "listdir")
        recorder.set_condition("second")
        for _ in range(200):
            classify_external("subprocess", "run")
    finally:
        trace = recorder.stop()
    assert trace.n_frames >= 1
    assert set(trace.conditions) <= {"first", "second"}
    assert trace.summary()["events"] > 0


def test_a_recording_never_raises_into_the_code_it_is_watching():
    """A callback that raises fails the program, not the recording.

    ``__code__`` on a class is a descriptor, not a code object, and reading
    ``co_filename`` off it raised inside whatever was running. Building a
    function through types.FunctionType is what networkx does at import, and it
    is what broke 88 test files while a recording was on.
    """
    import types

    recorder = ActivityRecorder(
        Path(__file__).resolve().parents[1],
        RecorderConfig(frame_seconds=0.05, capture_edges=True, max_wall_seconds=20.0),
    )
    if not recorder.start("hostile"):
        pytest.skip("sys.monitoring slot unavailable in this process")
    try:
        def _target(value):
            return value

        for _ in range(20):
            clone = types.FunctionType(
                _target.__code__, _target.__globals__, "clone", None, _target.__closure__
            )
            assert clone(1) == 1
        # A class passed where a callable is expected resolves __code__ to a
        # descriptor, which is the exact shape that raised.
        assert isinstance(getattr(types.FunctionType, "__code__", None), object)
    finally:
        trace = recorder.stop()
    assert trace.attrs["callback_failures"] == 0


def test_the_calcium_kernel_decays_and_normalises():
    trace = ActivityTrace(
        uids=("a",),
        conditions=tuple(["c"] * 6),
        spikes=[[10.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
    )
    signal = trace.calcium()
    assert signal.shape == (6, 1)
    values = [float(row[0]) for row in signal]
    assert values[0] > values[1] > values[2]
    assert values[-1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Segmentation and proofreading
# ---------------------------------------------------------------------------


def test_expected_run_length_weights_long_runs():
    chain = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    whole, mean_whole, runs_whole = expected_run_length([chain], set(chain))
    assert whole == pytest.approx(4.0)
    assert runs_whole == 1
    broken = set(chain) - {("c", "d")}
    split, _, runs_split = expected_run_length([chain], broken)
    assert runs_split == 2
    assert split < whole


def test_a_cell_does_not_make_a_synapse_onto_itself():
    """A comprehension carries its parent's identity, on both sides.

    The static reconstruction folds a closure into the cell that contains it, so
    on the recording side the same closure shows up as that cell calling itself.
    On one real recording that was 792 pairs carrying 51 million of 140 million
    observed calls, all of them read as edges the map was missing.
    """
    observed = ObservedEdges()
    observed.add("a", "a")
    observed.add("a", "b")
    assert ("a", "a") not in observed.counts
    assert observed.counts[("a", "b")] == 1

    legacy = ObservedEdges()
    legacy.counts[("a", "a")] = 500
    legacy.counts[("a", "b")] = 2
    cleaned = legacy.without_self_pairs()
    assert set(cleaned.counts) == {("a", "b")}
    assert legacy.counts[("a", "a")] == 500


def test_an_observed_edge_the_graph_lacks_is_a_split_error():
    snapshot = _graph_snapshot([("a", "b", 1)])
    observed = ObservedEdges()
    for _ in range(10):
        observed.add("a", "b")
        observed.add("b", "c")
    score = score_against_observation(snapshot, observed)
    assert score.observed_pairs == 2
    assert score.recovered == 1
    assert score.split_errors == 1
    assert score.recall == pytest.approx(0.5)


def test_the_ledger_replays_to_the_same_graph_and_can_be_withdrawn():
    snapshot = _graph_snapshot([("a", "b", 1)])
    ledger = EditLedger(base_digest=snapshot.digest())
    edit = ledger.join("b", "a", author="test", evidence="observed")
    first = ledger.apply(snapshot)
    second = ledger.apply(snapshot)
    assert first.digest() == second.digest()
    assert first.digest() != snapshot.digest()
    ledger.withdraw(edit.edit_id, author="test", evidence="wrong")
    assert ledger.apply(snapshot).digest() == snapshot.digest()
    assert len(ledger) == 2


def test_the_focused_queue_ranks_by_traffic():
    snapshot = _graph_snapshot([("a", "b", 1), ("x", "y", 1)])
    observed = ObservedEdges()
    observed.counts[("a", "y")] = 5
    observed.counts[("x", "b")] = 500
    queue = focused_queue(snapshot, observed)
    joins = [row for row in queue if row.kind.value == "join"]
    assert joins[0].pre == "x" and joins[0].post == "b"
    ledger = repair_observed_splits(snapshot, observed)
    assert len(ledger) == 2


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_rewiring_preserves_every_degree():
    edges = [(f"n{i}", f"n{(i * 7 + 3) % 60}", 1) for i in range(60)]
    graph = DiGraphView.from_snapshot(_graph_snapshot(edges))
    before_out = {node: len(graph.out[node]) for node in graph.nodes}
    before_in = {node: len(graph.inbound[node]) for node in graph.nodes}
    rewired = degree_preserving_rewire(graph, swaps_per_edge=6, seed=3)
    assert {node: len(rewired.out[node]) for node in rewired.nodes} == before_out
    assert {node: len(rewired.inbound[node]) for node in rewired.nodes} == before_in
    assert rewired.m == graph.m


def test_reciprocity_counts_mutual_pairs():
    snapshot = _graph_snapshot([("a", "b", 1), ("b", "a", 1), ("b", "c", 1)])
    assert reciprocity(DiGraphView.from_snapshot(snapshot)) == pytest.approx(2 / 3)


def test_the_power_law_fit_recovers_a_planted_exponent():
    import numpy as np

    rng = np.random.default_rng(11)
    alpha = 2.5
    samples = (rng.pareto(alpha - 1.0, 20_000) + 1.0) * 4.0
    fit = power_law_fit([int(v) for v in samples])
    assert fit["alpha"] == pytest.approx(alpha, abs=0.35)
    assert fit["ks"] < 0.1


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------


def test_cells_the_circuit_cannot_tell_apart_get_one_type():
    snapshot = _graph_snapshot(
        [("src", "a", 1), ("src", "b", 1), ("a", "sink", 1), ("b", "sink", 1)]
    )
    typing = refine_types(snapshot, rounds=1)
    assert typing.labels["a"] == typing.labels["b"]
    assert typing.labels["a"] != typing.labels["src"]


def test_adjusted_rand_is_one_for_identical_and_near_zero_for_unrelated():
    left = {f"c{i}": f"g{i % 4}" for i in range(80)}
    assert adjusted_rand_index(left, dict(left)) == pytest.approx(1.0)
    import random as _random

    shuffled = list(left.values())
    _random.Random(7).shuffle(shuffled)
    right = dict(zip(left, shuffled, strict=True))
    assert abs(adjusted_rand_index(left, right)) < 0.2


# ---------------------------------------------------------------------------
# Synaptology
# ---------------------------------------------------------------------------


def test_the_multiplicity_law_counts_pairs_and_compares_to_cortex():
    snapshot = _graph_snapshot([("a", "b", 1), ("a", "c", 1), ("b", "c", 9)])
    law = measure_multiplicity(snapshot)
    assert law.pairs == 3
    assert law.contacts == 11
    assert law.single_fraction == pytest.approx(2 / 3)
    assert law.four_or_more_fraction == pytest.approx(1 / 3)
    assert law.maximum == 9
    assert law.heavy_excess > 1.0
    strong = strong_connections(snapshot)
    assert len(strong) == 1 and strong[0].contacts == 9


def test_the_ei_report_finds_the_most_inhibited_region():
    units = {
        "e1": _unit("e1", region="calm"),
        "e2": _unit("e2", region="calm"),
        "i1": _unit("i1", cell_class=CellClass.INHIBITORY, region="calm"),
    }
    for index in range(30):
        units[f"g{index}"] = _unit(
            f"g{index}",
            cell_class=CellClass.INHIBITORY if index % 2 else CellClass.EXCITATORY,
            region="tense",
        )
    snapshot = ConnectomeSnapshot(version=1, units=units, connections={}, neuropils={})
    report = ei_report(snapshot)
    assert report["cortical_ei_ratio"] == pytest.approx(4.035, abs=0.001)
    assert report["most_inhibited_regions"][0]["region"] == "tense"


def test_compartments_separate_a_veto_from_a_vote(tiny_snapshot):
    profile = compartment_profile(tiny_snapshot)
    assert profile.total > 0
    assert profile.by_compartment.get("axon_initial_segment", 0) > 0
    assert profile.inhibitory_on_initial_segment > 0


# ---------------------------------------------------------------------------
# Microcircuit
# ---------------------------------------------------------------------------


def test_trophic_levels_climb_a_chain():
    snapshot = _graph_snapshot([("a", "b", 1), ("b", "c", 1), ("c", "d", 1)])
    heights = trophic_levels(DiGraphView.from_snapshot(snapshot))
    assert heights["a"] < heights["b"] < heights["c"] < heights["d"]
    assert heights["d"] - heights["a"] == pytest.approx(3.0, abs=0.05)


def test_the_cortex_comparison_reports_both_orientations():
    snapshot = _graph_snapshot([(f"n{i}", f"n{i + 1}", 1) for i in range(40)])
    assignment = assign_layers(snapshot)
    comparison = compare_to_cortex(connection_probabilities(snapshot, assignment))
    assert "spearman" in comparison
    assert "spearman_if_orientation_reversed" in comparison
    assert comparison["orientation_free"]["cortex_within_over_between"] == pytest.approx(
        5.945, abs=0.01
    )


# ---------------------------------------------------------------------------
# Criticality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("truth", [0.7, 0.9, 0.99])
def test_the_regression_estimator_recovers_a_known_branching_ratio(truth):
    import numpy as np

    rng = np.random.default_rng(4)
    series = [10.0]
    for _ in range(12_000):
        series.append(float(rng.poisson(truth * series[-1] + 2.0)))
    estimate = branching_ratio_mr(series[1:])
    assert estimate.m == pytest.approx(truth, abs=0.06)
    assert estimate.r_squared > 0.8


def test_the_naive_estimator_is_biased_where_the_regression_one_is_not():
    import numpy as np

    rng = np.random.default_rng(5)
    series = [10.0]
    for _ in range(12_000):
        series.append(float(rng.poisson(0.95 * series[-1] + 2.0)))
    full = np.asarray(series[1:])
    subsampled = rng.binomial(full.astype(int), 0.05).astype(float)
    assert branching_ratio_mr(subsampled).m == pytest.approx(0.95, abs=0.06)
    assert abs(naive_branching_ratio(subsampled) - 0.95) > 0.06


def test_avalanches_are_runs_above_the_quiet_threshold():
    activity = [0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0, 0.0]
    avalanches = extract_avalanches(activity, percentile=25.0)
    assert len(avalanches.sizes) == 2
    assert avalanches.durations == [2, 1]


# ---------------------------------------------------------------------------
# Gating, development, lesions
# ---------------------------------------------------------------------------


def test_closing_the_only_path_reroutes_and_closing_a_spare_one_does_not():
    snapshot = _graph_snapshot([("a", "b", 1), ("b", "c", 1)])
    gates = GateSet()
    gates.add(
        Gate(
            name="cut",
            opens="quiet",
            closes_on="alarm",
            predicate=lambda s: 0.0 if s.get("alarm") else 1.0,
            edges=(("b", "c"),),
        )
    )
    report = routing_change(snapshot, gates, ["a"], {"alarm": 1.0}, baseline_state={})
    assert report.rerouted is True
    assert report.lost_cells == 1

    spare = _graph_snapshot([("a", "b", 1), ("b", "c", 1), ("a", "c", 1)])
    quiet = routing_change(spare, gates, ["a"], {"alarm": 1.0}, baseline_state={})
    assert quiet.edges_closed == 1
    assert quiet.rerouted is False


def test_a_gate_with_no_closing_condition_is_reported_as_always_open():
    gates = GateSet()
    gates.add(Gate(name="open", opens="always", closes_on="", predicate=lambda s: 1.0))
    assert gates.always_open() == ["open"]


def test_pruning_by_use_keeps_more_traffic_than_pruning_at_random():
    edges = [(f"n{i}", f"n{i + 1}", 1) for i in range(200)]
    snapshot = _graph_snapshot(edges)
    observed = ObservedEdges()
    for index in range(0, 200, 4):
        observed.counts[(f"n{index}", f"n{index + 1}")] = 100
    used = prune_by_use(snapshot, observed, fraction=0.5)
    random_pruned = prune_at_random(snapshot, observed, fraction=0.5, seed=1)
    assert used.traffic_retained == pytest.approx(1.0)
    assert used.traffic_retained > random_pruned.traffic_retained
    comparison = compare_pruning(snapshot, observed, fraction=0.5)
    assert comparison["traffic_advantage"] > 0


def test_a_cut_vertex_costs_more_reach_than_a_matched_control():
    edges = [("src", "bridge", 1)]
    edges += [("bridge", f"far{i}", 1) for i in range(12)]
    edges += [("src", f"near{i}", 1) for i in range(12)]
    snapshot = _graph_snapshot(edges)
    for unit in snapshot.units.values():
        if unit.uid == "src":
            unit.attrs["afferent"] = 1
        if unit.uid.startswith("far"):
            unit.attrs["efferent"] = 1
    effect = measure_effect(snapshot, ["bridge"], null_samples=6)
    assert effect.reach_after < effect.reach_before
    assert effect.excess_reach_loss > 0.0


# ---------------------------------------------------------------------------
# Beyond biology
# ---------------------------------------------------------------------------


def test_the_delay_compiler_beats_doing_nothing_and_beats_random_holds():
    edges = [("src", "fast", 1), ("fast", "join", 1)]
    edges += [("src", f"slow{i}", 1) for i in range(1)]
    edges += [("slow0", "slow1", 1), ("slow1", "slow2", 1), ("slow2", "join", 1)]
    for index in range(6):
        edges.append((f"pad{index}", "join", 1))
        edges.append(("src", f"pad{index}", 1))
    schedule = compile_delays(_graph_snapshot(edges), seed=2)
    assert schedule.convergence_cells >= 1
    assert schedule.jitter_after < schedule.jitter_before
    assert schedule.jitter_after < schedule.jitter_random


def test_the_whorl_census_finds_a_planted_cycle():
    snapshot = _graph_snapshot(
        [("a", "b", 1), ("b", "c", 1), ("c", "a", 1), ("c", "d", 1), ("d", "e", 1)]
    )
    whorls = whorl_census(snapshot)
    assert whorls and whorls[0].size == 3
    assert set(whorls[0].members) == {"a", "b", "c"}
    assert whorls[0].external_out >= 1


def test_a_rewiring_carries_its_own_inverse():
    snapshot = _graph_snapshot([("a", "b", 1), ("b", "c", 1)])
    proposals = propose_rewiring(snapshot, candidates=4)
    assert proposals
    changed, inverse = apply_rewiring(snapshot, proposals[0])
    assert changed.digest() != snapshot.digest()
    restored, _ = apply_rewiring(changed, inverse)
    assert restored.digest() == snapshot.digest()


# ---------------------------------------------------------------------------
# Two individuals
# ---------------------------------------------------------------------------


def test_an_individual_does_not_differ_from_itself():
    snapshot = _graph_snapshot([("a", "b", 1), ("b", "c", 2)])
    divergence = compare_individuals(snapshot, snapshot, include_typing=False)
    assert divergence.changed_cells == ()
    assert divergence.divergent_fraction == 0.0
    assert divergence.rewired_pairs == 0


def test_a_changed_contact_count_counts_as_a_change():
    left = _graph_snapshot([("a", "b", 1)])
    right = _graph_snapshot([("a", "b", 9)])
    divergence = compare_individuals(left, right, include_typing=False)
    assert set(divergence.changed_cells) == {"a", "b"}


# ---------------------------------------------------------------------------
# Neuromodulation
# ---------------------------------------------------------------------------


def test_an_observational_fit_may_not_make_a_causal_claim():
    field = ReceptorField()
    field.set(
        fit_observational("brain", Modulator.DOPAMINE, [0.1, 0.4, 0.7, 0.9], [1.0, 1.3, 1.7, 2.0])
    )
    claim = field.claim("brain", Modulator.DOPAMINE)
    assert "no causal claim" in claim
    assert field.sensitivities[("brain", Modulator.DOPAMINE)].evidence is Evidence.OBSERVATIONAL


def test_one_assigned_level_is_not_a_dose_response():
    demoted = fit_interventional("m", Modulator.SEROTONIN, [(0.5, 1.0), (0.5, 1.1), (0.5, 0.9)])
    assert demoted.evidence is Evidence.OBSERVATIONAL
    promoted = fit_interventional("m", Modulator.SEROTONIN, [(0.1, 1.0), (0.5, 1.4), (0.9, 1.9)])
    assert promoted.evidence is Evidence.INTERVENTIONAL


def test_an_unmeasured_pair_has_no_effect_and_says_so():
    field = ReceptorField()
    assert field.gain("nowhere", Modulator.NORADRENALINE, 0.9) == 1.0
    assert "not measured" in field.claim("nowhere", Modulator.NORADRENALINE)


# ---------------------------------------------------------------------------
# The benchmark, validated on data where the answer is known
# ---------------------------------------------------------------------------


def _coupled_system(cells: int, frames: int, *, coupling: float, seed: int):
    """A connectome and a recording generated through it.

    With ``coupling`` above zero a cell's next value depends on the cells wired
    into it, so structure genuinely predicts activity. At zero every cell is its
    own independent process and the wiring predicts nothing, which is the case
    the benchmark has to be unable to find an effect in.
    """
    import numpy as np

    from core.connectome.activity import ActivityTrace

    rng = np.random.default_rng(seed)
    edges: list[tuple[str, str, int]] = []
    inputs: dict[int, list[int]] = {i: [] for i in range(cells)}
    for target in range(cells):
        for source in rng.choice(cells, size=3, replace=False):
            if int(source) == target:
                continue
            edges.append((f"c{int(source)}", f"c{target}", 1))
            inputs[target].append(int(source))
    snapshot = _graph_snapshot(edges)

    values = np.zeros((frames, cells), dtype=np.float64)
    values[0] = rng.normal(4.0, 1.0, size=cells)
    for t in range(1, frames):
        neighbour = np.array(
            [
                values[t - 1][inputs[i]].mean() if inputs[i] else 0.0
                for i in range(cells)
            ]
        )
        values[t] = (
            0.35 * values[t - 1]
            + coupling * neighbour
            + (1.0 - 0.35 - coupling) * 4.0
            + rng.normal(0.0, 0.25, size=cells)
        )
    trace = ActivityTrace(
        uids=tuple(f"c{i}" for i in range(cells)),
        conditions=tuple(f"stim{t // 40 % 3}" for t in range(frames)),
        spikes=[list(row) for row in values],
    )
    return snapshot, trace


def test_the_benchmark_finds_structure_when_the_activity_flows_through_it():
    from core.connectome.zapbench import BenchmarkConfig, run_benchmark

    snapshot, trace = _coupled_system(60, 400, coupling=0.55, seed=3)
    report = run_benchmark(
        trace,
        snapshot,
        BenchmarkConfig(contexts=(4,), horizon=8, bootstrap=200, signal="spikes"),
    )
    by_arm = {arm.arm: arm.mae for arm in report.arms if arm.context == 4}
    assert by_arm["connectome"] < by_arm["rewired"]
    assert by_arm["connectome"] < by_arm["blind"]
    test = report.structure_test["context_4"]["connectome_vs_rewired"]
    assert test["median_significant"] is True
    assert test["median_difference"] < 0
    assert test["share_better"] > 0.6
    assert "wiring predicts activity" in report.structure_test["context_4"]["verdict"]


def test_the_benchmark_does_not_find_structure_that_is_not_there():
    from core.connectome.zapbench import BenchmarkConfig, run_benchmark

    snapshot, trace = _coupled_system(60, 400, coupling=0.0, seed=4)
    report = run_benchmark(
        trace,
        snapshot,
        BenchmarkConfig(contexts=(4,), horizon=8, bootstrap=200, signal="spikes"),
    )
    verdict = report.structure_test["context_4"]["verdict"]
    assert "no detectable effect" in verdict or "rewiring beats" in verdict
    by_arm = {arm.arm: arm.mae for arm in report.arms if arm.context == 4}
    assert abs(by_arm["connectome"] - by_arm["rewired"]) < 0.05 * by_arm["blind"]


def test_the_naive_baselines_are_reported_and_beatable():
    from core.connectome.zapbench import BenchmarkConfig, run_benchmark

    snapshot, trace = _coupled_system(40, 300, coupling=0.5, seed=5)
    report = run_benchmark(
        trace,
        snapshot,
        BenchmarkConfig(contexts=(4,), horizon=8, bootstrap=100, signal="spikes"),
    )
    arms = {arm.arm: arm.mae for arm in report.arms}
    for baseline in ("mean", "condition_mean", "persistence"):
        assert baseline in arms
    assert arms["connectome"] < arms["mean"]
    assert report.dataset["cells"] == 40


# ---------------------------------------------------------------------------
# The layers a call graph cannot see
# ---------------------------------------------------------------------------


_LAYERED_MODULE = '''
def announce(bus):
    bus.publish("core/thing/happened", {})


def listen(bus):
    bus.subscribe("core/thing/happened", handle)


def orphan_publisher(bus):
    bus.publish("core/thing/nobody_hears", {})


def handle(event):
    return event


def writer(container):
    container.set("shared_key", object())


def reader(container):
    return container.get("shared_key")


def shows_a_message(ui):
    ui.publish("Awaiting confirmation from the person")
'''


@pytest.fixture
def layered_repo(tmp_path: Path) -> Path:
    package = tmp_path / "core" / "layered"
    package.mkdir(parents=True)
    (tmp_path / "core" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(_LAYERED_MODULE)
    return tmp_path


def test_the_volume_and_gap_layers_find_what_the_call_graph_cannot(layered_repo):
    from core.connectome.layers import Layer, extract_layers, multilink_census

    reconstructor = VolumeReconstructor(layered_repo, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    multilayer = extract_layers(snapshot, layered_repo, roots=("core",))

    names = {uid: unit.name.rsplit(":", 1)[1] for uid, unit in snapshot.units.items()}
    volume_named = {
        (names[pre], names[post]) for pre, post in multilayer.volume if pre in names and post in names
    }
    assert ("announce", "listen") in volume_named
    gap_named = {
        tuple(sorted((names[pre], names[post])))
        for pre, post in multilayer.gap
        if pre in names and post in names
    }
    assert ("reader", "writer") in gap_named
    # Neither pair is joined by a call.
    assert multilayer.unique_fraction(Layer.VOLUME) == pytest.approx(1.0)
    census = multilink_census(multilayer)
    assert census["volume_only"] >= 1
    assert census["gap_only"] >= 1


def test_a_sentence_is_not_a_topic(layered_repo):
    from core.connectome.layers import Layer, extract_layers

    reconstructor = VolumeReconstructor(layered_repo, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    multilayer = extract_layers(snapshot, layered_repo, roots=("core",))
    topics = {
        key.partition(":")[2]
        for key in multilayer.channels
        if key.startswith(str(Layer.VOLUME))
    }
    assert "core/thing/happened" in topics
    assert not any(" " in topic for topic in topics)


def test_a_half_wired_topic_is_a_candidate_and_a_heavy_pair_is_measured(layered_repo):
    from core.connectome.layers import extract_layers
    from core.connectome.pathology import Confidence, diagnose

    reconstructor = VolumeReconstructor(layered_repo, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    multilayer = extract_layers(snapshot, layered_repo, roots=("core",))
    report = diagnose(snapshot, multilayer=multilayer)
    half_wired = [f for f in report.findings if f.kind == "half_wired_channel"]
    assert any(f.subject == "core/thing/nobody_hears" for f in half_wired)
    assert all(f.confidence is Confidence.CANDIDATE for f in half_wired)
    assert all(f.closes_when for f in report.findings)


def test_diagnosis_with_nothing_wrong_reports_nothing():
    snapshot = _graph_snapshot([("a", "b", 1)])
    from core.connectome.pathology import diagnose

    report = diagnose(snapshot)
    assert report.findings == []
    assert report.as_json()["total"] == 0


# ---------------------------------------------------------------------------
# The local loop
# ---------------------------------------------------------------------------


def _noisy_trials(count: int, options: int, noise: float, seed: int, easy_share: float = 0.5):
    import random

    rng = random.Random(seed)
    from core.connectome.laminar import Candidate

    trials = []
    for index in range(count):
        truth = f"c{rng.randrange(options)}"
        separation = 0.55 if rng.random() < easy_share else 0.12
        base = {
            f"c{i}": (1.0 if f"c{i}" == truth else 1.0 - separation) for i in range(options)
        }
        state = random.Random(seed * 1000 + index)

        def evidence(candidate, prior, base=base, state=state):
            return base[candidate.key] + state.gauss(0.0, noise)

        trials.append(([Candidate(f"c{i}") for i in range(options)], evidence, truth))
    return trials


@pytest.mark.parametrize("noise", [0.10, 0.25, 0.40])
def test_the_local_loop_matches_a_fixed_budget_for_fewer_calls(noise):
    from core.connectome.laminar import LaminarConfig, compare_against_fixed_budget

    comparison = compare_against_fixed_budget(
        _noisy_trials(400, 4, noise, seed=7), config=LaminarConfig()
    )
    assert comparison.laminar_accuracy >= comparison.fixed_accuracy - 0.03
    assert comparison.call_saving > 0.15
    assert comparison.as_json()["verdict"] == "same accuracy for fewer calls"


def test_a_lead_inside_the_noise_is_not_a_decision():
    from core.connectome.laminar import decisive_margin

    tied = decisive_margin({"a": 0.81, "b": 0.80, "c": 0.79})
    assert tied.decisive is False
    clear = decisive_margin({"a": 0.95, "b": 0.40, "c": 0.35})
    assert clear.decisive is True
    assert clear.winner == "a"


def test_the_noise_is_estimated_from_the_candidates_that_lost():
    """The spread of everything includes the leader, which hides a real winner."""
    from core.connectome.laminar import decisive_margin

    decision = decisive_margin({"a": 5.0, "b": 1.01, "c": 1.00, "d": 0.99})
    assert decision.decisive is True
    assert decision.z > 10.0


def test_a_single_candidate_costs_one_call():
    from core.connectome.laminar import Candidate, settle

    calls = []

    def evidence(candidate, prior):
        calls.append(candidate.key)
        return 1.0

    settled = settle([Candidate("only")], evidence)
    assert settled.winner is not None and settled.winner.key == "only"
    assert settled.evidence_calls == 1
    assert len(calls) == 1


def test_the_calibrated_bound_moves_with_the_error_rate():
    from core.connectome.laminar import calibrate_threshold

    loose = calibrate_threshold([0.3], 0.25, target_error=0.10, candidates=4)
    tight = calibrate_threshold([0.3], 0.25, target_error=0.001, candidates=4)
    assert tight > loose
    assert 1.0 <= loose < tight <= 6.0


# ---------------------------------------------------------------------------
# Warming what is about to run
# ---------------------------------------------------------------------------


def test_the_prefetch_rule_names_what_the_active_cells_can_reach():
    from core.connectome.prefetch import downstream_of, predict_next_active

    snapshot = _graph_snapshot([("a", "b", 1), ("b", "c", 1), ("x", "y", 1)])
    assert downstream_of(snapshot, ["a"], hops=1) == {"b"}
    assert downstream_of(snapshot, ["a"], hops=2) == {"b", "c"}
    predicted = predict_next_active(snapshot, ["a"], hops=1)
    assert predicted == {"a", "b"}


def test_weighting_by_contacts_prefers_the_heavier_partner():
    from core.connectome.prefetch import weighted_next_active

    snapshot = _graph_snapshot([("a", "heavy", 20), ("a", "light", 1)])
    chosen = weighted_next_active(snapshot, ["a"], budget=2, hops=1)
    assert "heavy" in chosen
    assert "light" not in chosen


def test_prefetch_scores_every_rule_on_the_same_frames():
    from core.connectome.activity import ActivityTrace
    from core.connectome.prefetch import evaluate_prefetch

    snapshot = _graph_snapshot([("a", "b", 1), ("b", "c", 1), ("c", "a", 1)])
    spikes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    trace = ActivityTrace(
        uids=("a", "b", "c"), conditions=tuple(["c"] * 6), spikes=spikes
    )
    plan = evaluate_prefetch(trace, snapshot, hops=1, max_frames=5)
    rules = {rule.rule for rule in plan.rules}
    assert rules == {"connectome", "connectome_weighted", "frequent", "persistent"}
    connectome = next(r for r in plan.rules if r.rule == "connectome")
    # Activity here walks the ring, so the downstream rule has perfect recall.
    assert connectome.recall == pytest.approx(1.0)
    assert plan.as_json()["verdict"]


def test_a_warmer_that_raises_does_not_stop_the_warm_up():
    from core.connectome.prefetch import warm

    snapshot = _graph_snapshot([("a", "b", 3), ("a", "c", 2)])
    seen: list[str] = []

    def warmer(uid: str) -> None:
        if uid == "b":
            raise RuntimeError("cold")
        seen.append(uid)

    warmed = warm(snapshot, ["a"], warmer, hops=1, budget=8)
    assert "b" not in warmed
    assert set(seen) == set(warmed)
    assert warmed


# ---------------------------------------------------------------------------
# Two variants at once, and a circuit that moves
# ---------------------------------------------------------------------------


def test_running_both_variants_pairs_the_trials():
    from core.connectome.beyond import evaluate_variants

    trial = evaluate_variants(
        ("tight", {"scale": 1.0}),
        ("loose", {"scale": 2.0}),
        list(range(30)),
        lambda config, item: (item % 5) * config["scale"],
    )
    payload = trial.as_json()
    assert payload["trials"] == 30
    assert payload["decisive"] is True
    assert "tight is better" in payload["verdict"]


def test_a_circuit_lifts_out_with_its_pattern_and_its_edges():
    from core.connectome.beyond import extract_circuit, graft_report

    snapshot = _graph_snapshot(
        [("in", "p", 1), ("p", "q", 2), ("q", "p", 1), ("q", "out", 1)]
    )
    circuit = extract_circuit(snapshot, ["p", "q"], label="pair")
    assert circuit.inputs == ("in",)
    assert circuit.outputs == ("out",)
    assert len(circuit.internal) == 2
    assert graft_report(circuit, snapshot)["graftable"] is True

    poorer = _graph_snapshot([("p", "other", 1)])
    report = graft_report(circuit, poorer)
    assert report["graftable"] is False
    assert report["cells_missing"] == 1
    assert report["edges_to_create"] == 2


def test_a_confidence_interval_that_is_one_point_is_not_significant():
    """A spike in the distribution is not a tight interval.

    Most cells have no connectome neighbour, so the two arms differ for them
    only through a shared weight — the same number for every such cell. The
    median of every resample lands on it and the interval collapses.
    """
    import numpy as np

    from core.connectome.zapbench import _paired_bootstrap

    shared = np.concatenate([np.full(900, 0.001), np.random.default_rng(0).normal(0, 0.01, 100)])
    zeros = np.zeros_like(shared)
    result = _paired_bootstrap(shared, zeros, draws=200, seed=1)
    assert result["largest_shared_value_share"] > 0.2
    assert result["median_significant"] is False


def test_the_paired_comparison_can_be_restricted_to_the_cells_that_differ():
    import numpy as np

    from core.connectome.zapbench import _paired_bootstrap

    rng = np.random.default_rng(3)
    right = rng.normal(0.0, 0.01, 400)
    left = right.copy()
    subset = np.zeros(400, dtype=bool)
    subset[:200] = True
    # A real per-cell effect never lands on identical floats, so the shift
    # varies; a constant one would trip the degeneracy guard, which is what the
    # guard is for.
    left[:200] -= 0.02 + rng.normal(0.0, 0.003, 200)
    result = _paired_bootstrap(left, right, draws=200, seed=2, subset=subset)
    assert result["cells_compared"] == 200
    assert result["cells_identical"] == 200
    assert result["median_significant"] is True
    assert result["median_difference"] < 0


# ---------------------------------------------------------------------------
# The same individual over time
# ---------------------------------------------------------------------------


def test_a_template_separates_what_holds_still_from_what_moves():
    from core.connectome.longitudinal import build_template, drift_against

    first = _graph_snapshot([("a", "b", 1), ("b", "c", 1), ("c", "d", 1)])
    second = _graph_snapshot([("a", "b", 1), ("b", "c", 1), ("c", "e", 1)])
    template = build_template([first, second])
    assert template.timepoints == 2
    assert ("a", "b") in template.stable_edges()
    assert ("c", "d") not in template.stable_edges()

    drift = drift_against(template, second)
    assert drift["core_edges_lost"] == 0
    assert "core is intact" in drift["verdict"]

    third = _graph_snapshot([("a", "b", 1)])
    later = drift_against(template, third)
    assert later["core_edges_lost"] >= 1
    assert "had never changed" in later["verdict"]


def test_a_rename_survives_connectivity_alignment_and_a_stranger_does_not():
    from core.connectome.longitudinal import align_by_connectivity

    left = _graph_snapshot([("src", "old_name", 1), ("old_name", "sink", 1)])
    right = _graph_snapshot([("src", "new_name", 1), ("new_name", "sink", 1)])
    result = align_by_connectivity(left, right, minimum_overlap=0.5)
    assert result["matched_by_connectivity"] == 1
    assert result["pairs"][0]["same_module"] is True

    stranger = _graph_snapshot([("src", "unrelated", 1)])
    weak = align_by_connectivity(left, stranger, minimum_overlap=0.9)
    assert weak["matched_by_connectivity"] == 0


def test_the_projection_matrix_normalises_by_the_size_of_the_source():
    from core.connectome.longitudinal import projection_matrix

    units = {
        "big1": _unit("big1", region="big"),
        "big2": _unit("big2", region="big"),
        "big3": _unit("big3", region="big"),
        "big4": _unit("big4", region="big"),
        "small1": _unit("small1", region="small"),
        "target": _unit("target", region="target"),
    }
    connections = {}
    for pre, contacts in (("big1", 4), ("small1", 4)):
        connections[(pre, "target", str(EdgeKind.DRIVE))] = Connection(
            pre=pre, post="target", contacts=contacts, sign=1, kind=EdgeKind.DRIVE
        )
    snapshot = ConnectomeSnapshot(
        version=1, units=units, connections=connections, neuropils={}
    )
    matrix = projection_matrix(snapshot)
    weights = {(row["source"], row["target"]): row["weight"] for row in matrix["strongest"]}
    # The same four contacts from a one-cell package weigh four times what they
    # weigh from a four-cell one.
    assert weights[("small", "target")] == pytest.approx(4.0)
    assert weights[("big", "target")] == pytest.approx(1.0)
