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


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    # The census keys are the set of layers joining a pair, so a pair joined only
    # by a topic is "volume" and one joined by a call as well is "volume+wired".
    assert census.get("volume", 0) >= 1
    assert census.get("gap", 0) >= 1
    assert all("+" not in key or "wired" in key for key in census)


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


# ---------------------------------------------------------------------------
# Do cells that do the same thing wire together?
# ---------------------------------------------------------------------------


def _wired_activity(cells: int, frames: int, *, coupled: bool, seed: int):
    """A graph and a recording that either flows through it or does not."""
    import numpy as np

    from core.connectome.activity import ActivityTrace

    rng = np.random.default_rng(seed)
    edges = [(f"c{i}", f"c{(i + 1) % cells}", 1) for i in range(cells)]
    snapshot = _graph_snapshot(edges)
    values = rng.normal(5.0, 1.0, size=(frames, cells))
    if coupled:
        # Neighbours on the ring share a driver, so connected pairs move
        # together and pairs that a rewiring would create do not.
        for i in range(cells):
            shared = rng.normal(0.0, 3.0, size=frames)
            values[:, i] += shared
            values[:, (i + 1) % cells] += shared
    trace = ActivityTrace(
        uids=tuple(f"c{i}" for i in range(cells)),
        conditions=tuple(f"stim{t // 30 % 2}" for t in range(frames)),
        spikes=[],
        array=values.astype("float32"),
    )
    return snapshot, trace


def test_like_to_like_is_found_when_it_is_there():
    from core.connectome.likewise import test_like_to_like

    snapshot, trace = _wired_activity(80, 240, coupled=True, seed=5)
    result = test_like_to_like(trace, snapshot, nulls=6, min_frames_active=2)
    assert result.connected_mean > result.null_mean
    assert result.z > 3.0
    assert "like-to-like" in result.verdict


def test_like_to_like_is_not_found_when_it_is_not():
    from core.connectome.likewise import test_like_to_like

    snapshot, trace = _wired_activity(80, 240, coupled=False, seed=6)
    result = test_like_to_like(trace, snapshot, nulls=6, min_frames_active=2)
    assert abs(result.z) < 3.0
    assert "no detectable" in result.verdict


def test_the_within_condition_control_removes_the_workload():
    """Two cells busy in the same condition and nothing else must not count."""
    import numpy as np

    from core.connectome.activity import ActivityTrace
    from core.connectome.likewise import test_like_to_like

    cells = 60
    frames = 240
    rng = np.random.default_rng(9)
    values = rng.normal(1.0, 0.2, size=(frames, cells))
    # Every cell is loud in one condition and quiet in the other, so every pair
    # correlates through the workload alone.
    values[:120] += 20.0
    snapshot = _graph_snapshot([(f"c{i}", f"c{(i + 1) % cells}", 1) for i in range(cells)])
    trace = ActivityTrace(
        uids=tuple(f"c{i}" for i in range(cells)),
        conditions=tuple("loud" if t < 120 else "quiet" for t in range(frames)),
        spikes=[],
        array=values.astype("float32"),
    )
    raw = test_like_to_like(trace, snapshot, nulls=6, min_frames_active=2, within_condition=False)
    controlled = test_like_to_like(trace, snapshot, nulls=6, min_frames_active=2)
    assert raw.connected_mean > 0.9
    assert controlled.connected_mean < 0.3
    assert "no detectable" in controlled.verdict


def test_how_directly_cognition_reaches_the_actuators():
    from core.connectome.spine import descending_directness

    # A deep cell reaching an effector straight, and one reaching it through a
    # local circuit, so the share is a half.
    edges = [
        ("sense", "mid", 1),
        ("mid", "deep", 1),
        ("deep", "act_direct", 1),
        ("deep", "local", 1),
        ("local", "act_indirect", 1),
    ]
    snapshot = _graph_snapshot(edges)
    for unit in snapshot.units.values():
        if unit.uid == "sense":
            unit.attrs["afferent"] = 1
        if unit.uid.startswith("act_"):
            unit.attrs["efferent"] = 1
    report = descending_directness(snapshot)
    assert report["effectors"] == 2
    assert report["fly_typical_motor_neuron_share"] == pytest.approx(0.07)
    assert 0.0 <= report["mean_direct_share"] <= 1.0
    assert report["verdict"]


def test_stereotypy_of_an_individual_with_itself_is_one():
    from core.connectome.celltypes import stereotypy

    # Varied degrees and contact counts, so colour refinement produces enough
    # distinct types for the comparison to have something to correlate.
    edges: list[tuple[str, str, int]] = []
    for i in range(200):
        for step in range(1, (i % 5) + 2):
            edges.append((f"n{i}", f"n{(i * 7 + step) % 200}", (i + step) % 6 + 1))
    snapshot = _graph_snapshot(edges)
    result = stereotypy(snapshot, snapshot)
    assert result["shared_type_pairs"] >= 8
    assert result["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert result["only_in_left"] == 0


def test_serial_homology_counts_the_regions_a_type_spans():
    from core.connectome.celltypes import refine_types, serial_homology

    units = {}
    connections = {}
    for region in ("head", "trunk", "tail"):
        units[f"{region}_src"] = _unit(f"{region}_src", region=region)
        units[f"{region}_worker"] = _unit(f"{region}_worker", region=region)
        connections[(f"{region}_src", f"{region}_worker", str(EdgeKind.DRIVE))] = Connection(
            pre=f"{region}_src", post=f"{region}_worker", contacts=1, sign=1,
            kind=EdgeKind.DRIVE,
        )
    snapshot = ConnectomeSnapshot(
        version=1, units=units, connections=connections, neuropils={}
    )
    typing = refine_types(snapshot, rounds=1)
    report = serial_homology(snapshot, typing, minimum_regions=3)
    assert report["types_spanning_regions"] >= 1
    assert report["widest"][0]["regions"] == 3


# ---------------------------------------------------------------------------
# The wiring: modulator to bound, gate to route, measurement to rule
# ---------------------------------------------------------------------------


def test_noradrenaline_lowers_the_decision_bound_by_a_measured_amount():
    from core.connectome.laminar import LaminarConfig, config_for
    from core.connectome.neuromodulation import (
        Modulator,
        ReceptorField,
        fit_interventional,
    )

    field = ReceptorField()
    field.set(
        fit_interventional(
            "brain", Modulator.NORADRENALINE, [(0.1, 1.0), (0.5, 0.6), (0.9, 0.2)]
        )
    )
    field.set(
        fit_interventional(
            "calm", Modulator.NORADRENALINE, [(0.1, 1.0), (0.5, 1.02), (0.9, 0.99)]
        )
    )
    base = LaminarConfig()
    low = config_for("brain", {"noradrenaline": 0.1}, field, base=base).z
    high = config_for("brain", {"noradrenaline": 0.9}, field, base=base).z
    # Published direction: more noradrenaline, less evidence needed.
    assert high < base.z < low
    # A region whose activity barely moves with it barely moves its bound.
    flat = config_for("calm", {"noradrenaline": 0.9}, field, base=base).z
    assert abs(flat - base.z) < abs(high - base.z)
    # A region with nothing measured does not move at all.
    assert config_for("nowhere", {"noradrenaline": 0.9}, field, base=base).z == base.z


def test_the_direction_does_not_depend_on_the_sign_of_the_fit():
    """A region whose correlation came out negative must not respond backwards."""
    from core.connectome.laminar import LaminarConfig, config_for
    from core.connectome.neuromodulation import (
        Modulator,
        ReceptorField,
        fit_interventional,
    )

    rising = ReceptorField()
    rising.set(
        fit_interventional(
            "r", Modulator.NORADRENALINE, [(0.1, 0.2), (0.5, 0.6), (0.9, 1.0)]
        )
    )
    falling = ReceptorField()
    falling.set(
        fit_interventional(
            "r", Modulator.NORADRENALINE, [(0.1, 1.0), (0.5, 0.6), (0.9, 0.2)]
        )
    )
    base = LaminarConfig()
    assert config_for("r", {"noradrenaline": 0.9}, rising, base=base).z < base.z
    assert config_for("r", {"noradrenaline": 0.9}, falling, base=base).z < base.z


def test_a_gated_candidate_is_never_sampled():
    from core.connectome.laminar import Candidate, settle

    sampled: list[str] = []

    def evidence(candidate, prior):
        sampled.append(candidate.key)
        return 1.0 if candidate.key == "blocked" else 0.4

    result = settle(
        [Candidate("blocked"), Candidate("open1"), Candidate("open2")],
        evidence,
        gate=lambda candidate: candidate.key != "blocked",
    )
    assert "blocked" not in sampled
    assert result.gated_out == 1
    assert result.winner is not None and result.winner.key != "blocked"


def test_gating_everything_out_runs_the_race_anyway():
    """A gate set that closes every route is a bug, not an instruction."""
    from core.connectome.laminar import Candidate, settle

    result = settle(
        [Candidate("a"), Candidate("b")], lambda c, p: 1.0, gate=lambda c: False
    )
    assert result.winner is not None
    assert result.gated_out == 0


def test_the_warm_up_does_not_pull_in_a_cell_the_gate_closed():
    from core.connectome.gating import Gate, GateSet
    from core.connectome.prefetch import warm

    snapshot = _graph_snapshot([("a", "near", 5), ("a", "far", 5)])
    snapshot.units["far"].region = "elsewhere"
    snapshot.units["far"].neuropil = "elsewhere.m"
    gates = GateSet()
    gates.add(
        Gate(
            name="cut_elsewhere",
            opens="quiet",
            closes_on="alarm",
            predicate=lambda s: 0.0 if s.get("alarm") else 1.0,
            regions=(("r", "elsewhere"),),
        )
    )
    open_warm: list[str] = []
    warm(snapshot, ["a"], open_warm.append, gates=gates, state={})
    closed_warm: list[str] = []
    warm(snapshot, ["a"], closed_warm.append, gates=gates, state={"alarm": 1.0})
    assert "far" in open_warm
    assert "far" not in closed_warm
    assert "near" in closed_warm


def test_the_rule_comes_from_the_measurement():
    from core.connectome.prefetch import PrefetchPlan, SetPrediction, best_rule

    plan = PrefetchPlan(hops=1)
    plan.rules = [
        SetPrediction("connectome", 0.4, 0.9, 0.55, 10, 8, 100),
        SetPrediction("persistent", 0.7, 0.7, 0.70, 8, 8, 100),
    ]
    assert best_rule(plan) == "persistent"
    plan.rules[0] = SetPrediction("connectome", 0.8, 0.9, 0.85, 9, 8, 100)
    assert best_rule(plan) == "connectome"
    assert best_rule(PrefetchPlan(hops=1)) == "persistent"


def test_live_levels_are_empty_without_a_running_system():
    from core.connectome.neuromodulation import live_levels

    levels = live_levels()
    assert isinstance(levels, dict)
    # Every consumer treats an empty reading as "do not move the parameter".
    from core.connectome.laminar import LaminarConfig, config_for

    assert config_for("anywhere", levels, None).z == LaminarConfig().z


# ---------------------------------------------------------------------------
# A call count is not a weight
# ---------------------------------------------------------------------------


_FLOW_MODULE = '''
import logging

logger = logging.getLogger(__name__)


def make():
    return 1


def check():
    return True


def discards():
    """Ten calls, nothing read."""
    for _ in range(10):
        make()


def logs_it():
    logger.info("value %s", make())


def keeps_it():
    value = make()
    return value + 1


def branches_on_it():
    if check():
        return 1
    return 0


def returns_it():
    return make()


class Holder:
    def escapes(self):
        self.value = make()
'''


@pytest.fixture
def flow_repo(tmp_path: Path) -> Path:
    package = tmp_path / "core" / "flow"
    package.mkdir(parents=True)
    (tmp_path / "core" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(_FLOW_MODULE)
    return tmp_path


def test_what_happens_to_a_returned_value_is_classified(flow_repo):
    from core.connectome.dataflow import Consequence, extract_dataflow

    flow = extract_dataflow(flow_repo, roots=("core",))
    text = (flow_repo / "core" / "flow" / "mod.py").read_text().splitlines()
    found: dict[str, str] = {}
    for locus, consequence in flow.by_locus.items():
        line = int(locus.split(":")[1])
        found.setdefault(text[line - 1].strip(), str(consequence))
    assert found["make()"] == str(Consequence.DISCARDED)
    assert found['logger.info("value %s", make())'] == str(Consequence.LOGGED)
    assert found["value = make()"] == str(Consequence.LOCAL)
    assert found["if check():"] == str(Consequence.BRANCH)
    assert found["return make()"] == str(Consequence.RETURNED)
    assert found["self.value = make()"] == str(Consequence.ESCAPES)


def test_ten_calls_that_carry_nothing_weigh_nothing(flow_repo):
    from core.connectome.dataflow import extract_dataflow, weight_edges

    reconstructor = VolumeReconstructor(flow_repo, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    flow = extract_dataflow(flow_repo, roots=("core",))
    weighted = weight_edges(snapshot, flow, reconstructor.contact_loci)

    names = {uid: unit.name.rsplit(":", 1)[1] for uid, unit in snapshot.units.items()}
    by_pair = {
        (names[pre], names[post]): entry
        for (pre, post), entry in weighted.items()
        if pre in names and post in names
    }
    discarding = by_pair[("discards", "make")]
    assert discarding["contacts"] == 1  # one site, in a loop
    assert discarding["mean_weight"] == pytest.approx(0.0)
    assert discarding["carries_nothing"] is True

    deciding = by_pair[("branches_on_it", "check")]
    assert deciding["carries_nothing"] is False
    assert deciding["mean_weight"] > discarding["mean_weight"]


def test_the_report_separates_heavy_and_empty_from_light_and_decisive(flow_repo):
    from core.connectome.dataflow import dataflow_report, extract_dataflow, weight_edges

    reconstructor = VolumeReconstructor(flow_repo, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    flow = extract_dataflow(flow_repo, roots=("core",))
    weighted = weight_edges(snapshot, flow, reconstructor.contact_loci)
    report = dataflow_report(snapshot, weighted)
    assert report["edges_weighted"] > 0
    assert 0.0 <= report["carries_nothing_share"] <= 1.0
    assert any("branches_on_it" in row["pair"] for row in report["single_call_decisive"])


# ---------------------------------------------------------------------------
# Durable stores and the process boundary
# ---------------------------------------------------------------------------


_STORE_MODULE = '''
import json
import subprocess


def writes(out):
    out.write_text("payload")
    (out / "report.json").write_text("{}")


def reads(out):
    return (out / "report.json").read_text()


def writes_nobody_reads(out):
    (out / "orphan.json").write_text("{}")


def spawns():
    return subprocess.run(["helper-binary"], check=False)


def spawns_too():
    return subprocess.run(["helper-binary", "--flag"], check=False)
'''


def test_a_store_written_here_and_read_there_is_an_edge(tmp_path: Path):
    from core.connectome.layers import Layer, extract_layers

    package = tmp_path / "core" / "store"
    package.mkdir(parents=True)
    (tmp_path / "core" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(_STORE_MODULE)

    reconstructor = VolumeReconstructor(tmp_path, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    multilayer = extract_layers(snapshot, tmp_path, roots=("core",))

    names = {uid: unit.name.rsplit(":", 1)[1] for uid, unit in snapshot.units.items()}
    io_named = {
        (names[pre], names[post])
        for pre, post in multilayer.io
        if pre in names and post in names
    }
    assert ("writes", "reads") in io_named
    stores = {
        key.partition(":")[2]: value
        for key, value in multilayer.channels.items()
        if key.startswith(str(Layer.IO))
    }
    assert "report.json" in stores
    orphan = stores.get("orphan.json")
    assert orphan is not None and orphan["write"] and not orphan["read"]


def test_two_cells_that_spawn_the_same_helper_meet_at_the_boundary(tmp_path: Path):
    from core.connectome.layers import extract_layers

    package = tmp_path / "core" / "store"
    package.mkdir(parents=True)
    (tmp_path / "core" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(_STORE_MODULE)

    reconstructor = VolumeReconstructor(tmp_path, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    multilayer = extract_layers(snapshot, tmp_path, roots=("core",))
    names = {uid: unit.name.rsplit(":", 1)[1] for uid, unit in snapshot.units.items()}
    ipc_named = {
        tuple(sorted((names[pre], names[post])))
        for pre, post in multilayer.ipc
        if pre in names and post in names
    }
    assert ("spawns", "spawns_too") in ipc_named


# ---------------------------------------------------------------------------
# The effective connectome: what a connection does, in the state she is in
# ---------------------------------------------------------------------------


def _driven_trace(frames: int, *, driven: bool, condition: str, seed: int):
    """Two cells wired a->b, with b either driven by a's past or not."""
    import numpy as np

    from core.connectome.activity import ActivityTrace

    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, frames)
    noise = rng.normal(0.0, 0.3, frames)
    b = np.empty(frames)
    b[0] = noise[0]
    for t in range(1, frames):
        b[t] = (0.9 * a[t - 1] if driven else 0.0) + 0.2 * b[t - 1] + noise[t]
    values = np.stack([a, b], axis=1).astype("float32")
    snapshot = _graph_snapshot([("a", "b", 1)])
    trace = ActivityTrace(
        uids=("a", "b"),
        conditions=tuple([condition] * frames),
        spikes=[],
        array=values,
    )
    return snapshot, trace


def test_an_edge_that_drives_survives_the_rotation_null():
    from core.connectome.effective import Grade, predictive_influence

    snapshot, trace = _driven_trace(400, driven=True, condition="c", seed=1)
    graph = predictive_influence(trace, snapshot, "c", nulls=8)
    edge = graph.edges[("a", "b")]
    assert edge.grade is Grade.PREDICTIVE
    assert edge.survives_null is True
    assert edge.weight > 0.3
    assert graph.summary()["edges_surviving_null"] == 1


def test_an_edge_that_drives_nothing_does_not():
    from core.connectome.effective import predictive_influence

    snapshot, trace = _driven_trace(400, driven=False, condition="c", seed=2)
    graph = predictive_influence(trace, snapshot, "c", nulls=8)
    assert graph.edges[("a", "b")].survives_null is False


def test_a_condition_with_too_few_frames_is_skipped_not_guessed():
    from core.connectome.effective import MIN_FRAMES_PER_CONDITION, predictive_influence

    snapshot, trace = _driven_trace(40, driven=True, condition="c", seed=3)
    graph = predictive_influence(trace, snapshot, "c")
    assert graph.edges == {}
    assert str(MIN_FRAMES_PER_CONDITION) in graph.skipped
    assert graph.summary()["edges_measured"] == 0


def test_the_same_anatomy_can_run_different_circuits():
    import numpy as np

    from core.connectome.activity import ActivityTrace
    from core.connectome.effective import compare_conditions, predictive_influence

    frames = 400
    rng = np.random.default_rng(7)
    # Under "left" the a->b edge carries; under "right" the c->d edge does.
    a = rng.normal(0, 1, frames * 2)
    c = rng.normal(0, 1, frames * 2)
    b = np.zeros(frames * 2)
    d = np.zeros(frames * 2)
    for t in range(1, frames * 2):
        in_left = t < frames
        b[t] = (0.9 * a[t - 1] if in_left else 0.0) + rng.normal(0, 0.3)
        d[t] = (0.0 if in_left else 0.9 * c[t - 1]) + rng.normal(0, 0.3)
    values = np.stack([a, b, c, d], axis=1).astype("float32")
    snapshot = _graph_snapshot([("a", "b", 1), ("c", "d", 1)])
    trace = ActivityTrace(
        uids=("a", "b", "c", "d"),
        conditions=tuple(["left"] * frames + ["right"] * frames),
        spikes=[],
        array=values,
    )
    left = predictive_influence(trace, snapshot, "left", nulls=8)
    right = predictive_influence(trace, snapshot, "right", nulls=8)
    assert left.edges[("a", "b")].survives_null is True
    assert left.edges[("c", "d")].survives_null is False
    assert right.edges[("c", "d")].survives_null is True
    assert right.edges[("a", "b")].survives_null is False

    # Two edges is not enough to correlate two effective graphs, and the
    # comparison says so rather than reporting a correlation over two points.
    comparison = compare_conditions(left, right, snapshot=snapshot, limit=4)
    assert comparison["shared_edges"] == 2
    assert "too few edges" in comparison["verdict"]
    assert "correlation" not in comparison


def test_comparing_two_states_names_what_each_recruits():
    import numpy as np

    from core.connectome.activity import ActivityTrace
    from core.connectome.effective import compare_conditions, predictive_influence

    frames = 400
    cells = 12
    rng = np.random.default_rng(11)
    values = rng.normal(0.0, 1.0, size=(frames * 2, cells))
    edges = [(f"c{i}", f"c{i + 1}", 1) for i in range(cells - 1)]
    # The first half of the chain carries in "left", the second half in "right".
    for t in range(1, frames * 2):
        in_left = t < frames
        for i in range(cells - 1):
            early = i < (cells - 1) // 2
            carries = early if in_left else not early
            if carries:
                values[t, i + 1] = 0.9 * values[t - 1, i] + rng.normal(0, 0.3)
    snapshot = _graph_snapshot(edges)
    trace = ActivityTrace(
        uids=tuple(f"c{i}" for i in range(cells)),
        conditions=tuple(["left"] * frames + ["right"] * frames),
        spikes=[],
        array=values.astype("float32"),
    )
    left = predictive_influence(trace, snapshot, "left", nulls=8)
    right = predictive_influence(trace, snapshot, "right", nulls=8)
    comparison = compare_conditions(left, right, snapshot=snapshot, limit=8)
    assert comparison["shared_edges"] >= 8
    assert comparison["surviving_in_left_only"] >= 1
    assert comparison["surviving_in_right_only"] >= 1
    assert "active in one state and not the other" in comparison["verdict"]


def test_a_predictive_weight_may_not_be_read_as_a_cause():
    from core.connectome.effective import Grade

    assert "predict" in Grade.PREDICTIVE.licenses
    assert "model" in Grade.MODEL.licenses
    assert "disabling" in Grade.INTERVENTIONAL.licenses
    # The grades are ordered by what they license, and nothing promotes one.
    assert Grade.PREDICTIVE.licenses != Grade.INTERVENTIONAL.licenses


def test_the_model_tier_ranks_removals_and_says_it_is_a_model():
    from core.connectome.effective import Grade, model_influence

    snapshot = _graph_snapshot(
        [("hub", "x", 5), ("x", "y", 5), ("y", "z", 5), ("side", "z", 1)]
    )
    graph = model_influence(snapshot, ["hub", "side"], steps=4)
    assert graph.grade is Grade.MODEL
    assert graph.summary()["licenses"].startswith("that a model")
    assert graph.edges


# ---------------------------------------------------------------------------
# Merge errors: a builtin method name is not a cell here
# ---------------------------------------------------------------------------


_MERGE_MODULE = '''
class Field:
    def strip(self):
        """A real method that shares a name with str.strip."""
        return "clean"

    def warning(self):
        return None


def uses_the_real_one(field):
    holder = Field()
    return holder.strip()


def uses_a_string(raw):
    return str(raw or "").strip().lower()


def uses_a_logger(logger):
    logger.warning("something")
'''


def test_a_string_method_is_not_resolved_to_a_class_that_shares_its_name(tmp_path: Path):
    """The largest sink in the combined graph was every .strip() in the tree.

    A method call on an expression has a receiver whose type is unknown, and
    resolving it by name attaches thousands of standard-library calls to
    whichever class happens to define that name.
    """
    package = tmp_path / "core" / "merge"
    package.mkdir(parents=True)
    (tmp_path / "core" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(_MERGE_MODULE)

    reconstructor = VolumeReconstructor(tmp_path, ReconstructionConfig(roots=("core",)))
    reconstructor.scan()
    snapshot = reconstructor.build()
    names = {uid: unit.name.rsplit(":", 1)[1] for uid, unit in snapshot.units.items()}
    pairs = {
        (names[conn.pre], names[conn.post])
        for conn in snapshot.edges(EdgeKind.DRIVE)
        if conn.pre in names and conn.post in names
    }
    # The local constructor gives the receiver a type, so this one resolves.
    assert ("uses_the_real_one", "Field.strip") in pairs
    # These two have receivers whose type is unknown and must not.
    assert ("uses_a_string", "Field.strip") not in pairs
    assert ("uses_a_logger", "Field.warning") not in pairs


def test_the_unsafe_names_cover_builtins_and_the_common_library_objects():
    from core.connectome.volume import UNSAFE_ATTRIBUTE_NAMES

    for name in ("strip", "replace", "extend", "append", "keys", "items", "split"):
        assert name in UNSAFE_ATTRIBUTE_NAMES
    for name in ("warning", "exception", "exists", "is_dir", "resolve"):
        assert name in UNSAFE_ATTRIBUTE_NAMES
    # Names this system owns must stay resolvable.
    for name in ("record_degradation", "reconstruct", "publish_telemetry"):
        assert name not in UNSAFE_ATTRIBUTE_NAMES


# ---------------------------------------------------------------------------
# The coalition
# ---------------------------------------------------------------------------


def test_a_station_claims_each_cell_once():
    from core.connectome.coalition import assign_stations

    units = {}
    for name, module in (
        ("a", "core.affect.emotion_engine"),
        ("b", "core.consciousness.global_workspace"),
        ("c", "core.agency.goal_planner"),
        ("d", "core.utils.unrelated"),
    ):
        unit = _unit(name, region=module.split(".")[1])
        unit.neuropil = module
        units[name] = unit
    snapshot = ConnectomeSnapshot(version=1, units=units, connections={}, neuropils={})
    stations = assign_stations(snapshot)
    claimed = [uid for station in stations.values() for uid in station.cells]
    assert len(claimed) == len(set(claimed))
    assert "a" in stations["affect"].cells
    assert "b" in stations["workspace"].cells
    assert "c" in stations["planning"].cells
    assert "d" not in claimed


def test_a_ring_that_carries_beats_a_shuffled_one():
    from core.connectome.coalition import Station, test_closure

    # A clean ring: each station reaches only the next.
    ring = ["s0", "s1", "s2", "s3"]
    edges = []
    for i, name in enumerate(ring):
        nxt = ring[(i + 1) % len(ring)]
        edges.append((f"{name}_out", f"{nxt}_in", 4))
        edges.append((f"{name}_in", f"{name}_out", 4))
    snapshot = _graph_snapshot(edges)
    stations = {
        name: Station(name=name, patterns=(), cells=(f"{name}_in", f"{name}_out"))
        for name in ring
    }
    report = test_closure(
        snapshot, None, stations, order=ring, use_effective=False, nulls=200, max_hops=3
    )
    assert report.closed is True
    assert report.ring_enrichment > report.null_enrichment
    assert report.enrichment_z > 1.0
    assert "carries more than a ring drawn at random" in report.as_json()["verdict"]


def test_a_ring_nobody_wired_does_not():
    from core.connectome.coalition import Station, test_closure

    # A star: everything reaches s0 and nothing else.
    ring = ["s0", "s1", "s2", "s3"]
    edges = [(f"{name}_out", "s0_in", 4) for name in ring[1:]]
    edges += [(f"{name}_in", f"{name}_out", 4) for name in ring]
    snapshot = _graph_snapshot(edges)
    stations = {
        name: Station(name=name, patterns=(), cells=(f"{name}_in", f"{name}_out"))
        for name in ring
    }
    report = test_closure(
        snapshot, None, stations, order=ring, use_effective=False, nulls=200, max_hops=3
    )
    assert report.closed is False
    assert report.enrichment_z < 2.0
    assert "carry nothing" in report.as_json()["verdict"]


def test_a_recording_the_stations_did_not_fire_in_is_refused():
    from core.connectome.coalition import Station, test_closure

    snapshot = _graph_snapshot([("a_out", "b_in", 1), ("b_out", "a_in", 1)])
    stations = {
        "a": Station(name="a", patterns=(), cells=("a_in", "a_out")),
        "b": Station(name="b", patterns=(), cells=("b_in", "b_out")),
        "c": Station(name="c", patterns=(), cells=()),
    }

    class _Effective:
        condition = "quiet"
        edges: dict = {}

    report = test_closure(
        snapshot,
        _Effective(),
        stations,
        order=["a", "b", "c"],
        use_effective=True,
        recorded=["a_in"],
    )
    assert report.links == []
    assert "did not fire" in report.skipped
    assert report.as_json()["verdict"] == report.skipped


def test_every_lesion_prediction_says_what_survives_and_what_does_not():
    from core.connectome.coalition import LESION_PREDICTIONS

    # Three from the theory, three from what the first run of them found. The
    # list only grows: a prediction is deleted when the mechanism it describes
    # is gone, never because it came out wrong.
    assert len(LESION_PREDICTIONS) >= 6
    assert len({prediction.name for prediction in LESION_PREDICTIONS}) == len(
        LESION_PREDICTIONS
    )
    for prediction in LESION_PREDICTIONS:
        assert prediction.predicted_intact
        assert prediction.predicted_lost
        assert prediction.readout
        assert prediction.predicted_intact != prediction.predicted_lost


# ---------------------------------------------------------------------------
# What she can ask about her own machinery
# ---------------------------------------------------------------------------


def _effective_with(edges: dict, condition: str = "c"):
    from core.connectome.effective import EffectiveConnectome, EffectiveEdge, Grade

    graph = EffectiveConnectome(condition=condition, grade=Grade.PREDICTIVE, frames=400)
    for (pre, post), (weight, z) in edges.items():
        graph.edges[(pre, post)] = EffectiveEdge(
            pre=pre,
            post=post,
            condition=condition,
            weight=weight,
            null_mean=0.0,
            null_spread=1.0,
            z=z,
            samples=400,
            grade=Grade.PREDICTIVE,
        )
    return graph


def test_she_can_ask_which_circuit_dominates():
    from core.connectome.introspect import dominant_circuit

    snapshot = _graph_snapshot([("a", "b", 1), ("c", "d", 1)])
    graph = _effective_with({("a", "b"): (0.6, 9.0), ("c", "d"): (0.1, 0.4)})
    answer = dominant_circuit(snapshot, graph)
    assert "a" in answer.finding and "b" in answer.finding
    assert answer.detail["edges_surviving"] == 1
    assert "not by intervention" in answer.caveat


def test_a_predictive_answer_never_words_itself_as_a_cause():
    from core.connectome.effective import Grade
    from core.connectome.introspect import does_it_influence

    snapshot = _graph_snapshot([("mech", "out", 1)])
    graph = _effective_with({("mech", "out"): (0.5, 8.0)})
    answer = does_it_influence(snapshot, graph, ["mech"], ["out"], belief="I thought this did it")
    assert answer.grade is Grade.PREDICTIVE
    assert "influence" in answer.finding
    assert "cause" not in answer.finding.lower()
    assert "not by intervention" in answer.caveat
    assert "predict" in answer.grade.licenses


def test_the_mechanism_she_believes_in_may_have_no_influence():
    from core.connectome.introspect import does_it_influence

    snapshot = _graph_snapshot([("believed", "out", 1), ("actual", "out", 1)])
    graph = _effective_with(
        {("believed", "out"): (0.02, 0.3), ("actual", "out"): (0.7, 11.0)}
    )
    believed = does_it_influence(snapshot, graph, ["believed"], ["out"])
    actual = does_it_influence(snapshot, graph, ["actual"], ["out"])
    assert "none of them influences" in believed.finding
    assert "influence it" in actual.finding


def test_asking_about_machinery_nobody_measured_says_so():
    from core.connectome.introspect import does_it_influence

    snapshot = _graph_snapshot([("a", "b", 1)])
    graph = _effective_with({("a", "b"): (0.5, 8.0)})
    answer = does_it_influence(snapshot, graph, ["unmeasured"], ["b"])
    assert "no edge" in answer.finding
    assert "absence of a measurement is not absence of an influence" in answer.caveat


def test_she_can_ask_which_pathways_her_failures_depend_on():
    from core.connectome.introspect import failure_correlates

    pairs = {(f"p{i}", f"q{i}") for i in range(10)}
    snapshot = _graph_snapshot([(p, q, 1) for p, q in pairs])
    failing = _effective_with({p: (0.6 if p == ("p3", "q3") else 0.1, 9.0) for p in pairs},
                              condition="failing")
    succeeding = _effective_with({p: (0.1, 9.0) for p in pairs}, condition="succeeding")
    answer = failure_correlates(snapshot, failing, succeeding)
    assert "p3" in answer.finding and "q3" in answer.finding
    assert "correlation" in answer.caveat


def test_what_would_change_is_scored_against_a_matched_control():
    from core.connectome.introspect import what_would_change

    edges = [("src", "bridge", 1)] + [("bridge", f"far{i}", 1) for i in range(10)]
    edges += [("src", f"near{i}", 1) for i in range(10)]
    snapshot = _graph_snapshot(edges)
    snapshot.units["src"].attrs["afferent"] = 1
    answer = what_would_change(snapshot, ["bridge"])
    assert "reachability in the graph, not behaviour" in answer.caveat
    assert answer.detail["excess_reach_loss"] >= 0.0


# ---------------------------------------------------------------------------
# A change that has to answer for what it did to the anatomy
# ---------------------------------------------------------------------------


def test_every_axis_says_which_direction_is_better():
    from core.connectome.anatomy_gate import AXES

    assert len(AXES) == 7
    for axis in AXES:
        assert axis.question
        assert axis.improved(1.0, 2.0) is axis.higher_is_better
        assert axis.improved(2.0, 1.0) is not axis.higher_is_better


def test_an_axis_that_was_not_measured_is_not_scored_as_unchanged():
    from core.connectome.anatomy_gate import Quality, compare_quality

    before = Quality(values={"local_recurrence": 1.0}, unmeasured=("dormant_machinery",))
    after = Quality(values={"local_recurrence": 2.0}, unmeasured=("dormant_machinery",))
    delta = compare_quality(before, after)
    assert "local_recurrence" in delta.moves
    assert "dormant_machinery" not in delta.moves
    assert "dormant_machinery" in delta.unmeasured
    assert delta.improved == ["local_recurrence"]


def test_a_change_that_makes_something_worse_says_so_first():
    from core.connectome.anatomy_gate import Quality, anatomical_evidence, compare_quality

    before = Quality(values={"local_recurrence": 1.0, "half_wired_channels": 10.0})
    after = Quality(values={"local_recurrence": 2.0, "half_wired_channels": 20.0})
    delta = compare_quality(before, after)
    assert delta.improved == ["local_recurrence"]
    assert delta.worsened == ["half_wired_channels"]
    evidence = anatomical_evidence(delta, change="added a broadcast")
    assert evidence.index("worse:") < evidence.index("better:")
    assert "and worse in" not in delta.verdict
    assert "half_wired_channels" in delta.verdict


def test_a_change_that_moves_nothing_measured_says_that_too():
    from core.connectome.anatomy_gate import Quality, anatomical_evidence, compare_quality

    same = Quality(values={"local_recurrence": 1.0})
    delta = compare_quality(same, Quality(values={"local_recurrence": 1.0}))
    assert delta.improved == [] and delta.worsened == []
    assert "did not move" in delta.verdict
    assert "no axis moved" in anatomical_evidence(delta) or "not measured" in anatomical_evidence(
        delta
    )


def test_quality_measures_what_it_has_and_names_what_it_does_not():
    from core.connectome.anatomy_gate import measure_quality

    edges = [("src", "hub", 4)] + [("hub", f"leaf{i}", 2) for i in range(12)]
    snapshot = _graph_snapshot(edges)
    quality = measure_quality(snapshot, spof_sample=4)
    assert "local_recurrence" in quality.values
    assert "single_points_of_failure" in quality.values
    for name in ("coupling_carrying_nothing", "half_wired_channels", "dormant_machinery"):
        assert name in quality.unmeasured
    assert quality.detail["spof_sampled"] <= 4


def test_a_promotion_can_carry_what_the_change_did_to_the_shape():
    from core.cognition.how_a_change_is_promoted import a_ledger_of_its_own, promote
    from core.connectome.anatomy_gate import Quality, compare_quality

    before = Quality(
        values={"local_recurrence": 1.0, "half_wired_channels": 10.0},
        unmeasured=("dormant_machinery",),
    )
    # Worse by 0.4 of a channel rather than by ten of them. A regression larger
    # than a change to this part is allowed to cost is refused outright now, and
    # what this test is about is the LINE the receipt carries when a change is
    # kept — see the anatomical-law tests for the refusal.
    after = Quality(
        values={"local_recurrence": 2.0, "half_wired_channels": 10.4},
        unmeasured=("dormant_machinery",),
    )
    delta = compare_quality(before, after)
    with a_ledger_of_its_own():
        receipt = promote(
            "a.change",
            became="canary",
            started_by="test",
            evidence="the probe paid",
            anatomy=delta,
        )
        plain = promote(
            "b.change", became="canary", started_by="test", evidence="the probe paid"
        )
    assert "the probe paid" in receipt.evidence
    assert "worse: half_wired_channels" in receipt.evidence
    assert receipt.evidence.index("worse:") < receipt.evidence.index("better:")
    assert "not measured" in receipt.evidence
    # A promotion with nothing to say about the anatomy still says that.
    assert plain.evidence.startswith("the probe paid")
    assert "was not measured" in plain.evidence


def test_a_promotion_never_fails_because_the_shape_could_not_be_measured():
    from core.cognition.how_a_change_is_promoted import a_ledger_of_its_own, promote

    class _Broken:
        def as_json(self):
            raise ValueError("no")

    with a_ledger_of_its_own():
        receipt = promote(
            "c.change",
            became="canary",
            started_by="test",
            evidence="the probe paid",
            anatomy=_Broken(),
        )
    assert "the probe paid" in receipt.evidence
    assert "anatomy:" in receipt.evidence


# ---------------------------------------------------------------------------
# The stations are the phases that run, not the modules that share their names
# ---------------------------------------------------------------------------


def _kernel_phase_names() -> list[str]:
    """Every phase the kernel actually assembles, in order."""
    import tempfile

    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.state.state_repository import StateRepository

    with tempfile.TemporaryDirectory() as raw:
        vault = StateRepository(db_path=f"{raw}/stations.db", is_vault_owner=True)
        kernel = AuraKernel(config=KernelConfig(), vault=vault)
        kernel._setup_phases()
        return [type(phase).__name__ for phase in kernel._phases]


@pytest.mark.slow
def test_every_phase_that_runs_a_turn_is_placed_in_the_ring_or_outside_it():
    """A phase missing from the table and one deliberately outside it differ.

    The station table was written from module names and none of them ran. Nought
    of 129 workspace cells and nought of 272 action cells fired in a recording of
    240 turns, so the coalition test reported an architecture and measured a
    dictionary. This is the check that stops that recurring: the kernel's own
    phase list is the ground truth, and every phase in it has to be placed —
    either at a station or explicitly at none, with the reason in the comment.
    """
    from core.connectome.coalition import COALITION_ORDER, PHASE_STATIONS

    running = _kernel_phase_names()
    unplaced = [name for name in running if name not in PHASE_STATIONS]
    assert not unplaced, (
        f"these phases run a turn and the station table does not mention them: {unplaced}"
    )
    stale = [name for name in PHASE_STATIONS if name not in running]
    assert not stale, f"the table places phases the kernel no longer runs: {stale}"
    placed = {station for station in PHASE_STATIONS.values() if station}
    assert placed == set(COALITION_ORDER), (
        f"the ring wants {sorted(COALITION_ORDER)} and the phases supply {sorted(placed)}"
    )


@pytest.mark.slow
def test_each_station_claims_the_phase_assigned_to_it():
    """The patterns have to reach the phase they were written for.

    Placing ``UnitaryResponsePhase`` at ``action`` in one dictionary and failing
    to write a pattern that matches its module in the other is the same defect as
    before, one indirection along.
    """
    from core.connectome.coalition import PHASE_STATIONS, assign_stations
    from core.connectome.volume import VolumeReconstructor

    reconstructor = VolumeReconstructor(Path(__file__).resolve().parents[1])
    reconstructor.scan()
    snapshot = reconstructor.build()
    stations = assign_stations(snapshot)
    where: dict[str, str] = {}
    for name, station in stations.items():
        for uid in station.cells:
            unit = snapshot.units.get(uid)
            if unit is not None:
                where[f"{unit.neuropil}:{unit.name}"] = name

    missing: list[str] = []
    for phase, expected in PHASE_STATIONS.items():
        if not expected:
            continue
        found = {
            station
            for key, station in where.items()
            if key.rsplit(":", 1)[-1].split(".")[0] == phase
            or f":{phase}." in key
            or key.endswith(f":{phase}")
        }
        if expected not in found:
            missing.append(f"{phase} wanted {expected}, patterns gave {sorted(found) or 'nothing'}")
    assert not missing, "\n".join(missing)


# ---------------------------------------------------------------------------
# do(i): the cut is exact, reversible, and scored against a comparable cut
# ---------------------------------------------------------------------------


def test_a_lesion_puts_the_cell_back_even_when_the_body_raises():
    """A cut that does not heal turns one experiment into every later one."""
    from core.connectome import intervene
    from core.connectome.volume import VolumeReconstructor

    before = VolumeReconstructor.build
    try:
        with intervene.silence("core.connectome.volume:VolumeReconstructor.build"):
            assert VolumeReconstructor.build is not before
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert VolumeReconstructor.build is before


def test_a_silenced_cell_absorbs_its_calls_and_says_how_many():
    from core.connectome import intervene

    with intervene.silence("core.connectome.types:CellClass"):
        pass  # a class is callable; the point is the count below

    class _Holder:
        @staticmethod
        def work(value):
            return value * 2

    import core.connectome.types as types_module

    types_module._probe_holder = _Holder  # type: ignore[attr-defined]
    try:
        with intervene.silence("core.connectome.types:_probe_holder.work", returns=0):
            assert types_module._probe_holder.work(21) == 0
            assert types_module._probe_holder.work(3) == 0
        assert types_module._probe_holder.work(21) == 42
        assert intervene.silenced_calls("core.connectome.types:_probe_holder.work") == 2
    finally:
        del types_module._probe_holder


def test_an_async_cell_is_replaced_by_something_awaitable():
    """Handing a coroutine's caller a plain value measures the crash, not the cut."""
    import asyncio

    from core.connectome import intervene
    import core.connectome.types as types_module

    class _Holder:
        @staticmethod
        async def work():
            return "real"

    types_module._probe_async = _Holder  # type: ignore[attr-defined]
    try:
        with intervene.silence("core.connectome.types:_probe_async.work", returns="cut"):
            assert asyncio.run(types_module._probe_async.work()) == "cut"
        assert asyncio.run(types_module._probe_async.work()) == "real"
    finally:
        del types_module._probe_async


def test_a_lesion_refuses_what_it_cannot_cut():
    from core.connectome import intervene

    for uid in (
        "core.connectome.types",
        "core.connectome.types:NotThere",
        "no.such.module:thing",
        "core.connectome.types:CORTICAL_EI_RATIO",
    ):
        with pytest.raises(intervene.LesionRefusedError):
            with intervene.silence(uid):
                pass


def test_the_control_is_matched_on_both_degrees():
    from core.connectome.intervene import degree_matched_control
    from core.connectome.types import (
        CellClass,
        Connection,
        ConnectomeSnapshot,
        EdgeKind,
        Unit,
    )

    units = {
        name: Unit(uid=name, name=name, neuropil="m", region="r", cell_class=CellClass.EXCITATORY)
        for name in ("hub", "twin", "leaf", "a", "b", "c")
    }
    connections = {}
    for pre, post in (
        ("a", "hub"), ("b", "hub"), ("c", "hub"), ("hub", "leaf"),
        ("a", "twin"), ("b", "twin"), ("c", "twin"), ("twin", "leaf"),
    ):
        connections[(pre, post, EdgeKind.DRIVE)] = Connection(
            pre=pre, post=post, kind=EdgeKind.DRIVE, sign=1.0, contacts=1
        )
    snapshot = ConnectomeSnapshot(
        version=1, units=units, connections=connections, neuropils={"m": tuple(units)}
    )
    assert degree_matched_control(snapshot, "hub") == "twin"


def test_an_intervention_reports_a_lesion_that_never_bit():
    """A cell nothing called is not a lesion, and looks exactly like a null one."""
    from core.connectome.intervene import run_intervention
    import core.connectome.types as types_module

    class _Holder:
        @staticmethod
        def never_called():
            return 1

    types_module._probe_unused = _Holder  # type: ignore[attr-defined]
    try:
        report = run_intervention(
            lambda: {"score": 1.0},
            "core.connectome.types:_probe_unused.never_called",
            control="core.connectome.types:_probe_unused.never_called",
            repeats=2,
        )
        assert not report.bit
        assert "never called" in report.verdict()
    finally:
        del types_module._probe_unused


def test_a_registered_lesion_prediction_names_a_readout_that_exists():
    """A prediction with no readout is a design, not an experiment.

    Three of the six say what should be lost and were run; this pins that a
    prediction claiming an available readout has one in the runner, on both
    sides, so the claim and the machinery cannot drift apart.
    """
    import importlib.util

    from core.connectome.coalition import LESION_PREDICTIONS

    spec = importlib.util.spec_from_file_location(
        "aura_run_lesions", Path(__file__).resolve().parents[1] / "tools" / "run_lesions.py"
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    for prediction in LESION_PREDICTIONS:
        if not prediction.readout_available:
            continue
        assert prediction.name in runner.INTACT_READOUTS, prediction.name
        lost = set(runner.LOST_READOUTS.get(prediction.name, ())) | set(
            runner.RISING_READOUTS.get(prediction.name, ())
        )
        assert lost, f"{prediction.name} claims a readout and names nothing to lose"
        assert prediction.station in runner.STATION_PHASES, prediction.station


# ---------------------------------------------------------------------------
# The mesh, as a layer of the same graph
# ---------------------------------------------------------------------------


def test_what_the_code_injects_can_reach_what_the_code_reads():
    """The two ends of the mesh the rest of the system touches must connect.

    ``EmbodiedInteroception._push_to_mesh`` injects into the sensory tier and
    ``ConsciousnessBridge._integration_tick`` reads the executive projection. For
    a long time nothing injected could arrive: inter-column probability decayed
    as exp(-|i - j| * 0.15) and a tier boundary is where that distance is
    largest, so on eight seeds nought of sixteen executive columns was reachable
    from any sensory column, 7 to 16 columns were isolated, and the mesh came
    apart into 13 to 25 pieces.

    Several seeds, because the topology is a random draw and one draw is one
    draw.
    """
    import numpy as np

    from core.connectome.neural import build_mesh_layer, signal_can_cross
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    reached = []
    isolated = []
    for seed in range(4):
        mesh = NeuralMesh(MeshConfig())
        mesh._rng = np.random.default_rng(seed=seed)
        mesh._inter_W = mesh._build_inter_column_weights() + mesh._build_feedforward_weights()
        mesh._build_feedback_weights()
        crossing = signal_can_cross(build_mesh_layer(mesh))
        reached.append(crossing["executive_share_reached"])
        isolated.append(crossing["isolated_columns"])
    assert min(reached) > 0.5, (
        f"a signal injected into the sensory tier reaches {min(reached):.0%} of the "
        "executive columns on the worst of four draws"
    )
    assert max(isolated) <= 8, f"{max(isolated)} columns are wired to nothing"


def test_the_mesh_seam_names_the_cells_that_touch_it():
    """A mesh method called on something that is not a mesh is not a seam edge.

    ``get_field_state`` is also a method of the unified field. Matching on the
    method name alone attributed its call sites to the mesh, which is the merge
    error this package already paid for once in the call graph.
    """
    from core.connectome.neural import SEAM_CALLS, _find_callers, build_mesh_layer, join_to_code
    from core.connectome.volume import VolumeReconstructor

    reconstructor = VolumeReconstructor(Path(__file__).resolve().parents[1])
    reconstructor.scan()
    snapshot = reconstructor.build()
    callers = _find_callers(snapshot, SEAM_CALLS)
    assert callers["inject_sensory"], "nothing drives the mesh"
    assert callers["get_executive_projection"], "nothing reads the mesh"
    for uid in callers["get_field_state"]:
        assert "unified_field" not in snapshot.units[uid].neuropil

    layer = join_to_code(build_mesh_layer(), snapshot)
    summary = layer.summary()
    assert summary["code_cells_driving"] >= 1
    assert summary["code_cells_driven"] >= 1
    assert summary["columns"] == 64


# ---------------------------------------------------------------------------
# The shape of the system refuses, rather than only being recorded
# ---------------------------------------------------------------------------


def _delta(before: dict, after: dict, unmeasured: tuple = ()):
    from core.connectome.anatomy_gate import Quality, compare_quality

    return compare_quality(
        Quality(values=dict(before), unmeasured=unmeasured),
        Quality(values=dict(after), unmeasured=unmeasured),
    )


def test_the_tolerance_is_read_off_the_part_rather_than_picked():
    """Further-reaching parts are allowed to cost less shape, from one ladder."""
    from core.cognition.how_a_change_is_promoted import HOW_FAR
    from core.connectome.anatomy_law import tolerated_regression

    ladder = {part: tolerated_regression(part) for part in HOW_FAR}
    assert ladder["word"] > ladder["the search"] > ladder["the deciding"]
    for part, allowance in ladder.items():
        assert allowance == pytest.approx(HOW_FAR[part]), part


def test_a_regression_past_the_tolerance_refuses_the_promotion():
    from core.cognition.how_a_change_is_promoted import (
        AnatomyRefusedError,
        a_ledger_of_its_own,
        promote,
        the_receipts,
    )

    delta = _delta({"single_points_of_failure": 3.0}, {"single_points_of_failure": 9.0})
    with a_ledger_of_its_own():
        with pytest.raises(AnatomyRefusedError) as refusal:
            promote(
                "word overreach",
                became="canary",
                started_by="test",
                evidence="the probe paid",
                anatomy=delta,
            )
        # No receipt for a promotion that did not happen.
        assert the_receipts() == ()
    assert "single point of failure" in str(refusal.value)
    assert "tolerance" in str(refusal.value)


def test_a_regression_inside_the_tolerance_is_promoted_and_recorded():
    from core.cognition.how_a_change_is_promoted import a_ledger_of_its_own, promote

    delta = _delta({"single_points_of_failure": 3.0}, {"single_points_of_failure": 3.2})
    with a_ledger_of_its_own():
        receipt = promote(
            "word small cost",
            became="canary",
            started_by="test",
            evidence="the probe paid",
            anatomy=delta,
        )
    assert "worse: single_points_of_failure" in receipt.evidence


def test_the_same_regression_is_refused_where_every_decision_runs_through_it():
    from core.cognition.how_a_change_is_promoted import (
        AnatomyRefusedError,
        a_ledger_of_its_own,
        promote,
    )

    delta = _delta({"single_points_of_failure": 3.0}, {"single_points_of_failure": 3.2})
    with a_ledger_of_its_own(), pytest.raises(AnatomyRefusedError):
        promote(
            "the deciding/what a change is worth",
            became="canary",
            started_by="test",
            evidence="the probe paid",
            anatomy=delta,
        )


def test_an_unmeasured_change_passes_except_where_nobody_looking_is_not_an_answer():
    from core.cognition.how_a_change_is_promoted import (
        AnatomyRefusedError,
        a_ledger_of_its_own,
        promote,
    )

    with a_ledger_of_its_own():
        receipt = promote(
            "word unmeasured", became="canary", started_by="test", evidence="paid"
        )
        assert "not measured" in receipt.evidence
        with pytest.raises(AnatomyRefusedError) as refusal:
            promote(
                "the search/the proposer",
                became="canary",
                started_by="test",
                evidence="paid",
            )
    assert "nobody can account" in str(refusal.value)


def test_putting_something_back_is_never_refused():
    """A law that could block the remedy would trap the state it exists to stop."""
    from core.cognition.how_a_change_is_promoted import a_ledger_of_its_own, promote

    delta = _delta({"single_points_of_failure": 3.0}, {"single_points_of_failure": 99.0})
    with a_ledger_of_its_own():
        for became in ("rolled back", "would not go back", "retired"):
            receipt = promote(
                "the deciding/whatever",
                became=became,
                started_by="she",
                evidence="the probe did not hold",
                anatomy=delta,
            )
            assert receipt.became == became


def test_a_counted_axis_is_counted_and_a_ratio_axis_is_scaled():
    """One more brittle cell is one more, whatever the total was.

    Dividing a count by a total that the same edit changed measures the
    denominator. Ratio axes are scaled because a shift of 0.02 means something
    different against 0.05 than against 5.0.
    """
    from core.connectome.anatomy_law import anatomy_permits

    counted = _delta({"half_wired_channels": 100.0}, {"half_wired_channels": 100.4})
    assert anatomy_permits(counted, at="word").allowed
    ratio = _delta({"local_recurrence": 1.0}, {"local_recurrence": 0.4})
    verdict = anatomy_permits(ratio, at="word")
    assert not verdict.allowed
    assert verdict.regressions["local_recurrence"] == pytest.approx(0.6)


def test_the_law_is_not_an_outage_when_its_own_machinery_is_missing():
    """A constitution that blocks everything when it cannot run is a failure mode."""
    from core.cognition.how_a_change_is_promoted import a_ledger_of_its_own, promote

    class _NotAReading:
        pass

    with a_ledger_of_its_own():
        receipt = promote(
            "word x",
            became="canary",
            started_by="test",
            evidence="paid",
            anatomy=_NotAReading(),
        )
    assert receipt.became == "canary"


# ---------------------------------------------------------------------------
# The reading hierarchy, against a meta-analysis of 163 human studies
# ---------------------------------------------------------------------------


def _reading_profile(name: str, cells: set, regions: dict | None = None):
    from core.connectome.reading import ReadingProfile

    return ReadingProfile(
        condition=name,
        cells=frozenset(cells),
        by_region=regions or {"r": len(cells)},
        frames=500,
    )


def test_every_human_reading_finding_carries_what_would_refuse_it():
    from core.connectome.reading import HUMAN_READING

    assert len(HUMAN_READING) == 5
    for finding in HUMAN_READING:
        assert finding.in_humans and finding.predicted_here and finding.falsifier
        assert finding.predicted_here != finding.in_humans
        # The prediction is about her, so it may not simply name a brain region.
        assert "cortex" not in finding.predicted_here.lower()


def test_a_core_is_what_every_level_fires_and_specific_is_what_only_one_does():
    from core.connectome.reading import core_and_specific

    profiles = {
        "letter": _reading_profile("letter", {"a", "b", "L"}),
        "word": _reading_profile("word", {"a", "b", "W"}),
        "sentence": _reading_profile("sentence", {"a", "b", "S"}),
        "text": _reading_profile("text", {"a", "b", "T"}),
    }
    shape = core_and_specific(profiles)
    assert shape["core"] == 2
    assert shape["specific"] == {"letter": 1, "word": 1, "sentence": 1, "text": 1}
    assert shape["core_share_of_each"]["letter"] == pytest.approx(2 / 3, abs=1e-4)


def test_the_frame_cap_is_what_stops_a_longer_condition_looking_bigger():
    """A condition recorded twice as long fires more cells for that reason.

    Without the cap, "text recruits the most machinery" is a statement about how
    many frames text got.
    """
    import numpy as np

    from core.connectome.activity import ActivityTrace
    from core.connectome.reading import profile_condition
    from core.connectome.types import CellClass, ConnectomeSnapshot, Unit

    # Two conditions firing the same cell, one recorded four times as long, and
    # a second cell that only fires in the long one's later frames.
    rows = []
    conditions = []
    for index in range(500):
        rows.append([1.0, 1.0 if index > 400 else 0.0])
        conditions.append("long")
    for _ in range(100):
        rows.append([1.0, 0.0])
        conditions.append("short")
    trace = ActivityTrace(
        uids=("one", "two"),
        conditions=tuple(conditions),
        spikes=[],
        array=np.asarray(rows),
    )
    units = {
        name: Unit(uid=name, name=name, neuropil="m", region="r", cell_class=CellClass.EXCITATORY)
        for name in ("one", "two")
    }
    snapshot = ConnectomeSnapshot(
        version=1, units=units, connections={}, neuropils={"m": ("one", "two")}
    )
    uncapped = profile_condition(trace, snapshot, "long")
    capped = profile_condition(trace, snapshot, "long", frames_cap=100, seed=3)
    assert uncapped.size == 2
    assert capped.frames == 100
    assert capped.size <= uncapped.size


def test_two_routes_needs_both_sides_to_have_something_of_their_own():
    from core.connectome.reading import dual_route

    known = _reading_profile("word", {"a", "b", "K"})
    unknown = _reading_profile("pseudoword", {"a", "b", "U"})
    both = dual_route(known, unknown)
    assert both["two_routes"] is True
    assert both["only_known"] == 1 and both["only_unknown"] == 1

    nested = dual_route(known, _reading_profile("pseudoword", {"a", "b"}))
    assert nested["two_routes"] is False, "one route inside another is one route"


def test_the_task_and_the_stimulus_are_measured_the_same_way():
    """Comparing a distance with a count would decide the answer in advance."""
    from core.connectome.reading import task_over_stimulus

    profiles = {
        "word": _reading_profile("word", {"a", "b", "c", "d"}),
        "sentence": _reading_profile("sentence", {"a", "b", "c", "e"}),
        "judge_word": _reading_profile("judge_word", {"x", "y", "z", "w"}),
    }
    measured = task_over_stimulus(
        profiles,
        same_task_pairs=[("word", "sentence")],
        same_stimulus_pairs=[("word", "judge_word")],
    )
    assert measured["task_beats_stimulus"] is True
    assert 0.0 <= measured["distance_when_the_text_changes"] <= 1.0
    assert 0.0 <= measured["distance_when_the_question_changes"] <= 1.0


def test_the_reading_analysis_that_was_run_is_the_one_that_is_published():
    """The numbers in CONNECTOME.md come out of the artefact, not out of prose."""
    import json

    path = Path(__file__).resolve().parents[1] / "artifacts" / "connectome" / "reading" / "reading_analysis.json"
    if not path.exists():
        pytest.skip("no reading recording on this checkout")
    payload = json.loads(path.read_text())
    from core.connectome.reading import HUMAN_READING

    assert set(payload["findings"]) == {finding.name for finding in HUMAN_READING}
    for row in payload["findings"].values():
        assert isinstance(row["holds"], bool)
        assert row["falsifier"]


# ---------------------------------------------------------------------------
# What is measured, what was chosen, and what nothing here pins
# ---------------------------------------------------------------------------


def test_every_term_of_the_mind_says_where_its_value_comes_from():
    from core.science.reference_mind import CONSTRAINTS, Provenance, Term

    covered = {one.term for one in CONSTRAINTS}
    assert covered == set(Term), f"missing: {sorted(set(Term) - covered)}"
    for one in CONSTRAINTS:
        assert one.evidence and one.what_it_pins
        assert one.uses in set(Provenance)
        if one.uses in (Provenance.MEASURED, Provenance.DERIVED):
            assert one.where, f"{one.term} claims a measurement and names no module"
            assert one.falsifier or one.note, f"{one.term} claims a measurement bare"


def test_a_term_that_claims_a_measurement_names_a_module_that_exists():
    """A citation to a file nobody wrote is the same defect as no citation."""
    from core.science.reference_mind import CONSTRAINTS, Provenance

    root = Path(__file__).resolve().parents[1]
    for one in CONSTRAINTS:
        if one.uses not in (Provenance.MEASURED, Provenance.DERIVED) or not one.where:
            continue
        assert (root / one.where).exists(), f"{one.term} points at {one.where}"


def test_the_share_that_rests_on_evidence_is_reported_rather_than_assumed():
    from core.science.reference_mind import identifiability

    report = identifiability()
    assert 0.0 <= report["grounded_share"] <= 1.0
    assert report["grounded"] + len(report["chosen_or_absent"]) == report["terms"]
    assert "engineer" in report["verdict"]


def test_the_mesh_says_how_many_of_its_numbers_anybody_measured():
    """Calling a boundary sensory does not make the index it sits at a finding."""
    from core.science.reference_mind import audit_mesh

    report = audit_mesh()
    assert report["fields"] >= 10
    assert report["chosen"] > 0, (
        "every structural number in the mesh now claims a basis; if that is real "
        "the bases belong in _MESH_MEASURED with their sources"
    )
    assert report["chosen"] + report["with_a_basis"] == report["fields"]


def test_a_transmitter_can_land_somewhere_rather_than_everywhere():
    """One scalar for 4,096 units made dopamine at the sensory tier and dopamine
    at the executive tier the same event. Cortex is not like that: receptor
    densities vary by area and a transmitter's effect depends on where it lands.

    Uniform stays the default, because saying the spatial structure has not been
    measured is honest and guessing at it is not.
    """
    import numpy as np

    from core.consciousness.neural_mesh import NeuralMesh

    mesh = NeuralMesh()
    assert set(mesh._tier_names) == {"sensory", "association", "executive"}
    assert all(pair == (1.0, 1.0) for pair in mesh.regional_modulation().values())
    assert np.allclose(mesh._tier_vector(0), 1.0)

    mesh.set_regional_modulation({"executive": (2.0, 0.5), "sensory": (0.7, 1.4)})
    gain = mesh._tier_vector(0)
    noise = mesh._tier_vector(1)
    assert gain[0] == pytest.approx(0.7) and gain[-1] == pytest.approx(2.0)
    assert noise[0] == pytest.approx(1.4) and noise[-1] == pytest.approx(0.5)
    assert gain[mesh.cfg.sensory_end + 1] == pytest.approx(1.0), "association untouched"

    for _ in range(4):
        mesh._tick_inner()
    assert np.isfinite(mesh.get_field_state()).all()

    mesh.set_regional_modulation(None)
    assert np.allclose(mesh._tier_vector(0), 1.0)


def test_a_bad_regional_multiplier_cannot_switch_the_mesh_off():
    from core.consciousness.neural_mesh import NeuralMesh

    mesh = NeuralMesh()
    mesh.set_regional_modulation(
        {
            "executive": (float("nan"), 1.0),
            "sensory": (1e9, -4.0),
            "nowhere": (2.0, 2.0),
        }
    )
    applied = mesh.regional_modulation()
    assert applied["executive"] == (1.0, 1.0), "a non-finite pair is refused whole"
    assert applied["sensory"][0] <= 4.0 and applied["sensory"][1] >= 0.0
    assert "nowhere" not in applied
