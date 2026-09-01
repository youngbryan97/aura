"""The mind should get simpler, and a result should survive leaving this machine.

Cards 013, 030, 043, 057, 098, 103, 165, 190, 193, A12.14, A12.15.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from core.knowledge.atomspace import AtomSpace, TruthValue, concept, implication
from core.science.replication_pack import (
    Environment,
    ReplicationPack,
    ReplicationRegistry,
)

ROOT = Path(__file__).resolve().parent.parent


def _complexity():
    spec = importlib.util.spec_from_file_location(
        "cognitive_complexity", ROOT / "tools/cognitive_complexity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cognitive_complexity"] = module
    spec.loader.exec_module(module)
    return module


# ── complexity ────────────────────────────────────────────────────────────

def test_the_complexity_measures_are_all_present_and_positive():
    current = _complexity().measure()
    assert set(current) == {
        "organs", "cross_package_edges", "dependency_entropy", "kernel_lines"
    }
    assert all(value > 0 for value in current.values())


def test_the_baseline_exists_and_today_is_inside_it():
    module = _complexity()
    baseline = json.loads((ROOT / "config/cognitive_complexity_baseline.json").read_text())
    current = module.measure()
    assert all(current[key] <= baseline[key] for key in current), {
        key: (current[key], baseline[key]) for key in current if current[key] > baseline[key]
    }


def test_entropy_distinguishes_a_line_from_a_mesh():
    from collections import Counter

    module = _complexity()
    line = Counter({("a", "b"): 10, ("b", "c"): 10})
    mesh = Counter({(a, b): 5 for a in "abcd" for b in "abcd" if a != b})
    assert module._entropy(mesh) > module._entropy(line), (
        "twenty packages in a line are simpler than ten in a mesh"
    )


def test_organs_sharing_an_invariant_are_the_merge_candidates():
    shared = _complexity().redundancy()
    assert shared, "an architecture with no two organs doing the same job is unusual"
    assert all(len(names) > 1 for names in shared.values())
    assert ["agency", "skills"] in shared.values()


def test_an_organ_with_no_declared_invariant_is_listed_not_ignored():
    unmapped = _complexity().unmapped()
    assert unmapped, "the compression programme starts from this list"
    assert "cognition" not in unmapped


def test_every_declared_invariant_names_something_a_reader_can_check():
    module = _complexity()
    for package, invariant in module.INVARIANTS.items():
        assert len(invariant.split()) >= 3, f"{package}: {invariant!r} is a label, not an invariant"


# ── the scaling question, measured before answering it ────────────────────

@pytest.mark.parametrize("count", [2_000, 8_000])
def test_atomspace_point_reads_do_not_degrade_with_size(count):
    """The report asks for a native backend. Measure the wall first."""
    space = AtomSpace(max_atoms=200_000)
    for i in range(count):
        space.add(concept(f"c{i}"), TruthValue(0.5, 1.0), source=f"s{i}")
    started = time.perf_counter()
    for i in range(200):
        space.get_tv(concept(f"c{i * (count // 200)}"))
    per_read = (time.perf_counter() - started) / 200
    assert per_read < 5e-5, f"{per_read * 1e6:.1f}us per point read at {count} atoms"


def test_adding_atoms_stays_roughly_linear():
    def add(count: int) -> float:
        space = AtomSpace(max_atoms=200_000)
        started = time.perf_counter()
        for i in range(count):
            space.add(concept(f"c{i}"), TruthValue(0.5, 1.0), source=f"s{i}")
        return (time.perf_counter() - started) / count

    small, large = add(2_000), add(20_000)
    assert large < small * 4, (
        f"{small * 1e6:.1f}us/atom at 2k against {large * 1e6:.1f}us/atom at 20k; "
        "a superlinear insert is the wall a native backend would be for"
    )


def test_the_source_map_bound_holds_under_many_witnesses():
    from core.knowledge.atomspace import _MAX_SOURCES_PER_ATOM

    space = AtomSpace()
    atom = concept("busy")
    for i in range(_MAX_SOURCES_PER_ATOM * 2):
        space.add(atom, TruthValue(0.5, 1.0), source=f"s{i}")
    assert len(space.evidence_sources(atom)) <= _MAX_SOURCES_PER_ATOM


# ── replication ───────────────────────────────────────────────────────────

def _pack(**kw):
    base = dict(
        claim="duplicate evidence cannot inflate belief",
        tasks=("t1", "t2"), seeds=(1, 2, 3), arms=("treatment", "null"),
        expected={"treatment": 0.80, "null": 0.40}, tolerance=0.05,
        origin=Environment("abc123", "qwen3.8-27b", "m4-max", "3.12"),
        entrypoint="tests/test_evidence_independence.py",
    )
    return ReplicationPack(**{**base, **kw})


def test_a_pack_with_no_tolerance_cannot_be_replicated():
    with pytest.raises(ValueError, match="willing to be wrong by"):
        _pack(tolerance=0.0)


def test_a_pack_with_no_seeds_cannot_be_rerun_the_same_way():
    with pytest.raises(ValueError, match="the same way twice"):
        _pack(seeds=())


def test_expected_results_have_to_cover_the_arms():
    with pytest.raises(ValueError, match="do not cover the arms"):
        _pack(expected={"treatment": 0.8})


def test_the_same_machine_is_a_rerun_not_a_replication():
    pack = _pack()
    registry = ReplicationRegistry()
    registry.publish(pack)
    attempt = registry.submit(
        pack.seal, {"treatment": 0.79, "null": 0.41}, pack.origin, replicator="same box"
    )
    assert attempt.within_tolerance
    assert not attempt.divergence()["independent"]
    assert registry.status(pack.claim)["independent_replications"] == 0


def test_different_hardware_is_an_independent_replication():
    pack = _pack()
    registry = ReplicationRegistry()
    registry.publish(pack)
    attempt = registry.submit(
        pack.seal, {"treatment": 0.78, "null": 0.42},
        Environment("abc123", "qwen3.8-27b", "linux-x86", "3.12"), replicator="external",
    )
    assert attempt.within_tolerance
    assert attempt.divergence()["differed_on"] == ["hardware"]
    assert registry.status(pack.claim)["externally_replicated"]


def test_a_result_outside_tolerance_fails_however_independent():
    pack = _pack()
    registry = ReplicationRegistry()
    registry.publish(pack)
    attempt = registry.submit(
        pack.seal, {"treatment": 0.50, "null": 0.40},
        Environment("zzz", "other-model", "linux-x86"), replicator="external",
    )
    assert not attempt.within_tolerance
    assert registry.status(pack.claim)["failed"]


def test_a_pack_that_moved_cannot_be_replicated_against():
    pack = _pack()
    registry = ReplicationRegistry()
    registry.publish(pack)
    moved = _pack(tasks=("t1", "t2", "t3"))
    assert moved.seal != pack.seal
    with pytest.raises(KeyError, match="a pack that moved is a rerun"):
        registry.submit(moved.seal, {"treatment": 0.8, "null": 0.4}, pack.origin)


def test_the_seal_covers_the_tolerance_so_it_cannot_be_loosened_afterwards():
    assert _pack(tolerance=0.05).seal != _pack(tolerance=0.5).seal


def test_divergence_names_what_differed_rather_than_passing_or_failing_on_it():
    pack = _pack()
    registry = ReplicationRegistry()
    registry.publish(pack)
    attempt = registry.submit(
        pack.seal, {"treatment": 0.8, "null": 0.4},
        Environment("def456", "other-model", "linux-x86", "3.13"), replicator="external",
    )
    assert set(attempt.divergence()["differed_on"]) == {"commit", "model", "hardware", "python"}
