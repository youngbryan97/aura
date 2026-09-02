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


# ── the contracts publish rather than sit ─────────────────────────────────

def test_both_cognitive_fragments_register_and_carry_real_counts():
    from core.cognition.contract_health import install
    from core.runtime.health_fragments import collect_health_fragments

    assert install() == {"contracts": True, "growth": True}
    fragments = collect_health_fragments()
    for name in ("cognitive_contracts", "cognitive_growth"):
        assert fragments[name]["registered"], name


def test_the_contract_fragment_reports_traffic_not_capability():
    from core.cognition.contract_health import contract_health_fragment

    fragment = contract_health_fragment()
    assert set(fragment["evidence"]) >= {"duplicate_assertions_refused", "unattributed_assertions"}
    assert "coverage" in fragment["state_handoffs"]
    assert "organs_reporting" in fragment["impasse_bus"]


def test_importing_the_fragment_is_what_registers_the_invariants():
    from core.cognition.contract_health import contract_health_fragment

    assert contract_health_fragment()["architecture_invariants"]["checked"] >= 5


def test_a_ledger_that_raises_is_reported_not_fatal(monkeypatch):
    import core.cognition.contract_health as module

    def explode():
        raise RuntimeError("store is closed")

    assert "unavailable" in module._safe("probe", explode)


def test_the_expected_fragments_include_both_cognitive_ones():
    from core.runtime.health_fragments import EXPECTED_FRAGMENTS

    assert "cognitive_contracts" in EXPECTED_FRAGMENTS
    assert "cognitive_growth" in EXPECTED_FRAGMENTS


def test_the_evidence_report_runs_and_finds_no_contradiction():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evidence_report", ROOT / "tools/evidence_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_report"] = module
    spec.loader.exec_module(module)
    picture = module.gather()
    assert picture["claims"]["claims"] >= 4
    assert module.contradictions(picture) == []


def test_the_evidence_report_names_the_campaigns_that_have_not_been_run():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evidence_report", ROOT / "tools/evidence_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_report"] = module
    spec.loader.exec_module(module)
    campaigns = module.campaign_surface()
    assert "developmental_campaign" in campaigns
    assert all("establishes" in entry and "needs" in entry for entry in campaigns.values())


def test_no_claim_stands_above_causal_without_a_campaign_behind_it():
    from core.science.claim_ladder import Rung, get_ladder

    for claim in get_ladder().claims():
        assert (claim.rung or 0) <= Rung.CAUSAL, (
            f"{claim.statement!r} claims {claim.rung}; no campaign has been run"
        )


def test_raising_the_complexity_ratchet_requires_a_reason():
    baseline = json.loads((ROOT / "config/cognitive_complexity_baseline.json").read_text())
    for entry in baseline.get("history", []):
        assert entry["because"].strip(), (
            "a ratchet that can be reset without a reason is a number, not a ratchet"
        )
        assert set(entry["raised"]) <= set(entry["to"])


# ── the gap atlas gates itself ────────────────────────────────────────────

def _atlas():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gap_atlas", ROOT / "tools/gap_atlas.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gap_atlas"] = module
    spec.loader.exec_module(module)
    return module


def test_every_card_is_adjudicated_and_every_closed_one_names_a_test():
    assert _atlas().check() == 0


def test_a_bar_claiming_something_about_the_system_needs_a_production_caller():
    module = _atlas()
    entry = {
        "bar": "every organ reports through one bus",
        "closed_by": ["core/cognition/substate.py", "tests/x.py"],
    }
    assert module._integration_problems("X", entry)
    entry["wired_by"] = ["core/cognition/contract_health.py"]
    assert not module._integration_problems("X", entry)


def test_a_bar_naming_a_demonstrated_result_needs_the_campaign_named():
    module = _atlas()
    entry = {"bar": "the learned policy beats the static one", "closed_by": ["tests/x.py"]}
    assert module._demonstration_problems("X", entry)
    entry["outstanding"] = "a matched-compute A/B"
    assert not module._demonstration_problems("X", entry)


def test_a_mechanism_bar_needs_neither():
    module = _atlas()
    entry = {"bar": "duplicate evidence cannot inflate belief", "closed_by": ["tests/x.py"]}
    assert not module._integration_problems("X", entry)
    assert not module._demonstration_problems("X", entry)


def test_a_campaign_that_was_run_answers_the_bar_a_harness_cannot():
    module = _atlas()
    entry = {"bar": "the learned policy beats the static one", "closed_by": ["tests/x.py"]}
    assert module._demonstration_problems("X", entry)
    # A run campaign satisfies the bar, and only when it names evidence that
    # is on disk and that a test reads - otherwise it is a sentence, which is
    # the thing `outstanding` exists to stop being mistaken for a result.
    entry["campaign_run"] = "it went well"
    assert not module._demonstration_problems("X", entry)
    assert module._campaign_problems("X", entry)
    entry["campaign_run"] = "beat it (docs/evidence/nothing_here_at_all.json)"
    assert module._campaign_problems("X", entry)


def test_every_card_that_closes_on_a_harness_says_which_campaign():
    """Not a count: the property. Counting broke the moment a campaign ran."""
    import json

    module = _atlas()
    adjudication = json.loads((ROOT / "docs/gap_atlas/adjudication.json").read_text())
    entries = adjudication["entries"]

    demonstration_bars = [
        cid
        for cid, entry in entries.items()
        if any(
            word in (entry.get("bar") or "").lower()
            for word in module.DEMONSTRATION_WORDS
        )
    ]
    assert demonstration_bars, "no card names a demonstrated result at all"

    for cid in demonstration_bars:
        entry = entries[cid]
        assert entry.get("outstanding") or entry.get("campaign_run"), (
            f"[{cid}] closes on a harness and says neither which campaign is "
            "still to run nor which one was"
        )
        assert not module._campaign_problems(cid, entry), module._campaign_problems(
            cid, entry
        )

    # And the two halves are both non-empty, so neither field is carrying the
    # whole list by accident.
    assert [c for c in demonstration_bars if entries[c].get("outstanding")]
    assert [c for c in demonstration_bars if entries[c].get("campaign_run")]


# ── the clean-room rule can fire ──────────────────────────────────────────

def _lint():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "architecture_lint", ROOT / "tools/architecture_lint.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["architecture_lint"] = module
    spec.loader.exec_module(module)
    return module


def test_a_rival_copyright_line_is_caught(tmp_path):
    """A rule that cannot match reports green forever."""
    module = _lint()
    probe = ROOT / "core" / "_cleanroom_probe"
    probe.mkdir(exist_ok=True)
    (probe / "vendored.py").write_text("# Copyright (c) OpenCog Foundation\n")
    try:
        findings = module.check_clean_room([str(probe.relative_to(ROOT))])
        assert any("copyright" in f.message for f in findings)
    finally:
        (probe / "vendored.py").unlink()
        probe.rmdir()


def test_importing_a_rival_runtime_is_caught():
    module = _lint()
    probe = ROOT / "core" / "_cleanroom_probe"
    probe.mkdir(exist_ok=True)
    (probe / "vendored.py").write_text("import nengo\n")
    try:
        findings = module.check_clean_room([str(probe.relative_to(ROOT))])
        assert any("vendored code" in f.message for f in findings)
    finally:
        (probe / "vendored.py").unlink()
        probe.rmdir()


def test_the_tree_carries_no_rival_source():
    module = _lint()
    coverage = json.loads((ROOT / "config/architecture_lint_coverage.json").read_text())
    assert module.check_clean_room(coverage["clean_room"]) == []


def test_the_clean_room_rule_covers_where_the_adaptation_happened():
    coverage = json.loads((ROOT / "config/architecture_lint_coverage.json").read_text())
    covered = set(coverage["clean_room"])
    assert {"core/cognition", "core/evidence", "core/knowledge", "core/science"} <= covered
