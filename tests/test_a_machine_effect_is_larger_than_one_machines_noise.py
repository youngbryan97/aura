"""A result that survives only on the machine it was measured on is about that machine.

Two runs on one host already differ — irreducibility moved 0.028 and the lesion
deficit 0.122 between two of them. So comparing one number here against one
number there says nothing: what has to be separated is how much of the gap is
the machine and how much is the run.

Every measure is reported twice. The spread between runs on one host, and the
spread between hosts. A between-host gap smaller than what one host already
produces on its own is not a machine effect; one that is larger is the case the
specification calls scientifically important, where the apparent integrated
subject disappears under small timing differences.

Two runs of different campaigns are two experiments, and reading across them is
the thing the fingerprint exists to prevent — so that is refused unless it is
asked for explicitly.
"""

from __future__ import annotations

import pytest

from tools.compare_across_machines import COMPARED, _comparable, _host, compare

pytestmark = pytest.mark.unit


def _report(host: str, *, phi: float, campaign: str = "abc", tree: str = "t1", **over):
    base = {
        "campaign": {
            "fingerprint": campaign, "tree_hash": tree, "platform": host,
            "frozen": {"schema": {"hash": "s1"}},
        },
        "environment": {"os": host, "hardware": {"cpus": 8, "memory_gb": 32.0}},
        "phi": {"phi_do": phi, "lower_bound": phi - 0.01},
        "differentiation": {"d_eff": 12.0, "d_eff_normalised": 0.09},
        "perturbation": {"mean_spread": 0.5, "mean_pci": 0.2},
        "intrinsic": {"delta_intrinsic": 0.5},
        "graph": {"vertex_connectivity": 2, "edge_count": 40},
        "lesion": {"deltas": {"phi_do": 0.1, "spread": 0.1}, "rescue": {"phi_do": 0.08}},
        "agency": {"ownership_divergence": 0.5},
        "closure": {"leak": 0.0},
        "edges": [
            {"source": "A", "target": "G", "kept": True},
            {"source": "G", "target": "A", "kept": True},
        ],
        "verdict": {"criteria": [{"criterion": "phi", "passed": phi > 0.05}]},
    }
    base.update(over)
    return base


def test_two_runs_on_one_host_report_a_within_spread_and_no_between() -> None:
    out = compare([("a", _report("mac", phi=0.05)), ("b", _report("mac", phi=0.09))])
    assert out["hosts"] == ["mac / 8 / 32.0"]
    row = out["measures"]["phi_do"]
    assert row["within_machine_spread"] == pytest.approx(0.04)
    assert row["between_machine_spread"] is None
    assert row["machine_effect"] is None


def test_a_gap_smaller_than_one_hosts_own_noise_is_not_a_machine_effect() -> None:
    out = compare([
        ("a1", _report("mac", phi=0.02)),
        ("a2", _report("mac", phi=0.12)),   # one host spans 0.10 on its own
        ("b1", _report("linux", phi=0.08)),
        ("b2", _report("linux", phi=0.09)),
    ])
    assert len(out["hosts"]) == 2
    assert out["measures"]["phi_do"]["machine_effect"] is False


def test_a_gap_larger_than_it_is_named() -> None:
    out = compare([
        ("a1", _report("mac", phi=0.50)),
        ("a2", _report("mac", phi=0.51)),
        ("b1", _report("linux", phi=0.01)),
        ("b2", _report("linux", phi=0.02)),
    ])
    assert out["measures"]["phi_do"]["machine_effect"] is True
    assert "phi_do" in out["measures_with_a_machine_effect"]


def test_different_campaigns_are_two_experiments() -> None:
    rows = _comparable([_report("mac", phi=0.1), _report("mac", phi=0.1, campaign="zzz")])
    assert rows["same_campaign"] is False


def test_the_same_campaign_on_different_code_is_also_two_experiments() -> None:
    rows = _comparable([_report("mac", phi=0.1), _report("mac", phi=0.1, tree="t2")])
    assert rows["same_campaign"] is True
    assert rows["same_code"] is False


def test_the_topology_is_compared_edge_by_edge() -> None:
    other = _report("mac", phi=0.1)
    other["edges"] = [{"source": "A", "target": "G", "kept": True}]
    out = compare([("a", _report("mac", phi=0.1)), ("b", other)])
    assert out["topology"]["kept_everywhere"] == ["A->G"]
    assert out["topology"]["differing"] == ["G->A"]


def test_a_criterion_that_answered_differently_is_named() -> None:
    out = compare([("a", _report("mac", phi=0.02)), ("b", _report("mac", phi=0.09))])
    assert out["criteria_that_disagreed"] == ["phi"]


def test_a_run_that_recorded_no_machine_does_not_count_as_its_own_host() -> None:
    """Reports written before the runner recorded an environment carry only
    the campaign's platform string."""
    thin = _report("mac", phi=0.1)
    thin["environment"] = {}
    assert _host(thin) == "mac"


def test_every_compared_measure_is_declared_up_front() -> None:
    """So one cannot be dropped after seeing that it disagreed."""
    labels = {label for label, _ in COMPARED}
    for wanted in ("phi_do", "spread", "vertex_connectivity", "lesion_deficit_phi", "ownership"):
        assert wanted in labels


def _with_synergy(host: str, rows):
    report = _report(host, phi=0.1)
    report["synergy"] = rows
    report["metastability"] = {"dwell_mean": 60.0, "largest_regime_share": 0.67}
    report["global_access"] = {"consumers": 8}
    return report


def test_synergy_is_compared_triple_by_triple() -> None:
    """Not by each run's own best triple, which would compare two different
    measurements and call them one."""
    out = compare([
        ("a", _with_synergy("mac", [
            {"sources": ["A", "S"], "target": "G", "synergy_fraction": 0.03, "passes": False},
            {"sources": ["P", "M"], "target": "W", "synergy_fraction": 0.27, "passes": True},
        ])),
        ("b", _with_synergy("mac", [
            {"sources": ["A", "S"], "target": "G", "synergy_fraction": 0.05, "passes": False},
            {"sources": ["P", "M"], "target": "W", "synergy_fraction": 0.24, "passes": True},
        ])),
    ])
    assert out["synergy"]["triples_compared"] == ["A+S->G", "P+M->W"]
    assert out["measures"]["synergy[P+M->W]"]["within_machine_spread"] == pytest.approx(0.03)
    assert out["synergy"]["passing_everywhere"] == ["P+M->W"]


def test_a_triple_that_passed_on_one_run_only_is_named() -> None:
    out = compare([
        ("a", _with_synergy("mac", [
            {"sources": ["P", "M"], "target": "W", "synergy_fraction": 0.27, "passes": True},
        ])),
        ("b", _with_synergy("linux", [
            {"sources": ["P", "M"], "target": "W", "synergy_fraction": 0.11, "passes": False},
        ])),
    ])
    assert out["synergy"]["passing_only_on_some_runs"] == ["P+M->W"]
    assert out["synergy"]["passing_everywhere"] == []


def test_metastability_and_broadcast_are_carried() -> None:
    out = compare([
        ("a", _with_synergy("mac", [])),
        ("b", _with_synergy("mac", [])),
    ])
    assert "dwell_mean" in out["measures"]
    assert "broadcast_consumers" in out["measures"]
