"""One run gives a number, and the difference between two of them is noise.

Reporting the run that came out highest is picking the draw. So the scorecard
asks, per criterion, whether it held every time, failed every time, or changed
its answer — and for the continuous measures underneath, how far they moved.

Those spreads are the point. Across run_017 and run_019 the irreducibility
estimate ran from 0.022 to 0.062 against a bar of 0.05, and ownership from 0.84
to 0.22. A criterion reading yes on one of those and no on the other is not
evidence either way, and a total hides which.

It also names, per domain, the thinnest channel in and the thinnest channel
out. A domain can sit inside the strongly connected component hanging off one
weak edge, and the component alone does not say which one.
"""

from __future__ import annotations

import pytest

from tools.subject_core_scorecard import SYNERGY_TRIPLES, TRACKED, _couplings, scorecard

pytestmark = pytest.mark.unit


def _report(**over):
    base = {
        "verdict": {"passed": 2, "total": 3, "criteria": [
            {"criterion": "a", "passed": True, "bar": ">0", "section": "1", "requires": "x"},
            {"criterion": "b", "passed": True, "bar": ">0", "section": "1", "requires": "y"},
            {"criterion": "c", "passed": False, "bar": ">0", "section": "1", "requires": "z"},
        ]},
        "campaign": {"fingerprint": "abc", "frozen": {"seed": 7}},
        "phi": {"phi_do": 0.05, "lower_bound": 0.01, "standard_error": 0.02},
        "differentiation": {"d_eff": 12.0, "d_eff_normalised": 0.09, "largest_component_share": 0.25},
        "graph": {"vertex_connectivity": 2, "edge_count": 40},
        "agency": {"ownership_divergence": 0.5, "ownership_floor": 0.03, "self_to_action": 0.07},
        "lesion": {"deltas": {"phi_do": 0.1, "spread": 0.1, "synergy": 0.1},
                   "rescue": {"phi_do": 0.08}},
        "synergy": [
            {"sources": ["A", "S"], "target": "G", "synergy_fraction": 0.08},
            {"sources": ["P", "M"], "target": "W", "synergy_fraction": 0.44},
        ],
        "edges": [
            {"source": "A", "target": "G", "effect": 0.9, "kept": True},
            {"source": "A", "target": "C", "effect": 0.4, "kept": True},
            {"source": "G", "target": "A", "effect": 0.6, "kept": True},
            {"source": "D", "target": "A", "effect": 0.2, "kept": False},
        ],
        "perturbation": {"mean_spread": 0.5, "mean_pci": 0.2},
        "nulls": {"surrogate_floor": 0.01},
        "intrinsic": {"delta_intrinsic": 0.5},
        "closure": {"leak": 0.0},
        "metastability": {"regimes": 2},
        "global_access": {"consumers": 7},
    }
    base.update(over)
    return base


def test_a_criterion_that_changed_answer_is_unresolved() -> None:
    first = _report()
    second = _report(verdict={"passed": 1, "total": 3, "criteria": [
        {"criterion": "a", "passed": True, "bar": ">0", "section": "1", "requires": "x"},
        {"criterion": "b", "passed": False, "bar": ">0", "section": "1", "requires": "y"},
        {"criterion": "c", "passed": False, "bar": ">0", "section": "1", "requires": "z"},
    ]})
    card = scorecard([first, second])
    verdicts = {row["criterion"]: row["verdict"] for row in card["criteria"]}
    assert verdicts == {"a": "holds", "b": "unresolved", "c": "fails"}


def test_every_continuous_measure_carries_its_spread() -> None:
    card = scorecard([_report(), _report()])
    for row in card["numbers"].values():
        assert {"values", "spread", "median", "mean", "sd", "min", "max"} <= set(row)


def test_the_measures_a_criterion_can_flip_on_are_tracked() -> None:
    labels = {label for label, _ in TRACKED}
    for wanted in (
        "phi_do", "phi_lower_bound", "spread", "ownership",
        "lesion_deficit_phi", "rescue_phi", "vertex_connectivity",
    ):
        assert wanted in labels


def test_the_triples_are_keyed_by_name_not_by_position() -> None:
    """A run that reordered them is not a run that changed them."""
    flipped = _report(synergy=[
        {"sources": ["P", "M"], "target": "W", "synergy_fraction": 0.44},
        {"sources": ["A", "S"], "target": "G", "synergy_fraction": 0.08},
    ])
    card = scorecard([_report(), flipped])
    assert card["synergy_triples"]["A+S->G"]["sd"] == pytest.approx(0.0)
    assert card["synergy_triples"]["P+M->W"]["sd"] == pytest.approx(0.0)


def test_the_four_triples_are_named_up_front() -> None:
    assert SYNERGY_TRIPLES == ("A+S->G", "P+M->W", "W+A->D", "S+D->C")


def test_each_domain_gets_its_thinnest_channel_in_and_out() -> None:
    rows = _couplings(_report())
    assert rows["A"]["outgoing"] == 2
    assert rows["A"]["weakest_outgoing"] == {"to": "C", "effect": 0.4}
    assert rows["A"]["weakest_incoming"] == {"from": "G", "effect": 0.6}


def test_an_edge_that_was_not_kept_is_not_a_channel() -> None:
    rows = _couplings(_report())
    assert rows["D"]["outgoing"] == 0
    assert rows["D"]["weakest_outgoing"] is None


def test_runs_from_different_campaigns_are_recorded_as_different(  ) -> None:
    """Reading across two campaigns is reading across two experiments."""
    other = _report(campaign={"fingerprint": "zzz", "frozen": {"seed": 11}})
    card = scorecard([_report(), other])
    assert len(card["campaigns"]) == 2
    assert card["independent_initialisations"] == 2
