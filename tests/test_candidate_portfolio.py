"""The shared selector retains declared alternatives and their evidence."""

import pytest

from core.evidence.candidate_portfolio import select_candidate_portfolio
from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    build_necessary_condition_selector,
)
from core.evidence.packet import observe


def test_all_methods_are_considered_with_stable_ties():
    rows = {"base": {"valid": 0.}, "second": {"valid": 1.}, "third": {"valid": 1.}}
    selector = build_necessary_condition_selector((NecessaryEvidenceCondition("valid", 1., "contract"),))
    result = select_candidate_portfolio(selector, incumbent="base", measurements=rows,
        provenance={k: observe(1., origin=k, ref=k, subject="same") for k in rows})
    assert result.selected == "second" and result.candidate_order == tuple(rows)
    assert [x.receipt["incumbent"] for x in result.comparisons] == ["base", "second"]
    assert [x.receipt["challenger"] for x in result.comparisons] == ["second", "third"]


def test_cross_observation_selection_rejected():
    with pytest.raises(ValueError, match="different observations"):
        select_candidate_portfolio(None, incumbent="a", measurements={"a": {"x": 1.}, "b": {"x": 1.}},
            provenance={k: observe(1., origin=k, ref=k, subject=k) for k in ("a", "b")})


def test_even_sole_candidate_requires_finite_evidence():
    with pytest.raises(ValueError, match="finite"):
        select_candidate_portfolio(None, incumbent="a", measurements={"a": {"x": float("nan")}},
            provenance={"a": observe(1., origin="a", ref="a", subject="same")})
