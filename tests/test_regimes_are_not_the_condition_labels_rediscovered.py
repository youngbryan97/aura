"""The regimes metastability finds are checked against the conditions the life ran under.

P39.6 asks that clustering not merely rediscover condition labels. A life run
under eight named conditions can be clustered into exactly those eight, and a
regime that is a relabelled condition says nothing about her. These pin the
reading: regimes that are a function of the conditions leave no entropy once the
condition is known and fail, and regimes that cut across conditions pass.
"""

from __future__ import annotations

import pytest

from core.subject.metastability import MetastabilityReport, condition_alignment

pytestmark = pytest.mark.unit

CONDITIONS = ["conversation", "idle", "stress", "memory"] * 30


def _report(labels: list[int]) -> MetastabilityReport:
    return MetastabilityReport(
        regimes=len(set(labels)), dwell_mean=2.0, dwell_max=4.0, transition_entropy=1.0,
        entropy_ceiling=2.0, top_share=0.5, labels=tuple(labels),
        alignment=condition_alignment(labels, CONDITIONS),
    )


def test_regimes_that_are_the_conditions_relabelled_fail() -> None:
    relabelled = [{"conversation": 0, "idle": 1, "stress": 2, "memory": 3}[c] for c in CONDITIONS]
    report = _report(relabelled)
    assert report.alignment["condition_nmi"] == pytest.approx(1.0)
    assert report.alignment["regime_given_condition_bits"] == pytest.approx(0.0)
    assert report.not_the_conditions is False


def test_regimes_that_cut_across_conditions_pass() -> None:
    across = [0 if index < 60 else 1 for index in range(len(CONDITIONS))]
    report = _report(across)
    assert report.alignment["conditions_spanning_regimes"] == 4.0
    assert report.alignment["regime_given_condition_bits"] > 0.0
    assert report.not_the_conditions is True
    assert report.as_dict()["not_the_conditions"] is True


def test_no_conditions_is_unmeasured_not_passed() -> None:
    report = MetastabilityReport(2, 2.0, 4.0, 1.0, 2.0, 0.5, (0, 1))
    assert report.not_the_conditions is None
