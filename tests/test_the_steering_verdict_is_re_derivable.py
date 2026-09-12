"""The verdict beside the samples must be recomputable from them.

`campaign_verdict.json` was written by hand. Every number in it was real, and
none of it could be re-derived — which is the one property a record of a
negative result has to have, because a negative is what somebody will want to
re-check hardest.

`tools/read_the_steering_campaign.py` recomputes it through the same
independent replay the campaign uses. This holds the two together: if the
committed verdict and the committed samples ever disagree, one of them has
been edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.read_the_steering_campaign import read

RECOVERY = Path(__file__).resolve().parents[1] / "artifacts/migration/27b/recovery"
RESULT = RECOVERY / "campaign_result.json"
VERDICT = RECOVERY / "campaign_verdict.json"


@pytest.fixture(scope="module")
def committed() -> tuple[dict, dict]:
    if not RESULT.exists() or not VERDICT.exists():
        pytest.skip("no committed 27B campaign to check")
    return (
        json.loads(RESULT.read_text(encoding="utf-8")),
        json.loads(VERDICT.read_text(encoding="utf-8")),
    )


def test_the_condition_means_come_back_the_same(committed):
    result, verdict = committed
    recomputed = read(result)
    for name, mean in verdict["condition_means"].items():
        assert recomputed["condition_means"][name] == pytest.approx(mean, abs=5e-4), name


def test_the_win_counts_come_back_the_same(committed):
    result, verdict = committed
    recomputed = read(result)
    assert recomputed["treatment_successes"] == verdict["treatment_successes"]
    assert recomputed["matched_control_successes"] == verdict["matched_control_successes"]
    assert recomputed["lesion_successes"] == verdict["lesion_successes"]
    assert recomputed["no_regression"] == verdict["no_regression"]


def test_the_refusal_comes_back_the_same(committed):
    """Serving authority is refused, and for the two stated reasons."""
    result, verdict = committed
    recomputed = read(result)
    assert recomputed["causal_effect_positive"] is False
    assert recomputed["causal_effect_positive"] == verdict["causal_effect_positive"]
    assert set(recomputed["unmet_requirements"]) == set(verdict["unmet_requirements"])


def test_adds_to_text_is_not_part_of_the_pass(committed):
    """It answers a different question and must never be a back door."""
    result, _verdict = committed
    recomputed = read(result)
    assert "adds_to_text" in recomputed
    # This campaign predates the combined condition, so there is no answer yet.
    assert recomputed["adds_to_text"] is None
    assert recomputed["causal_effect_positive"] is False
