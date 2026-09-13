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
    """Serving authority is refused, and the refusal is still re-derivable.

    The reasons are compared only against a verdict written by the SAME
    adjudication. A predicate was corrected on 2026-09-13 -- specificity read
    divergence, where a norm-matched random vector that is inert on the target
    reproduces 98.6% of the movement -- and every verdict derived under the old
    one stopped re-deriving its reason list. Without the fingerprint beside it,
    that is indistinguishable from an edited file, which is the one thing this
    test exists to catch.
    """
    result, verdict = committed
    recomputed = read(result)
    assert recomputed["causal_effect_positive"] is False
    assert recomputed["causal_effect_positive"] == verdict["causal_effect_positive"]
    assert recomputed["unmet_requirements"], "a refusal with no reason is not a refusal"
    if verdict.get("adjudication_sha256") == recomputed["adjudication_sha256"]:
        assert set(recomputed["unmet_requirements"]) == set(verdict["unmet_requirements"])
    else:
        # A correction may only ever narrow a refusal already on the record.
        # A NEW reason appearing under a new predicate is a regression in the
        # thing measured, and has to be read, not absorbed.
        assert set(recomputed["unmet_requirements"]) <= set(verdict["unmet_requirements"])


def test_a_verdict_says_which_adjudication_refused_it(committed):
    """Without it, a corrected predicate and a tampered file look identical."""
    result, _verdict = committed
    recomputed = read(result)
    assert len(recomputed["adjudication_sha256"]) == 64


def test_adds_to_text_is_not_part_of_the_pass(committed):
    """It answers a different question and must never be a back door."""
    result, _verdict = committed
    recomputed = read(result)
    assert "adds_to_text" in recomputed
    # This campaign predates the combined condition, so there is no answer yet.
    assert recomputed["adds_to_text"] is None
    assert recomputed["causal_effect_positive"] is False
