"""Rescue failed on a channel the lesion had barely touched.

run_019 cut the cheapest partition and measured three things. Irreducibility
fell by 0.126 from 0.076 and came back 79% of the way. Spread fell by 0.074 and
came back past where it started. Synergy fell by 0.0166 from 0.245 — under
seven per cent — and its recovery read as minus a quarter of that, which failed
the whole criterion.

Two rules follow, and both are fixed before the experiment rather than read off
the result.

A measure enters the rescue verdict only when its own deficit was worth
rescuing. A channel the lesion did not damage has nothing to restore, and
judging a restoration on it is judging noise.

And a trivial improvement is not a rescue. "Rescued is larger than cut" is a
comparison two noisy readings pass half the time. Half the deficit has to come
back.
"""

from __future__ import annotations

import pytest

from core.subject.battery import DEFICIT_SHARE, RECOVERY_TOLERANCE

pytestmark = pytest.mark.unit

MEASURES = ("phi_do", "spread", "synergy")


def _verdict(intact: dict, cut: dict, rescued: dict) -> dict:
    """The rule as the runner applies it."""
    deltas = {k: intact[k] - cut[k] for k in MEASURES}
    recovery = {k: rescued[k] - cut[k] for k in MEASURES}
    real = {
        k: abs(deltas[k]) >= DEFICIT_SHARE * max(abs(intact[k]), 1e-9) for k in MEASURES
    }
    fractions = {
        k: (recovery[k] / deltas[k] if abs(deltas[k]) > 1e-9 else None) for k in MEASURES
    }
    judged = [k for k in MEASURES if real[k]]
    return {
        "judged_on": judged,
        "recovery_fraction": fractions,
        "deficit_worth_rescuing": real,
        "rescued_ok": bool(judged)
        and all((fractions[k] or 0.0) >= RECOVERY_TOLERANCE for k in judged),
    }


RUN_019 = (
    {"phi_do": 0.07578, "spread": 0.40741, "synergy": 0.24508},
    {"phi_do": -0.05027, "spread": 0.33333, "synergy": 0.22851},
    {"phi_do": 0.04982, "spread": 0.44444, "synergy": 0.22393},
)


def test_the_channel_the_lesion_barely_touched_is_not_judged() -> None:
    out = _verdict(*RUN_019)
    assert out["deficit_worth_rescuing"]["synergy"] is False
    assert "synergy" not in out["judged_on"]


def test_the_two_it_did_damage_are_judged_and_recover() -> None:
    out = _verdict(*RUN_019)
    assert out["judged_on"] == ["phi_do", "spread"]
    assert out["recovery_fraction"]["phi_do"] == pytest.approx(0.794, abs=0.01)
    assert out["recovery_fraction"]["spread"] == pytest.approx(1.5, abs=0.01)
    assert out["rescued_ok"] is True


def test_a_trivial_improvement_is_not_a_rescue() -> None:
    """The old rule was `rescued > cut`, which this passes."""
    intact = {"phi_do": 1.0, "spread": 1.0, "synergy": 1.0}
    cut = {"phi_do": 0.0, "spread": 0.0, "synergy": 0.0}
    barely = {"phi_do": 0.01, "spread": 0.01, "synergy": 0.01}
    assert all(barely[k] > cut[k] for k in MEASURES)
    assert _verdict(intact, cut, barely)["rescued_ok"] is False


def test_half_the_deficit_coming_back_is_a_rescue() -> None:
    intact = {"phi_do": 1.0, "spread": 1.0, "synergy": 1.0}
    cut = {"phi_do": 0.0, "spread": 0.0, "synergy": 0.0}
    half = {k: RECOVERY_TOLERANCE for k in MEASURES}
    assert _verdict(intact, cut, half)["rescued_ok"] is True


def test_a_lesion_that_damaged_nothing_cannot_be_rescued() -> None:
    """Nothing to restore is not a successful restoration."""
    same = {"phi_do": 0.5, "spread": 0.5, "synergy": 0.5}
    out = _verdict(same, dict(same), dict(same))
    assert out["judged_on"] == []
    assert out["rescued_ok"] is False


def test_one_judged_measure_failing_fails_the_rescue() -> None:
    intact = {"phi_do": 1.0, "spread": 1.0, "synergy": 1.0}
    cut = {"phi_do": 0.0, "spread": 0.0, "synergy": 0.0}
    lopsided = {"phi_do": 0.9, "spread": 0.9, "synergy": 0.1}
    assert _verdict(intact, cut, lopsided)["rescued_ok"] is False


def test_the_tolerances_are_fixed_before_the_experiment() -> None:
    """And they are in the campaign fingerprint, so moving one starts a new one."""
    from core.subject.provenance import campaign

    assert 0.0 < RECOVERY_TOLERANCE <= 1.0
    assert 0.0 < DEFICIT_SHARE < 1.0
    frozen = campaign(seed=1, rounds=1, trials=1, turns=1)["frozen"]
    assert "thresholds" in frozen
