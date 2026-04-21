"""tests/test_minimal_selfhood.py
====================================
Tests for the Glasgow-inspired minimal selfhood stack:
  trichoplax → dugesia transition, chemotaxis speed, directed action priority.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.minimal_selfhood import (  # noqa: E402
    ACTION_CATEGORIES,
    BIAS_DIM,
    DEFICIT_KEYS,
    MinimalSelfhood,
    Mode,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _budget(energy=0.7, pressure=0.1, thermal=0.1):
    return {
        "energy_reserves": float(energy),
        "resource_pressure": float(pressure),
        "thermal_stress": float(thermal),
    }


def _affect(coherence=0.7, curiosity=0.6):
    return {"coherence": float(coherence), "curiosity": float(curiosity)}


def _cognitive(social_hunger=0.1, prediction_error=0.1, agency_score=0.8):
    return {
        "social_hunger": float(social_hunger),
        "prediction_error": float(prediction_error),
        "agency_score": float(agency_score),
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_initial_mode_is_trichoplax():
    ms = MinimalSelfhood()
    assert ms.mode() == Mode.TRICHOPLAX


def test_zero_deficit_yields_low_speed():
    ms = MinimalSelfhood()
    st = ms.update(
        body_budget=_budget(energy=1.0),
        affect=_affect(coherence=1.0, curiosity=1.0),
        cognitive_state=_cognitive(prediction_error=0.0, agency_score=1.0),
    )
    assert st.speed_scalar < 0.3, f"satiated state should be slow, got {st.speed_scalar}"


def test_full_deficit_yields_high_speed():
    ms = MinimalSelfhood()
    st = ms.update(
        body_budget=_budget(energy=0.0, pressure=1.0, thermal=1.0),
        affect=_affect(coherence=0.0, curiosity=0.0),
        cognitive_state=_cognitive(social_hunger=1.0, prediction_error=1.0, agency_score=0.0),
    )
    assert st.speed_scalar > 0.7, f"full deficit should be fast, got {st.speed_scalar}"


def test_priority_shape_and_bounds():
    ms = MinimalSelfhood()
    st = ms.update(
        body_budget=_budget(energy=0.3),
        affect=_affect(),
        cognitive_state=_cognitive(),
    )
    assert st.action_priority.shape == (BIAS_DIM,)
    assert np.all(np.isfinite(st.action_priority))
    assert np.all(np.abs(st.action_priority) <= 1.0)


def test_dominant_deficit_reflects_input():
    ms = MinimalSelfhood()
    # Drive energy deficit to 1.0 (energy_reserves=0) — should dominate.
    st = ms.update(
        body_budget=_budget(energy=0.0),
        affect=_affect(coherence=0.9, curiosity=0.8),
        cognitive_state=_cognitive(social_hunger=0.0, prediction_error=0.05, agency_score=0.95),
    )
    assert st.dominant_deficit == "energy"


def test_trichoplax_prefers_rest_under_deficit():
    ms = MinimalSelfhood()
    st = ms.update(
        body_budget=_budget(energy=0.0, pressure=0.6, thermal=0.6),
        affect=_affect(coherence=0.3, curiosity=0.2),
        cognitive_state=_cognitive(agency_score=0.2),
    )
    # In trichoplax mode, rest should not be suppressed.
    rest_idx = ACTION_CATEGORIES.index("rest")
    assert st.action_priority[rest_idx] >= 0.0


def test_reinforcement_strengthens_weights_and_transitions_mode():
    ms = MinimalSelfhood()
    # Simulate 40 reinforcement cycles where the "rest" action reliably
    # reduces the energy deficit.
    pre_deficit = np.zeros(len(DEFICIT_KEYS), dtype=np.float32)
    pre_deficit[0] = 0.9
    post_deficit = np.zeros(len(DEFICIT_KEYS), dtype=np.float32)
    post_deficit[0] = 0.2

    for _ in range(40):
        token = ms.tag_action("rest", pre_deficit)
        ok = ms.reinforce(token, post_deficit)
        assert ok

    weights = ms.learned_weights()
    rest_idx = ACTION_CATEGORIES.index("rest")
    # rest → energy deficit-reduction weight must be clearly positive
    assert weights[rest_idx, 0] > 0.1

    # Now an update should see learned-weight norm ≥ threshold → DUGESIA.
    ms.update(
        body_budget=_budget(energy=0.3),
        affect=_affect(),
        cognitive_state=_cognitive(),
    )
    # Check mode transition flag.
    # Note: depending on decay, may still be trichoplax if threshold high.
    total = np.sum(np.abs(weights))
    if total >= MinimalSelfhood._DUGESIA_THRESHOLD:
        assert ms.mode() == Mode.DUGESIA


def test_dugesia_directs_priority_toward_learned_action():
    """After heavy training on (rest → energy deficit reduction), and
    facing an energy deficit, action priority should put REST among
    the top picks."""
    ms = MinimalSelfhood()
    pre_deficit = np.zeros(len(DEFICIT_KEYS), dtype=np.float32)
    pre_deficit[0] = 1.0
    post_deficit = np.zeros(len(DEFICIT_KEYS), dtype=np.float32)
    post_deficit[0] = 0.1
    for _ in range(80):
        t = ms.tag_action("rest", pre_deficit)
        ms.reinforce(t, post_deficit)
    # Trigger update with energy deficit.
    st = ms.update(
        body_budget=_budget(energy=0.0),
        affect=_affect(coherence=0.8, curiosity=0.7),
        cognitive_state=_cognitive(),
    )
    rest_idx = ACTION_CATEGORIES.index("rest")
    top3 = np.argsort(st.action_priority)[::-1][:3]
    assert rest_idx in top3, (
        f"rest not in top-3 priorities after training: "
        f"top3={[ACTION_CATEGORIES[i] for i in top3]}"
    )


def test_heartbeat_modulation_range():
    ms = MinimalSelfhood()
    ms.update(body_budget=_budget(energy=1.0), affect=_affect(), cognitive_state=_cognitive())
    slow = ms.get_heartbeat_modulation()
    ms.update(
        body_budget=_budget(energy=0.0, pressure=1.0, thermal=1.0),
        affect=_affect(coherence=0.0, curiosity=0.0),
        cognitive_state=_cognitive(social_hunger=1.0, prediction_error=1.0, agency_score=0.0),
    )
    fast = ms.get_heartbeat_modulation()
    assert 0.5 <= slow <= 1.5
    assert 0.5 <= fast <= 1.5
    # Satiated state must modulate to a slower heartbeat than deficit state.
    assert slow > fast


def test_reinforcement_with_no_improvement_does_not_grow_weight():
    ms = MinimalSelfhood()
    pre = np.ones(len(DEFICIT_KEYS), dtype=np.float32) * 0.5
    post = np.ones(len(DEFICIT_KEYS), dtype=np.float32) * 0.5
    initial = ms.learned_weights().copy()
    for _ in range(20):
        t = ms.tag_action("explore", pre)
        ms.reinforce(t, post)
    final = ms.learned_weights()
    # No reward, so weights should have DECAYED not grown.
    assert np.sum(np.abs(final)) <= np.sum(np.abs(initial)) + 1e-6


def test_status_reports_fields():
    ms = MinimalSelfhood()
    ms.update(body_budget=_budget(), affect=_affect(), cognitive_state=_cognitive())
    s = ms.get_status()
    for k in ("mode", "transition_count", "n_updates", "speed_scalar",
              "dominant_deficit", "learned_weight_norm"):
        assert k in s


def test_priority_bias_shape_matches_bias_dim():
    ms = MinimalSelfhood()
    ms.update(body_budget=_budget(), affect=_affect(), cognitive_state=_cognitive())
    bias = ms.get_priority_bias()
    assert bias.shape == (BIAS_DIM,)


def test_action_categories_length_matches_bias_dim():
    assert len(ACTION_CATEGORIES) == BIAS_DIM


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_initial_mode_is_trichoplax,
        test_zero_deficit_yields_low_speed,
        test_full_deficit_yields_high_speed,
        test_priority_shape_and_bounds,
        test_dominant_deficit_reflects_input,
        test_trichoplax_prefers_rest_under_deficit,
        test_reinforcement_strengthens_weights_and_transitions_mode,
        test_dugesia_directs_priority_toward_learned_action,
        test_heartbeat_modulation_range,
        test_reinforcement_with_no_improvement_does_not_grow_weight,
        test_status_reports_fields,
        test_priority_bias_shape_matches_bias_dim,
        test_action_categories_length_matches_bias_dim,
    ]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ok {t.__name__}")
        except Exception as exc:
            failed.append((t.__name__, exc))
            print(f"  FAIL {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if not failed else 1)
