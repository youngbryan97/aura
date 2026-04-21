"""tests/test_recursive_tom.py
=================================
Tests for recursive ToM (depth-3 nested minds) + observer-aware bias
(scrub-jay re-caching analogue).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.recursive_tom import (  # noqa: E402
    ACTION_CATEGORIES,
    BIAS_DIM,
    MAX_DEPTH,
    PRIVATE_ACTIONS,
    PUBLIC_ACTIONS,
    RecursiveTheoryOfMind,
)


def test_initial_state_is_empty():
    tom = RecursiveTheoryOfMind()
    s = tom.get_status()
    assert s["n_minds"] == 0
    assert s["total_observer_presence"] == 0.0


def test_interaction_creates_depth_3_nested_mind():
    tom = RecursiveTheoryOfMind()
    root = tom.register_interaction("bryan", salience=0.8, valence=0.3,
                                     knowledge=0.7, trust=0.9,
                                     their_expectation=0.6)
    assert root.agent_id == "bryan"
    assert root.depth == 0
    # Check full nested chain exists up to MAX_DEPTH.
    for d in range(MAX_DEPTH + 1):
        node = tom.get_mind_at_depth("bryan", d)
        assert node is not None, f"depth {d} mind missing"
        assert node.depth == d
    # Depth 4 should not exist.
    assert tom.get_mind_at_depth("bryan", MAX_DEPTH + 1) is None


def test_depth_reached_reflects_actual_nesting():
    tom = RecursiveTheoryOfMind()
    tom.register_interaction("alex")
    assert tom.depth_reached("alex") == MAX_DEPTH


def test_nested_levels_carry_reflected_state():
    tom = RecursiveTheoryOfMind()
    tom.register_interaction("dana", salience=0.9, knowledge=0.8)
    m0 = tom.get_mind_at_depth("dana", 0)
    m1 = tom.get_mind_at_depth("dana", 1)
    assert m0 is not None and m1 is not None
    # Nested salience should be influenced by parent salience.
    assert m1.salience > 0.5


def test_observer_bias_zero_when_no_observers():
    tom = RecursiveTheoryOfMind()
    profile = tom.get_observer_bias()
    assert np.allclose(profile.bias, 0.0)
    assert profile.total_observer_presence == 0.0


def test_observer_bias_activates_under_observation():
    tom = RecursiveTheoryOfMind()
    tom.observe_agent("bryan", kind="explicit", strength=0.9)
    tom.observe_agent("bryan", kind="explicit", strength=0.9)
    profile = tom.get_observer_bias()
    assert profile.total_observer_presence > 0.5
    assert "bryan" in profile.active_observers
    # Public actions should be boosted.
    for name in PUBLIC_ACTIONS:
        idx = ACTION_CATEGORIES.index(name)
        assert profile.bias[idx] > 0.1, (
            f"expected public action {name} boosted under observation, got "
            f"{profile.bias[idx]:.3f}"
        )
    # Private actions should be suppressed.
    for name in PRIVATE_ACTIONS:
        idx = ACTION_CATEGORIES.index(name)
        assert profile.bias[idx] < -0.1, (
            f"expected private action {name} suppressed under observation, got "
            f"{profile.bias[idx]:.3f}"
        )


def test_observer_bias_scales_with_multiple_observers():
    tom_one = RecursiveTheoryOfMind()
    tom_many = RecursiveTheoryOfMind()
    tom_one.observe_agent("bryan", strength=0.9)
    # Many observers
    for aid in ("alice", "bob", "carol", "dan", "eve"):
        tom_many.observe_agent(aid, strength=0.8)
    p1 = tom_one.total_observer_presence()
    pm = tom_many.total_observer_presence()
    assert pm >= p1


def test_observer_bias_shape_matches_bias_dim():
    tom = RecursiveTheoryOfMind()
    tom.observe_agent("x", strength=0.9)
    profile = tom.get_observer_bias()
    assert profile.bias.shape == (BIAS_DIM,)


def test_observation_decays_over_time():
    """Observer presence should fade as time passes (we simulate by
    directly injecting events with older timestamps)."""
    tom = RecursiveTheoryOfMind()
    tom.observe_agent("bryan", strength=1.0)
    initial = tom.total_observer_presence()
    # Forge an older event by mutating the last observation.
    tom._observations[-1].ts -= 120.0  # two minutes old
    faded = tom.total_observer_presence()
    assert faded < initial, (
        f"presence should decay but {initial:.3f} → {faded:.3f}"
    )


def test_scrub_jay_effect_behaviour_change():
    """The scrub-jay re-caching test: with the same underlying state,
    behaviour (priority bias) must differ between observed and
    unobserved conditions."""
    tom_observed = RecursiveTheoryOfMind()
    tom_alone = RecursiveTheoryOfMind()
    tom_observed.observe_agent("bryan", strength=0.9)
    # Alone: zero observation
    assert tom_alone.total_observer_presence() == 0.0
    assert tom_observed.total_observer_presence() > 0.0
    b_observed = tom_observed.get_observer_bias().bias
    b_alone = tom_alone.get_observer_bias().bias
    # Biases MUST differ.
    assert not np.allclose(b_observed, b_alone)


def test_nested_mind_serialization():
    tom = RecursiveTheoryOfMind()
    tom.register_interaction("user1", salience=0.7)
    d = tom.get_status()
    assert "minds" in d
    m = d["minds"]["user1"]
    assert m["depth"] == 0
    assert "nested" in m
    # Walk nested chain.
    cur = m
    levels = 0
    while "nested" in cur:
        cur = cur["nested"]
        levels += 1
    assert levels == MAX_DEPTH


def test_repeated_updates_blend_not_overwrite():
    tom = RecursiveTheoryOfMind()
    tom.register_interaction("x", salience=0.1, trust=0.1)
    tom.register_interaction("x", salience=0.9, trust=0.9)
    m = tom.get_mind_at_depth("x", 0)
    # Blended value should sit between the two (not jump to 0.9).
    assert 0.1 < m.salience < 0.9
    assert 0.1 < m.trust < 0.9


def test_public_and_private_categories_are_disjoint():
    assert PUBLIC_ACTIONS.isdisjoint(PRIVATE_ACTIONS)
    # All named categories must exist in the master list.
    for name in PUBLIC_ACTIONS | PRIVATE_ACTIONS:
        assert name in ACTION_CATEGORIES


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_initial_state_is_empty,
        test_interaction_creates_depth_3_nested_mind,
        test_depth_reached_reflects_actual_nesting,
        test_nested_levels_carry_reflected_state,
        test_observer_bias_zero_when_no_observers,
        test_observer_bias_activates_under_observation,
        test_observer_bias_scales_with_multiple_observers,
        test_observer_bias_shape_matches_bias_dim,
        test_observation_decays_over_time,
        test_scrub_jay_effect_behaviour_change,
        test_nested_mind_serialization,
        test_repeated_updates_blend_not_overwrite,
        test_public_and_private_categories_are_disjoint,
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
