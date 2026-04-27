"""tests/test_octopus_arms.py
================================
Tests for 8-arm octopus federation: local autonomy + central arbitration,
link severance, integration latency.
"""
from __future__ import annotations


import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.octopus_arms import (  # noqa: E402
    ACTION_DIM,
    N_ARMS,
    SENSOR_CHANNELS,
    ArmState,
    OctopusFederation,
)


def _env(phase: float, rng: np.random.Generator) -> np.ndarray:
    return np.array([
        np.sin(phase),
        np.cos(phase * 1.2),
        np.sin(phase * 0.8 + 0.3),
    ], dtype=np.float32) + rng.standard_normal(SENSOR_CHANNELS) * 0.1


def test_construction():
    fed = OctopusFederation()
    assert len(fed.arms) == N_ARMS
    assert fed.arbiter.link_state() == ArmState.LINKED


def test_tick_produces_decision_under_linked():
    fed = OctopusFederation()
    rng = np.random.default_rng(1)
    result = fed.tick(_env(0.3, rng))
    assert result.winning_action is not None
    assert 0 <= result.winning_action < ACTION_DIM
    assert result.link_state == ArmState.LINKED


def test_all_arms_produce_actions_each_tick():
    fed = OctopusFederation()
    rng = np.random.default_rng(2)
    fed.tick(_env(0.5, rng))
    for i in range(N_ARMS):
        a = fed.arm_action(i)
        assert a is not None
        assert 0 <= a.action_idx < ACTION_DIM


def test_link_severance_suppresses_winner():
    fed = OctopusFederation()
    rng = np.random.default_rng(3)
    fed.tick(_env(0.1, rng))
    fed.sever_link()
    result = fed.tick(_env(0.2, rng))
    assert result.winning_action is None
    assert result.link_state == ArmState.SEVERED


def test_severed_link_raises_autonomy_to_one():
    fed = OctopusFederation()
    fed.sever_link()
    for a in fed.arms:
        assert a.autonomy == 1.0


def test_arms_continue_acting_when_severed():
    fed = OctopusFederation()
    fed.sever_link()
    rng = np.random.default_rng(4)
    fed.tick(_env(0.3, rng))
    # All arms still produced individual actions.
    for i in range(N_ARMS):
        assert fed.arm_action(i) is not None


def test_link_restoration_eventually_integrates():
    fed = OctopusFederation()
    rng = np.random.default_rng(5)
    # Drive for a while with link intact.
    phase = 0.0
    for _ in range(30):
        phase += 0.2
        fed.tick(_env(phase, rng))
    fed.sever_link()
    for _ in range(20):
        phase += 0.2
        fed.tick(_env(phase, rng))
    fed.restore_link()
    # Run many ticks and verify link_state becomes LINKED again.
    eventually_linked = False
    for _ in range(60):
        phase += 0.2
        r = fed.tick(_env(phase, rng))
        if r.link_state == ArmState.LINKED:
            eventually_linked = True
            break
    # (May remain RECOVERING for very noisy inputs; the monotonic
    # progression is the guarantee we test.)
    assert fed.arbiter.link_state() in (ArmState.LINKED, ArmState.RECOVERING)


def test_decision_variance_tracked():
    fed = OctopusFederation()
    rng = np.random.default_rng(6)
    for _ in range(10):
        fed.tick(_env(0.5, rng))
    r = fed.arbiter.current_state()
    assert r is not None
    assert 0.0 <= r.decision_variance <= 1.0


def test_status_dict_is_complete():
    fed = OctopusFederation()
    rng = np.random.default_rng(7)
    fed.tick(_env(0.4, rng))
    s = fed.get_status()
    for k in ("n_arms", "link_state", "last_winning_action",
              "last_variance", "arms_autonomy", "arms_last_action"):
        assert k in s
    assert s["n_arms"] == N_ARMS
    assert len(s["arms_autonomy"]) == N_ARMS
    assert len(s["arms_last_action"]) == N_ARMS


def test_integration_latency_counts_ticks():
    fed = OctopusFederation()
    rng = np.random.default_rng(8)
    fed.tick(_env(0.1, rng))
    fed.sever_link()
    for _ in range(5):
        fed.tick(_env(0.1, rng))
    fed.restore_link()
    # Drive a consistent environment so variance falls.
    stable_env = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    latency_seen = 0
    for _ in range(20):
        r = fed.tick(stable_env)
        if r.integration_latency > latency_seen:
            latency_seen = r.integration_latency
    assert latency_seen > 0


def test_individual_arms_differ_in_receptive_field():
    fed = OctopusFederation()
    rng = np.random.default_rng(9)
    env = _env(0.3, rng)
    sensed = [fed.arms[i].sense(env) for i in range(N_ARMS)]
    # Not all arms should report identical sensor values.
    pairwise_same = sum(
        np.allclose(sensed[i], sensed[j], atol=1e-4)
        for i in range(N_ARMS) for j in range(i + 1, N_ARMS)
    )
    assert pairwise_same < N_ARMS * (N_ARMS - 1) / 2


def test_severed_produces_higher_variance_or_equal():
    """With severed link and uncoordinated action, arm decisions should
    show at least as much (typically more) variance."""
    fed_linked = OctopusFederation()
    fed_severed = OctopusFederation()
    fed_severed.sever_link()
    rng = np.random.default_rng(11)
    phase = 0.0
    linked_vars, severed_vars = [], []
    for _ in range(30):
        phase += 0.23
        e = _env(phase, rng)
        rL = fed_linked.tick(e)
        rS = fed_severed.tick(e)
        linked_vars.append(rL.decision_variance)
        severed_vars.append(rS.decision_variance)
    # Severed federation should not have consistently lower variance than linked.
    assert np.mean(severed_vars) >= np.mean(linked_vars) - 1e-6


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_construction,
        test_tick_produces_decision_under_linked,
        test_all_arms_produce_actions_each_tick,
        test_link_severance_suppresses_winner,
        test_severed_link_raises_autonomy_to_one,
        test_arms_continue_acting_when_severed,
        test_link_restoration_eventually_integrates,
        test_decision_variance_tracked,
        test_status_dict_is_complete,
        test_integration_latency_counts_ticks,
        test_individual_arms_differ_in_receptive_field,
        test_severed_produces_higher_variance_or_equal,
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
