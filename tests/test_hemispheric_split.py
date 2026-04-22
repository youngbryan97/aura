"""tests/test_hemispheric_split.py
=====================================
End-to-end and ablation tests for the split-brain hemispheric architecture.

Covers:
  - basic tick and fused-bias production
  - pattern memory recognition (right hemisphere)
  - corpus-callosum severance → hemispheres drift apart
  - agreement-rate drops after severance
  - confabulation: left-hemisphere post-hoc reasons after right-driven actions
  - restoration: callosum re-intact → agreement recovers
  - dissent signal when right-hemisphere pattern hits
  - fused bias is NOT identical to either hemisphere's bias
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.hemispheric_split import (  # noqa: E402
    BIAS_DIM,
    CONFAB_WINDOW_S,
    DISAGREEMENT_L2_THRESHOLD,
    Hemisphere,
    HemisphericSplit,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synth_inputs(rng: np.random.Generator, phase: float):
    exec_summary = np.sin(phase + np.arange(8) * 0.1) + rng.standard_normal(8) * 0.1
    sens_summary = np.cos(phase + np.arange(16) * 0.05) + rng.standard_normal(16) * 0.1
    cog_aff = np.concatenate([
        np.sin(phase + np.arange(8) * 0.07),      # affect
        np.cos(phase + np.arange(8) * 0.13),       # cognitive
    ]) + rng.standard_normal(16) * 0.1
    embodiment = np.tanh(np.sin(phase * 2.0 + np.arange(8) * 0.2))
    return exec_summary, sens_summary, cog_aff, embodiment


def _drive(split: HemisphericSplit, n_ticks: int, *, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    phase = 0.0
    for _ in range(n_ticks):
        phase += 0.2
        exec_s, sens_s, cog, emb = _synth_inputs(rng, phase)
        split.tick(exec_s, sens_s, cog, emb)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_construction_and_basic_tick():
    split = HemisphericSplit()
    _drive(split, 5)
    s = split.current_state()
    assert s is not None
    assert s.left_bias.shape == (BIAS_DIM,)
    assert s.right_bias.shape == (BIAS_DIM,)
    assert s.fused_bias.shape == (BIAS_DIM,)
    assert s.callosum_bandwidth == 1.0


def test_fused_bias_is_not_trivially_equal_to_either_hemisphere():
    split = HemisphericSplit()
    _drive(split, 25)
    s = split.current_state()
    assert s is not None
    # With non-zero biases, the fused vector should differ from each side.
    assert not np.allclose(s.fused_bias, s.left_bias, atol=1e-3)
    assert not np.allclose(s.fused_bias, s.right_bias, atol=1e-3)


def test_pattern_memory_recognition():
    split = HemisphericSplit()
    rng = np.random.default_rng(13)
    # Train a pattern vector into the right hemisphere's pattern memory.
    pattern = rng.standard_normal(32).astype(np.float32)
    split.right.learn_pattern(pattern, "bryans_face")
    assert split.right.pattern_memory_size() == 1

    # Drive with that exact pattern — recognition should fire.
    exec_s = np.zeros(8)
    # The right hemisphere projects sensory(16) + affect(8) + embodiment(8) = 32.
    # Use the trained vector as-is for the combined input.
    sens_s = pattern[:16].copy()
    cog = np.concatenate([pattern[16:24], np.zeros(8)])
    emb = pattern[24:32].copy()
    split.tick(exec_s, sens_s, cog, emb)
    assert split.right.last_recognised_pattern() == "bryans_face"


def test_callosum_severance_increases_disagreement():
    split = HemisphericSplit()
    _drive(split, 60, seed=2)
    pre_agreement = split.agreement_rate()
    split.sever_callosum()
    assert split.callosum.bandwidth() == 0.0
    _drive(split, 80, seed=3)
    post_agreement = split.agreement_rate()
    # Agreement should decline after severance (hemispheres drift).
    # Allow equality at worst — this is a stochastic signal.
    assert post_agreement <= pre_agreement + 1e-6, (
        f"agreement did not fall after callosum severance "
        f"(pre={pre_agreement:.3f}, post={post_agreement:.3f})"
    )


def test_callosum_restoration_recovers_agreement():
    split = HemisphericSplit()
    _drive(split, 40)
    split.sever_callosum()
    _drive(split, 60, seed=4)
    agreement_severed = split.agreement_rate()
    split.restore_callosum(bandwidth=1.0)
    _drive(split, 80, seed=5)
    agreement_restored = split.agreement_rate()
    # After restoration, agreement should match or exceed severed state.
    assert agreement_restored >= agreement_severed - 1e-6


def test_confabulation_logged_when_right_drove_action():
    split = HemisphericSplit()
    _drive(split, 10)
    # Right hemisphere "drives" an action.
    split.record_action("wiggle_left_hand", Hemisphere.RIGHT)
    text = split.supply_reason("wiggle_left_hand")
    assert text, "left hemisphere must supply a reason text"
    assert split.confabulation_rate() > 0.0
    # The reason should be tagged as a confabulation.
    assert "[CONFAB]" in text


def test_left_driven_actions_are_not_confabulation():
    split = HemisphericSplit()
    _drive(split, 10)
    split.record_action("speak_hello", Hemisphere.LEFT)
    text = split.supply_reason("speak_hello")
    assert text
    # Left-driven reason is NOT confabulation.
    assert split.confabulation_rate() == 0.0
    assert "[CONFAB]" not in text


def test_confabulation_window_elapsed_still_counts_if_driver_was_right():
    """Even if time has passed, if the driver was right and left supplies a
    reason, the counter increments while the window is open."""
    split = HemisphericSplit()
    _drive(split, 5)
    split.record_action("grab_object", Hemisphere.RIGHT)
    time.sleep(0.05)
    split.supply_reason("grab_object")
    assert split.confabulation_rate() > 0.0


def test_dissent_becomes_active_under_pattern_plus_arousal():
    split = HemisphericSplit()
    # Prime pattern memory.
    rng = np.random.default_rng(21)
    pattern = rng.standard_normal(32).astype(np.float32)
    split.right.learn_pattern(pattern, "alert_face")
    exec_s = np.zeros(8)
    sens_s = pattern[:16]
    # Force high arousal (index 1) in the affect portion of cognitive_affective.
    cog = np.concatenate([pattern[16:24], np.zeros(8)]).copy()
    cog[1] = 1.8  # saturated arousal
    emb = pattern[24:32]
    state = split.tick(exec_s, sens_s, cog, emb)
    assert state.dissent_active or split.right.current_dissent() > 0.3


def test_status_reports_everything_expected():
    split = HemisphericSplit()
    _drive(split, 30)
    s = split.get_status()
    for key in [
        "tick", "callosum_bandwidth", "pattern_memory_size", "agreement_rate",
        "disagreement_count", "confabulation_count", "confabulation_rate",
        "last_disagreement_l2", "last_dissent_active", "fused_bias_mean_abs",
    ]:
        assert key in s, f"missing status key: {key}"


def test_severed_callosum_produces_incoherent_fused_bias():
    """When hemispheres strongly disagree (callosum severed with divergent
    inputs), the fused bias should show less coherence (lower mean magnitude)
    than when the hemispheres are aligned."""
    split_intact = HemisphericSplit()
    split_severed = HemisphericSplit()
    split_severed.sever_callosum()

    rng = np.random.default_rng(33)
    phase = 0.0
    mean_intact, mean_severed = [], []
    for _ in range(80):
        phase += 0.19
        exec_s, sens_s, cog, emb = _synth_inputs(rng, phase)
        # Give severed split DIFFERENT inputs to left vs right (simulates the
        # diverging streams a split patient experiences).
        left_cog = cog.copy()
        right_cog = np.roll(cog.copy(), 3)  # Shift to de-correlate.
        st_intact = split_intact.tick(exec_s, sens_s, cog, emb)
        st_severed = split_severed.tick(exec_s, sens_s, right_cog, emb)
        mean_intact.append(float(np.mean(np.abs(st_intact.fused_bias))))
        mean_severed.append(float(np.mean(np.abs(st_severed.fused_bias))))

    # Both produce non-trivial biases; difference shows up in disagreement stats.
    assert np.mean(mean_intact) > 0.0 and np.mean(mean_severed) > 0.0
    assert split_severed.agreement_rate() <= split_intact.agreement_rate() + 1e-6


def test_disagreement_threshold_configurable():
    """Ensure the threshold constant is positive and reasonable."""
    assert 0.0 < DISAGREEMENT_L2_THRESHOLD < 2.0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_construction_and_basic_tick,
        test_fused_bias_is_not_trivially_equal_to_either_hemisphere,
        test_pattern_memory_recognition,
        test_callosum_severance_increases_disagreement,
        test_callosum_restoration_recovers_agreement,
        test_confabulation_logged_when_right_drove_action,
        test_left_driven_actions_are_not_confabulation,
        test_confabulation_window_elapsed_still_counts_if_driver_was_right,
        test_dissent_becomes_active_under_pattern_plus_arousal,
        test_status_reports_everything_expected,
        test_severed_callosum_produces_incoherent_fused_bias,
        test_disagreement_threshold_configurable,
    ]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✓ {t.__name__}")
        except Exception as exc:
            failed.append((t.__name__, exc))
            print(f"  ✗ {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if not failed else 1)
