"""tests/test_substrate_stability_long_horizon.py
==================================================
Long-horizon substrate-stability defenses:

A. Entropy decay threat — over 72h+ of continuous operation, computational
   entropy creeps past what passive idle dissipation can clear, acting as a
   slow cognitive poison. The organism must autonomously select an allostatic
   consolidation ("rest") to clear the buffer when it crosses a critical line.

B. Attractor saturation — the recurrent tanh UnifiedField can settle into a
   degenerate attractor (all values near ±1): the eigenvalue spectrum of recent
   states collapses, Phi → 0, the Will locks down. A SATURATED attractor reads
   as HIGH coherence, so the legacy low-coherence recovery is blind to it.
   Anti-degeneracy chaos must scale dynamically with the field's eigenvalue
   entropy (and the ControlledChaos engine must scale its noise the same way).
"""
from __future__ import annotations

import numpy as np

from core.consciousness.alife_dynamics import ALifeDynamics, EntropyTracker
from core.consciousness.controlled_chaos import ChaosConfig, ChaosEngine
from core.consciousness.unified_field import FieldConfig, UnifiedField, spectral_entropy_of


# ── A. Entropy allostatic consolidation ─────────────────────────────────────

def test_entropy_creep_then_allostatic_reset():
    et = EntropyTracker(max_entropy=100.0, allostatic_threshold=0.80, allostatic_target=0.30)
    # Long-horizon creep: generation outpaces idle dissipation.
    for _ in range(300):
        et.add_entropy(1.0, "work")
        et.dissipate_entropy(0.1, "idle_tick")
    assert et.needs_allostatic_reset() is True
    cleared = et.allostatic_reset()
    assert cleared > 0.0
    assert et.get_entropy() == 30.0  # driven to allostatic_target * max
    assert et.needs_allostatic_reset() is False


def test_allostatic_reset_is_noop_when_already_rested():
    et = EntropyTracker(max_entropy=100.0, allostatic_target=0.30)
    et.add_entropy(10.0, "work")  # below target (30)
    assert et.allostatic_reset() == 0.0
    assert et.get_entropy() == 10.0


def test_entropy_stats_expose_allostatic_signals():
    et = EntropyTracker(max_entropy=100.0)
    stats = et.get_stats()
    assert "needs_allostatic_reset" in stats
    assert "allostatic_resets" in stats


def test_alife_dynamics_autonomously_consolidates_under_load():
    """The substrate self-selects rest: push entropy past the threshold via the
    external API, then a tick clears it (cooldown starts fresh)."""
    alife = ALifeDynamics(max_entropy=100.0)
    alife._consolidation_cooldown_s = 0.0  # allow immediate autonomous rest
    alife.add_entropy(95.0, "heavy_long_run")
    assert alife.needs_allostatic_reset() is True

    cols = np.zeros(64, dtype=np.float32)
    proj = np.ones(64, dtype=np.float32)
    weights = np.zeros((64, 64), dtype=np.float32)

    import asyncio

    state = asyncio.run(alife.tick(cols, weights, proj))
    assert state.entropy_consolidating is True
    assert alife.get_entropy() <= 31.0  # consolidated toward target
    assert alife.needs_allostatic_reset() is False


def test_will_can_select_rest_explicitly():
    alife = ALifeDynamics(max_entropy=100.0)
    alife.add_entropy(90.0, "load")
    cleared = alife.allostatic_reset("will_selected_rest")
    assert cleared > 0.0
    assert alife.get_entropy() <= 31.0


# ── B. Attractor saturation defense ─────────────────────────────────────────

def test_field_noise_scales_with_eigenvalue_entropy():
    uf = UnifiedField(FieldConfig(dim=64))

    # Rich dynamics → high spectral entropy → baseline noise.
    rng = np.random.default_rng(0)
    for _ in range(40):
        uf._history.append((rng.standard_normal(64) * 0.3).astype(np.float32))
    uf._spectral_entropy = spectral_entropy_of(uf._history)
    uf.F = (rng.standard_normal(64) * 0.2).astype(np.float32)
    healthy_entropy = uf.get_spectral_entropy()
    healthy_scale = uf._anti_degeneracy_scale()

    # Degenerate rail-saturated attractor → spectrum collapses → big noise boost.
    uf._history.clear()
    for _ in range(40):
        uf._history.append(np.ones(64, dtype=np.float32) * 0.999)
    uf._spectral_entropy = spectral_entropy_of(uf._history)
    uf.F = np.ones(64, dtype=np.float32) * 0.999
    degenerate_entropy = uf.get_spectral_entropy()
    degenerate_scale = uf._anti_degeneracy_scale()

    assert healthy_entropy > 0.7
    assert degenerate_entropy < 0.2
    assert healthy_scale < 1.5
    assert degenerate_scale > 3.0
    assert degenerate_scale > healthy_scale


def test_field_saturation_rescue_breaks_the_rails():
    """A saturated attractor (high coherence, invisible to the legacy recovery)
    must be pulled off the rails by the saturation guard."""
    uf = UnifiedField(FieldConfig(dim=64))
    uf._history.clear()
    for _ in range(40):
        uf._history.append(np.ones(64, dtype=np.float32) * 0.999)
    uf._spectral_entropy = spectral_entropy_of(uf._history)
    uf.F = np.ones(64, dtype=np.float32) * 0.999
    before = float(np.mean(np.abs(uf.F)))
    uf._update_coherence()
    after = float(np.mean(np.abs(uf.F)))
    assert before > 0.95
    assert after < before  # pulled toward the interior
    assert uf._saturation_recoveries >= 1


def test_chaos_intensity_scales_with_degeneracy_pressure():
    engine = ChaosEngine(ChaosConfig(state_dim=64, intensity=0.005, degeneracy_gain=6.0))
    base = engine.effective_intensity()
    engine.set_complexity_pressure(1.0)
    boosted = engine.effective_intensity()
    assert base == 0.005
    assert boosted == 0.005 * 7.0  # 1 + degeneracy_gain
    assert boosted > base


def test_chaos_pressure_is_clamped_and_robust():
    engine = ChaosEngine(ChaosConfig(state_dim=64))
    engine.set_complexity_pressure(5.0)   # over-range → clamps to 1.0
    assert engine._complexity_pressure == 1.0
    engine.set_complexity_pressure(-1.0)  # under-range → clamps to 0.0
    assert engine._complexity_pressure == 0.0
    engine.set_complexity_pressure(float("nan"))  # ignored
    assert engine._complexity_pressure == 0.0
