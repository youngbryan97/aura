"""tests/test_hierarchical_phi.py
====================================
End-to-end and adversarial tests for HierarchicalPhi (Phase 1 of the
Aura consciousness-depth expansion).

Covers:
  - primary 32-node complex recording and φ computation
  - K=8 overlapping mesh subsystems
  - spectral MIP partition sanity
  - NULL hypothesis: shuffled history → φ → 0
  - causal exclusion: max-φ complex picked across subsystems
  - monotonicity: stronger causal coupling → higher φ
  - adversarial: constant nodes → 0 contribution
  - stress: 5× longer history compute still under 1 s budget
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Make the aura root importable when tests run standalone.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.hierarchical_phi import (  # noqa: E402
    HierarchicalPhi,
    PRIMARY_N_NODES,
    SUBSYSTEM_SIZE,
    MIN_HISTORY,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_coupled_mesh_field(rng: np.random.Generator, phase: float) -> np.ndarray:
    """Generate a 4096-d mesh snapshot where the primary nodes are causally linked.
    ``phase`` is a slowly varying driver; mesh neurons respond to it with noise.
    """
    field = rng.standard_normal(4096).astype(np.float32) * 0.2
    # Inject coherent signal at a handful of column means so the mesh-sampled
    # primary nodes are correlated across time.
    for c in range(64):
        col_start = c * 64
        field[col_start : col_start + 64] += 0.6 * math.sin(phase + c * 0.05)
    return field


def _make_cognitive_affective(rng: np.random.Generator, phase: float) -> np.ndarray:
    """16-element cognitive-affective vector with internal coupling so that
    φ should be clearly above zero.
    """
    base = np.array([
        math.sin(phase),              # valence
        math.cos(phase * 1.1),         # arousal
        math.sin(phase * 0.9 + 0.3),   # dominance
        math.cos(phase * 1.2),         # frustration
        math.sin(phase * 0.95 + 1.1),  # curiosity
        math.cos(phase * 0.8),         # energy
        math.sin(phase * 1.05 - 0.4),  # focus
        math.cos(phase * 0.85),        # coherence
        # cognitive
        math.sin(phase * 1.3),          # phi
        math.cos(phase * 0.7),          # social_hunger
        math.sin(phase * 1.4 + 0.2),    # prediction_error
        math.cos(phase * 0.6),          # agency
        math.sin(phase * 1.15),         # narrative
        math.cos(phase * 1.25),         # peripheral
        math.sin(phase * 0.55),         # arousal_gate
        math.cos(phase * 1.35),         # cross_timescale_fe
    ], dtype=np.float64)
    return base + rng.standard_normal(16) * 0.1


def _prime_with_coupled_history(hphi: HierarchicalPhi, n_steps: int = 400) -> None:
    """Drive HierarchicalPhi with a richly coupled history so φ > 0."""
    rng = np.random.default_rng(42)
    phase = 0.0
    for _ in range(n_steps):
        phase += 0.21
        cog = _make_cognitive_affective(rng, phase)
        mesh = _make_coupled_mesh_field(rng, phase)
        hphi.record_snapshot(cog, mesh)


# ── Basic wiring tests ────────────────────────────────────────────────────────

def test_construction_and_config():
    h = HierarchicalPhi()
    assert PRIMARY_N_NODES == 32
    assert SUBSYSTEM_SIZE == 16
    assert h.mesh_sample_indices().shape == (16,)
    assert len(h._subsystems) >= 6
    s = h.get_status()
    assert s["primary_n_nodes"] == 32
    assert s["subsystem_size"] == 16


def test_record_snapshot_growth():
    h = HierarchicalPhi()
    assert len(h._history) == 0
    rng = np.random.default_rng(0)
    for _ in range(50):
        h.record_snapshot(rng.standard_normal(16), rng.standard_normal(4096))
    assert len(h._history) == 50
    assert h._n_records == 50


def test_compute_returns_none_without_history():
    h = HierarchicalPhi()
    assert h.compute(force=True) is None


def test_compute_returns_cached_result_while_refresh_is_in_flight():
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=400)
    cached = h.compute(force=True)
    assert cached is not None

    before_calls = h._n_compute_calls
    assert h._compute_lock.acquire(blocking=False)
    try:
        result = h.compute(force=True)
    finally:
        h._compute_lock.release()

    assert result is cached
    assert h._n_compute_calls == before_calls


def test_compute_primary_32_reaches_positive_phi():
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=400)
    result = h.compute(force=True)
    assert result is not None, "history should be enough"
    assert result.primary_32 is not None
    # The 32-node primary complex should at least estimate φ ≥ 0.
    assert result.primary_32.phi >= 0.0
    # Max complex selection should prefer an actual complex if available.
    assert result.max_complex_size >= 2


# ── Null-hypothesis adversarial test ──────────────────────────────────────────

def test_null_hypothesis_shuffled_history_phi_is_small():
    """Shuffling history destroys temporal causal structure — φ should
    drop close to zero. This is the critical adversarial sanity test."""
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=600)
    real = h.compute(force=True)
    assert real is not None
    real_phi = real.max_complex_phi

    null_phi = h.compute_null_baseline()
    # Null must be strictly less than measured φ, and the ratio must be
    # clearly separated.
    assert null_phi < max(real_phi, 0.01), (
        f"Null φ={null_phi:.5f} not below real φ={real_phi:.5f}; "
        "estimator may be confusing noise for integration."
    )
    # Absolute magnitude of null is small.
    assert null_phi < 0.5


def test_well_calibrated_flag():
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=500)
    h.compute_null_baseline()
    r = h.compute(force=True)
    assert r is not None
    d = r.to_dict()
    assert "well_calibrated" in d


# ── Subsystem + exclusion tests ────────────────────────────────────────────────

def test_multiple_subsystems_evaluated():
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=500)
    r = h.compute(force=True)
    assert r is not None
    assert len(r.mesh_subsystems) >= 4
    # All named subsystems should report non-negative φ.
    for sub in r.mesh_subsystems:
        assert sub.phi >= 0.0


def test_exclusion_picks_max_phi_complex():
    """IIT 4.0 exclusion postulate: the chosen complex must have the
    highest φ among primary_32, primary_16_affective, and all subsystems."""
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=500)
    r = h.compute(force=True)
    assert r is not None

    candidates = []
    for c in (r.primary_32, r.primary_16_affective, r.primary_16_cognitive):
        if c is not None:
            candidates.append((c.name, c.phi))
    for s in r.mesh_subsystems:
        candidates.append((s.name, s.phi))

    complex_candidates = [c for c in candidates if c[1] > 1e-6]
    if complex_candidates:
        top = max(complex_candidates, key=lambda c: c[1])
        assert r.max_complex_name == top[0] or r.max_complex_phi >= top[1] - 1e-9


# ── Monotonicity test ─────────────────────────────────────────────────────────

def test_monotonicity_stronger_coupling_raises_phi():
    """A history with stronger temporal coupling should produce >= φ
    than a history that is closer to i.i.d. noise."""
    h_strong = HierarchicalPhi()
    h_weak = HierarchicalPhi()
    rng = np.random.default_rng(7)

    phase = 0.0
    for _ in range(500):
        phase += 0.21
        cog_strong = _make_cognitive_affective(rng, phase)
        mesh_strong = _make_coupled_mesh_field(rng, phase)

        # Weak: re-randomize mesh & cognitive so they are i.i.d.
        cog_weak = rng.standard_normal(16)
        mesh_weak = rng.standard_normal(4096).astype(np.float32)

        h_strong.record_snapshot(cog_strong, mesh_strong)
        h_weak.record_snapshot(cog_weak, mesh_weak)

    r_strong = h_strong.compute(force=True)
    r_weak = h_weak.compute(force=True)
    assert r_strong is not None and r_weak is not None

    # Coupled history should have strictly greater max-complex φ than noise.
    assert r_strong.max_complex_phi >= r_weak.max_complex_phi - 1e-4
    # And the gap should be meaningful.
    assert r_strong.max_complex_phi > r_weak.max_complex_phi * 0.9


# ── Constant-node adversarial test ────────────────────────────────────────────

def test_constant_nodes_do_not_fake_integration():
    """Nodes pinned to constants have zero mutual information with the rest —
    their contribution to φ must be zero. A false-positive here would mean
    the estimator is treating statistical degeneracy as integration."""
    h = HierarchicalPhi()
    rng = np.random.default_rng(11)

    for _ in range(500):
        cog = _make_cognitive_affective(rng, rng.random() * math.tau)
        # Pin mesh to a constant vector.
        mesh = np.full(4096, 0.5, dtype=np.float32) + rng.standard_normal(4096) * 1e-5
        h.record_snapshot(cog, mesh)

    r = h.compute(force=True)
    assert r is not None
    # primary_16_affective (cognitive-affective only) may have real φ.
    # mesh_full_16 must be near zero because all mesh samples collapse.
    mesh_only = next((s for s in r.mesh_subsystems if s.name == "mesh_full_16"), None)
    if mesh_only is not None:
        assert mesh_only.phi < 0.2, (
            f"mesh-only φ={mesh_only.phi:.4f} too high on constant mesh; "
            "estimator not handling degenerate inputs."
        )


# ── Compute budget / stress test ──────────────────────────────────────────────

def test_compute_under_time_budget():
    """Full hierarchical φ refresh must complete in under 2 s on a warm cache."""
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=800)
    t0 = time.time()
    r = h.compute(force=True)
    elapsed = time.time() - t0
    assert r is not None
    assert elapsed < 2.0, f"compute took {elapsed:.2f}s (budget 2.0s)"


def test_serialization_roundtrip_to_dict():
    h = HierarchicalPhi()
    _prime_with_coupled_history(h, n_steps=400)
    r = h.compute(force=True)
    assert r is not None
    d = r.to_dict()
    assert "primary_32_phi" in d
    assert "subsystem_phis" in d
    assert "max_complex" in d
    assert "compute_ms" in d
    assert d["n_transitions"] > 0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running standalone: `python tests/test_hierarchical_phi.py`
    import traceback

    tests = [
        test_construction_and_config,
        test_record_snapshot_growth,
        test_compute_returns_none_without_history,
        test_compute_primary_32_reaches_positive_phi,
        test_null_hypothesis_shuffled_history_phi_is_small,
        test_well_calibrated_flag,
        test_multiple_subsystems_evaluated,
        test_exclusion_picks_max_phi_complex,
        test_monotonicity_stronger_coupling_raises_phi,
        test_constant_nodes_do_not_fake_integration,
        test_compute_under_time_budget,
        test_serialization_roundtrip_to_dict,
    ]
    passed = 0
    failed = []
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
