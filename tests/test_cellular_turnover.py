"""tests/test_cellular_turnover.py
=====================================
Tests for neuron turnover + pattern-identity preservation on the
NeuralMesh (Theseus / Kurzgesagt cell-replacement thought experiment).
"""
from __future__ import annotations


import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.cellular_turnover import (  # noqa: E402
    CellularTurnover,
    DEFAULT_TURNOVER_RATE,
    IdentityFingerprint,
    THRESHOLD_IDENTITY,
)
from core.consciousness.neural_mesh import NeuralMesh  # noqa: E402


def _make_mesh() -> NeuralMesh:
    """Construct a mesh without starting its background loop."""
    return NeuralMesh()


def _drive_mesh(mesh: NeuralMesh, n_ticks: int = 20) -> None:
    """Inject a little signal so the mesh isn't all zeros."""
    rng = np.random.default_rng(3)
    for c in mesh.columns:
        c.x = rng.standard_normal(c.n).astype(np.float32) * 0.3


def test_attach_and_capture_initial_fingerprint():
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover(turnover_rate=0.01)
    turn.attach(mesh)
    assert turn.fingerprints_count() >= 1


def test_tick_with_no_mesh_returns_none():
    turn = CellularTurnover()
    assert turn.tick() is None


def test_single_tick_replaces_expected_count():
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover(turnover_rate=0.01)   # 1% per tick = ~40 of 4096
    turn.attach(mesh)
    # Run enough ticks that at least one replacement occurs.
    replaced_total = 0
    for _ in range(40):
        ev = turn.tick()
        if ev is not None:
            replaced_total += ev.n_replaced
    assert replaced_total > 0, "expected some neurons replaced after 40 ticks"


def test_forced_20pct_turnover_preserves_identity():
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover(turnover_rate=0.0)   # No auto-turnover
    turn.attach(mesh)
    fp_before = turn._fingerprints[-1]
    fp_after = turn.force_turnover(0.20)   # 20% of neurons
    sim = fp_after.similarity(fp_before)
    assert sim >= THRESHOLD_IDENTITY, (
        f"identity drift too large after 20% turnover: "
        f"similarity={sim:.3f} < {THRESHOLD_IDENTITY}"
    )


def test_forced_100pct_turnover_diverges_identity():
    """Full replacement SHOULD drift identity (lose the pattern)."""
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover(turnover_rate=0.0)
    turn.attach(mesh)
    fp_before = turn._fingerprints[-1]
    fp_after = turn.force_turnover(1.0)
    sim = fp_after.similarity(fp_before)
    assert sim < 1.0, "100% turnover should alter fingerprint"


def test_fingerprint_shape():
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover()
    turn.attach(mesh)
    fp = turn._fingerprints[-1]
    assert isinstance(fp, IdentityFingerprint)
    assert len(fp.tier_energies) == 3
    assert fp.projection_signature.shape == (16,)


def test_events_logged_per_tick():
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover(turnover_rate=0.05)
    turn.attach(mesh)
    for _ in range(20):
        turn.tick()
    ev = turn.recent_events(5)
    # Events recorded (not every tick yields one at tiny rate; ensure SOME).
    assert len(ev) > 0


def test_replacement_respects_neighbourhood_pattern():
    """Replaced neurons should have activations in the same general range
    as their neighbours — NOT reset to zero."""
    mesh = _make_mesh()
    _drive_mesh(mesh)
    # Set column 0 activations to a strong pattern.
    col = mesh.columns[0]
    col.x = np.full(col.n, 0.5, dtype=np.float32)
    turn = CellularTurnover(turnover_rate=0.5)
    turn.attach(mesh)
    # Force a burst that will hit column 0 with high probability.
    for _ in range(30):
        turn.tick()
    # At least some neurons still carry values near 0.5 (the neighbourhood pattern).
    # Not all — turnover replaces some — but median should remain close.
    post_median = float(np.median(col.x))
    assert abs(post_median - 0.5) < 0.35, (
        f"median {post_median:.3f} drifted too far from neighbourhood pattern 0.5"
    )


def test_status_dict_complete():
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover()
    turn.attach(mesh)
    s = turn.get_status()
    for k in ("tick", "turnover_rate", "total_replaced", "fingerprint_count",
              "last_identity_similarity", "identity_stable", "mesh_attached"):
        assert k in s
    assert s["mesh_attached"] is True


def test_identity_stable_under_modest_turnover():
    """Over many ticks at the default rate, identity similarity should
    stay above the threshold."""
    mesh = _make_mesh()
    _drive_mesh(mesh)
    turn = CellularTurnover(turnover_rate=DEFAULT_TURNOVER_RATE)
    turn.attach(mesh)
    low_marks = []
    for _ in range(200):
        turn.tick()
        low_marks.append(turn.last_similarity())
    # With low per-tick rate, similarity should mostly stay near 1.0.
    assert np.mean(low_marks) > 0.8


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_attach_and_capture_initial_fingerprint,
        test_tick_with_no_mesh_returns_none,
        test_single_tick_replaces_expected_count,
        test_forced_20pct_turnover_preserves_identity,
        test_forced_100pct_turnover_diverges_identity,
        test_fingerprint_shape,
        test_events_logged_per_tick,
        test_replacement_respects_neighbourhood_pattern,
        test_status_dict_complete,
        test_identity_stable_under_modest_turnover,
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
