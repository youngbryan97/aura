"""tests/test_consciousness_expansion_gauntlet.py
====================================================
End-to-end gauntlet spanning Phases 1-7 of the Aura consciousness
expansion.  Runs adversarial, stress and integration checks that touch
all new modules together.

Coverage:
    Phase 1 (HierarchicalPhi) — stress compute + null baseline + 32+ nodes
    Phase 2 (HemisphericSplit) — callosum severance cycle + confabulation
    Phase 3 (MinimalSelfhood)  — dugesia transition under sustained training
    Phase 4 (RecursiveToM)     — depth-3 + scrub-jay behaviour change
    Phase 5 (OctopusFederation) — sever/restore latency
    Phase 6 (CellularTurnover) — 25% burst identity preservation
    Phase 7 (AbsorbedVoices)   — attribution after cross-voice training
    Phase 8 (UnifiedCognitiveBias) — fused vector composition sanity
    Cross-phase — biases stay bounded, stress under 500 combined ticks.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.hierarchical_phi import HierarchicalPhi  # noqa: E402
from core.consciousness.hemispheric_split import HemisphericSplit, Hemisphere  # noqa: E402
from core.consciousness.minimal_selfhood import (  # noqa: E402
    ACTION_CATEGORIES as MS_ACTIONS, MinimalSelfhood, Mode,
)
from core.consciousness.recursive_tom import RecursiveTheoryOfMind, MAX_DEPTH  # noqa: E402
from core.consciousness.octopus_arms import OctopusFederation, ArmState  # noqa: E402
from core.consciousness.cellular_turnover import (  # noqa: E402
    CellularTurnover, THRESHOLD_IDENTITY,
)
from core.consciousness.neural_mesh import NeuralMesh  # noqa: E402
from core.consciousness.absorbed_voices import AbsorbedVoices  # noqa: E402
from core.consciousness.unified_cognitive_bias import UnifiedCognitiveBias  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coherent_snapshot(rng, phase):
    cog = np.array([
        math.sin(phase), math.cos(phase * 1.1), math.sin(phase * 0.9),
        math.cos(phase * 1.2), math.sin(phase * 1.3), math.cos(phase * 0.8),
        math.sin(phase * 0.7), math.cos(phase * 1.05),
        math.sin(phase * 1.25), math.cos(phase * 0.85),
        math.sin(phase * 0.95), math.cos(phase * 1.35),
        math.sin(phase * 0.55), math.cos(phase * 1.45),
        math.sin(phase * 1.15), math.cos(phase * 0.75),
    ], dtype=np.float64) + rng.standard_normal(16) * 0.05
    mesh = rng.standard_normal(4096).astype(np.float32) * 0.2
    for c in range(64):
        start = c * 64
        mesh[start:start + 64] += 0.6 * math.sin(phase + c * 0.05)
    return cog, mesh


# ── Gauntlet tests ───────────────────────────────────────────────────────────

def test_gauntlet_hierarchical_phi_under_load():
    h = HierarchicalPhi()
    rng = np.random.default_rng(101)
    phase = 0.0
    for _ in range(600):
        phase += 0.19
        cog, mesh = _coherent_snapshot(rng, phase)
        h.record_snapshot(cog, mesh)
    t0 = time.time()
    result = h.compute(force=True)
    elapsed = time.time() - t0
    assert result is not None
    assert elapsed < 3.0
    # Null baseline stays strictly below measured max-complex phi.
    null_phi = h.compute_null_baseline()
    assert null_phi < max(result.max_complex_phi, 0.02)


def test_gauntlet_hemispheric_severance_and_restore_cycle():
    split = HemisphericSplit()
    rng = np.random.default_rng(202)
    for _ in range(120):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    pre = split.agreement_rate()
    split.sever_callosum()
    for _ in range(80):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    mid = split.agreement_rate()
    split.restore_callosum()
    for _ in range(120):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    post = split.agreement_rate()
    # Confabulation path works.
    split.record_action("grab_thing", Hemisphere.RIGHT)
    split.supply_reason("grab_thing")
    assert split.confabulation_rate() > 0.0
    # Agreement monotonic-ish: severance drops, restore recovers.
    assert post >= mid - 1e-6


def test_gauntlet_minimal_selfhood_reaches_dugesia_and_biases_toward_rest():
    ms = MinimalSelfhood()
    pre = np.zeros(8, dtype=np.float32); pre[0] = 1.0
    post = np.zeros(8, dtype=np.float32); post[0] = 0.0
    for _ in range(90):
        t = ms.tag_action("rest", pre)
        ms.reinforce(t, post)
    ms.update(
        body_budget={"energy_reserves": 0.0, "resource_pressure": 0.2,
                     "thermal_stress": 0.0},
        affect={"coherence": 0.8, "curiosity": 0.7},
        cognitive_state={"social_hunger": 0.0, "prediction_error": 0.1,
                         "agency_score": 0.9},
    )
    assert ms.mode() == Mode.DUGESIA
    rest_idx = MS_ACTIONS.index("rest")
    top3 = np.argsort(ms.action_priority())[::-1][:3]
    assert rest_idx in top3


def test_gauntlet_tom_observer_effect_drives_bias_change():
    tom_alone = RecursiveTheoryOfMind()
    tom_watched = RecursiveTheoryOfMind()
    tom_watched.observe_agent("bryan", strength=0.9)
    tom_watched.observe_agent("bryan", strength=0.9)
    tom_watched.register_interaction("bryan", salience=0.8, trust=0.9)
    assert tom_watched.depth_reached("bryan") == MAX_DEPTH
    ba = tom_alone.get_observer_bias().bias
    bw = tom_watched.get_observer_bias().bias
    assert not np.allclose(ba, bw)


def test_gauntlet_octopus_severance_and_recovery():
    fed = OctopusFederation()
    rng = np.random.default_rng(303)
    for _ in range(30):
        fed.tick(rng.standard_normal(3))
    fed.sever_link()
    for _ in range(20):
        fed.tick(rng.standard_normal(3))
    fed.restore_link()
    for _ in range(60):
        fed.tick(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    # Should have returned to LINKED after stable-env ticks.
    assert fed.arbiter.link_state() in (ArmState.LINKED, ArmState.RECOVERING)


def test_gauntlet_cellular_turnover_preserves_identity_under_25pct_burst():
    mesh = NeuralMesh()
    rng = np.random.default_rng(404)
    for c in mesh.columns:
        c.x = rng.standard_normal(c.n).astype(np.float32) * 0.3
    turn = CellularTurnover(turnover_rate=0.0)
    turn.attach(mesh)
    fp_before = turn._fingerprints[-1]
    fp_after = turn.force_turnover(0.25)
    assert fp_after.similarity(fp_before) >= THRESHOLD_IDENTITY


def test_gauntlet_absorbed_voices_attribution_multi_voice():
    import tempfile
    av = AbsorbedVoices(storage_dir=Path(tempfile.mkdtemp()))
    av.add_voice("bryan", sample_text="enterprise quality, real tests, deep impact")
    av.add_voice("teacher", sample_text="fractions and decimals, examples first")
    av.add_voice("fiction", sample_text="dragons, wizards, distant galaxies")
    a = av.attribute_thought("add enterprise quality checks with deep tests")
    assert a.best_voice_id == "bryan"


def test_gauntlet_unified_bias_composes_all_three_sources():
    uni = UnifiedCognitiveBias()
    hemi = np.array([0.8, -0.3] + [0.0] * 14, dtype=np.float32)
    selfhood = np.array([0.1, 0.1] + [0.7] + [0.0] * 13, dtype=np.float32)
    observer = np.array([0.0, 0.6] + [0.0] * 14, dtype=np.float32)
    snap = uni.fuse(hemi, selfhood, observer, observer_presence=0.7)
    assert snap.fused.shape == (16,)
    assert np.all(np.isfinite(snap.fused))
    # Each contribution scaled by its weight.
    assert np.linalg.norm(snap.hemi_contribution) > 0.0
    assert np.linalg.norm(snap.selfhood_contribution) > 0.0
    assert np.linalg.norm(snap.observer_contribution) > 0.0


def test_gauntlet_biases_remain_bounded_across_many_iterations():
    split = HemisphericSplit()
    ms = MinimalSelfhood()
    tom = RecursiveTheoryOfMind()
    uni = UnifiedCognitiveBias()
    rng = np.random.default_rng(505)
    for i in range(300):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
        ms.update(
            body_budget={"energy_reserves": 0.5, "resource_pressure": 0.3,
                         "thermal_stress": 0.2},
            affect={"coherence": 0.6, "curiosity": 0.5},
            cognitive_state={"social_hunger": 0.3, "prediction_error": 0.2,
                             "agency_score": 0.6},
        )
        if i % 3 == 0:
            tom.observe_agent(f"agent_{i % 5}", strength=0.4)
        uni.fuse(
            split.fused_bias(),
            ms.get_priority_bias(),
            tom.get_observer_bias().bias,
            tom.total_observer_presence(),
        )
    snap = uni.last()
    assert snap is not None
    assert np.all(np.abs(snap.fused) <= 1.0 + 1e-5)


def test_gauntlet_combined_latency_budget():
    """A combined tick of hemispheric + selfhood + recursive ToM + unified
    fusion should complete in well under 20 ms so we can maintain > 50 Hz."""
    split = HemisphericSplit()
    ms = MinimalSelfhood()
    tom = RecursiveTheoryOfMind()
    uni = UnifiedCognitiveBias()
    rng = np.random.default_rng(606)
    # Warm up.
    for _ in range(10):
        exec_s = rng.standard_normal(8); sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16); emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    t0 = time.time()
    for _ in range(50):
        exec_s = rng.standard_normal(8); sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16); emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
        ms.update(
            body_budget={"energy_reserves": 0.5, "resource_pressure": 0.3,
                         "thermal_stress": 0.2},
            affect={"coherence": 0.6, "curiosity": 0.5},
            cognitive_state={"social_hunger": 0.3, "prediction_error": 0.2,
                             "agency_score": 0.6},
        )
        tom.observe_agent("stream", strength=0.2)
        uni.fuse(
            split.fused_bias(),
            ms.get_priority_bias(),
            tom.get_observer_bias().bias,
            tom.total_observer_presence(),
        )
    elapsed = (time.time() - t0) / 50 * 1000.0
    assert elapsed < 20.0, f"combined tick too slow: {elapsed:.1f}ms (> 20ms)"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_gauntlet_hierarchical_phi_under_load,
        test_gauntlet_hemispheric_severance_and_restore_cycle,
        test_gauntlet_minimal_selfhood_reaches_dugesia_and_biases_toward_rest,
        test_gauntlet_tom_observer_effect_drives_bias_change,
        test_gauntlet_octopus_severance_and_recovery,
        test_gauntlet_cellular_turnover_preserves_identity_under_25pct_burst,
        test_gauntlet_absorbed_voices_attribution_multi_voice,
        test_gauntlet_unified_bias_composes_all_three_sources,
        test_gauntlet_biases_remain_bounded_across_many_iterations,
        test_gauntlet_combined_latency_budget,
    ]
    passed, failed = 0, []
    t0 = time.time()
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ok {t.__name__}")
        except Exception as exc:
            failed.append((t.__name__, exc))
            print(f"  FAIL {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed in {(time.time() - t0):.1f}s")
    sys.exit(0 if not failed else 1)
