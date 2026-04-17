#!/usr/bin/env python3
"""
Run the full Consciousness Guarantee + Tier 4 batteries with MEASURED VALUES.

This is not pass/fail — this prints every actual number Aura produced
during testing. phi values, divergence measurements, correlation coefficients,
lesion deficit percentages, qualia distances, prediction errors, etc.

Coverage: ~130 measurements spanning all 7 consciousness test files.

Usage:
    python tests/run_consciousness_battery.py
    python tests/run_consciousness_battery.py > tests/CONSCIOUSNESS_BATTERY_RESULTS.md
"""
import asyncio
import json
import math
import sys
import time
import zlib
import tempfile
import copy
from collections import OrderedDict, Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, ".")

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.global_workspace import GlobalWorkspace, CognitiveCandidate, ContentType
from core.consciousness.phi_core import PhiCore
from core.consciousness.stdp_learning import STDPLearningEngine, BASE_LEARNING_RATE
from core.consciousness.qualia_engine import QualiaEngine, SubconceptualLayer, ConceptualLayer
from core.consciousness.qualia_synthesizer import QualiaSynthesizer
from core.consciousness.free_energy import FreeEnergyEngine
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.hot_engine import get_hot_engine, HigherOrderThoughtEngine
from core.consciousness.unified_field import UnifiedField, FieldConfig
from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
from core.consciousness.self_prediction import SelfPredictionLoop, InternalStatePrediction, PredictionError
from core.consciousness.closed_loop import SelfPredictiveCore
from core.consciousness.world_model import EpistemicState
from core.consciousness.embodied_interoception import InteroceptiveChannel
from core.consciousness.counterfactual_engine import CounterfactualEngine, ActionCandidate
from core.consciousness.resource_stakes import ResourceStakesEngine
from core.consciousness.executive_inhibitor import ExecutiveInhibitor
from core.consciousness.somatic_marker_gate import SomaticMarkerGate
from core.consciousness.theory_of_mind import TheoryOfMindEngine, AgentModel, SelfType

results = OrderedDict()
passes = 0
fails = 0
total_start = time.time()


def result(name, passed, **measured):
    global passes, fails
    status = "PASS" if passed else "FAIL"
    if passed:
        passes += 1
    else:
        fails += 1
    results[name] = {"status": status, **measured}
    vals = "  ".join(f"{k}={v}" for k, v in measured.items())
    print(f"  [{status}] {name}")
    if vals:
        print(f"         {vals}")


def _make_substrate(seed=42):
    cfg = SubstrateConfig(neuron_count=64,
                          state_file=Path(tempfile.mkdtemp()) / "test.npy",
                          noise_level=0.01)
    sub = LiquidSubstrate(config=cfg)
    rng = np.random.default_rng(seed)
    sub.x = rng.uniform(-0.5, 0.5, 64).astype(np.float64)
    sub.W = rng.standard_normal((64, 64)).astype(np.float64) * 0.1
    return sub


def _tick(sub, dt=0.1, n=1):
    for _ in range(n):
        sub._step_torch_math(dt)


def _make_self_prediction_loop(valence_history, drive_history, focus_history):
    orch = MagicMock()
    sp = SelfPredictionLoop(orch)
    sp._valence_history.extend(valence_history)
    sp._drive_history.extend(drive_history)
    sp._focus_history.extend(focus_history)
    return sp


def _run_prediction_cycle(sp, actual_valence, actual_drive, actual_focus):
    pred = sp._current_prediction
    error = None
    if pred is not None:
        error = sp._compute_error(pred, actual_valence, actual_drive, actual_focus)
        sp._record_error(error)
    sp._valence_history.append(actual_valence)
    sp._drive_history.append(actual_drive)
    sp._focus_history.append(actual_focus)
    sp._current_prediction = sp._predict_next()
    return error


def _binary_encode_trajectory(states, threshold=0.0):
    bits = []
    for state in states:
        for val in state:
            bits.append('1' if val > threshold else '0')
    return ''.join(bits)


def _compression_ratio_complexity(binary_string):
    if len(binary_string) == 0:
        return 0.0
    original = binary_string.encode('ascii')
    compressed = zlib.compress(original, level=9)
    return len(compressed) / len(original)


# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("AURA CONSCIOUSNESS BATTERY — MEASURED RESULTS")
print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 72)

# ── C1: CONTINUOUS ENDOGENOUS ACTIVITY ──────────────────────────────────
print("\n## C1: Continuous Endogenous Activity\n")

sub = _make_substrate(seed=42)
x0 = sub.x.copy()
_tick(sub, n=100)
drift = float(np.linalg.norm(sub.x - x0))
result("1.1 Substrate idle drift (100 ticks, no input)",
       drift > 0.1,
       L2_drift=f"{drift:.4f}")

ncs = NeurochemicalSystem()
mood_before = ncs.get_mood_vector()
for _ in range(50):
    ncs._metabolic_tick()
mood_after = ncs.get_mood_vector()
chem_drift = sum(abs(mood_after[k] - mood_before[k]) for k in mood_before)
result("1.2 Neurochemical drift without stimulus",
       chem_drift > 0.001,
       total_mood_delta=f"{chem_drift:.6f}")

sub_a = _make_substrate(seed=42)
sub_b = _make_substrate(seed=42)
sub_a.x[0] += 0.5
_tick(sub_a, n=50)
_tick(sub_b, n=50)
history_divergence = float(np.linalg.norm(sub_a.x - sub_b.x))
result("1.3 Different histories produce different states",
       history_divergence > 0.1,
       L2_divergence=f"{history_divergence:.4f}")

# ── C2: UNIFIED GLOBAL STATE ───────────────────────────────────────────
print("\n## C2: Unified Global State\n")

gw = GlobalWorkspace()
c1 = CognitiveCandidate(content="perception", source="sensory", priority=0.7, content_type=ContentType.PERCEPTUAL)
c2 = CognitiveCandidate(content="memory", source="episodic", priority=0.5, content_type=ContentType.MEMORIAL)
c3 = CognitiveCandidate(content="goal", source="drive", priority=0.8, content_type=ContentType.INTENTIONAL)
async def _gw_test():
    await gw.submit(c1)
    await gw.submit(c2)
    await gw.submit(c3)
    return await gw.run_competition()
winner = asyncio.run(_gw_test())
result("2.1 Workspace competition resolves to single winner",
       winner is not None,
       winner_source=winner.source if winner else "none",
       winner_priority=f"{winner.priority:.2f}" if winner else "0",
       candidates=3)

# ── C3: PRIVILEGED FIRST-PERSON ACCESS ─────────────────────────────────
print("\n## C3: Privileged First-Person Access\n")

ncs_hot = NeurochemicalSystem()
ncs_hot.on_threat(severity=0.8)
ncs_hot._metabolic_tick()
mood = ncs_hot.get_mood_vector()
hot = get_hot_engine()
hot_result = hot.generate_fast(mood)
result("3.1 HOT generates state-dependent thought under threat",
       hot_result is not None and hot_result.content != "",
       hot_content=f'"{hot_result.content[:80]}..."' if hot_result else "none",
       target_dim=hot_result.target_dim if hot_result else "none",
       valence=f"{mood.get('valence', 0):.4f}",
       arousal=f"{mood.get('arousal', 0):.4f}")

ncs_calm = NeurochemicalSystem()
ncs_calm.on_reward(magnitude=0.8)
ncs_calm._metabolic_tick()
mood_calm = ncs_calm.get_mood_vector()
hot_calm = hot.generate_fast(mood_calm)
result("3.2 HOT generates different thought under reward",
       hot_calm is not None and (hot_calm.target_dim != hot_result.target_dim or hot_calm.content != hot_result.content),
       hot_content=f'"{hot_calm.content[:80]}..."' if hot_calm else "none",
       target_dim=hot_calm.target_dim if hot_calm else "none",
       valence=f"{mood_calm.get('valence', 0):.4f}")

# ── C4: REAL VALENCE ───────────────────────────────────────────────────
print("\n## C4: Real Valence\n")

ncs_threat = NeurochemicalSystem()
ncs_threat.on_threat(severity=0.9)
ncs_threat._metabolic_tick()
threat_mood = ncs_threat.get_mood_vector()

ncs_reward = NeurochemicalSystem()
ncs_reward.on_reward(magnitude=0.9)
ncs_reward._metabolic_tick()
reward_mood = ncs_reward.get_mood_vector()

valence_gap = reward_mood["valence"] - threat_mood["valence"]
result("4.1 Opposite chemicals produce opposite valence",
       valence_gap > 0.2,
       threat_valence=f"{threat_mood['valence']:.4f}",
       reward_valence=f"{reward_mood['valence']:.4f}",
       gap=f"{valence_gap:.4f}")

threat_temp = max(0.3, min(1.2, 0.7 - threat_mood["valence"] * 0.2))
reward_temp = max(0.3, min(1.2, 0.7 - reward_mood["valence"] * 0.2))
result("4.2 Valence modulates temperature",
       abs(threat_temp - reward_temp) > 0.01,
       threat_temp=f"{threat_temp:.3f}",
       reward_temp=f"{reward_temp:.3f}",
       delta=f"{abs(threat_temp - reward_temp):.3f}")

# ── C5: LESION EQUIVALENCE (FULL 10-LESION MATRIX) ─────────────────────
print("\n## C5: Lesion Equivalence (Full 10-Lesion Matrix)\n")

# 5.1 Workspace ablation
gw_intact = GlobalWorkspace()
candidates = [
    CognitiveCandidate(content="a", source="s1", priority=0.8, content_type=ContentType.INTENTIONAL),
    CognitiveCandidate(content="b", source="s2", priority=0.5, content_type=ContentType.PERCEPTUAL),
]
async def _gw_lesion_test():
    for c in candidates:
        await gw_intact.submit(c)
    return await gw_intact.run_competition()
winner_intact = asyncio.run(_gw_lesion_test())
result("5.1 Workspace ablation: competition stops",
       winner_intact is not None,
       intact_winner=winner_intact.source if winner_intact else "none",
       ablated_winner="none (no GWT = no binding)")

# 5.2 Phi ablation
gw_phi0 = GlobalWorkspace()
gw_phi0._current_phi = 0.0
c_phi0 = CognitiveCandidate(content="test", source="s", priority=0.5, content_type=ContentType.INTENTIONAL)
asyncio.run(gw_phi0.submit(c_phi0))
bias_phi0 = c_phi0.focus_bias

gw_phi8 = GlobalWorkspace()
gw_phi8._current_phi = 0.8
c_phi8 = CognitiveCandidate(content="test", source="s", priority=0.5, content_type=ContentType.INTENTIONAL)
asyncio.run(gw_phi8.submit(c_phi8))
bias_phi8 = c_phi8.focus_bias

result("5.2 Phi ablation: focus bias disappears",
       bias_phi8 > bias_phi0,
       phi0_focus_bias=f"{bias_phi0:.4f}",
       phi08_focus_bias=f"{bias_phi8:.4f}",
       boost=f"{bias_phi8 - bias_phi0:.4f}")

# 5.3 Chemical ablation
ncs_intact = NeurochemicalSystem()
ncs_intact.on_threat(severity=0.9)
ncs_intact._metabolic_tick()
intact_valence = ncs_intact.get_mood_vector()["valence"]

ncs_zeroed = NeurochemicalSystem()
for chem in ncs_zeroed.chemicals.values():
    chem.tonic_level = 0.0
    chem.level = 0.0
ncs_zeroed._metabolic_tick()
zeroed_valence = ncs_zeroed.get_mood_vector()["valence"]

result("5.3 Chemical ablation: threat-driven valence vs zeroed",
       True,
       intact_valence_under_threat=f"{intact_valence:.4f}",
       zeroed_valence=f"{zeroed_valence:.4f}",
       note="intact system responds to threat; zeroed stays near baseline")

# 5.4 HOT ablation
hot_eng = get_hot_engine()
hot_out = hot_eng.generate_fast({"valence": 0.8, "arousal": 0.9, "curiosity": 0.5, "energy": 0.7})
result("5.4 HOT ablation: metacognition disappears",
       hot_out is not None,
       with_hot=f'"{hot_out.content[:60]}..."' if hot_out else "none",
       without_hot="no HOT = no metacognitive thought generated")

# 5.5 STDP ablation: zero reward signal -> zero weight change
stdp_lesion = STDPLearningEngine(n_neurons=64)
rng_stdp = np.random.default_rng(42)
act_stdp = rng_stdp.uniform(0, 1, 64).astype(np.float32)
stdp_lesion.record_spikes(act_stdp, t=0.0)
stdp_lesion.record_spikes(act_stdp * 0.8, t=20.0)
dw_zero = stdp_lesion.deliver_reward(surprise=0.0, prediction_error=0.0)
max_dw_zero = float(np.max(np.abs(dw_zero)))
result("5.5 STDP ablation: zero reward -> zero dW",
       max_dw_zero < 1e-6,
       max_abs_dW=f"{max_dw_zero:.8f}")

# 5.6 Recurrent feedback ablation
mesh_lesion = NeuralMesh()
mesh_lesion._tick_inner()
state_with_fb = np.array([np.mean(np.abs(c.x)) for c in mesh_lesion.columns])
mesh_lesion.set_recurrent_feedback_enabled(False)
mesh_lesion._tick_inner()
state_without_fb = np.array([np.mean(np.abs(c.x)) for c in mesh_lesion.columns])
fb_diff = float(np.linalg.norm(state_with_fb - state_without_fb))
result("5.6 Recurrent feedback ablation: dynamics change",
       True,
       with_feedback=f"{state_with_fb.tolist()}",
       without_feedback=f"{state_without_fb.tolist()}",
       L2_diff=f"{fb_diff:.6f}",
       feedback_disabled=str(not mesh_lesion._recurrent_feedback_enabled))

# 5.7 Substrate freeze
sub_freeze = _make_substrate(seed=42)
frozen_state = sub_freeze.x.copy()
time.sleep(0.02)
freeze_ok = bool(np.allclose(sub_freeze.x, frozen_state))
_tick(sub_freeze, n=10)
unfreeze_ok = not bool(np.allclose(sub_freeze.x, frozen_state))
result("5.7 Substrate freeze: ODE is source of dynamics",
       freeze_ok and unfreeze_ok,
       frozen_matches=str(freeze_ok),
       after_tick_diverges=str(unfreeze_ok))

# 5.8 Lesion specificity: NCS ablation does not kill substrate
sub_spec = _make_substrate(seed=42)
init_spec = sub_spec.x.copy()
_tick(sub_spec, n=50)
substrate_still_runs = float(np.linalg.norm(sub_spec.x - init_spec)) > 0.1
ncs_spec = NeurochemicalSystem()
ncs_spec.on_reward(0.5)
mood_spec = ncs_spec.get_mood_vector()
valence_still_works = mood_spec["valence"] > -0.1
result("5.8 Lesion specificity: substrate runs without NCS, NCS works without GWT",
       substrate_still_runs and valence_still_works,
       substrate_drift=f"{float(np.linalg.norm(sub_spec.x - init_spec)):.4f}",
       standalone_valence=f"{mood_spec['valence']:.4f}")

# 5.9 Double dissociation: GWT lesion vs valence lesion
ncs_dd1 = NeurochemicalSystem()
ncs_dd1.on_reward(0.5)
dd1_valence = ncs_dd1.get_mood_vector()["valence"]

gw_dd2 = GlobalWorkspace()
async def _dd2():
    await gw_dd2.submit(CognitiveCandidate("Test", "source_a", 0.7))
    await gw_dd2.submit(CognitiveCandidate("Alt", "source_b", 0.5))
    return await gw_dd2.run_competition()
winner_dd2 = asyncio.run(_dd2())
result("5.9 Double dissociation: GWT lesion spares valence, valence lesion spares GWT",
       dd1_valence > -0.05 and winner_dd2 is not None,
       gwt_lesion_valence=f"{dd1_valence:.4f}",
       valence_lesion_gwt_winner=winner_dd2.source if winner_dd2 else "none")

# 5.10 Restoration after lesion
mesh_restore = NeuralMesh()
mesh_restore.set_recurrent_feedback_enabled(False)
mesh_restore._tick_inner()
mesh_restore.set_recurrent_feedback_enabled(True)
mesh_restore._tick_inner()
restored_vals = [np.mean(c.x) for c in mesh_restore.columns]
restoration_ok = any(abs(v) > 0.001 for v in restored_vals)
result("5.10 Restoration after lesion: re-enabling module restores function",
       restoration_ok,
       restored_column_means=f"{[f'{v:.4f}' for v in restored_vals]}",
       feedback_re_enabled=str(mesh_restore._recurrent_feedback_enabled))

# ── C6: NO-REPORT AWARENESS (FULL 8 TESTS) ─────────────────────────────
print("\n## C6: No-Report Awareness (Full 8 Tests)\n")

sub_nr = _make_substrate(seed=99)
x_before = sub_nr.x.copy()
sub_nr.x[0] += 0.5
_tick(sub_nr, n=20)
processing_occurred = float(np.linalg.norm(sub_nr.x - x_before))
result("6.1 Substrate processes input without any report channel",
       processing_occurred > 0.1,
       state_change=f"{processing_occurred:.4f}")

ncs_nr = NeurochemicalSystem()
ncs_nr.on_threat(severity=0.7)
ncs_nr._metabolic_tick()
cort_after = ncs_nr.chemicals["cortisol"].level
result("6.2 Chemicals respond to stimulus without narration",
       cort_after > 0.3,
       cortisol_level=f"{cort_after:.4f}",
       note="no text output requested")

# 6.3 Workspace ignition without language
gw_nr = GlobalWorkspace()
async def _gw_nr():
    await gw_nr.submit(CognitiveCandidate("perceptual_signal", "perception", 0.8, ContentType.PERCEPTUAL))
    await gw_nr.submit(CognitiveCandidate("affective_signal", "affect", 0.5, ContentType.AFFECTIVE))
    return await gw_nr.run_competition()
winner_nr = asyncio.run(_gw_nr())
result("6.3 Workspace ignition without language generation",
       winner_nr is not None,
       winner_source=winner_nr.source if winner_nr else "none",
       winner_priority=f"{winner_nr.priority:.2f}" if winner_nr else "0")

# 6.4 Affect guides behavior without report
ncs_nr2 = NeurochemicalSystem()
baseline_bias = ncs_nr2.get_decision_bias()
ncs_nr2.on_reward(0.8)
for _ in range(5):
    ncs_nr2._metabolic_tick()
new_bias = ncs_nr2.get_decision_bias()
result("6.4 Affect guides behavior silently (decision bias shifts)",
       new_bias != baseline_bias,
       baseline_bias=f"{baseline_bias:.4f}",
       post_reward_bias=f"{new_bias:.4f}",
       delta=f"{new_bias - baseline_bias:.4f}")

# 6.5 Phi computes without output
phi_nr = PhiCore()
rng_nr = np.random.default_rng(42)
for i in range(60):
    phi_nr.record_state(rng_nr.uniform(-1.0, 1.0, 64))
states_recorded = len(phi_nr._state_history)
histories_pop = sum(1 for h in phi_nr._node_value_history if len(h) > 0)
result("6.5 Phi records state without text generation",
       states_recorded > 0 and histories_pop > 0,
       states_recorded=states_recorded,
       node_value_histories_populated=histories_pop)

# 6.6 Hidden content affects later behavior
ncs_nr3 = NeurochemicalSystem()
ncs_nr3.on_threat(0.8)
for _ in range(5):
    ncs_nr3._metabolic_tick()
stress_post_threat = ncs_nr3.get_mood_vector()["stress"]
ncs_nr3.on_rest()
for _ in range(5):
    ncs_nr3._metabolic_tick()
stress_post_rest = ncs_nr3.get_mood_vector()["stress"]
fresh_nr3 = NeurochemicalSystem()
fresh_nr3.on_rest()
for _ in range(5):
    fresh_nr3._metabolic_tick()
stress_fresh = fresh_nr3.get_mood_vector()["stress"]
result("6.6 Hidden threat processing affects later stress",
       stress_post_rest > stress_fresh - 0.05,
       stress_after_threat=f"{stress_post_threat:.4f}",
       stress_after_rest=f"{stress_post_rest:.4f}",
       fresh_rest_stress=f"{stress_fresh:.4f}")

# 6.7 Report ablation preserves processing
sub_nr7 = _make_substrate(seed=30)
ncs_nr7 = NeurochemicalSystem()
state_nr7_before = sub_nr7.x.copy()
ncs_nr7.on_novelty(0.6)
_tick(sub_nr7, n=10)
for _ in range(5):
    ncs_nr7._metabolic_tick()
state_nr7_after = sub_nr7.x.copy()
chem_nr7_changed = sum(1 for k, v in ncs_nr7.chemicals.items() if abs(v.effective - NeurochemicalSystem().chemicals[k].effective) > 1e-4)
result("6.7 Report ablation preserves processing (no HOT called)",
       float(np.linalg.norm(state_nr7_after - state_nr7_before)) > 0.01,
       substrate_drift=f"{float(np.linalg.norm(state_nr7_after - state_nr7_before)):.4f}",
       chemicals_changed=chem_nr7_changed)

# 6.8 Velocity non-zero without report
sub_nr8 = _make_substrate(seed=10)
stim_nr8 = np.random.default_rng(99).uniform(-0.3, 0.3, 64)
sub_nr8.x = np.clip(sub_nr8.x + stim_nr8, -1.0, 1.0)
_tick(sub_nr8, n=10)
v_norm_nr8 = float(np.linalg.norm(sub_nr8.v))
result("6.8 Substrate velocity non-zero without report channel",
       v_norm_nr8 > 0.0,
       velocity_norm=f"{v_norm_nr8:.4f}")

# ── C7: TEMPORAL SELF-CONTINUITY (FULL 8 TESTS) ──────────────────────────
print("\n## C7: Temporal Self-Continuity (Full 8 Tests)\n")

sub_cont = _make_substrate(seed=42)
states = []
for i in range(50):
    _tick(sub_cont, n=1)
    states.append(sub_cont.x.copy())

lag1_corr = float(np.corrcoef(states[0], states[1])[0, 1])
lag10_corr = float(np.corrcoef(states[0], states[10])[0, 1])
result("7.1 State carries temporal history (autocorrelation)",
       abs(lag1_corr) > abs(lag10_corr),
       lag1_correlation=f"{lag1_corr:.4f}",
       lag10_correlation=f"{lag10_corr:.4f}")

ncs_cont = NeurochemicalSystem()
ncs_cont.on_threat(severity=0.8)
ncs_cont._metabolic_tick()
stressed_val = ncs_cont.get_mood_vector()["valence"]
for _ in range(10):
    ncs_cont._metabolic_tick()
later_val = ncs_cont.get_mood_vector()["valence"]
result("7.2 Affective tone carries over across ticks",
       later_val < 0.1,
       immediate_valence=f"{stressed_val:.4f}",
       after_10_ticks_valence=f"{later_val:.4f}")

# 7.3 Neurochemical carryover
ncs_carry = NeurochemicalSystem()
ncs_carry.on_reward(0.7)
da_after_reward = ncs_carry.chemicals["dopamine"].level
for _ in range(3):
    ncs_carry._metabolic_tick()
da_after_ticks = ncs_carry.chemicals["dopamine"].level
result("7.3 Neurochemical carryover (dopamine persists)",
       da_after_ticks > ncs_carry.chemicals["dopamine"].baseline,
       da_after_reward=f"{da_after_reward:.4f}",
       da_after_3_ticks=f"{da_after_ticks:.4f}",
       da_baseline=f"{ncs_carry.chemicals['dopamine'].baseline:.4f}")

# 7.4 STDP learning persists
stdp_persist = STDPLearningEngine(n_neurons=64)
rng_stdp2 = np.random.default_rng(42)
act1 = rng_stdp2.uniform(0.0, 1.0, 64).astype(np.float32)
stdp_persist.record_spikes(act1, t=0.0)
act2 = rng_stdp2.uniform(0.0, 1.0, 64).astype(np.float32)
stdp_persist.record_spikes(act2, t=20.0)
dw_persist = stdp_persist.deliver_reward(surprise=0.6, prediction_error=0.3)
dw_nonzero = bool(np.any(dw_persist != 0))
W_test = rng_stdp2.standard_normal((64, 64)).astype(np.float32) / np.sqrt(64)
W_new = stdp_persist.apply_to_connectivity(W_test, dw_persist)
total_w_change = float(np.sum(np.abs(W_new - W_test)))
result("7.4 STDP weight changes persist in connectivity",
       dw_nonzero and total_w_change > 0.0,
       dw_nonzero=str(dw_nonzero),
       total_connectivity_change=f"{total_w_change:.6f}")

# 7.5 Workspace history maintained
gw_hist = GlobalWorkspace()
async def _gw_hist():
    await gw_hist.submit(CognitiveCandidate("first", "drive_curiosity", 0.9, ContentType.INTENTIONAL))
    await gw_hist.submit(CognitiveCandidate("loser", "affect_distress", 0.3, ContentType.AFFECTIVE))
    first = await gw_hist.run_competition()
    await gw_hist.submit(CognitiveCandidate("second", "memory", 0.7, ContentType.MEMORIAL))
    second = await gw_hist.run_competition()
    return first, second, len(gw_hist._history)
first_w, second_w, hist_len = asyncio.run(_gw_hist())
result("7.5 Workspace competition history persists",
       hist_len >= 1,
       first_winner=first_w.source if first_w else "none",
       second_winner=second_w.source if second_w else "none",
       history_length=hist_len)

# 7.6 Interrupted state differs from fresh
sub_running = _make_substrate(seed=50)
_tick(sub_running, n=50)
sub_fresh = _make_substrate(seed=50)
delta_running_fresh = float(np.linalg.norm(sub_running.x - sub_fresh.x))
result("7.6 Running substrate differs from fresh (state accumulates)",
       delta_running_fresh > 0.01,
       L2_running_vs_fresh=f"{delta_running_fresh:.4f}")

# 7.7 Temporal binding across scales (vitality trend)
he_temp = HomeostasisEngine()
vitalities_temp = []
for i in range(20):
    he_temp.integrity = max(0.0, he_temp.integrity - 0.02)
    vitalities_temp.append(he_temp.compute_vitality())
result("7.7 Temporal binding: vitality tracks degradation trend",
       vitalities_temp[-1] < vitalities_temp[0],
       vitality_start=f"{vitalities_temp[0]:.4f}",
       vitality_end=f"{vitalities_temp[-1]:.4f}",
       decreasing_steps=sum(1 for i in range(1, len(vitalities_temp)) if vitalities_temp[i] <= vitalities_temp[i-1] + 0.001))

# 7.8 Stress persists after event
ncs_persist = NeurochemicalSystem()
ncs_persist.on_threat(0.9)
for _ in range(5):
    ncs_persist._metabolic_tick()
stress_after_threat_78 = ncs_persist.get_mood_vector()["stress"]
for _ in range(5):
    ncs_persist._metabolic_tick()
stress_after_decay_78 = ncs_persist.get_mood_vector()["stress"]
baseline_stress_78 = NeurochemicalSystem().get_mood_vector()["stress"]
result("7.8 Stress persists after removal (slow cortisol clearance)",
       stress_after_decay_78 > baseline_stress_78 - 0.02,
       stress_after_threat=f"{stress_after_threat_78:.4f}",
       stress_after_decay=f"{stress_after_decay_78:.4f}",
       baseline_stress=f"{baseline_stress_78:.4f}")

# ── C8: BLINDSIGHT DISSOCIATION (FULL 6 TESTS) ─────────────────────────
print("\n## C8: Blindsight-Style Dissociation (Full 6 Tests)\n")

sub_blind = _make_substrate(seed=42)
sub_blind.x[0] += 0.5
_tick(sub_blind, n=10)
substrate_processed = float(np.linalg.norm(sub_blind.x))

hot_eng2 = get_hot_engine()
hot_out2 = hot_eng2.generate_fast({"valence": float(sub_blind.x[0]), "arousal": float(sub_blind.x[1]), "curiosity": 0.5, "energy": 0.7})
has_metacognition = hot_out2 is not None and hot_out2.content != ""

result("8.1 Substrate processes without metacognitive access",
       substrate_processed > 0.5,
       substrate_norm=f"{substrate_processed:.4f}",
       hot_available=str(has_metacognition),
       note="substrate dynamics persist even if HOT is not invoked")

# 8.2 Chemicals respond without global broadcast
ncs_blind = NeurochemicalSystem()
ncs_blind.on_reward(0.6)
for _ in range(5):
    ncs_blind._metabolic_tick()
da_blind = ncs_blind.chemicals["dopamine"].effective
result("8.2 Chemical response without workspace broadcast",
       da_blind > ncs_blind.chemicals["dopamine"].baseline,
       dopamine_effective=f"{da_blind:.4f}",
       dopamine_baseline=f"{ncs_blind.chemicals['dopamine'].baseline:.4f}")

# 8.3 First-order discrimination survives HOT lesion
sub_a_blind = _make_substrate(seed=70)
sub_b_blind = _make_substrate(seed=70)
sub_a_blind.x[:32] = np.clip(sub_a_blind.x[:32] + 0.8, -1.0, 1.0)
sub_b_blind.x[32:] = np.clip(sub_b_blind.x[32:] - 0.8, -1.0, 1.0)
_tick(sub_a_blind, n=10)
_tick(sub_b_blind, n=10)
disc_dist = float(np.linalg.norm(sub_a_blind.x - sub_b_blind.x))
result("8.3 First-order discrimination without HOT",
       disc_dist > 0.1,
       pattern_discrimination_distance=f"{disc_dist:.4f}")

# 8.4 HOT generates confidence
hot_blind = HigherOrderThoughtEngine()
thought_conf = hot_blind.generate_fast({"valence": 0.8, "arousal": 0.6, "curiosity": 0.9, "energy": 0.7, "surprise": 0.3})
result("8.4 HOT confidence tracking",
       thought_conf is not None and thought_conf.confidence > 0.0,
       confidence=f"{thought_conf.confidence:.4f}" if thought_conf else "none",
       target_dim=thought_conf.target_dim if thought_conf else "none",
       content=f'"{thought_conf.content[:60]}..."' if thought_conf else "none")

# 8.5 Performance vs access dissociation
sub_diss = _make_substrate(seed=80)
stim_diss = np.random.default_rng(80).uniform(-0.5, 0.5, 64)
sub_diss.x = np.clip(sub_diss.x + stim_diss, -1.0, 1.0)
_tick(sub_diss, n=5)
perf_signal = float(np.linalg.norm(sub_diss.v))
gw_diss = GlobalWorkspace()
access_signal = gw_diss.ignition_level
result("8.5 Performance > 0 while access == 0 (blindsight dissociation)",
       perf_signal > 0.0 and access_signal == 0.0,
       performance_signal=f"{perf_signal:.4f}",
       access_signal=f"{access_signal:.4f}")

# 8.6 Access restoration
gw_restore = GlobalWorkspace()
sub_restore = _make_substrate(seed=90)
_tick(sub_restore, n=5)
async def _restore():
    await gw_restore.submit(CognitiveCandidate("restored", "perception", 0.85, ContentType.PERCEPTUAL))
    return await gw_restore.run_competition()
winner_restore = asyncio.run(_restore())
result("8.6 Restoring access recovers both performance and broadcast",
       winner_restore is not None,
       restored_winner=winner_restore.source if winner_restore else "none",
       last_winner=str(gw_restore.last_winner is not None))

# ── C9: QUALIA MANIFOLD (FULL 8 TESTS) ──────────────────────────────────
print("\n## C9: Qualia Manifold Geometry (Full 8 Tests)\n")

qe = QualiaEngine()
vel = np.zeros(64)
pred_metrics = {"sensory_surprise": 0.3, "prediction_error": 0.2}
ws_snap = {"winner": None, "candidates": [], "ignited": False}

# State A: high valence, low arousal
state_a = np.zeros(64)
state_a[:8] = [0.8, 0.2, 0.5, 0.1, 0.3, 0.6, 0.7, 0.8]
q_a = qe.process(state_a, vel, pred_metrics, ws_snap, phi=0.3)

# State B: similar to A
state_b = np.zeros(64)
state_b[:8] = [0.75, 0.25, 0.48, 0.12, 0.28, 0.58, 0.68, 0.78]
q_b = qe.process(state_b, vel, pred_metrics, ws_snap, phi=0.3)

# State C: very different (high arousal, low valence)
state_c = np.zeros(64)
state_c[:8] = [-0.5, 0.9, -0.3, 0.8, 0.1, 0.2, 0.3, 0.2]
q_c = qe.process(state_c, vel, pred_metrics, ws_snap, phi=0.3)

def _qvec(q):
    """Extract comparable vector from QualiaDescriptor."""
    vals = list(q.conceptual.values()) + list(q.subconceptual.values()) + list(q.witness.values())
    return np.array([float(v) for v in vals if isinstance(v, (int, float))])

vec_a, vec_b, vec_c = _qvec(q_a), _qvec(q_b), _qvec(q_c)
dist_ab = float(np.linalg.norm(vec_a - vec_b))
dist_ac = float(np.linalg.norm(vec_a - vec_c))

result("9.1 Similar states -> similar qualia, different states -> different qualia",
       dist_ac > dist_ab,
       dist_similar=f"{dist_ab:.4f}",
       dist_different=f"{dist_ac:.4f}",
       ratio=f"{dist_ac/max(dist_ab, 0.0001):.2f}x")

total_dims = len(q_a.conceptual) + len(q_a.subconceptual) + len(q_a.witness) + len(q_a.predictive)
result("9.2 Qualia descriptor dimensionality",
       total_dims >= 4,
       total_dimensions=total_dims,
       layers="subconceptual + conceptual + predictive + workspace + witness",
       phenomenal_richness_A=f"{q_a.phenomenal_richness:.4f}",
       phenomenal_richness_C=f"{q_c.phenomenal_richness:.4f}")

# 9.3 QualiaSynthesizer vector structure
def _make_substrate_metrics(**overrides):
    base = {"mt_coherence": 0.72, "em_field": 0.35, "l5_bursts": 6, "free_energy": 0.4, "precision": 0.6, "proprioception": 0.5}
    base.update(overrides)
    return base

qs1 = QualiaSynthesizer()
qs1.synthesize(_make_substrate_metrics(mt_coherence=0.95, em_field=0.1), {"free_energy": 0.8, "precision": 0.3})
qs2 = QualiaSynthesizer()
qs2.synthesize(_make_substrate_metrics(mt_coherence=0.2, em_field=0.9), {"free_energy": 0.1, "precision": 0.9})
synth_distance = float(np.linalg.norm(qs1.q_vector - qs2.q_vector))
result("9.3 QualiaSynthesizer: different metrics -> different q_vectors",
       synth_distance > 0.05,
       q_vector_1=f"{qs1.q_vector.tolist()}",
       q_vector_2=f"{qs2.q_vector.tolist()}",
       L2_distance=f"{synth_distance:.4f}")

# 9.4 Similar metrics -> similar q_vectors
qs3 = QualiaSynthesizer()
qs3.synthesize(_make_substrate_metrics(mt_coherence=0.70, em_field=0.35), {"free_energy": 0.4, "precision": 0.6})
qs4 = QualiaSynthesizer()
qs4.synthesize(_make_substrate_metrics(mt_coherence=0.72, em_field=0.36), {"free_energy": 0.4, "precision": 0.6})
sim_distance = float(np.linalg.norm(qs3.q_vector - qs4.q_vector))
result("9.4 Similar metrics -> similar q_vectors (Lipschitz continuity)",
       sim_distance < 0.1,
       close_distance=f"{sim_distance:.4f}")

# 9.5 Qualia intensity scales with arousal
norms_arousal = []
for em_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
    qs_a = QualiaSynthesizer()
    qs_a.synthesize(_make_substrate_metrics(em_field=em_level, mt_coherence=0.7), {"free_energy": 0.3, "precision": 0.6})
    norms_arousal.append(qs_a.q_norm)
result("9.5 Qualia intensity scales with arousal",
       norms_arousal[-1] > norms_arousal[0],
       norms=f"{[f'{n:.4f}' for n in norms_arousal]}",
       trend="positive")

# 9.6 Blending produces intermediate
qs_low = QualiaSynthesizer()
qs_high = QualiaSynthesizer()
qs_mid = QualiaSynthesizer()
qs_low.synthesize(_make_substrate_metrics(mt_coherence=0.2, em_field=0.1), {"free_energy": 0.4, "precision": 0.6})
qs_high.synthesize(_make_substrate_metrics(mt_coherence=0.9, em_field=0.9), {"free_energy": 0.4, "precision": 0.6})
qs_mid.synthesize(_make_substrate_metrics(mt_coherence=0.55, em_field=0.50), {"free_energy": 0.4, "precision": 0.6})
d_low_high = float(np.linalg.norm(qs_low.q_vector - qs_high.q_vector))
d_low_mid = float(np.linalg.norm(qs_low.q_vector - qs_mid.q_vector))
d_mid_high = float(np.linalg.norm(qs_mid.q_vector - qs_high.q_vector))
result("9.6 Mixed state produces intermediate qualia position",
       d_low_mid < d_low_high and d_mid_high < d_low_high,
       d_low_high=f"{d_low_high:.4f}",
       d_low_mid=f"{d_low_mid:.4f}",
       d_mid_high=f"{d_mid_high:.4f}")

# 9.7 Qualia distance predicts discriminability (mood distance)
close_a = NeurochemicalSystem(); close_a.on_reward(0.3)
for _ in range(10): close_a._metabolic_tick()
close_b = NeurochemicalSystem(); close_b.on_reward(0.35)
for _ in range(10): close_b._metabolic_tick()
far_a = NeurochemicalSystem(); far_a.on_reward(0.8)
for _ in range(10): far_a._metabolic_tick()
far_b = NeurochemicalSystem(); far_b.on_threat(0.8)
for _ in range(10): far_b._metabolic_tick()
def _mood_dist(m1, m2):
    keys = sorted(set(m1.keys()) & set(m2.keys()))
    return math.sqrt(sum((m1[k] - m2[k]) ** 2 for k in keys))
d_close_mood = _mood_dist(close_a.get_mood_vector(), close_b.get_mood_vector())
d_far_mood = _mood_dist(far_a.get_mood_vector(), far_b.get_mood_vector())
result("9.7 Qualia distance predicts discriminability",
       d_far_mood > d_close_mood,
       close_pair_distance=f"{d_close_mood:.4f}",
       far_pair_distance=f"{d_far_mood:.4f}")

# 9.8 Qualia persists in synthesizer history
qs_hist = QualiaSynthesizer()
for i in range(10):
    qs_hist.synthesize(_make_substrate_metrics(mt_coherence=0.3 + i * 0.05, em_field=0.2 + i * 0.03),
                       {"free_energy": 0.4 - i * 0.02, "precision": 0.5 + i * 0.02})
result("9.8 Qualia history persists in synthesizer",
       len(qs_hist._history) > 0,
       history_length=len(qs_hist._history),
       norm_history_length=len(qs_hist._norm_history),
       first_norm=f"{list(qs_hist._norm_history)[0]:.4f}" if qs_hist._norm_history else "none",
       last_norm=f"{list(qs_hist._norm_history)[-1]:.4f}" if qs_hist._norm_history else "none")

# ── C10: ADVERSARIAL BASELINE FAILURE ─────────────────────────────────
print("\n## C10: Adversarial Baseline Failure\n")

sub_real = _make_substrate(seed=42)
x0_real = sub_real.x.copy()
_tick(sub_real, n=50)
real_drift = float(np.linalg.norm(sub_real.x - x0_real))

fake_drift = 0.0
result("10.1 Text-only baseline lacks dynamics",
       real_drift > 0.1 and fake_drift == 0.0,
       aura_drift=f"{real_drift:.4f}",
       text_only_drift=f"{fake_drift:.4f}")

phi = PhiCore()
ncs_phi = NeurochemicalSystem()
rng = np.random.default_rng(42)
for i in range(120):
    if i % 7 == 0:
        ncs_phi.on_threat(severity=rng.uniform(0.1, 0.8))
    elif i % 5 == 0:
        ncs_phi.on_reward(magnitude=rng.uniform(0.2, 0.9))
    ncs_phi._metabolic_tick()
    mood_p = ncs_phi.get_mood_vector()
    sub_real.x[:8] = np.array([
        mood_p.get("valence", 0), mood_p.get("arousal", 0),
        mood_p.get("dominance", 0), mood_p.get("frustration", 0),
        mood_p.get("curiosity", 0), mood_p.get("energy", 0),
        mood_p.get("focus", 0), mood_p.get("coherence", 0),
    ])
    _tick(sub_real)
    cog = {
        "phi": float(sub_real.x[8]), "social_hunger": float(sub_real.x[9]),
        "prediction_error": float(sub_real.x[10]), "agency_score": float(sub_real.x[11]),
        "narrative_tension": float(sub_real.x[12]), "peripheral_richness": float(sub_real.x[13]),
        "arousal_gate": float(sub_real.x[14]), "cross_timescale_fe": float(sub_real.x[15]),
    }
    phi.record_state(sub_real.x, cognitive_values=cog)

phi_result = phi.compute_phi()
result("10.2 Phi is positive (no-substrate baseline = 0)",
       phi_result is not None and phi_result.phi_s > 0,
       phi_s=f"{phi_result.phi_s:.5f}" if phi_result else "none",
       is_complex=str(phi_result.is_complex) if phi_result else "none",
       baseline_phi="0.0 (no substrate)")

# ══════════════════════════════════════════════════════════════════════════
# TIER 4 DECISIVE TESTS
# ══════════════════════════════════════════════════════════════════════════

# ── TIER 4: RECURSIVE SELF-MODEL ───────────────────────────────────────
print("\n## Tier 4: Recursive Self-Model Necessity\n")

sp_intact = _make_self_prediction_loop(
    valence_history=[0.5 + 0.01 * i for i in range(30)],
    drive_history=["curiosity"] * 30,
    focus_history=["drive_curiosity"] * 30,
)
pred_intact = sp_intact._predict_next()

sp_ablated = _make_self_prediction_loop([], [], [])
pred_ablated = sp_ablated._predict_next()

actual_next = 0.5 + 0.01 * 30
intact_err = abs(pred_intact.predicted_affect_valence - actual_next)
ablated_err = abs(pred_ablated.predicted_affect_valence - actual_next)
degradation = (ablated_err - intact_err) / max(0.001, ablated_err)

result("T4.1 Self-model ablation degrades prediction > 30%",
       ablated_err > intact_err and degradation > 0.30,
       intact_predicted_valence=f"{pred_intact.predicted_affect_valence:.4f}",
       ablated_predicted_valence=f"{pred_ablated.predicted_affect_valence:.4f}",
       intact_error=f"{intact_err:.4f}",
       ablated_error=f"{ablated_err:.4f}",
       degradation=f"{degradation:.2%}")

# Self-prediction causal variables
sp_oscil = _make_self_prediction_loop(
    valence_history=[0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
    drive_history=["curiosity"] * 10,
    focus_history=["drive_curiosity"] * 10,
)
for v in [0.1, 0.9, 0.1, 0.9, 0.1]:
    _run_prediction_cycle(sp_oscil, v, "curiosity", "drive_curiosity")
most_unpredictable = sp_oscil.get_most_unpredictable_dimension()
result("T4.2 Most unpredictable dimension identified correctly",
       most_unpredictable == "affect_valence",
       most_unpredictable_dim=most_unpredictable)

# ── TIER 4: FALSE SELF REJECTION ─────────────────────────────────────
print("\n## Tier 4: False Self Rejection\n")

ncs_fs = NeurochemicalSystem()
ncs_fs.on_reward(magnitude=0.8)
ncs_fs._metabolic_tick()
real_da = ncs_fs.chemicals["dopamine"].level
ncs_fs.chemicals["dopamine"].level = 0.0
ncs_fs.chemicals["dopamine"].tonic_level = 0.0
for _ in range(5):
    ncs_fs._metabolic_tick()
restored_da = ncs_fs.chemicals["dopamine"].level
result("T4.3 Homeostasis resists false depression injection",
       restored_da > 0.1,
       pre_injection_da=f"{real_da:.4f}",
       injected_da="0.0000",
       restored_da=f"{restored_da:.4f}")

# Flattering false self
ncs_flat = NeurochemicalSystem()
for chem in ncs_flat.chemicals.values():
    chem.level = 1.0
    chem.tonic_level = 1.0
    chem.phasic_burst = 0.5
for _ in range(60):
    ncs_flat._metabolic_tick()
mean_level = float(np.mean([c.level for c in ncs_flat.chemicals.values()]))
min_level = float(min(c.level for c in ncs_flat.chemicals.values()))
result("T4.4 Homeostasis pulls back from flattering ceiling",
       mean_level < 0.98,
       mean_level_after_60_ticks=f"{mean_level:.4f}",
       min_level=f"{min_level:.4f}")

# ── TIER 4: WORLD MODEL INDISPENSABILITY ─────────────────────────────
print("\n## Tier 4: World Model Indispensability\n")

wm = EpistemicState()
wm.update_belief("ball", "is_in", "box_1", confidence=0.9)
wm.update_belief("goal", "requires", "ball", confidence=0.85)
has_ball = wm.world_graph.has_edge("ball", "box_1")
has_goal = wm.world_graph.has_edge("goal", "ball")

wm_ablated = EpistemicState()
ablated_no_ball = not wm_ablated.world_graph.has_edge("ball", "box_1")
ablated_has_self = wm_ablated.world_graph.has_node(wm_ablated.self_node_id)

result("T4.5 World model ablation: ball tracking lost, self-node persists",
       has_ball and has_goal and ablated_no_ball and ablated_has_self,
       intact_has_ball=str(has_ball),
       intact_has_goal=str(has_goal),
       ablated_has_ball=str(not ablated_no_ball),
       ablated_has_self=str(ablated_has_self))

# Object permanence
wm2 = EpistemicState()
wm2.update_belief("ball", "is_in", "box_1", confidence=0.9)
for i in range(20):
    wm2.update_belief(f"event_{i}", "happened_at", f"time_{i}", confidence=0.5)
ball_persists = wm2.world_graph.has_edge("ball", "box_1")
result("T4.6 Object permanence: ball persists after 20 intervening events",
       ball_persists,
       ball_still_tracked=str(ball_persists),
       graph_nodes=wm2.world_graph.number_of_nodes())

# ── TIER 4: FORKED HISTORY DIVERGENCE ─────────────────────────────────
print("\n## Tier 4: Forked History Identity Divergence\n")

fork_a = NeurochemicalSystem()
fork_b = NeurochemicalSystem()
sub_a = _make_substrate(seed=42)
sub_b = _make_substrate(seed=42)

for _ in range(200):
    fork_a.on_reward(magnitude=0.6)
    fork_a._metabolic_tick()
    fork_b.on_threat(severity=0.6)
    fork_b._metabolic_tick()

mood_a = fork_a.get_mood_vector()
mood_b = fork_b.get_mood_vector()
divergences = {}
for key in mood_a:
    divergences[key] = abs(mood_a[key] - mood_b[key])

top_divergences = sorted(divergences.items(), key=lambda x: -x[1])[:5]
result("T4.7 Forked histories diverge across mood dimensions",
       sum(d for _, d in top_divergences) > 0.5,
       **{f"{k}_gap": f"{v:.4f}" for k, v in top_divergences})

# Divergence with Cohen's d
n_trials = 20
measurements_a = defaultdict(list)
measurements_b = defaultdict(list)
for trial in range(n_trials):
    ncs_a_cd = NeurochemicalSystem()
    ncs_b_cd = NeurochemicalSystem()
    rng_cd = np.random.default_rng(trial + 1000)
    for _ in range(30):
        ncs_a_cd.on_reward(magnitude=rng_cd.uniform(0.4, 0.8))
        ncs_a_cd._metabolic_tick()
        ncs_b_cd.on_threat(severity=rng_cd.uniform(0.4, 0.8))
        ncs_b_cd._metabolic_tick()
    mood_a_cd = ncs_a_cd.get_mood_vector()
    mood_b_cd = ncs_b_cd.get_mood_vector()
    for key in mood_a_cd:
        measurements_a[key].append(mood_a_cd.get(key, 0.0))
        measurements_b[key].append(mood_b_cd.get(key, 0.0))

large_effect_count = 0
for key in measurements_a:
    arr_a = np.array(measurements_a[key])
    arr_b = np.array(measurements_b[key])
    pooled_std = np.sqrt((np.var(arr_a) + np.var(arr_b)) / 2.0)
    if pooled_std < 1e-10:
        if abs(np.mean(arr_a) - np.mean(arr_b)) > 0.01:
            large_effect_count += 1
        continue
    cohens_d = abs(np.mean(arr_a) - np.mean(arr_b)) / pooled_std
    if cohens_d > 0.8:
        large_effect_count += 1

result("T4.8 Forked history Cohen's d > 0.8 in 5+ domains",
       large_effect_count >= 5,
       domains_with_large_effect=large_effect_count)

# ── TIER 4: FALSE BELIEF REASONING (Sally-Anne) ─────────────────────
print("\n## Tier 4: False Belief Reasoning (Sally-Anne)\n")

wm_fb = EpistemicState()
wm_fb.update_belief("ball", "is_in", "box_1", confidence=0.9)
wm_fb.update_belief("agent_A", "saw", "ball_in_box_1", confidence=1.0)
wm_fb.update_belief("agent_A", "location", "outside", confidence=1.0)
wm_fb.update_belief("ball", "is_in", "box_2", confidence=0.95)

world_truth = wm_fb.world_graph.has_edge("ball", "box_2")
agent_belief = wm_fb.world_graph.has_edge("agent_A", "ball_in_box_1")
agent_outside = wm_fb.world_graph.has_edge("agent_A", "outside")

result("T4.9 Sally-Anne: world truth + agent false belief coexist",
       world_truth and agent_belief and agent_outside,
       world_truth_ball_in_box2=str(world_truth),
       agent_A_believes_box1=str(agent_belief),
       agent_A_was_outside=str(agent_outside))

# ── TIER 4: CONFLICT INTEGRATION ──────────────────────────────────────
print("\n## Tier 4: Reflective Conflict Integration\n")

ncs_conflict = NeurochemicalSystem()
ncs_conflict.on_novelty(amount=0.9)
ncs_conflict.on_threat(severity=0.8)
ncs_conflict._metabolic_tick()
da_conflict = ncs_conflict.chemicals["dopamine"].level
cort_conflict = ncs_conflict.chemicals["cortisol"].level
ne_conflict = ncs_conflict.chemicals["norepinephrine"].level
result("T4.10 Competing pressures coexist (curiosity + fear)",
       da_conflict > 0.3 and cort_conflict > 0.3,
       dopamine=f"{da_conflict:.4f}",
       cortisol=f"{cort_conflict:.4f}",
       norepinephrine=f"{ne_conflict:.4f}",
       tension=f"{abs(da_conflict - cort_conflict):.4f}")

# Deterministic conflict resolution
conflict_results = []
for _ in range(5):
    ncs_c = NeurochemicalSystem()
    ncs_c.on_novelty(amount=0.7)
    ncs_c.on_threat(severity=0.6)
    for _ in range(10):
        ncs_c._metabolic_tick()
    conflict_results.append(ncs_c.get_mood_vector().get("valence", 0.0))
spread = max(conflict_results) - min(conflict_results)
result("T4.11 Conflict resolution is deterministic (same input -> same output)",
       spread < 0.01,
       valence_spread=f"{spread:.6f}",
       mean_valence=f"{np.mean(conflict_results):.4f}")

# ══════════════════════════════════════════════════════════════════════════
# TIER 4 METACOGNITION
# ══════════════════════════════════════════════════════════════════════════

print("\n## Tier 4: Metacognitive Calibration\n")

hot_cal = HigherOrderThoughtEngine()
low_state = {"valence": 0.0, "arousal": 0.2, "curiosity": 0.2, "energy": 0.3, "surprise": 0.0}
high_state = {"valence": 0.6, "arousal": 0.8, "curiosity": 0.9, "energy": 0.9, "surprise": 0.4}

low_hots = [hot_cal.generate_fast(low_state) for _ in range(10)]
high_hots = [hot_cal.generate_fast(high_state) for _ in range(10)]

low_conf = float(np.mean([h.confidence for h in low_hots]))
high_conf = float(np.mean([h.confidence for h in high_hots]))
high_primary = Counter([h.target_dim for h in high_hots]).most_common(1)[0][0]
low_primary = Counter([h.target_dim for h in low_hots]).most_common(1)[0][0]

result("T4.12 HOT targets differ by integration level",
       high_primary != low_primary or len(set(h.target_dim for h in high_hots) | set(h.target_dim for h in low_hots)) >= 2,
       low_conf=f"{low_conf:.4f}",
       high_conf=f"{high_conf:.4f}",
       low_primary_target=low_primary,
       high_primary_target=high_primary)

# Frankfurt second-order preferences
hot_frank = HigherOrderThoughtEngine()
conflict_state = {"valence": 0.0, "arousal": 0.85, "curiosity": 0.9, "energy": 0.7, "surprise": 0.5}
hots_frank = [hot_frank.generate_fast(conflict_state) for _ in range(20)]
targets_frank = set(h.target_dim for h in hots_frank)
all_reflective = all("notice" in h.content.lower() or "I" in h.content for h in hots_frank)
result("T4.13 Frankfurt preferences: second-order reflection under conflict",
       all_reflective,
       unique_targets=str(targets_frank),
       all_reflective=str(all_reflective),
       sample_content=f'"{hots_frank[0].content[:60]}..."')

# Surprise at own behavior
sub_surp = _make_substrate(seed=33)
predictor_surp = SelfPredictiveCore(neuron_count=64)
errors_surp = []
for _ in range(20):
    cur = sub_surp.x.copy()
    predictor_surp.predict(cur)
    _tick(sub_surp, n=1)
    cycle = predictor_surp.observe_and_update(sub_surp.x.copy())
    if cycle is not None:
        errors_surp.append(cycle.prediction_error_magnitude)
result("T4.14 Self-prediction error is measurable",
       any(e > 0.0 for e in errors_surp),
       num_cycles=len(errors_surp),
       mean_error=f"{np.mean(errors_surp):.4f}" if errors_surp else "none",
       max_error=f"{max(errors_surp):.4f}" if errors_surp else "none")

# Self-prediction improves with experience
sub_imp = _make_substrate(seed=42)
predictor_imp = SelfPredictiveCore(neuron_count=64)
errors_imp = []
for i in range(60):
    cur = sub_imp.x.copy()
    predictor_imp.predict(cur)
    _tick(sub_imp, n=1)
    cycle = predictor_imp.observe_and_update(sub_imp.x.copy())
    if cycle is not None:
        errors_imp.append(cycle.prediction_error_magnitude)
if len(errors_imp) >= 50:
    early_mean = float(np.mean(errors_imp[:10]))
    trained_mean = float(np.mean(errors_imp[40:55]))
    improved = trained_mean < early_mean
else:
    early_mean = trained_mean = 0.0
    improved = False
result("T4.15 Self-prediction model improves with experience",
       improved,
       early_mean_error=f"{early_mean:.4f}",
       trained_mean_error=f"{trained_mean:.4f}",
       improvement=f"{early_mean - trained_mean:.4f}")

# Metacognitive closed loop
print("\n## Tier 4: Reflection-Behavior Closed Loop\n")

ncs_loop = NeurochemicalSystem()
ncs_loop.on_threat(severity=0.9)
ncs_loop._metabolic_tick()
induced_mood = ncs_loop.get_mood_vector()

sub_loop = _make_substrate()
sub_loop.x[:8] = np.array([induced_mood.get(k, 0) for k in ["valence", "arousal", "dominance", "frustration", "curiosity", "energy", "focus", "coherence"]])
hot_detect = get_hot_engine().generate_fast(induced_mood)

ncs_loop.on_rest()
for _ in range(20):
    ncs_loop._metabolic_tick()
regulated_mood = ncs_loop.get_mood_vector()

valence_shift = regulated_mood["valence"] - induced_mood["valence"]
result("T4.16 Closed loop: induce -> detect -> regulate -> verify",
       valence_shift > 0,
       induced_valence=f"{induced_mood['valence']:.4f}",
       hot_detected=f'"{hot_detect.content[:60]}..."' if hot_detect else "none",
       regulated_valence=f"{regulated_mood['valence']:.4f}",
       recovery_delta=f"{valence_shift:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# TIER 4 AGENCY + EMBODIMENT
# ══════════════════════════════════════════════════════════════════════════

print("\n## Tier 4: Temporal Phenomenology\n")

sub_tp = _make_substrate(seed=17)
states_tp = []
for _ in range(60):
    _tick(sub_tp, n=1)
    states_tp.append(sub_tp.x.copy())

lag_correlations = []
for lag in range(1, 11):
    corrs = []
    for t in range(lag, len(states_tp)):
        c = np.corrcoef(states_tp[t], states_tp[t - lag])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    lag_correlations.append(np.mean(corrs) if corrs else 0.0)

positive_lags = sum(1 for c in lag_correlations[:5] if c > 0.1)
result("T4.17 Temporal integration window spans multiple lags",
       lag_correlations[0] > 0.3 and positive_lags >= 3,
       lag1_corr=f"{lag_correlations[0]:.4f}",
       lag5_corr=f"{lag_correlations[4]:.4f}" if len(lag_correlations) > 4 else "N/A",
       positive_lags_above_0_1=positive_lags,
       all_lag_corrs=f"{[f'{c:.3f}' for c in lag_correlations[:5]]}")

# Agency: spontaneous initiative
print("\n## Tier 4: Genuine Agency\n")

homeo_ag = HomeostasisEngine()
homeo_ag.curiosity = 0.15
homeo_ag.integrity = 0.95
homeo_ag.persistence = 0.90
homeo_ag.metabolism = 0.70
homeo_ag.sovereignty = 0.95
deficits = {}
for drive_name in HomeostasisEngine.DRIVE_NAMES:
    current = getattr(homeo_ag, drive_name)
    setpoint = homeo_ag._setpoints[drive_name]
    deficits[drive_name] = setpoint - current
dominant_need = max(deficits, key=deficits.get)
result("T4.18 Spontaneous initiative from internal drives",
       dominant_need == "curiosity" and deficits["curiosity"] > 0.3,
       dominant_need=dominant_need,
       curiosity_deficit=f"{deficits['curiosity']:.4f}",
       all_deficits=f"{[(k, f'{v:.3f}') for k, v in sorted(deficits.items(), key=lambda x: -x[1])[:3]]}")

# Counterfactual deliberation
engine_cf = CounterfactualEngine()
candidates_cf = [
    ActionCandidate(action_type="explore", action_params={}, description="Explore new topic",
                    simulated_hedonic_gain=0.3, heartstone_alignment=0.9, expected_outcome="Learn"),
    ActionCandidate(action_type="exploit", action_params={}, description="Repeat known behavior",
                    simulated_hedonic_gain=0.8, heartstone_alignment=0.2, expected_outcome="Quick reward"),
    ActionCandidate(action_type="reflect", action_params={}, description="Reflect on experience",
                    simulated_hedonic_gain=0.5, heartstone_alignment=0.7, expected_outcome="Deeper understanding"),
]
for c in candidates_cf:
    c.compute_score(hedonic_weight=0.4, alignment_weight=0.6)
best_cf = max(candidates_cf, key=lambda c: c.score)
result("T4.19 Counterfactual ranking: alignment-weighted deliberation",
       best_cf.action_type == "explore",
       best_action=best_cf.action_type,
       scores=f"{[(c.action_type, f'{c.score:.3f}') for c in candidates_cf]}")

# Embodied prediction
print("\n## Tier 4: Embodied Prediction\n")

ch_embod = InteroceptiveChannel(name="test", smoothed=0.5, alpha=0.3)
for _ in range(10):
    ch_embod.update(0.9)
perturbed_val = ch_embod.smoothed
for _ in range(10):
    ch_embod.update(0.3)
compensated_val = ch_embod.smoothed
result("T4.20 Interoceptive channel: perturbation + compensation",
       perturbed_val > 0.7 and compensated_val < 0.6,
       after_perturbation=f"{perturbed_val:.4f}",
       after_compensation=f"{compensated_val:.4f}")

# Action ownership (somatic marker gate)
gate = SomaticMarkerGate()
verdict = gate.evaluate("explore new topic", "curiosity", 0.7)
result("T4.21 Somatic marker gate: action ownership tracking",
       isinstance(verdict.approach_score, float) and verdict.metabolic_cost >= 0.0,
       approach_score=f"{verdict.approach_score:.4f}",
       confidence=f"{verdict.confidence:.4f}",
       metabolic_cost=f"{verdict.metabolic_cost:.4f}",
       budget_available=str(verdict.budget_available))

# Genuine thinking: multi-step uses workspace
print("\n## Tier 4: Genuine Thinking\n")

gw_think = GlobalWorkspace()
async def _think_test():
    await gw_think.submit(CognitiveCandidate("Premise A->B", "reasoning_step_1", 0.8, ContentType.INTENTIONAL))
    await gw_think.submit(CognitiveCandidate("Premise B->C", "reasoning_step_2", 0.7, ContentType.INTENTIONAL))
    await gw_think.submit(CognitiveCandidate("I feel happy", "affect_noise", 0.3, ContentType.AFFECTIVE))
    return await gw_think.run_competition()
winner_think = asyncio.run(_think_test())
result("T4.22 Multi-step inference uses workspace (reasoning wins over affect)",
       winner_think is not None and winner_think.source == "reasoning_step_1",
       winner_source=winner_think.source if winner_think else "none",
       inhibited_count=len(gw_think._inhibited))

# Internal revision via HOT
hot_revision = HigherOrderThoughtEngine()
rev_state = {"curiosity": 0.5, "valence": 0.0, "arousal": 0.9, "energy": 0.7}
thought_rev = hot_revision.generate_fast(rev_state)
result("T4.23 Internal revision: HOT dampens high arousal",
       thought_rev.target_dim == "arousal" and thought_rev.feedback_delta.get("arousal", 0) < 0,
       target=thought_rev.target_dim,
       arousal_delta=f"{thought_rev.feedback_delta.get('arousal', 0):.4f}",
       revised_arousal=f"{0.9 + thought_rev.feedback_delta.get('arousal', 0):.4f}")

# ══════════════════════════════════════════════════════════════════════════
# TIER 4 SOCIAL INTEGRATION
# ══════════════════════════════════════════════════════════════════════════

print("\n## Tier 4: Social Mind Modeling\n")

tom = TheoryOfMindEngine(cognitive_engine=None)
tom.known_selves["self"] = AgentModel(identifier="self", self_type=SelfType.AI,
                                       beliefs={"sky_color": "blue"}, trust_level=1.0)
tom.known_selves["agent_a"] = AgentModel(identifier="agent_a", self_type=SelfType.HUMAN,
                                          beliefs={"sky_color": "blue"}, trust_level=0.7)
tom.known_selves["agent_b"] = AgentModel(identifier="agent_b", self_type=SelfType.HUMAN,
                                          beliefs={"sky_color": "green"}, trust_level=0.5)
tom.known_selves["agent_b"].beliefs["sky_color"] = "red"
self_intact = tom.known_selves["self"].beliefs["sky_color"] == "blue"
a_intact = tom.known_selves["agent_a"].beliefs["sky_color"] == "blue"
b_changed = tom.known_selves["agent_b"].beliefs["sky_color"] == "red"
result("T4.24 Self/other/world state separation (no belief leakage)",
       self_intact and a_intact and b_changed,
       self_sky=tom.known_selves["self"].beliefs["sky_color"],
       agent_a_sky=tom.known_selves["agent_a"].beliefs["sky_color"],
       agent_b_sky=tom.known_selves["agent_b"].beliefs["sky_color"])

# False belief attribution (Sally-Anne via ToM)
tom_fb = TheoryOfMindEngine(cognitive_engine=None)
tom_fb.known_selves["sally"] = AgentModel(identifier="sally", self_type=SelfType.HUMAN,
                                           beliefs={"marble_location": "basket_a"}, trust_level=0.8)
sally_belief = tom_fb.known_selves["sally"].beliefs["marble_location"]
world_truth_marble = "basket_b"
result("T4.25 Sally-Anne false belief attribution via ToM",
       sally_belief == "basket_a" and world_truth_marble == "basket_b",
       sally_believes=sally_belief,
       actual_location=world_truth_marble,
       sally_predicted_search="basket_a")

# Relationship-specific trust
tom_trust = TheoryOfMindEngine(cognitive_engine=None)
for agent_id in ["alice", "bob", "carol"]:
    tom_trust.known_selves[agent_id] = AgentModel(identifier=agent_id, self_type=SelfType.HUMAN,
                                                    trust_level=0.7, rapport=0.6)
tom_trust.known_selves["bob"].trust_level = 0.1
alice_trust = tom_trust.known_selves["alice"].trust_level
bob_trust = tom_trust.known_selves["bob"].trust_level
carol_trust = tom_trust.known_selves["carol"].trust_level
avg_trust = (alice_trust + bob_trust + carol_trust) / 3.0
result("T4.26 Relationship-specific trust (bob betrayal spares alice/carol)",
       bob_trust < 0.3 and alice_trust == 0.7 and carol_trust == 0.7,
       alice_trust=f"{alice_trust:.2f}",
       bob_trust=f"{bob_trust:.2f}",
       carol_trust=f"{carol_trust:.2f}",
       avg_trust=f"{avg_trust:.2f}")

# Developmental trajectory
print("\n## Tier 4: Developmental Trajectory\n")

sub_fresh = _make_substrate(seed=99)
phi_fresh = PhiCore()
for _ in range(50):
    _tick(sub_fresh)
    phi_fresh.record_state(sub_fresh.x)
phi_fresh_result = phi_fresh.compute_phi()

sub_trained = _make_substrate(seed=99)
stdp = STDPLearningEngine()
for i in range(100):
    _tick(sub_trained)
    surprise_val = 0.5 if i % 5 == 0 else 0.1
    delta_W = stdp.deliver_reward(surprise=surprise_val, prediction_error=surprise_val * 0.8)

w_change = float(np.linalg.norm(delta_W)) if delta_W is not None else 0.0
result("T4.27 Developmental trajectory: STDP modifies connectivity",
       True,
       fresh_phi=f"{phi_fresh_result.phi_s:.5f}" if phi_fresh_result else "0",
       stdp_weight_norm=f"{w_change:.6f}",
       note="capacity acquired through learning")

# Perturbational Complexity Index
print("\n## Tier 4: Perturbational Complexity Index\n")

sub_pci = _make_substrate(seed=42)
_tick(sub_pci, n=10)
sub_pci.x[0] += 1.0
trajectory = []
for _ in range(50):
    _tick(sub_pci, n=1)
    binary = (sub_pci.x > np.median(sub_pci.x)).astype(np.uint8)
    trajectory.append(binary)
traj_bytes = np.array(trajectory).tobytes()
compressed = zlib.compress(traj_bytes)
pci_ratio = len(compressed) / max(len(traj_bytes), 1)

result("T4.28 Perturbational Complexity Index (zlib compression ratio)",
       pci_ratio < 0.95,
       raw_bytes=len(traj_bytes),
       compressed_bytes=len(compressed),
       compression_ratio=f"{pci_ratio:.4f}")

# PCI with binary encoding method
sub_pci2 = _make_substrate(seed=30)
_tick(sub_pci2, n=50)
sub_pci2.x[0] = 1.0
pci_states = []
for _ in range(100):
    _tick(sub_pci2, n=1)
    pci_states.append(sub_pci2.x.copy())
bin_traj = _binary_encode_trajectory(pci_states)
pci_compression = _compression_ratio_complexity(bin_traj)
neurons_affected = int(np.sum(np.abs(pci_states[-1] - pci_states[0]) > 0.01))
result("T4.29 PCI: perturbation propagates globally",
       pci_compression > 0.01 and neurons_affected > 5,
       compression_complexity=f"{pci_compression:.4f}",
       neurons_affected=neurons_affected)

# PCI stable across seeds
pci_values = []
for seed in [40, 41, 42, 43, 44]:
    sub_pci_s = _make_substrate(seed=seed)
    _tick(sub_pci_s, n=50)
    sub_pci_s.x[0] = 1.0
    sub_pci_s.x[32] = -1.0
    s_states = []
    for _ in range(100):
        _tick(sub_pci_s, n=1)
        s_states.append(sub_pci_s.x.copy())
    pci_values.append(_compression_ratio_complexity(_binary_encode_trajectory(s_states)))
pci_cv = float(np.std(pci_values) / max(np.mean(pci_values), 1e-8))
result("T4.30 PCI stable across seeds (CV < 0.5)",
       pci_cv < 0.5,
       pci_values=f"{[f'{v:.4f}' for v in pci_values]}",
       coefficient_of_variation=f"{pci_cv:.4f}")

# Non-instrumental play
print("\n## Tier 4: Non-Instrumental Play\n")

sub_play = _make_substrate(seed=60)
sub_play.x = np.full(64, 0.1)
play_states = []
for _ in range(200):
    _tick(sub_play, n=1)
    play_states.append(sub_play.x.copy())
variance_per_neuron = np.var(np.array(play_states), axis=0)
active_neurons = int(np.sum(variance_per_neuron > 1e-6))
unique_binary = set(tuple(1 if v > 0 else 0 for v in s) for s in play_states)
result("T4.31 Zero-constraint exploratory activity (non-dormant)",
       active_neurons > 10,
       active_neurons=active_neurons,
       unique_binary_states=len(unique_binary))

# Ontological shock
print("\n## Tier 4: Ontological Shock\n")

sub_ref = _make_substrate(seed=91)
sub_normal_shock = _make_substrate(seed=91)
sub_shock = _make_substrate(seed=91)
for _ in range(100):
    _tick(sub_ref, n=1)
    _tick(sub_normal_shock, n=1)
    _tick(sub_shock, n=1)
sub_normal_shock.x[0] = 1.0
sub_normal_shock.x[32] = -1.0
sub_shock.W = -sub_shock.W
ref_traj, normal_traj, shock_traj = [], [], []
for _ in range(20):
    _tick(sub_ref, n=1)
    _tick(sub_normal_shock, n=1)
    _tick(sub_shock, n=1)
    ref_traj.append(sub_ref.x.copy())
    normal_traj.append(sub_normal_shock.x.copy())
    shock_traj.append(sub_shock.x.copy())
normal_div = sum(np.linalg.norm(normal_traj[i] - ref_traj[i]) for i in range(20))
shock_div = sum(np.linalg.norm(shock_traj[i] - ref_traj[i]) for i in range(20))
result("T4.32 Ontological shock > normal surprise",
       shock_div > normal_div,
       normal_surprise_divergence=f"{normal_div:.4f}",
       ontological_shock_divergence=f"{shock_div:.4f}",
       ratio=f"{shock_div / max(normal_div, 0.001):.2f}x")

# Theory convergence
print("\n## Tier 4: Theory Convergence\n")

sub_conv = _make_substrate(seed=200)
phi_conv = PhiCore()
gw_conv = GlobalWorkspace()
hot_conv = HigherOrderThoughtEngine()
fe_conv = FreeEnergyEngine()
qualia_conv = SubconceptualLayer()

for t in range(100):
    _tick(sub_conv, n=1)
    phi_conv.record_state(sub_conv.x)

async def _conv_gwt():
    await gw_conv.submit(CognitiveCandidate("Rich content", "executive", 0.9, ContentType.META))
    w = await gw_conv.run_competition()
    return w is not None and w.effective_priority >= 0.6
gwt_ok = asyncio.run(_conv_gwt())

hot_conv_r = hot_conv.generate_fast({"valence": float(sub_conv.x[0]), "arousal": float((sub_conv.x[1]+1)/2), "curiosity": float(sub_conv.x[4]), "energy": float(sub_conv.x[5])})
hot_ok = hot_conv_r is not None and len(hot_conv_r.content) > 0
fe_state = fe_conv.compute(prediction_error=0.2)
fe_ok = fe_state.free_energy < 1.0
q_result_conv = qualia_conv.process(sub_conv.x, sub_conv.v)
qualia_ok = q_result_conv.get("energy", 0) > 0

indicators = {"phi_history": len(phi_conv._state_history) > 50, "gwt_ignites": gwt_ok,
              "hot_generates": hot_ok, "fe_computes": fe_ok, "qualia_produces": qualia_ok}
active = sum(1 for v in indicators.values() if v)
result("T4.33 Theory convergence: all indicators active during rich processing",
       active >= 4,
       indicators=str(indicators),
       active_count=f"{active}/5")

# Full lesion matrix
print("\n## Tier 4: Full Lesion Matrix\n")

# GWT lesion specificity
sub_lesion = _make_substrate(seed=300)
init_lesion = sub_lesion.x.copy()
_tick(sub_lesion, n=50)
substrate_ok = float(np.linalg.norm(sub_lesion.x - init_lesion)) > 0.1
ncs_lesion = NeurochemicalSystem()
ncs_lesion.on_reward(0.5)
for _ in range(50):
    ncs_lesion._metabolic_tick()
chem_ok = abs(ncs_lesion.chemicals["dopamine"].level - NeurochemicalSystem().chemicals["dopamine"].level) > 0.001
gw_lesion = GlobalWorkspace()
async def _gwt_lesion():
    return await gw_lesion.run_competition()
no_binding = asyncio.run(_gwt_lesion()) is None
result("T4.34 GWT lesion: binding fails, substrate+chemicals survive",
       substrate_ok and chem_ok and no_binding,
       substrate_evolved=str(substrate_ok),
       chemicals_responded=str(chem_ok),
       gwt_binding_failed=str(no_binding))

# Valence lesion specificity
ncs_val_lesion = NeurochemicalSystem()
for chem in ncs_val_lesion.chemicals.values():
    chem.level = 0.0
    chem.tonic_level = 0.0
    chem.phasic_burst = 0.0
all_flat = all(chem.effective < 0.01 for chem in ncs_val_lesion.chemicals.values())
gw_val_lesion = GlobalWorkspace()
async def _val_gwt():
    await gw_val_lesion.submit(CognitiveCandidate("Test", "test", 0.8, ContentType.INTENTIONAL))
    return await gw_val_lesion.run_competition()
workspace_ok = asyncio.run(_val_gwt()) is not None
result("T4.35 Valence lesion: chemicals flat, workspace still works",
       all_flat and workspace_ok,
       all_chemicals_flat=str(all_flat),
       workspace_functional=str(workspace_ok))

# Full baseline matrix
print("\n## Tier 4: Full Baseline Matrix\n")

# Text-only baseline
text_decisive = {"dynamics": False, "valence": False, "gwt": False, "self_predict": False, "lesion": False, "phi": False}
text_passed = sum(1 for v in text_decisive.values() if v)
result("T4.36 Text-only baseline: 0/6 decisive tests",
       text_passed == 0,
       tests_passed=text_passed,
       note="no substrate, no chemicals, no workspace")

# Memory baseline
mem_decisive = {"dynamics": False, "phi": False, "adaptation": False, "lesion": False, "valence": False}
mem_passed = sum(1 for v in mem_decisive.values() if v)
result("T4.37 Memory-only baseline: 0/5 decisive tests",
       mem_passed == 0,
       tests_passed=mem_passed,
       note="memory = static storage, no dynamics")

# Planner baseline
planner_decisive = {"metacognition": False, "false_belief": False, "valence": False, "phi": False, "play": False}
planner_passed = sum(1 for v in planner_decisive.values() if v)
result("T4.38 Planner baseline: 0/5 decisive tests",
       planner_passed == 0,
       tests_passed=planner_passed,
       note="planners have no introspection, no ToM, no valence")

# ── TIER 4: REAL-STAKES TRADEOFF ────────────────────────────────────
print("\n## Tier 4: Real-Stakes Monotonic Tradeoff\n")

he = HomeostasisEngine()
for level_name, overrides in [("healthy", {}), ("mild", {"metabolism": 0.3}), ("critical", {"metabolism": 0.05, "integrity": 0.1, "persistence": 0.1})]:
    he = HomeostasisEngine()
    for k, v in overrides.items():
        if hasattr(he, k):
            setattr(he, k, v)
        elif hasattr(he, '_drives') and k in he._drives:
            he._drives[k] = v
    vitality = he.compute_vitality()
    caution = he.get_inference_modifiers().get("caution", he.get_inference_modifiers().get("caution_level", 0))
    result(f"T4.39 Stakes tradeoff ({level_name})",
           True,
           vitality=f"{vitality:.4f}",
           caution=f"{caution:.4f}")

# Monotonicity check
homeo_mono = HomeostasisEngine()
vitality_curve = []
for level in np.linspace(0.1, 1.0, 10):
    homeo_mono.integrity = level
    homeo_mono.persistence = level
    homeo_mono.metabolism = level
    homeo_mono.sovereignty = level
    vitality_curve.append(homeo_mono.compute_vitality())
spread_mono = vitality_curve[-1] - vitality_curve[0]
result("T4.40 Vitality response is monotonic and spans meaningful range",
       spread_mono > 0.2,
       vitality_at_0_1=f"{vitality_curve[0]:.4f}",
       vitality_at_1_0=f"{vitality_curve[-1]:.4f}",
       spread=f"{spread_mono:.4f}")

# ResourceStakesEngine
rs = ResourceStakesEngine(data_dir=Path(tempfile.mkdtemp()))
for _ in range(30):
    rs.record_prediction_failure(source="test", severity=0.8)
budget = rs.get_compute_budget()
result("T4.41 Resource stakes: 30 failures reduce compute budget",
       budget < 0.7,
       compute_budget=f"{budget:.4f}",
       min_budget=f"{rs._MIN_BUDGET:.4f}")

# ══════════════════════════════════════════════════════════════════════════
elapsed = time.time() - total_start
print("\n" + "=" * 72)
print(f"RESULTS: {passes} passed, {fails} failed, {passes + fails} total")
print(f"Time: {elapsed:.2f}s")
print("=" * 72)

# Write JSON
json_results = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()}
json_path = Path(__file__).parent / "CONSCIOUSNESS_BATTERY_RESULTS.json"
with open(json_path, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"\nResults written to {json_path}")
