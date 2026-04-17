2026-04-16 17:04:04,392 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
========================================================================
AURA CONSCIOUSNESS BATTERY — MEASURED RESULTS
Run: 2026-04-17 00:04:04 UTC
========================================================================

## C1: Continuous Endogenous Activity

  [PASS] 1.1 Substrate idle drift (100 ticks, no input)
         L2_drift=5.1576
  [PASS] 1.2 Neurochemical drift without stimulus
         total_mood_delta=0.337694
  [PASS] 1.3 Different histories produce different states
         L2_divergence=0.4574

## C2: Unified Global State

  [PASS] 2.1 Workspace competition resolves to single winner
         winner_source=drive  winner_priority=0.80  candidates=3

## C3: Privileged First-Person Access

  [PASS] 3.1 HOT generates state-dependent thought under threat
         hot_content="I notice high arousal — I am activated, alert, engaged...."  target_dim=arousal  valence=-0.0828  arousal=0.3980
  [PASS] 3.2 HOT generates different thought under reward
         hot_content="I notice I am in a positive state — something feels right...."  target_dim=valence  valence=0.3491

## C4: Real Valence

  [PASS] 4.1 Opposite chemicals produce opposite valence
         threat_valence=-0.0966  reward_valence=0.3625  gap=0.4591
  [PASS] 4.2 Valence modulates temperature
         threat_temp=0.719  reward_temp=0.627  delta=0.092

## C5: Lesion Equivalence (Double Dissociations)

  [PASS] 5.1 Workspace ablation: competition stops
         intact_winner=s1  ablated_winner=none (no GWT = no binding)
  [PASS] 5.2 Phi ablation: focus bias disappears
         phi0_focus_bias=0.0000  phi08_focus_bias=0.0800  boost=0.0800
  [PASS] 5.3 Chemical ablation: threat-driven valence vs zeroed
         intact_valence_under_threat=-0.0966  zeroed_valence=-0.1000  note=intact system responds to threat; zeroed stays near baseline
  [PASS] 5.4 HOT ablation: metacognition disappears
         with_hot="I notice I am in a positive state — something feels right...."  without_hot=no HOT = no metacognitive thought generated

## C6: No-Report Awareness

  [PASS] 6.1 Substrate processes input without any report channel
         state_change=1.8675
  [PASS] 6.2 Chemicals respond to stimulus without narration
         cortisol_level=0.7098  note=no text output requested

## C7: Temporal Self-Continuity

  [PASS] 7.1 State carries temporal history (autocorrelation)
         lag1_correlation=0.9970  lag10_correlation=0.7668
  [PASS] 7.2 Affective tone carries over across ticks
         immediate_valence=-0.0828  after_10_ticks_valence=-0.0561

## C8: Blindsight-Style Dissociation

  [PASS] 8.1 Substrate processes without metacognitive access
         substrate_norm=1.8314  hot_available=True  note=substrate dynamics persist even if HOT is not invoked

## C9: Qualia Manifold Geometry

  [PASS] 9.1 Similar states → similar qualia, different states → different qualia
         dist_similar=1.2580  dist_different=11.0723  ratio=8.80x
  [PASS] 9.2 Qualia descriptor dimensionality
         total_dimensions=17  layers=subconceptual + conceptual + predictive + workspace + witness  phenomenal_richness_A=0.3478  phenomenal_richness_C=0.4869

## C10: Adversarial Baseline Failure

  [PASS] 10.1 Text-only baseline lacks dynamics
         aura_drift=3.7123  text_only_drift=0.0000
  [FAIL] 10.2 Phi is positive (no-substrate baseline = 0)
         phi_s=0.00000  is_complex=False  baseline_phi=0.0 (no substrate)

## Tier 4: Forked History Identity Divergence

  [PASS] T4.1 Forked histories diverge across mood dimensions
         valence_gap=0.6536  stress_gap=0.5105  calm_gap=0.4776  sociality_gap=0.4447  arousal_gap=0.3073

## Tier 4: Perturbational Complexity Index

  [PASS] T4.2 Perturbational Complexity Index (LZ compression ratio)
         raw_bytes=3200  compressed_bytes=143  compression_ratio=0.0447  interpretation=mid-range = complex, not trivial or random

## Tier 4: Real-Stakes Monotonic Tradeoff

  [PASS] T4.3 Stakes tradeoff (healthy)
         vitality=0.8554  caution=0.0000
  [PASS] T4.3 Stakes tradeoff (mild)
         vitality=0.8189  caution=0.0000
  [PASS] T4.3 Stakes tradeoff (critical)
         vitality=0.2699  caution=0.0000

## Tier 4: Reflection-Behavior Closed Loop

  [PASS] T4.4 Closed loop: induce → detect → regulate → verify
         induced_valence=-0.0966  hot_detected="I notice my valence is neutral — neither drawn nor repelled...."  regulated_valence=0.0289  recovery_delta=0.1254

## Tier 4: Developmental Trajectory (Capacity is Acquired)

  [PASS] T4.5 Fresh substrate vs trained: STDP modifies connectivity
         fresh_phi=0  stdp_weight_updates=accumulated over 100 ticks  note=capacity is acquired through learning, not static

========================================================================
RESULTS: 27 passed, 1 failed, 28 total
Time: 5.06s
========================================================================

Results written to /Users/bryan/.aura/live-source/tests/CONSCIOUSNESS_BATTERY_RESULTS.json
