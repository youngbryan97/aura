2026-04-14 12:05:56,824 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
========================================================================
AURA CAUSAL EXCLUSION + PHENOMENAL CONVERGENCE SUITE — MEASURED RESULTS
Run: 2026-04-14 19:05:57 UTC
========================================================================

## Causal Exclusion: Cryptographic State Binding

  [PASS] Param diversity: temperature std
         temperature_std=0.0123  temperature_range=[0.500, 0.540]
  [PASS] Param diversity: token budget std
         token_std=22.4  token_range=[657, 748]
  [PASS] Mood vector pairwise distance
         mean_dist=0.2907  min_dist=0.0175  max_dist=0.7989
  [PASS] Temperature tracks arousal
         calm_temp=0.5  excited_temp=0.718  delta=0.218

## Causal Exclusion: Counterfactual Injection

  [PASS] State distance predicts param distance
         pearson_r=0.9408  p_value=0.0  mean_state_dist=0.257  mean_param_dist=0.0539

## Causal Exclusion: Receptor Adaptation (Temporal Dynamics)

  [PASS] Receptor adaptation attenuates DA
         da_fresh=0.8605999946594238  da_saturated=0.6776999831199646  attenuation_pct=21.299999237060547

## Grounding: Multi-Dimensional Specificity

  [PASS] Valence->tokens correlation
         pearson_r=0.9999  p_value=0.0

## Grounding: STDP Trajectory Divergence

  [PASS] STDP trajectory divergence
         divergence=0.2994

## Embodied: Free Energy Active Inference

  [PASS] FE scales with prediction error
         fe_low_pe=0.263  fe_high_pe=0.2787  urgency_low=0.2641
  [PASS] Action urgency scales with FE
         urgency_high_pe=0.3136

## Embodied: Homeostatic Override

  [PASS] Critical depletion changes inference
         healthy_caution=0.1  critical_caution=0.95  healthy_vitality=0.877  critical_vitality=0.457

## Phenomenology: GWT Broadcast Signatures

  [PASS] GWT broadcast reaches processors
         winner_content=test insight  processor_received=True

## Phenomenology: HOT Meta-Cognitive Accuracy

  [PASS] HOT produces state-specific thoughts
         curious_dim=valence  stressed_dim=arousal  curious_hot=I notice I am in a positive state — something feels right.  stressed_hot=I notice high arousal — I am activated, alert, engaged.

## Convergence: Pre-Report Quality Space

  [PASS] Quality space has categorical structure
         within_mean=0.2771  between_mean=0.3817  separation_ratio=1.377
  [PASS] Quality space is multi-dimensional
         pc1_var=0.9017  pc2_var=0.0861  pc3_var=0.0075

## Convergence: Perturbational Integration

  [PASS] Perturbation complexity (intact system)
         complexity=0.3243
  [PASS] Intact vs shuffled divergence
         final_state_divergence=4.9437

## Convergence: Phenomenal Tethering

  [PASS] Phi=0 removes GWT boost (anesthesia)
         awake_priority=0.68  anesthetized_priority=0.6

## Convergence: UnifiedWill Decision Gating

  [PASS] Will produces valid decisions
         outcome=proceed  receipt=will_a98e1736512

========================================================================
RESULTS: 19 passed, 0 failed, 19 total
Time: 0.07s
========================================================================

Results written to tests/CAUSAL_EXCLUSION_RESULTS.json
