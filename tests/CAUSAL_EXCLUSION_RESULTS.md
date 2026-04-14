2026-04-14 12:10:30,595 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Integrity breach: Severity high reported. Current: 0.85
Integrity breach: Severity medium reported. Current: 0.80
Integrity breach: Severity medium reported. Current: 0.75
Integrity breach: Severity medium reported. Current: 0.70
Integrity breach: Severity medium reported. Current: 0.65
Integrity breach: Severity medium reported. Current: 0.60
========================================================================
AURA CAUSAL EXCLUSION + PHENOMENAL CONVERGENCE SUITE — MEASURED RESULTS
Run: 2026-04-14 19:10:31 UTC
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
         divergence=0.3004

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

## Causal Exclusion: Narrative Diversity

  [PASS] Circumplex produces diverse narratives
         distinct_narratives=4  examples=['Somatically stable.', 'Somatically comfortable and engaged.', 'Somatically calm and settled.']

## Causal Exclusion: State Reversal

  [PASS] State reversal: positive -> more tokens
         pos_mean_tokens=756.0  neg_mean_tokens=650.0

## Causal Exclusion: Extreme State Param Divergence

  [PASS] Extreme states diverge from human baseline
         mean_param_divergence=0.5640000104904175

## Grounding: Substrate Idle Drift

  [PASS] Substrate idle drift (100 ticks)
         L2_drift=6.0075

## Grounding: Homeostasis Context Block

  [PASS] Homeostasis degradation changes context
         healthy_vitality=0.855  degraded_vitality=0.297

## Grounding: FE Prediction Error Response

  [PASS] FE responds to prediction error
         fe_low=0.263  fe_high=0.2719  action_low=rest  action_high=rest

## Embodied: Error Compounding

  [PASS] Errors compound integrity loss
         initial=1.0  after_1=0.85  after_6=0.6

## Embodied: STDP Surprise Ratio

  [PASS] STDP surprise ratio
         low_surprise_change=0.00705  high_surprise_change=0.18579  ratio=26.35

## Embodied: Cross-Subsystem Coherence

  [PASS] Threat vs reward produce different cascades
         threat_valence=-0.07000000029802322  reward_valence=0.2720000147819519  threat_stress=0.34200000762939453  reward_stress=0.02500000037252903

## Phenomenology: GWT Emotion Competition

  [PASS] Different emotions win over noise
         winners=['curiosity', 'anxiety', 'excitement']

## Phenomenology: HOT Feedback Loop

  [PASS] HOT produces feedback deltas
         target_dim=surprise  deltas={'curiosity': 0.05, 'arousal': 0.03}

## Phenomenology: IIT Perturbation Propagation

  [PASS] Perturbation propagates to other neurons
         neurons_affected=62  total_neurons=64

## Phenomenology: Shuffled Connectivity Divergence

  [PASS] Shuffled W produces different trajectory
         divergence=6.9201

## Convergence: Counterfactual State Transfer

  [PASS] State transfer carries behavioral bias
         transferred_valence=-0.18700000643730164  neg_source_valence=-0.18700000643730164  pos_source_valence=0.49399998784065247  dist_to_neg_source=0.0010000000474974513  dist_to_pos_source=0.6800000071525574

## Convergence: Baselines Fail

  [PASS] Real NCS has stronger valence-stress structure
         real_corr=-0.9326000213623047  random_corr=0.0569

## Convergence: Zero Connectivity Degeneracy

  [PASS] Zero W produces degenerate dynamics
         divergence_vs_real=2.3342

## Convergence: Full Stack vs Single Subsystem

  [PASS] Full stack effective dimensionality
         full_stack_edim=1.31  ncs_only_edim=1.29

## Convergence: Multi-Theory Indicators Present

  [PASS] All 6 theory indicators present
         GWT=True  IIT=True  HOT=True  PP=True  Embodied=True  Will=True

## Convergence: Pre-Report Quality Space

  [PASS] Quality space has categorical structure
         within_mean=0.2771  between_mean=0.3817  separation_ratio=1.377
  [PASS] Quality space is multi-dimensional
         pc1_var=0.9017  pc2_var=0.0861  pc3_var=0.0075

## Convergence: Perturbational Integration

  [PASS] Perturbation complexity (intact system)
         complexity=0.3238
  [PASS] Intact vs shuffled divergence
         final_state_divergence=4.928

## Convergence: Phenomenal Tethering

  [PASS] Phi=0 removes GWT boost (anesthesia)
         awake_priority=0.68  anesthetized_priority=0.6

## Convergence: UnifiedWill Decision Gating

  [PASS] Will produces valid decisions
         outcome=proceed  receipt=will_0f8917a20c6

========================================================================
RESULTS: 37 passed, 0 failed, 37 total
Time: 0.21s
========================================================================

Results written to tests/CAUSAL_EXCLUSION_RESULTS.json
