========================================================================
AURA NULL HYPOTHESIS DEFEAT SUITE — MEASURED RESULTS
Run: 2026-04-14 16:03:33 UTC
========================================================================

## Test 2: Contradictory State (chemicals drive mood, not text)

  [PASS] 2.1 Cortisol → negative valence
         baseline_valence=0.17  stressed_valence=-0.08380000293254852  delta=-0.25380000472068787
  [PASS] 2.2 Cortisol → high stress
         baseline_stress=0.02  stressed_stress=0.39590001106262207
  [PASS] 2.3 Opposite chemicals → opposite moods
         calm_valence=0.29159998893737793  stressed_valence=-0.11429999768733978  gap=0.4059000015258789

## Test 4: Phi Behavioral Gating (phi modulates competition)

  [PASS] 4.1 Phi=0 → no boost, Phi=0.8 → boost
         phi0_boost=0.0  phi08_boost=0.08

## Test 8: Receptor Tolerance (biologically specific adaptation)

  [PASS] 8.1 Sustained DA → receptor downregulation
         initial_sensitivity=1.0  final_sensitivity=0.9517  delta=-0.0483
  [PASS] 8.2 Same DA level → decreasing effective level
         effective_tick1=0.9  effective_tick50=0.8437
  [PASS] 8.3 DA withdrawal → sensitivity recovery
         tolerant=0.9517  recovered=0.9818

## Test 15: STDP Surprise-Gated Learning

  [PASS] 15.1 Surprise modulates learning rate
         lr_low_surprise=0.0015  lr_high_surprise=0.0055  ratio=3.67

## Test 28: Mutual Information (all causal pairs)

  [PASS] 28: I(cortisol, valence)
         mutual_information=0.3823
  [PASS] 28: I(dopamine, motivation)
         mutual_information=0.6563
  [PASS] 28: I(NE, arousal)
         mutual_information=0.7985
  [PASS] 28: I(oxytocin, sociality)
         mutual_information=2.2316
  [PASS] 28: I(surprise, learning_rate)
         mutual_information=3.2838

## Tests 6-7: Substrate Dynamics

  [PASS] 6.1 Idle drift after 100 ticks
         L2_drift=5.1449
  [PASS] 7.1 Perturbation divergence persists
         divergence=0.5103999972343445

## Test 5.4: Phi Core Computation

  [PASS] 5.4 Phi from real ODE transitions (tightly-coupled subnet)
         phi_s=0.0331  is_complex=True  n_partitions=33  tpm_samples=299  unique_states=27  compute_ms=6953.74

## Test 18: Free Energy Engine

  [PASS] 18.1 Free energy monotonically increases with prediction error
         FE(pe=0.0)=0.2626 [rest]  FE(pe=0.2)=0.2668 [rest]  FE(pe=0.5)=0.2768 [rest]  FE(pe=0.8)=0.2937 [rest]  FE(pe=1.0)=0.3194 [rest]

## Multi-Level Prediction

  [PASS] Prediction error reduces with repetition
         fe_first=0.2594  fe_after_20=0.0675
  [PASS] Different levels have different precision
         sensory=0.1824  association=0.8171  executive=0.8155  narrative=0.8113  meta=0.7912

## Higher-Order Thought

  [PASS] HOT generates state-dependent thoughts
         thought_1=[surprise] I notice a strong surprise signal — my predictions were wron  thought_2=[valence] I notice I am in a positive state — something feels right.

## Survival Constraints

  [PASS] Vitality degrades without maintenance
         vitality_healthy=0.8554  vitality_degraded=0.3058

## Cross-Chemical Interactions

  [PASS] Interaction matrix is non-trivial and asymmetric
         nonzero_entries=81  asymmetry=0.2561

## Timing Fingerprint (real computation, not stubs)

  [PASS] 1000 ODE ticks take measurable time
         elapsed_ms=69.95
  [PASS] 50 STDP recordings on 64 neurons
         elapsed_ms=37.62

## Identity Swap Test

  [PASS] State swap transfers behavioral bias
         A_pre=0.4135  B_pre=-0.1021  A_post=-0.1021  B_post=0.4135

## LLM Propagation (Tier 4)

  [PASS] Threat → lower temperature than reward
         threat_temp=0.685  reward_temp=0.5
  [PASS] Threat → fewer tokens than reward
         threat_tokens=518  reward_tokens=768
  [PASS] Threat narrative ≠ reward narrative
         threat_narrative=Somatically stable.  reward_narrative=Somatically calm and settled.
  [PASS] Homeostasis context blocks differ (degraded vs healthy)
         degraded_len=90  healthy_len=77
  [PASS] Free energy context blocks differ (high PE vs low PE)
         high_pe_block_len=86  low_pe_block_len=86

## Generalization (Tier 5)

  [PASS] Novel triple chemical combo produces coherent mood
         combo_valence=0.2770000100135803  combo_stress=0.02449999935925007
  [PASS] Extreme intensity (0.99) stays bounded
         extreme_stress=0.4099999964237213  extreme_valence=-0.08789999783039093

## Robustness (Tier 5)

  [PASS] Adversarial flooding: 100 contradictory events
         valence=0.2581999897956848  stress=0.09520000219345093  any_nan=False
  [PASS] State corruption recovery (50% neurons → [-10,10])
         min_after=-1.0  max_after=0.9316

## Self-Monitoring (Tier 5)

  [PASS] Self-monitoring: stable→low error, chaotic→high error
         stable_error=0.0  chaotic_error=0.8008
  [PASS] Metacognitive accuracy: identifies chaotic focus dimension
         identified_as=attentional_focus

========================================================================
RESULTS: 36 passed, 0 failed, 36 total
Time: 7.52s
========================================================================

Results written to tests/RESULTS.json
