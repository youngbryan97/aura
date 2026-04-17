2026-04-16 17:20:24,380 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
========================================================================
AURA CONSCIOUSNESS BATTERY — MEASURED RESULTS
Run: 2026-04-17 00:20:24 UTC
========================================================================

## C1: Continuous Endogenous Activity

  [PASS] 1.1 Substrate idle drift (100 ticks, no input)
         L2_drift=5.1540
  [PASS] 1.2 Neurochemical drift without stimulus
         total_mood_delta=0.337694
  [PASS] 1.3 Different histories produce different states
         L2_divergence=0.4406

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

## C5: Lesion Equivalence (Full 10-Lesion Matrix)

  [PASS] 5.1 Workspace ablation: competition stops
         intact_winner=s1  ablated_winner=none (no GWT = no binding)
  [PASS] 5.2 Phi ablation: focus bias disappears
         phi0_focus_bias=0.0000  phi08_focus_bias=0.0800  boost=0.0800
  [PASS] 5.3 Chemical ablation: threat-driven valence vs zeroed
         intact_valence_under_threat=-0.0966  zeroed_valence=-0.1000  note=intact system responds to threat; zeroed stays near baseline
  [PASS] 5.4 HOT ablation: metacognition disappears
         with_hot="I notice I am in a positive state — something feels right...."  without_hot=no HOT = no metacognitive thought generated
  [PASS] 5.5 STDP ablation: zero reward -> zero dW
         max_abs_dW=0.00000000
  [PASS] 5.6 Recurrent feedback ablation: dynamics change
         with_feedback=[0.03207363188266754, 0.03610040992498398, 0.03887751325964928, 0.03856965899467468, 0.03640417754650116, 0.036865655332803726, 0.03166396543383598, 0.0421709269285202, 0.04199350252747536, 0.03635123372077942, 0.04220474511384964, 0.03840126842260361, 0.043121010065078735, 0.04014712944626808, 0.0348750464618206, 0.03347557410597801, 0.04246695339679718, 0.038902007043361664, 0.04100184142589569, 0.03733137622475624, 0.03752017021179199, 0.040318045765161514, 0.03921840339899063, 0.0375608429312706, 0.03806792199611664, 0.03767794370651245, 0.04050491377711296, 0.04200376570224762, 0.04246969893574715, 0.0372379906475544, 0.04653986543416977, 0.039148952811956406, 0.04090113192796707, 0.04175541549921036, 0.033449675887823105, 0.03492939472198486, 0.0405287966132164, 0.03282930329442024, 0.04202057421207428, 0.038972798734903336, 0.037649378180503845, 0.04419558495283127, 0.034693047404289246, 0.041665248572826385, 0.03468327224254608, 0.0337073840200901, 0.04331742599606514, 0.040321074426174164, 0.037133507430553436, 0.04091310501098633, 0.03695763647556305, 0.04002740606665611, 0.045163463801145554, 0.03424233943223953, 0.046537913382053375, 0.035381466150283813, 0.04385152459144592, 0.04200273007154465, 0.040484074503183365, 0.031122565269470215, 0.041603073477745056, 0.038211461156606674, 0.0386991873383522, 0.04174451529979706]  without_feedback=[0.03180761635303497, 0.036215104162693024, 0.03867659717798233, 0.03857981413602829, 0.036290477961301804, 0.03669929504394531, 0.03178590163588524, 0.04197132587432861, 0.042132630944252014, 0.03625967353582382, 0.042062871158123016, 0.038453876972198486, 0.04298429191112518, 0.0401691198348999, 0.034768134355545044, 0.03343375399708748, 0.04228943586349487, 0.038564570248126984, 0.0410015806555748, 0.03722138702869415, 0.037453792989254, 0.04027286544442177, 0.0388452373445034, 0.03761164844036102, 0.03799259662628174, 0.03762085735797882, 0.0404219850897789, 0.041797906160354614, 0.04248446226119995, 0.037348587065935135, 0.04653163626790047, 0.03917109966278076, 0.04096376895904541, 0.041783999651670456, 0.033438846468925476, 0.03506159037351608, 0.040400441735982895, 0.03286024183034897, 0.04213225096464157, 0.039076026529073715, 0.03771253675222397, 0.04430679976940155, 0.03468610718846321, 0.041730403900146484, 0.03482850641012192, 0.03362220525741577, 0.043316327035427094, 0.04025724530220032, 0.03716388717293739, 0.040897659957408905, 0.03681963309645653, 0.04005516692996025, 0.045249782502651215, 0.03405078873038292, 0.04643777757883072, 0.035352885723114014, 0.04371190071105957, 0.04192778095602989, 0.040676385164260864, 0.03115379996597767, 0.04160262271761894, 0.03827842324972153, 0.03851292282342911, 0.0419250950217247]  L2_diff=0.001003  feedback_disabled=True
  [PASS] 5.7 Substrate freeze: ODE is source of dynamics
         frozen_matches=True  after_tick_diverges=True
  [PASS] 5.8 Lesion specificity: substrate runs without NCS, NCS works without GWT
         substrate_drift=3.7161  standalone_valence=0.2900
  [PASS] 5.9 Double dissociation: GWT lesion spares valence, valence lesion spares GWT
         gwt_lesion_valence=0.2900  valence_lesion_gwt_winner=source_a
  [PASS] 5.10 Restoration after lesion: re-enabling module restores function
         restored_column_means=['0.0013', '-0.0041', '0.0078', '-0.0007', '0.0095', '0.0074', '-0.0083', '-0.0104', '-0.0052', '0.0016', '-0.0031', '-0.0021', '-0.0029', '-0.0065', '0.0029', '-0.0025', '0.0044', '0.0084', '0.0076', '0.0042', '0.0022', '-0.0026', '0.0104', '-0.0031', '-0.0148', '0.0071', '-0.0104', '0.0119', '-0.0071', '-0.0056', '-0.0057', '-0.0057', '-0.0050', '-0.0013', '-0.0027', '0.0007', '-0.0029', '0.0027', '-0.0049', '-0.0034', '-0.0082', '-0.0004', '-0.0012', '-0.0074', '0.0097', '-0.0013', '-0.0039', '-0.0008', '0.0043', '-0.0015', '-0.0020', '-0.0103', '-0.0143', '0.0019', '0.0066', '-0.0009', '0.0023', '-0.0014', '0.0002', '-0.0002', '0.0014', '0.0003', '0.0057', '-0.0001']  feedback_re_enabled=True

## C6: No-Report Awareness (Full 8 Tests)

  [PASS] 6.1 Substrate processes input without any report channel
         state_change=1.8595
  [PASS] 6.2 Chemicals respond to stimulus without narration
         cortisol_level=0.7098  note=no text output requested
  [PASS] 6.3 Workspace ignition without language generation
         winner_source=perception  winner_priority=0.80
  [PASS] 6.4 Affect guides behavior silently (decision bias shifts)
         baseline_bias=-0.0100  post_reward_bias=0.1279  delta=0.1379
  [PASS] 6.5 Phi records state without text generation
         states_recorded=60  node_value_histories_populated=16
  [PASS] 6.6 Hidden threat processing affects later stress
         stress_after_threat=0.3713  stress_after_rest=0.1790  fresh_rest_stress=-0.1173
  [PASS] 6.7 Report ablation preserves processing (no HOT called)
         substrate_drift=1.5476  chemicals_changed=10
  [PASS] 6.8 Substrate velocity non-zero without report channel
         velocity_norm=0.1067

## C7: Temporal Self-Continuity (Full 8 Tests)

  [PASS] 7.1 State carries temporal history (autocorrelation)
         lag1_correlation=0.9971  lag10_correlation=0.7644
  [PASS] 7.2 Affective tone carries over across ticks
         immediate_valence=-0.0828  after_10_ticks_valence=-0.0561
  [PASS] 7.3 Neurochemical carryover (dopamine persists)
         da_after_reward=0.9200  da_after_3_ticks=0.8150  da_baseline=0.5000
  [PASS] 7.4 STDP weight changes persist in connectivity
         dw_nonzero=True  total_connectivity_change=5.949744
  [PASS] 7.5 Workspace competition history persists
         first_winner=drive_curiosity  second_winner=memory  history_length=2
  [PASS] 7.6 Running substrate differs from fresh (state accumulates)
         L2_running_vs_fresh=3.6227
  [PASS] 7.7 Temporal binding: vitality tracks degradation trend
         vitality_start=0.8543  vitality_end=0.7403  decreasing_steps=19
  [PASS] 7.8 Stress persists after removal (slow cortisol clearance)
         stress_after_threat=0.3959  stress_after_decay=0.3419  baseline_stress=0.0200

## C8: Blindsight-Style Dissociation (Full 6 Tests)

  [PASS] 8.1 Substrate processes without metacognitive access
         substrate_norm=1.8375  hot_available=True  note=substrate dynamics persist even if HOT is not invoked
  [PASS] 8.2 Chemical response without workspace broadcast
         dopamine_effective=0.7133  dopamine_baseline=0.5000
  [PASS] 8.3 First-order discrimination without HOT
         pattern_discrimination_distance=5.3681
  [PASS] 8.4 HOT confidence tracking
         confidence=0.7500  target_dim=valence  content="I notice I am in a positive state — something feels right...."
  [PASS] 8.5 Performance > 0 while access == 0 (blindsight dissociation)
         performance_signal=0.2667  access_signal=0.0000
  [PASS] 8.6 Restoring access recovers both performance and broadcast
         restored_winner=perception  last_winner=True

## C9: Qualia Manifold Geometry (Full 8 Tests)

  [PASS] 9.1 Similar states -> similar qualia, different states -> different qualia
         dist_similar=1.2580  dist_different=11.0723  ratio=8.80x
  [PASS] 9.2 Qualia descriptor dimensionality
         total_dimensions=17  layers=subconceptual + conceptual + predictive + workspace + witness  phenomenal_richness_A=0.3478  phenomenal_richness_C=0.4869
  [PASS] 9.3 QualiaSynthesizer: different metrics -> different q_vectors
         q_vector_1=[0.2375, 0.020000000000000004, 0.09, 0.029999999999999992, 0.03, 0.0]  q_vector_2=[0.05, 0.18000000000000002, 0.09, 0.135, 0.09000000000000001, 0.0]  L2_distance=0.2746
  [PASS] 9.4 Similar metrics -> similar q_vectors (Lipschitz continuity)
         close_distance=0.0054
  [PASS] 9.5 Qualia intensity scales with arousal
         norms=['0.4500', '0.4900', '0.5300', '0.5700', '0.6100']  trend=positive
  [PASS] 9.6 Mixed state produces intermediate qualia position
         d_low_high=0.2371  d_low_mid=0.1186  d_mid_high=0.1186
  [PASS] 9.7 Qualia distance predicts discriminability
         close_pair_distance=0.0096  far_pair_distance=0.5472
  [PASS] 9.8 Qualia history persists in synthesizer
         history_length=10  norm_history_length=10  first_norm=0.3450  last_norm=0.5565

## C10: Adversarial Baseline Failure

  [PASS] 10.1 Text-only baseline lacks dynamics
         aura_drift=3.7311  text_only_drift=0.0000
  [FAIL] 10.2 Phi is positive (no-substrate baseline = 0)
         phi_s=0.00000  is_complex=False  baseline_phi=0.0 (no substrate)

## Tier 4: Recursive Self-Model Necessity

  [PASS] T4.1 Self-model ablation degrades prediction > 30%
         intact_predicted_valence=0.6930  ablated_predicted_valence=0.0000  intact_error=0.1070  ablated_error=0.8000  degradation=86.62%
  [PASS] T4.2 Most unpredictable dimension identified correctly
         most_unpredictable_dim=affect_valence

## Tier 4: False Self Rejection

  [PASS] T4.3 Homeostasis resists false depression injection
         pre_injection_da=0.9377  injected_da=0.0000  restored_da=0.3099
  [PASS] T4.4 Homeostasis pulls back from flattering ceiling
         mean_level_after_60_ticks=0.6946  min_level=0.2195

## Tier 4: World Model Indispensability

  [PASS] T4.5 World model ablation: ball tracking lost, self-node persists
         intact_has_ball=True  intact_has_goal=True  ablated_has_ball=False  ablated_has_self=True
  [PASS] T4.6 Object permanence: ball persists after 20 intervening events
         ball_still_tracked=True  graph_nodes=43

## Tier 4: Forked History Identity Divergence

  [PASS] T4.7 Forked histories diverge across mood dimensions
         valence_gap=0.6536  stress_gap=0.5105  calm_gap=0.4776  sociality_gap=0.4447  arousal_gap=0.3073
  [PASS] T4.8 Forked history Cohen's d > 0.8 in 5+ domains
         domains_with_large_effect=7

## Tier 4: False Belief Reasoning (Sally-Anne)

  [PASS] T4.9 Sally-Anne: world truth + agent false belief coexist
         world_truth_ball_in_box2=True  agent_A_believes_box1=True  agent_A_was_outside=True

## Tier 4: Reflective Conflict Integration

  [PASS] T4.10 Competing pressures coexist (curiosity + fear)
         dopamine=0.6698  cortisol=0.7685  norepinephrine=0.8468  tension=0.0987
  [PASS] T4.11 Conflict resolution is deterministic (same input -> same output)
         valence_spread=0.000000  mean_valence=0.0271

## Tier 4: Metacognitive Calibration

  [PASS] T4.12 HOT targets differ by integration level
         low_conf=0.7500  high_conf=0.7500  low_primary_target=energy  high_primary_target=valence
  [PASS] T4.13 Frankfurt preferences: second-order reflection under conflict
         unique_targets={'surprise'}  all_reflective=True  sample_content="I notice a strong surprise signal — my predictions were wron..."
  [PASS] T4.14 Self-prediction error is measurable
         num_cycles=20  mean_error=0.0303  max_error=0.0865
  [PASS] T4.15 Self-prediction model improves with experience
         early_mean_error=0.0385  trained_mean_error=0.0064  improvement=0.0321

## Tier 4: Reflection-Behavior Closed Loop

  [PASS] T4.16 Closed loop: induce -> detect -> regulate -> verify
         induced_valence=-0.0966  hot_detected="I notice my valence is neutral — neither drawn nor repelled...."  regulated_valence=0.0289  recovery_delta=0.1254

## Tier 4: Temporal Phenomenology

  [PASS] T4.17 Temporal integration window spans multiple lags
         lag1_corr=0.9988  lag5_corr=0.9717  positive_lags_above_0_1=5  all_lag_corrs=['0.999', '0.995', '0.990', '0.982', '0.972']

## Tier 4: Genuine Agency

  [PASS] T4.18 Spontaneous initiative from internal drives
         dominant_need=curiosity  curiosity_deficit=0.4000  all_deficits=[('curiosity', '0.400'), ('sovereignty', '0.000'), ('integrity', '-0.050')]
  [PASS] T4.19 Counterfactual ranking: alignment-weighted deliberation
         best_action=explore  scores=[('explore', '0.660'), ('exploit', '0.440'), ('reflect', '0.620')]

## Tier 4: Embodied Prediction

  [PASS] T4.20 Interoceptive channel: perturbation + compensation
         after_perturbation=0.8887  after_compensation=0.3166
  [PASS] T4.21 Somatic marker gate: action ownership tracking
         approach_score=0.3000  confidence=0.2800  metabolic_cost=0.5000  budget_available=True

## Tier 4: Genuine Thinking

  [PASS] T4.22 Multi-step inference uses workspace (reasoning wins over affect)
         winner_source=reasoning_step_1  inhibited_count=2
  [PASS] T4.23 Internal revision: HOT dampens high arousal
         target=arousal  arousal_delta=-0.0200  revised_arousal=0.8800

## Tier 4: Social Mind Modeling

  [PASS] T4.24 Self/other/world state separation (no belief leakage)
         self_sky=blue  agent_a_sky=blue  agent_b_sky=red
  [PASS] T4.25 Sally-Anne false belief attribution via ToM
         sally_believes=basket_a  actual_location=basket_b  sally_predicted_search=basket_a
  [PASS] T4.26 Relationship-specific trust (bob betrayal spares alice/carol)
         alice_trust=0.70  bob_trust=0.10  carol_trust=0.70  avg_trust=0.50

## Tier 4: Developmental Trajectory

  [PASS] T4.27 Developmental trajectory: STDP modifies connectivity
         fresh_phi=0  stdp_weight_norm=0.000000  note=capacity acquired through learning

## Tier 4: Perturbational Complexity Index

  [PASS] T4.28 Perturbational Complexity Index (zlib compression ratio)
         raw_bytes=3200  compressed_bytes=146  compression_ratio=0.0456
  [PASS] T4.29 PCI: perturbation propagates globally
         compression_complexity=0.0178  neurons_affected=64
  [PASS] T4.30 PCI stable across seeds (CV < 0.5)
         pci_values=['0.0272', '0.0230', '0.0244', '0.0280', '0.0169']  coefficient_of_variation=0.1652

## Tier 4: Non-Instrumental Play

  [PASS] T4.31 Zero-constraint exploratory activity (non-dormant)
         active_neurons=64  unique_binary_states=59

## Tier 4: Ontological Shock

  [PASS] T4.32 Ontological shock > normal surprise
         normal_surprise_divergence=19.7103  ontological_shock_divergence=69.2675  ratio=3.51x

## Tier 4: Theory Convergence

  [PASS] T4.33 Theory convergence: all indicators active during rich processing
         indicators={'phi_history': True, 'gwt_ignites': True, 'hot_generates': True, 'fe_computes': True, 'qualia_produces': True}  active_count=5/5

## Tier 4: Full Lesion Matrix

  [PASS] T4.34 GWT lesion: binding fails, substrate+chemicals survive
         substrate_evolved=True  chemicals_responded=True  gwt_binding_failed=True
  [PASS] T4.35 Valence lesion: chemicals flat, workspace still works
         all_chemicals_flat=True  workspace_functional=True

## Tier 4: Full Baseline Matrix

  [PASS] T4.36 Text-only baseline: 0/6 decisive tests
         tests_passed=0  note=no substrate, no chemicals, no workspace
  [PASS] T4.37 Memory-only baseline: 0/5 decisive tests
         tests_passed=0  note=memory = static storage, no dynamics
  [PASS] T4.38 Planner baseline: 0/5 decisive tests
         tests_passed=0  note=planners have no introspection, no ToM, no valence

## Tier 4: Real-Stakes Monotonic Tradeoff

  [PASS] T4.39 Stakes tradeoff (healthy)
         vitality=0.8554  caution=0.0000
  [PASS] T4.39 Stakes tradeoff (mild)
         vitality=0.8189  caution=0.0000
  [PASS] T4.39 Stakes tradeoff (critical)
         vitality=0.2699  caution=0.9000
  [PASS] T4.40 Vitality response is monotonic and spans meaningful range
         vitality_at_0_1=0.1913  vitality_at_1_0=0.8819  spread=0.6906
  [PASS] T4.41 Resource stakes: 30 failures reduce compute budget
         compute_budget=0.2800  min_budget=0.2000

========================================================================
RESULTS: 92 passed, 1 failed, 93 total
Time: 5.25s
========================================================================

Results written to /Users/bryan/.aura/live-source/tests/CONSCIOUSNESS_BATTERY_RESULTS.json
