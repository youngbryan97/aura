# Intrinsic Subject Core — scorecard

15 hold, 2 unresolved, 7 fail across 3 runs. Per-run totals [17, 17, 15] of 24.

## The campaign

- commit `670f3e955a85` (clean tree)
- tree hash `0266ba017d9522f18ee589991b115203`
- fingerprint `8874aea9613abbbb2953c25bbc19284e`
- seeds [7, 11, 13] — 3 independent initialisations
- 60 rounds over 8 conditions; 6 trials, 2 turns per arm, delta 0.15

## Criteria

| | criterion | held | § | needs |
|---|---|---|---|---|
| ✓ | `causal_closure_scc` | 3/3 | 14 | one component covering all domains |
| ✗ | `robust_recurrence_kappa` | 0/3 | 17 | >= 2 |
| ✓ | `cycles_per_domain` | 3/3 | 15 | >= 2 each |
| ✓ | `reentry` | 3/3 | 16 | cycle length >= 3 |
| ✗ | `partition_irreducibility` | 0/3 | 11 | lower bound > 0.05 |
| ✗ | `partition_beats_nulls` | 0/3 | 18 | above all nulls |
| ✓ | `differentiation` | 3/3 | 18 | d_eff >= 3, top share < 0.5, ratio < 0.4 |
| ✓ | `differentiation_above_floor` | 3/3 | 18 | d_eff >= 3 and no component above half the variance |
| ~ | `intrinsic_persistence` | 2/3 | 19 | > 0, and above the shuffled-state control |
| ✓ | `causal_closure_of_the_core` | 3/3 | 3 | no improvement on K beyond what a shuffled periphery gives |
| ✗ | `perturbational_spread` | 0/3 | 24 | >= 0.6 |
| ✓ | `perturbational_complexity` | 3/3 | 26 | a response that reached somewhere, above the 99th percentile of matched nulls |
| ✗ | `synergy` | 0/3 | 27 | >= 0.1 and above its shifted null |
| ✓ | `metastability` | 3/3 | 29 | more than one regime, entropy strictly between zero and its ceiling |
| ✓ | `global_access` | 3/3 | 30 | >= 3 |
| ✓ | `recurrent_global_access` | 3/3 | 31 | at least one G -> X -> G path |
| ✓ | `self_drives_action` | 3/3 | 33 | above the same-arm floor |
| ✓ | `ownership` | 3/3 | 34 | above the same-arm floor |
| ✓ | `fast_to_slow` | 3/3 | 38 | at least one edge from a fast domain into a slow one |
| ✓ | `slow_to_fast` | 3/3 | 38 | at least one edge from a slow domain into a fast one |
| ✓ | `natural_runtime_replication` | 3/3 | 44 | strongly connected in at least three conditions |
| ✗ | `lesion_deficit` | 0/3 | 39 | irreducibility, complexity and synergy all fall |
| ✗ | `rescue` | 0/3 | 40 | a deficit first, then the measures return towards the intact values |
| ~ | `beats_every_null` | 2/3 | 41 | no null passes the conjunction |

## The numbers, across runs

| quantity | values | median | spread |
|---|---|---|---|
| phi_do | [0.00107, 0.01831, -0.03034] | 0.00107 | 0.04865 |
| d_eff_normalised | [0.0843, 0.0851, 0.0911] | 0.0851 | 0.0068 |
| intrinsic | [0.5153, 0.53353, 0.0012] | 0.5153 | 0.53233 |
| spread | [0.53333, 0.54444, 0.55556] | 0.54444 | 0.02222 |
| pci | [0.19718, 0.19601, 0.21461] | 0.19718 | 0.0186 |
| surrogate_floor | [0.02037, 0.02069, 0.01149] | 0.02037 | 0.0092 |
| phi_lower_bound | [-0.0274, -0.01482, -0.08321] | -0.0274 | 0.06838 |
| phi_standard_error | [0.01452, 0.01691, 0.02697] | 0.01691 | 0.01245 |
| d_eff | [12.0528, 11.7505, 13.2116] | 12.0528 | 1.4611 |
| largest_component_share | [0.2495, 0.254, 0.2329] | 0.2495 | 0.0211 |
| vertex_connectivity | [1.0, 1.0, 1.0] | 1.0 | 0.0 |
| edges_kept | [41.0, 43.0, 43.0] | 43.0 | 2.0 |
| lesion_deficit_phi | [0.0, 0.0, 0.0] | 0.0 | 0.0 |
| lesion_deficit_spread | [0.26543, 0.19753, 0.33951] | 0.26543 | 0.14198 |
| lesion_deficit_synergy | [0.0, 0.0, 0.0] | 0.0 | 0.0 |
| rescue_phi | [0.0, 0.0, 0.0] | 0.0 | 0.0 |
| ownership | [1.4161, 621054827684.2852, 1.3326] | 1.4161 | 621054827682.9525 |
| ownership_floor | [0.0066, 0.04065, 0.03255] | 0.03255 | 0.03405 |
| self_to_action | [0.0916, 0.1255, 0.16825] | 0.1255 | 0.07665 |
| global_access_consumers | [8.0, 8.0, 8.0] | 8.0 | 0.0 |
| metastability_regimes | [2.0, 3.0, 6.0] | 3.0 | 4.0 |
| closure_leak | [0.0, 0.0, 0.0] | 0.0 | 0.0 |

## Edges that survived every bar

| edge | kept in | effect across runs |
|---|---|---|
| `P->A` | 3/3 | 10.00, 10.00, 10.00 |
| `W->N` | 3/3 | 5.01, 4.66, 6.64 |
| `A->S` | 3/3 | 3.43, 6.46, 4.21 |
| `C->A` | 3/3 | 5.90, 5.57, 4.90 |
| `S->N` | 3/3 | 5.84, 5.58, 3.32 |
| `S->W` | 3/3 | 5.69, 4.76, 2.21 |
| `S->A` | 3/3 | 4.96, 4.84, 1.07 |
| `I->D` | 3/3 | 4.05, 2.50, 0.55 |
| `W->A` | 3/3 | 3.76, 2.93, 3.76 |
| `G->P` | 3/3 | 3.10, 3.38, 3.29 |
| `C->G` | 3/3 | 3.25, 2.81, 3.05 |
| `C->S` | 3/3 | 1.66, 3.05, 1.72 |
| `S->C` | 3/3 | 1.85, 2.22, 0.71 |
| `C->I` | 3/3 | 0.56, 2.14, 1.23 |
| `A->G` | 3/3 | 1.90, 1.73, 2.08 |
| `G->S` | 3/3 | 1.80, 2.02, 1.60 |
| `I->N` | 3/3 | 1.88, 1.28, 1.48 |
| `G->W` | 3/3 | 1.52, 1.76, 1.80 |
| `W->C` | 3/3 | 1.51, 1.47, 1.74 |
| `G->C` | 3/3 | 1.15, 1.74, 1.29 |
| `S->P` | 3/3 | 1.67, 1.34, 0.65 |
| `I->A` | 3/3 | 1.64, 0.95, 0.59 |
| `M->G` | 3/3 | 1.58, 1.60, 1.57 |
| `P->N` | 3/3 | 1.11, 1.07, 1.46 |
| `D->N` | 3/3 | 1.11, 1.14, 1.37 |
| `W->G` | 3/3 | 1.22, 0.59, 0.74 |
| `I->P` | 3/3 | 1.19, 1.08, 0.84 |
| `G->M` | 3/3 | 0.71, 0.85, 1.19 |
| `G->N` | 3/3 | 1.00, 0.89, 1.15 |
| `G->A` | 3/3 | 0.98, 1.02, 0.76 |
| `C->P` | 3/3 | 0.94, 0.76, 0.73 |
| `I->C` | 3/3 | 0.80, 0.53, 0.40 |
| `G->D` | 3/3 | 0.71, 0.70, 0.79 |
| `A->C` | 3/3 | 0.52, 0.64, 0.64 |
| `N->A` | 3/3 | 0.42, 0.49, 0.64 |
| `S->I` | 3/3 | 0.31, 0.32, 0.34 |
| `A->I` | 2/3 | 0.00, 8.96, 2.82 |
| `S->G` | 2/3 | 1.90, 1.46, 0.00 |
| `P->S` | 2/3 | 0.13, 0.73, 0.37 |
| `M->W` | 2/3 | 0.62, 0.52, 0.27 |
| `W->S` | 2/3 | 0.60, 0.39, 0.41 |
| `N->C` | 2/3 | 0.25, 0.36, 0.43 |
| `P->C` | 2/3 | 0.23, 0.37, 0.41 |
| `P->W` | 2/3 | 0.36, 0.35, 0.23 |
| `I->G` | 1/3 | 0.38, 0.22, 0.14 |
| `D->A` | 1/3 | 0.24, 0.26, 0.36 |
| `M->S` | 1/3 | 0.25, 0.23, 0.36 |

## The pairs closest to carrying

| pair | effect | q | conditions carrying |
|---|---|---|---|
| `I->S` | 0.50 | 0.0000 | stress |
| `M->C` | 0.44 | 0.0000 | idle, memory |
| `W->S` | 0.39 | 0.0000 | conversation, stress |
| `P->G` | 0.38 | 0.0000 | idle, stress |
| `D->W` | 0.31 | 0.0000 | autonomy, tool_use |
| `D->C` | 0.29 | 0.0000 | autonomy |
| `D->G` | 0.27 | 0.0000 | autonomy, stress |
| `M->W` | 0.27 | 0.0000 | idle |
| `D->A` | 0.26 | 0.0000 | stress |
| `M->S` | 0.25 | 0.0000 | autonomy, idle, tool_use |
| `N->C` | 0.25 | 0.0000 | conversation, problem_solving |
| `P->C` | 0.23 | 0.0000 | idle |
