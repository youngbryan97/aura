# Intrinsic Subject Core — scorecard

17 hold, 2 unresolved, 5 fail across 3 runs. Per-run totals [18, 19, 18] of 24.

## The campaign

- commit `45a74c9130db` (clean tree)
- tree hash `ae901290fe41dc99a76b5af811fd8402`
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
| ✓ | `lesion_deficit` | 3/3 | 39 | irreducibility, complexity and synergy all fall |
| ✓ | `rescue` | 3/3 | 40 | a deficit first, then the measures return towards the intact values |
| ~ | `beats_every_null` | 2/3 | 41 | no null passes the conjunction |

## The numbers, across runs

| quantity | values | median | spread |
|---|---|---|---|
| phi_do | [-0.03154, 0.00316, -0.02468] | -0.02468 | 0.0347 |
| d_eff_normalised | [0.091, 0.0889, 0.0853] | 0.0889 | 0.0057 |
| intrinsic | [-0.44016, 0.66841, 0.53451] | 0.53451 | 1.10857 |
| spread | [0.55556, 0.56667, 0.53333] | 0.55556 | 0.03333 |
| pci | [0.20221, 0.20775, 0.20009] | 0.20221 | 0.00766 |
| surrogate_floor | [0.03268, 0.02275, 0.02549] | 0.02549 | 0.00993 |
| phi_lower_bound | [-0.09947, -0.00247, -0.04932] | -0.04932 | 0.097 |
| phi_standard_error | [0.03466, 0.00288, 0.01257] | 0.01257 | 0.03178 |
| d_eff | [13.0108, 12.1746, 11.5978] | 12.1746 | 1.413 |
| largest_component_share | [0.2339, 0.2456, 0.2503] | 0.2456 | 0.0164 |
| vertex_connectivity | [1.0, 1.0, 1.0] | 1.0 | 0.0 |
| edges_kept | [48.0, 47.0, 44.0] | 47.0 | 4.0 |
| lesion_deficit_phi | [-0.15436, 0.00273, 0.0561] | 0.00273 | 0.21046 |
| lesion_deficit_spread | [0.14198, 0.37654, 0.29012] | 0.29012 | 0.23456 |
| lesion_deficit_synergy | [-0.00243, 0.09413, 0.05212] | 0.05212 | 0.09656 |
| rescue_phi | [0.08857, -0.00514, 0.07896] | 0.07896 | 0.09371 |
| ownership | [1.27605, 3.13435, 1.07465] | 1.27605 | 2.0597 |
| ownership_floor | [0.01435, 0.0099, 0.0] | 0.0099 | 0.01435 |
| self_to_action | [0.1209, 0.0772, 0.08845] | 0.08845 | 0.0437 |
| global_access_consumers | [7.0, 8.0, 7.0] | 7.0 | 1.0 |
| metastability_regimes | [2.0, 2.0, 6.0] | 2.0 | 4.0 |
| closure_leak | [0.0, 0.0, 0.0] | 0.0 | 0.0 |

## Edges that survived every bar

| edge | kept in | effect across runs |
|---|---|---|
| `P->A` | 3/3 | 10.00, 10.00, 10.00 |
| `A->I` | 3/3 | 2.90, 7.65, 8.96 |
| `S->N` | 3/3 | 7.22, 2.75, 7.64 |
| `W->N` | 3/3 | 5.97, 6.68, 7.38 |
| `A->S` | 3/3 | 3.00, 6.15, 6.15 |
| `W->A` | 3/3 | 4.39, 5.10, 6.13 |
| `S->W` | 3/3 | 4.10, 3.36, 5.47 |
| `C->A` | 3/3 | 5.44, 5.16, 5.35 |
| `S->A` | 3/3 | 5.35, 0.87, 5.10 |
| `C->G` | 3/3 | 3.29, 4.02, 4.01 |
| `I->D` | 3/3 | 3.32, 3.30, 3.10 |
| `G->P` | 3/3 | 2.46, 2.92, 3.23 |
| `C->S` | 3/3 | 1.41, 3.14, 3.19 |
| `C->I` | 3/3 | 1.05, 2.53, 2.72 |
| `W->C` | 3/3 | 2.11, 2.68, 2.71 |
| `W->G` | 3/3 | 0.69, 2.58, 2.36 |
| `S->C` | 3/3 | 2.47, 0.59, 2.32 |
| `A->G` | 3/3 | 1.98, 2.46, 2.24 |
| `D->A` | 3/3 | 1.89, 2.14, 0.35 |
| `G->S` | 3/3 | 1.90, 2.05, 1.64 |
| `P->N` | 3/3 | 1.73, 1.53, 1.74 |
| `W->S` | 3/3 | 0.48, 1.54, 1.72 |
| `D->N` | 3/3 | 1.70, 1.48, 1.39 |
| `I->P` | 3/3 | 0.81, 1.70, 1.61 |
| `G->W` | 3/3 | 1.50, 1.66, 1.60 |
| `G->C` | 3/3 | 1.34, 1.31, 1.50 |
| `M->G` | 3/3 | 1.39, 1.44, 1.39 |
| `G->N` | 3/3 | 1.03, 1.10, 1.21 |
| `C->P` | 3/3 | 0.83, 1.19, 1.10 |
| `S->P` | 3/3 | 1.05, 0.76, 1.13 |
| `A->P` | 3/3 | 1.07, 0.87, 0.69 |
| `G->A` | 3/3 | 0.83, 0.95, 1.05 |
| `A->M` | 3/3 | 0.85, 0.48, 0.45 |
| `A->C` | 3/3 | 0.67, 0.71, 0.65 |
| `N->A` | 3/3 | 0.67, 0.63, 0.69 |
| `A->N` | 3/3 | 0.66, 0.39, 0.41 |
| `G->M` | 3/3 | 0.62, 0.65, 0.64 |
| `N->C` | 3/3 | 0.48, 0.47, 0.51 |
| `P->C` | 3/3 | 0.44, 0.44, 0.45 |
| `I->N` | 3/3 | 0.37, 0.40, 0.41 |
| `D->P` | 2/3 | 1.37, 1.67, 0.00 |
| `S->G` | 2/3 | 1.34, 0.00, 1.58 |
| `D->M` | 2/3 | 0.98, 1.12, 0.06 |
| `M->W` | 2/3 | 0.38, 0.30, 0.53 |
| `P->W` | 2/3 | 0.34, 0.24, 0.50 |
| `I->A` | 2/3 | 0.21, 0.44, 0.35 |
| `D->C` | 2/3 | 0.33, 0.32, 0.23 |
| `S->I` | 2/3 | 0.32, 0.31, 0.30 |
| `G->D` | 1/3 | 0.07, 0.67, 0.06 |
| `S->M` | 1/3 | 0.38, 0.00, 0.24 |
| `D->G` | 1/3 | 0.29, 0.35, 0.28 |

## The pairs closest to carrying

| pair | effect | q | conditions carrying |
|---|---|---|---|
| `I->S` | 0.67 | 0.0049 | idle, stress |
| `I->G` | 0.33 | 0.0000 | idle, stress |
| `W->P` | 0.31 | 0.0376 | conversation, stress, tool_use |
| `M->W` | 0.30 | 0.0000 | idle, tool_use |
| `P->S` | 0.30 | 0.0000 | idle |
| `S->I` | 0.30 | 0.0000 | memory, salience, stress |
| `I->C` | 0.29 | 0.0000 | stress |
| `D->G` | 0.29 | 0.0000 | memory, tool_use |
| `D->W` | 0.26 | 0.0000 | tool_use |
| `C->N` | 0.24 | 0.0000 | autonomy, idle, salience, stress |
| `S->M` | 0.24 | 0.0050 | idle, problem_solving, stress, tool_use |
| `P->W` | 0.24 | 0.0000 | none |
