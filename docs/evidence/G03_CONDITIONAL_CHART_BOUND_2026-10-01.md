# Conditional Chart Bound, 2026-10-01

The selected-edge replay still timed out on the remaining development
request. Its [negative receipt](G03_SELECTED_EDGE_EXECUTION_2026-10-01.md)
is retained. Execution version five adds an admissible complete-chart bound
before learned scoring and constrained optimization. The thirty-second
request allowance, fitted tensors, candidate grammar and score policy stay
unchanged.

## Bound

For slot s, let b_si be each original option score and
L_s = log(sum_i exp(b_si)). The declared conditional learned update is

```
u_si = b_si + w*z_si - log(sum_j exp(b_sj + w*z_sj)) + L_s.
```

The normalized factor is at most one, so u_si <= L_s for every finite
learned score. This includes scores not yet computed. Each register's
definition attachment contributes at most its largest positive attachment
score, paid once. The existing optimizer enforces that single-definition
condition. Relaxing all graph, use and overlap constraints gives

```
U(chart) = sum_s L_s
         + sum_register max(0, max_definition attachment_score)
         - original_choice_log_normalizer
         + sum_operation operation_score
         - unchanged_length_penalty * operation_count.
```

The bridge creates the same context as before, with no positive pair bonus
or negative context cost. The bound is specific to that execution contract.
It is not valid for raw additive learned scores; raw replay does not prune.
No answer, target register or construction label appears in U.

A bound incumbent must have a complete accepted assignment. The bridge
reuses the parent's connected topological-order check and the same learned
register-use contract before retaining its joint score. It skips a chart
only if U plus a conservative floating-point guard is strictly below that
score. Equal and potentially better charts remain. Diagnostic receipts
record the upper bound, operation score, incumbent, guard and parent checks.
These records do not turn score margins into probabilities.

## Checks

Focused checks passed 103 tests in 23.57 seconds. The new cases cover
arbitrary learned logits and weights against the bound, optional positive
and negative definitions, strict losing charts, ties, better charts, raw
replay, connectivity rejection, register-use rejection and agreement of
the complete graph selector with unpruned execution. Existing fitted
pointer, offset normalization and native-evaluation checks remain green.

Standard gates passed: smoke 164 passed and one skipped in 56.80 seconds;
lint, compile, governance, layering and writing passed. No gate baseline or
request budget was relaxed. Training and checkpoint custody are unchanged.

The remaining source still needs a real replay. A fixture bound proof does
not establish useful pruning on real charts, language accuracy, broad gain,
general transfer, fusion or frontier performance. G03 remains open.
