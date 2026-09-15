# G03 semantic counterexample admission

The four training sources selected by the frozen score contradiction were
replayed against the parent decoder's retained argument charts. No validation
or test examples entered this diagnostic.

The search now excludes complete register graphs across all mention and
definition realizations. A negative requires different outputs from two
successful executions on the universal floor. Failed execution and finite
agreement do not establish a negative. Structural symmetry and exact integer
polynomial normalization establish supported positive equivalences.

## Measured result

- Three sources produced distinguishing execution witnesses.
- For source `e4a2801136d38287cec9283160fdcdc7ad1c48c654b533fb75d9c5754e855b27`,
  public values `[79,38,3]` produce target `114` and alternative `-114`.
  The target's relation-factor difference is `-2.3021699339151387`;
  role, proposal and pointer differences are zero.
- The exact frozen-score certificate remains infeasible with all three
  adjustable scales bounded below by `1e-6`. This is not a theorem about a
  refitted neural head or the best score over every correct interpretation.
- One source exhausted its eight-graph diagnostic allowance after eight
  equivalent alternatives: seven polynomial identities and one structural
  symmetry. Its status remains `search_incomplete`, not absence of errors.

[Machine receipt](../../artifacts/rlc/semantic_graph_factor_dev_20260915/semantic_counterexamples.json)
records source identities, comparisons, floor receipts and the exact score
certificate. The model-active runtime was not used or changed.

## Implementation boundary

The graph-factor refit now uses the same witnessed-negative admission. Its
new receipt is version 2; retained historical version-1 results are unchanged.
An empty set of witnessed errors preserves the coefficients and reports no
fit rather than claiming convergence. This is training infrastructure, not
completion of joint neural-head learning or iterative refinement.

Focused checks: 113 passed before refit integration; 31 passed after refit
integration. Smoke: 164 passed, one skipped in 100.86 seconds. Lint, compile,
governance-lint and layering passed. G03 and
S03-S08 remain open; this diagnostic is not a new 500-row accuracy result.
