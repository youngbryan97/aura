# Exact graph-scale conflict

This is a reproduction and trainer integration of the
[2026-09-15 capacity result](G03_CAPACITY_AND_SEARCH_2026-09-15.md), not a new
discovery. Its [semantic follow-up](G03_SEMANTIC_COUNTEREXAMPLES_2026-09-15.md)
already distinguishes truly different programs from equivalent assignments.
Those historical obligations and results remain unchanged.

The existing `core.learning.score_capacity` abstraction and linear-arithmetic
proof kernel were applied to all 728 historical source-training contrasts in
`~/.aura/rlc-evidence/semantic-graph-factor-supervisor-20260915/detached.log`.
No validation answers entered the measurement. The fixed proposal scale is
0.875; the three variable scales have lower bounds of 0.000001.

Both the required margin 0.1 and strict ranking with margin zero have exact
rational contradiction certificates. Independent `verify_score_capacity`
replay passes. Their problem identities are, respectively:

- `5ab62950bb6b1cb5836f64803667e7a0c0369fac904cd5a3d713781af511473e`
- `d41953b4e3791530c8f0e4c00f11a27561b7a82b52b253be9a2f062408c9a65b`

The contradiction uses contrast indices 35, 144, 548 and 551. Their source IDs
are:

- `e4a2801136d38287cec9283160fdcdc7ad1c48c654b533fb75d9c5754e855b27`
- `a63e42506e99d71c0533e6806acbb997616df4a7a8bc986316fcbc1b6cb34cfb`
- `268537d577dce9946018de47fddf1e6ee1236b0712f633c5a813ec52da13773c`
- `f177324ef4458fde66aad60977a5cf330688687cf92ec4650f2c1554566cf1ff`

Their frozen binary64 coefficients are retained in
`test_historical_factor_conflict_prevents_even_strict_ranking`. The exact
nonnegative combination cancels all three variables and produces a
contradiction. The certificate uses only these four comparisons, not the
coefficient-bound rows. This is the certificate approach described by
[Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf),
implemented by Aura's pre-existing kernel rather than a second prover.

The graph-scale trainer now calls that shared capacity mechanism when its
numerical optimizer reports infeasibility. Missing or rejected exact evidence
remains numerical only. This proves a limitation of these frozen score
comparisons, not a limitation of richer representations, the correctness of
their labels, or broad-language impossibility.

Separately, runtime-retention receipts now distinguish exhaustion of a finite
beam from completion of the existing bounded complete-search iterator. Both
states have regression tests. The currently running development campaign is
frozen at its earlier revision and is not modified during execution.
