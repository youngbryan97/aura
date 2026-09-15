# Frozen score capacity and complete operation search

The graph-factor refit retained 728 source-training records. The capacity
tool binds those records to their original receipt and converts their frozen
binary64 coefficients to exact rational constraints. Numerical optimization
only proposes a witness. Aura's existing Farkas kernel checks contradictions;
feasible weights must satisfy every constraint in exact arithmetic.

Both the unit-margin system and the strict-ranking system are infeasible.
Their certificates use these four source comparisons:

- `e4a2801136d38287cec9283160fdcdc7ad1c48c654b533fb75d9c5754e855b27`
- `a63e42506e99d71c0533e6806acbb997616df4a7a8bc986316fcbc1b6cb34cfb`
- `268537d577dce9946018de47fddf1e6ee1236b0712f633c5a813ec52da13773c`
- `f177324ef4458fde66aad60977a5cf330688687cf92ec4650f2c1554566cf1ff`

The strict problem digest is
`d41953b4e3791530c8f0e4c00f11a27561b7a82b52b253be9a2f062408c9a65b`.
Both complete constraint sets and replayable certificates are retained in
`artifacts/rlc/semantic_graph_factor_dev_20260915/`.

This proves a limit of three positive factor scales with fixed proposal
scores and annotated operations on these comparisons. It does not prove
semantic impossibility: register-assignment negatives can include equivalent
programs. Equivalence-aware contrasts and changes to learned representations
remain necessary. The checker cannot establish the truth of supplied labels.

## Search implementation

The opt-in `complete_bounded_v1` operation policy retains all source spans up
to the declared span length and all learned operation labels. A lazy interval
search enumerates all nonoverlapping charts up to the declared step limit.
Suffix bounds use exact rational values of the supplied scores. No top-16
chart cutoff or top-256 span cutoff applies in this policy.

The production transducer's joint selector consumes the iterator. An explicit
expansion allowance preserves the frontier and reports incomplete search;
it cannot emit an interim winner as a proved optimum. Existing models retain
their original policy unless explicitly migrated. Argument and definition
candidate construction is unchanged, so this is operation-grammar coverage,
not complete natural-language interpretation or full argument coverage.

Tests compare every chart against brute force across 12 randomized small
grammars, exercise cancellation-sensitive bounds, resume an interrupted
iterator, and recover a winning chart ranked below the old cutoff. Tests also
exercise serialized model identity and the actual transducer decode path.

The capacity/search suites pass 40 tests. Before the strict-ranking extension,
the broader focused run passed 133 tests. Repository smoke passes 164 with one
skip; lint, compile, governance-lint and layering pass. No live model was
loaded or promoted, and the 500-row candidate comparison has not been rerun
under this search policy. G03 remains open.
