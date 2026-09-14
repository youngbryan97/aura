# Operation-chart feasibility development

The operation beam previously spent capacity on charts that could not satisfy
the existing register-use contract. A chart with too few argument edges cannot
consume the required inputs and intermediate results, regardless of its learned
score. These necessary bounds now run before the final beam truncation.

The policy is receipt-versioned and opt-in. It changes neither coefficients nor
the graph contract. Unknown primitives do not receive a feasibility claim.
Legacy receipts retain their prior behavior. This is not exhaustive search:
per-cardinality proposals and the final beam remain bounded, and argument
assignment still decides types, ownership, overlap and graph consistency.

Candidate receipt:
`55dd181db139602ffa3e8b170f90ff34f03d29a4b6f50744675b8ad5bde6f91e`.

The 500 exposed source cases produce 464 correct answers, versus 456 for the
immediate literal-retention parent: eight gains and zero regressions. The source
run took 255.03 seconds. Exposed weave remains 47/48, taking 106.49 seconds.
These runs used cached representations, not the live resident model.
Coefficient and hidden-shuffle controls each score 0/48. Their combined run
took 203.82 seconds. Weave has zero gains and zero regressions against the
immediate parent. Reports and paired comparisons are preserved under
`artifacts/rlc/semantic_feasible_operation_dev_20260913/`.

The focused feasibility suite passes 26 tests, including enumeration of small
valid connected programs under two register-use contracts. Lint, compile,
governance, layering and writing checks passed. Smoke reports 163 passed,
one skipped and one failure: the existing resident-manifest drift alarm.
That failure has not been waived or relabeled.

This candidate has no serving authority. G03 is not closed; these exposed
development results do not establish fresh transfer or frontier performance.
