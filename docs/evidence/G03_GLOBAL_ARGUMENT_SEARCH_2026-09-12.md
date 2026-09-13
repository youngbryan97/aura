# G03 global argument search development

The opt-in `global_constraint_v1` candidate scores the same typed argument
options with the same learned coefficients as the parent. SciPy/HiGHS selects
the joint assignment. Constraints enforce one mention per argument, token
non-overlap, the source-learned register-use bounds, acyclicity and one connected
output. The existing floor and final graph validator remain authoritative.
The old beam search remains the default. No qualified package was changed.

The old search truncated local argument combinations before testing their
compatibility with the rest of the graph. It could discard valid continuations
while retaining high-scoring combinations that no complete graph could use.
The new solver optimizes the existing mention shortlist without those beam
truncations. It does not make the shortlist exhaustive over arbitrary language.
An incomplete solver run is reported separately from proven infeasibility.

On the 48 already exposed six-input, five-operation weave examples:

| Arm | Exact programs | Exact answers |
| --- | ---: | ---: |
| Parent | 19/48 | 21/48 |
| Global constraint candidate | 41/48 | 41/48 |
| Candidate, coefficients removed | 0/48 | 1/48 |
| Candidate, hidden tokens shuffled | 0/48 | 0/48 |

The candidate adds 21 exact answers and regresses one parent success. All 48
graphs are accepted; seven are wrong. This is exposed development, not fresh
replication. The result depends on learned features and explicit solver
assistance. It is not evidence of additional neural computation or frontier
reasoning. The incumbent is retained.

The remaining failures include argument mentions assigned across the wrong
operation clauses. In one case the first two operations exchange their first
input. Another selects a public definition as an argument mention and moves
the intended mention into another clause. One count operation is anchored to
an earlier token. Better feasibility search cannot correct wrong learned
rankings; scoring and operation anchoring still need development.

## Projection reuse

The relation scorer now computes each definition projection once per chart,
and each query projection once per reference. Scalar score arithmetic remains
unchanged. Tests compare scores bit-for-bit with the original implementation.
All 48 legacy decoded rows also match the parent exactly across two bounded
replays. A same-process four-repeat microbenchmark of 32 references and eight
registers with three definitions each measured old scoring at 0.134-0.182
seconds and reused scoring at 0.033-0.043 seconds. This is a scoring-operation
measurement, not an end-to-end speed claim. Concurrent host load varied.

The complete candidate replay took 167.27 seconds. Controls completed in two
bounded runs. No latency comparison is claimed from differently loaded runs.

## Evidence and checks

Generated rows, controls, legacy equivalence and source hashes are retained in
`artifacts/rlc/semantic_program_global_constraints_dev_20260912/`.
Candidate receipt:
`27f63ee27e3d9ddb98abb849a336c039ac8cdd15c51bd9b516fbe797d67c9b43`.

The solver tests compare with exhaustive enumeration across graph sizes,
arities, overlap patterns, argument distinctness and register-use bounds.
They also reject cycles, disconnected graphs, invalid integer solutions and
solver-limit results reported as optima. The combined solver, projection,
prefix, validation-selection and source-identity suites passed 63 tests.
Smoke passed 164 tests with one skip. Touched files pass Ruff and the full
tree compiles. Aggregate lint found 69 issues outside these changed files;
that gate is not reported green. G03 remains open.
