# G03: Literal atoms in the argument chart

The two candidates in the reproduced cataphoric regression selected identical
operation spans. The expanded argument search found a higher-scoring wrong
assignment by splitting a list literal into two fragments. One fragment became
a reference to the list input; the other became a reference to an unrelated
computed scalar. Oracle operation-span diagnostics had hidden this failure.

The existing source parser already identifies each literal as a whole atom.
The opt-in `argument_literal_boundaries=atomic_v1` policy carries that boundary
into the argument chart. A reference may contain a complete literal but may
not intersect only a fragment of it. Exact literal references retain the
existing register-identity restriction. Alias spans outside literals remain
learned proposals. No vocabulary, family, answer, or phrase rule is added.

The reproduced source now selects the correct program:
`sub(in2, at(in0, in1))`. Its prior expanded-search result reversed the
subtraction arguments. The candidate changes no learned coefficients and has
receipt `df295e6c56caf7427974fac2cc8e88eecbad4593de1ed00d7790e4907a59a7ea`.

The boundary invariant, partial/intersecting/containing spans, receipt replay,
and existing graph suites pass 150 tests. Lint, compile, governance, and layering
pass. Smoke is 163 passed, one skipped, and the existing resident-manifest
serving alarm remains failed. Full development and control comparisons are
pending. The candidate is not promoted and G03 remains open.
