# Overlap-complete argument development

The four-mentions-per-register cutoff could remove a reference needed by a
globally valid graph. The opt-in `overlap_dominance_v3` policy removes a mention
only when a same-slot, same-register, same-definition mention scores at least
as well and occupies a subset of its tokens. Replacement preserves the other
constraints. This is a proof about the proposed chart, not complete language
coverage.

The full chart initially spent excessive time in integer-solver presolve.
Conflict-bitset reduction and disabling presolve did not solve that cost and
were discarded. The retained implementation obtains a feasible incumbent from
a shortlist, then uses a continuous relaxation's dual lower bound to eliminate
only binary choices that cannot match it. Continuous ordering variables are
not screened. Search limits still raise incomplete-search errors.

The stronger 472/500 argument-ranking parent and its coefficients are unchanged.
The new candidate scored 474/500: four role-binding gains, two cataphoric
regressions. Arithmetic is 117/128; cataphoric 44/48; fork/join 192/192;
role binding 48/48; reserved aliases 37/48; natural source 24/24; natural aliases
12/12. Exposed weave remains 47/48 with no paired changes. Coefficient and
hidden-shuffle controls each score 0/48.

The completed source evaluation took 997.18 seconds, weave 521.72 seconds,
and controls 653.73 seconds under concurrent host load. These are not isolated
latency comparisons. Earlier stopped trials are not scored as completed runs.
Afterward, a test exposed redundant binary bounds absorbing useful dual prices.
The pricing relaxation now omits those bounds but certifies against the original
finite domain. The semantic objective and feasible integer programs are unchanged.
136 focused tests pass, including exhaustive small-graph comparisons with and
without screening, definition choices, overlaps, arities and dependency orders.
Lint, compile, governance, layering and writing checks pass. Smoke reports
163 passed, one skipped and the existing resident-manifest activation alarm.
That alarm remains visible and is not treated as semantic qualification.

Reports are in `artifacts/rlc/semantic_overlap_dominance_dev_20260914/`.
Candidate receipt: `1264806aa5197d63d09659f035703f24a9179e8772bea73821e30f4461836811`.
This exposed-development candidate is not promoted. G03 remains open because
two prior successes regress and 26 answers are wrong. No fresh-transfer,
broad-reasoning or live-serving claim follows from these measurements.
