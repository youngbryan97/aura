# Literal retention and failure attribution, 2026-09-13

## Measured mechanism

The production graph solver now receives an immutable `ScoredArgumentChart`.
Its solve method delegates to the existing optimizer. An optional observer can
retain the scored chart; labeled target attribution is a separate offline
method and cannot replace the runtime choice.

On the immediate parent's 62 structurally incorrect source-development
programs, supplying gold operations gives these results:

| Condition | Cases |
| --- | ---: |
| Target arguments selected | 35 |
| Target feasible but another assignment ranks higher | 23 |
| Target excluded by joint constraints | 4 |

These are labeled diagnostics, not successful autonomous answers. Some
structurally incorrect programs happen to return the correct answer on their
particular input.

All four constraint cases remain infeasible when definition consistency is
removed, and become feasible when mention exclusivity is removed. The latter
is diagnostic only: the production constraint is preserved. Inspection found
that ranked mention pruning can remove a literal's exact span, leaving
overlapping high-scoring mentions as the only options for a required input.

The separately identified `ranked_with_literal_anchors_v2` policy preserves
each available exact literal span alongside the existing top-four mentions.
It does not force the graph to select that span, invent an unavailable
candidate, change learned coefficients, or remove graph constraints.

## Development results

Candidate receipt:
`162a8933df3a1bbfa45802442755f9aaa0b66e362924dbf2c95cd207dc792607`.

Source answers: **456/500**, two gains and zero regressions against the
immediate parent's 454/500. Against the frozen 346/500 parent, there are 115
gains and five regressions. The complete source run took 180.73 seconds.

Exposed weave answers: **47/48**, with no answer changes against the immediate
parent. The run took 49.99 seconds. Its raw report compares against an older
categorical-relation candidate; the preservation receipt separately records
the immediate-parent comparison to avoid confusing those baselines.

Coefficient and hidden-token-shuffle controls each score 0/48. The control
run took 209.87 seconds. Reports and hashes are preserved under
`artifacts/rlc/semantic_literal_anchor_dev_20260913/`; the candidate tensor
receipt is retained outside Git at the path recorded there.

The 35 cases recovered by gold operations were examined at the operation
proposal boundary: 21 have all correct nodes but their chart lies outside the
retained beam; six lack a gold span; three misclassify a gold span; five contain
the gold chart in the beam. Increasing definition coverage does not address
those operation-stage failures. The next search repair must account for
complete graph evidence instead of accepting the first feasible chart.

## Verification boundary

98 focused chart, optimization, runtime, attachment and operation-view tests
passed. Lint, compile, governance-lint, layering and writing passed. Smoke
passed 163 tests, skipped one, and retained the resident-manifest-drift alarm.
No live model was loaded or restarted in this pass.

The candidate is development-only. Five regressions against the frozen parent
remain, and all these source and weave cases are exposed. G03-G12 remain open;
no scientific or serving qualification is implied by this repair.
