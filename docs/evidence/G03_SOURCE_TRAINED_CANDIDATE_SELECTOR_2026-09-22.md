# Source-trained selection against a retained program bank

The complete-request program ranker is a quarantined experiment. It consumes
frozen source features, measured input and operation spans, input types, and
candidate computation graphs. It does not receive a target, expected answer,
family label, or diagnostic comparison at inference. Source-training labels
are used only in the fit and after target-blind candidate acquisition.

The initial source-only contrast corpus changed primitives and typed register
edges and included same-depth source-training peer programs. Negatives needed
a witnessed output difference on generated counterfactual inputs; all
candidate programs passed the shared primitive type contract. Construction
groups were held out by the frozen three-fold source partition.

| Fixed comparison | Held-out source contrast selection | Exposed real-bank failures |
| --- | ---: | ---: |
| Global source/program scorer, fold 0, ten epochs | 150/256 | 4/23 |
| Operation-span anchored scorer, fold 0, ten epochs | 117/256 | 1/23 |
| Ordinary incumbent on the 23 selected failures | not this comparison | 0/23 |

The 23 development rows were selected precisely because the ordinary answer
failed. All 23 retained banks contain a verified correct candidate, but these
figures cannot establish net gain or non-regression on the other 477 rows.
Anchoring improved source-training fit while worsening both held-out measures.
This is evidence of a source-contrast versus real-decoder distribution gap,
not a basis for a serving change. Neither candidate is promoted.

The candidate bank also had a diagnostic defect: a bounded ordinary decode
refusal discarded all alternative graphs. It now retains alternatives only
for `decode_search_budget_exhausted`, without changing the ordinary outcome,
its refusal, or serving authority. In a two-row source-only pilot with four
operation charts and two argument graphs per chart, both ordinary decodes
exhausted their allowance, yet both retained banks contained a verified
equivalent program. Two rows are not a coverage estimate.

Next, a fixed source-only subset takes the lowest two source identities per
independent construction group, before any labels are inspected. It is a
pilot, not complete source coverage. Real decoder-generated alternatives
must improve selection on held-out source groups and then on the full frozen
development denominator before a promotion can be considered. G03 remains
open; G04 transfer and G05 public decoding are separate claims.

Artifacts: `~/.aura/rlc-evidence/semantic-candidate-ranker-source-fold0-20260922/`
and `~/.aura/rlc-evidence/semantic-candidate-ranker-anchored-fold0-20260922/`.
The frozen real-bank audit is
`~/.aura/rlc-evidence/semantic-gap-mixed-candidate-bank-20260922/report.json`.
