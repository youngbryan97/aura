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

That real-bank pilot has now completed. With four operation charts, two
argument graphs per chart, and a two-second ordinary-decode allowance, all
84 source-training rows retained a verified correct program; ordinary decode
selected one on 27. In this particular subset the first retained program was
correct on all 84 rows, whereas it was wrong on all 23 exposed development
misses. The 57 ordinary source refusals exhausted their decode allowance;
their subsequent diagnostic bank search is additional work, not an
equal-compute rescue. This is a substantial train/development distribution
shift and a reason not to teach a position-only rule.

The request-conditioned ranker was fitted for three epochs on real source
banks, holding out construction groups in each of the three frozen folds:

| Fold | Held rows | Ordinary selected | Ranker selected | Gains | Regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 24 | 10 | 9 | 7 | 8 |
| 1 | 28 | 8 | 6 | 5 | 7 |
| 2 | 32 | 9 | 12 | 9 | 6 |
| Total | 84 | 27 | 27 | 21 | 21 |

The fold-0 ranker selected a verified correct program on 8/23 exposed
development misses, but those rows were chosen precisely because ordinary
decode failed. The 84-row paired source result has no net gain, and the
development complement was not scored. Posterior mass over the same
candidate graph scores selected 84/84 on this easy source subset but only
4/23 on the exposed misses; it is not a general selector either. No serving
change or G03/G04 claim follows. The next source of training evidence must
include genuinely hard, answer-blind candidate choices and independent
preservation checks, not only more epochs on this separable source subset.

Mixing real retained banks with source-only witnessed counterfactual contrasts
improved the three-fold source pilot from 27/84 to 34/84, but still produced
27 gains and 20 regressions against ordinary decode. By fold, the mixed model
selected 15/24, 9/28, and 10/32 respectively; ordinary selected 10/24,
8/28, and 9/32. This is not an admitted policy. A separately trained direct
program decoder selected 23/24 on fold 0's same bank, versus 15/24 for the
mixed ranker and 10/24 for ordinary decode. That comparison is one pilot
fold, not a full cross-fold result or a live gain.

The method-comparison tool now executes the ordinary, ranker, and direct
choices through the pre-existing semantic program portfolio and records
their disagreements and unanswered inquiries. Its oracle-union count is
computed only after the candidate set is frozen. The union is a reachability
ceiling, not a runtime selector; execution by the floor is not proof that a
candidate answers the user's question. No G03 box is closed by these pilots.

Artifacts: `~/.aura/rlc-evidence/semantic-candidate-ranker-source-fold0-20260922/`
and `~/.aura/rlc-evidence/semantic-candidate-ranker-anchored-fold0-20260922/`.
Real-bank source pilot:
`~/.aura/rlc-evidence/semantic-candidate-training-4x2-stratified-20260922/`.
Three source-fold results and the exposed-failure replay:
`~/.aura/rlc-evidence/semantic-real-bank-ranker-fold0-20260922/`.
Mixed source-fold pilots:
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/`.
The frozen real-bank audit is
`~/.aura/rlc-evidence/semantic-gap-mixed-candidate-bank-20260922/report.json`.
