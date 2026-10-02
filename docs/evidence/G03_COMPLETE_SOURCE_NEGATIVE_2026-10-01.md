# Complete source-development negative

## Verified measurement

The frozen `cb5d2b807` checkout completed all 1,500 public decodes: five
hundred requests in each of the source-parent, global-chart and joint-native
arms. There were no refusals or interrupted decodes. Evaluation plan:
`27c715c32843380276cf3e3e2c37bea4b9346cb503c58684d08f3c6ef66c7f35`.

| Arm | Equivalent programs / 500 | Reference answers / 500 |
| --- | ---: | ---: |
| Source parent | 355 | 386 |
| Global chart | 327 | 343 |
| Joint native | 326 | 341 |

Reference answers are independently executed Python programs, not freely
generated public model answers. Numeric agreement does not establish the
intended program. All five hundred reference answers were measurable.

The joint candidate loses 38 parent successes and gains nine. Against the
global-chart control it loses one success and gains none. Global and native
produce the same operation/argument programs on 494 requests. The global
control alone introduces 37 parent regressions: 32 three-operation programs
become two-operation programs, and five two-operation programs acquire a
third operation. All 174 joint interpretation misses have a different
operation count from the annotation. These are measured failure categories;
they do not yet establish why every intended operation was lost.

## Custody

Evaluation root:
`/Users/bryan/.aura/rlc-evidence/semantic-grounded-joint-source-validation-v1-20261001`.
The evaluation report receipt is
`b69f1df6c7380aaf189cdff243ad521b93d07391947e16019bcb52be89a885bb`.
Its contained terminal receipt is
`0b7606c9ac36b230274746f9244242fef268df65645e9b512affa8f199aef134`:
exit two, 2957.714304 seconds, no timeout, empty process group and lineage.
Supervisor and child are independently observed dead. Negative acceptance
is not an infrastructure crash.

The already-launched adjudication handoff ran from frozen `75a57fc04` after
the model process exited. It loaded no backbone and repeated no decode or
fit. Its own contained terminal is
`36b2bcba9c112aa6298eb508bb78590418ee1d0ebca1c1f86a0fb7a6835ee085`,
exit zero, empty process group and lineage. Completion:
`a670dda6111e5ef08041a9b57fdeff623cab320da96fa8f9250a079d5906f0b2`.
Artifact verification:
`f89ecf7336486172acf327a1ee58fbdfd3eceb8661bbfe5422374ad6758c3130`.
Independent adjudication:
`51157b4023efeb7d80a7b6266214e7adb37770cc07ebd94e80693a3a05b9fff4`.
The handoff's successful processing retains `advance_development=false`.

## Repair boundary

The parent uses first-feasible operation selection. The new chart path
changes this to the sum of operation and argument scores without retraining
the operation model for that decision. Native fitting trains a role/filler
pointer and suffix on annotated operation and mention spans. It does not
train an operation presence/count head. These are code facts, separate from
the hypothesis that one of them explains every measured miss.

Before another native fit, distinguish missing public operation proposals
from misranked proposals across the complete failed population. The new
`tools/audit_semantic_grounded_operation_selection.py` freezes each actual
public inventory before reading annotations. It measures count, ordered
labels and exact span coverage separately, without loading a model or
admitting validation labels to fitting. Seventy focused checks and smoke
(164 passed, one skipped), lint, compile, governance, layering and writing
pass. The audit measurement itself is still pending at this checkpoint.

Retaining a parent candidate is necessary but does not certify a replacement.
A different program with a higher uncalibrated score may be worse. Serving
remains unchanged; this candidate is not promoted. G03 and later scientific
items remain open. No retry, larger budget, or fresh transfer run follows
automatically from this negative result.
