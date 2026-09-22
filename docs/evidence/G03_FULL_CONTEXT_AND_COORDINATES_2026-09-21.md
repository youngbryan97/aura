# Full-source context and common input coordinates

G03 remains open. The contextual candidate is not promoted. No fresh transfer,
broad reasoning, public decoding, fusion, or frontier result follows.

## Completed paired measurement

The context recognizer trained on all 764 source examples for eight fixed
epochs. It reached 764/764 training operation sets and 356/500 development
operation sets. Removing cross-token context also gave 356/500. Validation
labels did not enter optimization. This development set has been exposed in
earlier experiments; it is not fresh evaluation.

The frozen argument resolver converted predicted operations to complete
programs. The existing literal-identity incumbent ran on the same 500 tasks,
with alternating arm order and the same ten-second search allowance.

| Method | Correct complete programs | Different |
| --- | ---: | ---: |
| Context plus typed argument resolution | 437 | 63 |
| Literal-identity incumbent | 477 | 23 |
| Correct program available from either | 486 | 14 |
| Answer-blind necessary-condition selector | 477 | 23 |

The selector made zero gains and zero regressions. Nine correct alternatives
were available but not selected. Fourteen tasks lacked a correct candidate.
Keeping both methods improves available coverage; it does not yet improve
autonomous selection. Successful execution is insufficient evidence that a
program expresses the request.

The complete graph runs took 458.824 seconds in the context arm and 923.355
seconds in the incumbent arm. Portfolio execution and comparison took 0.456
seconds. These omit fitting, feature acquisition, and loading. They do not
establish equal-compute superiority.

## Corrected grading

The new graph probe compared input register indices directly. Grounding can
permute equal-valued literals while preserving their source anchors. Under a
counterfactual that assigns those occurrences different values, direct-index
grading can report a false difference. The established graph evaluator
already accounts for this; the new probe had omitted that step.

The original report remains intact: 425 context successes and 463 incumbent
successes. A separate regrading report normalizes programs by source anchors,
without changing operations, intermediate registers, or the selected program.
It corrects 12 context grades and 14 incumbent grades. All 1,000 rows aligned.
The incumbent then reproduces its previous 477/500 result.

`reanchor_program_inputs` requires a bijection and preserved public values.
Tests cover equal literals, every four-input permutation, independently varied
counterfactual values, invalid anchors, and invalid register references.
Future graph reports record both runtime and source anchors, retain the raw
program, and bind a whole-core implementation identity before and after the
measurement. A portfolio rejects mixed or missing coordinate declarations.

Earlier source-fold context reports used the same direct-index grading.
Their historical numbers remain recorded, but their common-coordinate grades
have not been recomputed. Do not use them as corrected transfer evidence.
Their frozen argument heads also saw the source fold.

## Exact search cost

Finite binary64 scores have power-of-two denominators. A common positive
denominator maps them to integers exactly. This preserves every sum, bound,
comparison, tie, and branch order. The public bound rounds outward.

The production operation search now uses this representation. Exhaustive
small charts and a separate rational reference verify candidate order and
expansion counts, including cancellation, ties, and subnormals. A five-seed
480-node benchmark against retained commit `b9472565c` measured median search
times of 0.667350 seconds with fractions and 0.059250 seconds with integers.
The median paired ratio was 10.3088. Each run yielded the same 256 charts,
performed the same expansions, and ended at the same bound. This measures
search arithmetic only, not whole-system latency or reasoning quality.

## Retained artifacts

Root: `/Users/bryan/.aura/rlc-evidence/`.

| Relative report path | SHA-256 |
| --- | --- |
| `semantic-context-full-source-20260921/report.json` | `41516285f7cd66cf95c8d33075e6638909ce3630ad0b3b13ee3207dd1fae8a18` |
| `semantic-context-full-graphs-20260921/report.json` | `a74ac77771996e6d13db927364c64cb71208182665e77a720ce4a590de9bc79e` |
| `semantic-context-full-graphs-aligned-20260921/report.json` | `c9063f0a081440133fe81891713cc2f598f1a6fb2f9a258b4537901b763dd583` |
| `semantic-context-full-portfolio-20260921/report.json` | `08261aea9ed708e7530a2626e3aa1751a3907a5d8af23833a416326ecd577152` |
| `semantic-exact-dyadic-search-20260921/report.json` | `1e3c34573f5065d739e3ea1c9574b189071f4830520f814ca0178d3be527d410` |

Focused tests: 119 passed; the final coordinate and provenance suite then
passed 31 tests, including two additional recovery-identity cases. The combined
smoke process was killed by signal 9 while the host was busy. Running its
unchanged targets in three bounded processes passed 88, 33, and 43 tests,
with one skip: 164 passed and one skipped in total.
The direct search benchmark excludes only the old invariant registration
function to avoid a duplicate registry key. Its inference code is unchanged.
