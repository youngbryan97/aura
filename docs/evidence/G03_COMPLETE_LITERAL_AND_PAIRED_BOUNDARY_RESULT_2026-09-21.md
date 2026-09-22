# Completed Literal and Paired Boundary Results

The literal-grammar identity candidate retains 764/764 training examples and
reaches 477/500 validation examples. Compared with the archived incumbent
decisions, it gains one case and loses none. This is exposed development
evidence, not fresh transfer or downstream public-answer measurement.

Report: `semantic-literal-identity-cohort-20260921/report.json` under the
local RLC evidence root. Receipt:
`31fa91a7f2f0b891816402c15f01e6950ca1c7148fe62909aef7b8752752dd41`.

The paired-boundary candidate converged after 110 iterations and 4,880.802
seconds of fitting. Its objective fell from 0.3415296634 to 0.0158113728.
It nevertheless reached only 37/54 selected development examples against
39/54 for conditional labels alone. Both retain 3/3 source cases and 15/16
controls. The candidate is rejected. It does not establish that paired
features lack capacity; it establishes that this source-trained fit did
not improve the declared development cohort.

Report: `semantic-paired-span-set-resumable-20260921/report.json`. Receipt:
`2962a8238d97614aa8919ff4032ede9301647af40bb108464082bfc4c09215ce`.
The fit predates literal identity in this candidate's policy; the separate
literal-policy experiment must not be silently counted into these results.

The follow-up reproduces all 17 failures: sixteen span differences and the
known literal binding difference. Every span failure gives the target a
lower operation score than the selected graph. Nine have argument evidence
favoring the target, but not enough to offset the operation deficit. The
diagnostic uses annotations only for attribution, never fitting. Receipt:
`0b7f40e293e6fa50a9168416664cd02e1e6375d1c9aa5f4d9a1a673098ec311d`
in `semantic-paired-boundary-attribution-20260921/report.json`.

## Exact Partition Cost

The span-set dynamic program now batches its independent count dimension.
Forward dependencies still proceed by endpoint; reverse dependencies still
proceed in reverse endpoint order. No intervals or candidate counts are
removed. Exhaustive partition, marginal, and finite-difference tests apply.

A local 20-repeat microbenchmark measured old versus batched seconds:

| Tokens | Maximum span | Maximum count | Old | Batched |
| --- | --- | --- | --- | --- |
| 40 | 8 | 4 | 0.002078 | 0.001333 |
| 120 | 16 | 8 | 0.014664 | 0.005002 |
| 240 | 24 | 12 | 0.043750 | 0.010775 |

This is partition timing, not whole-campaign speedup. The completed paired
candidate was fitted before this batching change. G03 remains open.
