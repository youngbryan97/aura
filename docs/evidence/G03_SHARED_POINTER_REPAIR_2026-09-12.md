# G03 shared pointer repair

The shared span detector gave contradictory labels to valid boundaries. While
fitting one correct span, its hard-negative sampler included the other correct
spans of the same role in that request. The same hidden vector therefore carried
both labels. This affected operation, argument and definition detectors.

The repair excludes every same-head positive boundary before applying the
negative quota. Other semantic roles remain eligible hard negatives. It changes
future fitting, not historical coefficients. New fit receipts name the policy.

## Source-label audit

The original 728 training examples contain these negative-label rows:

| Shared detector | Negative rows | Contradictory rows | After repair |
| --- | ---: | ---: | ---: |
| Operation | 37,984 | 4,448 | 0 |
| Argument | 79,384 | 24,128 | 0 |
| Definition | 95,812 | 35,552 | 0 |

Three regression tests fail before the repair and pass afterward. They inspect
the labels passed to the optimizer. A fourth test preserves hard negatives from
other roles. These counts describe the source supervision, not a claim that all
three detectors have been retrained in the candidate below.

## Measured candidate

The operation-pointer-only refit uses the frozen parent's exact 728 training
examples and 500 validation examples. Training fits boundary coefficients;
source validation recalibrates the operation-count penalty. Other coefficient
groups remain unchanged. No weave case enters either stage. The refit took
180.47 seconds on this host; this is one observation, not a serving latency.

The candidate also uses the previously tested constrained argument graph,
order-independent dependency admission and conditional argument scores. Its
relation scores now retain categorical log-odds from the cross-entropy-trained
relation tissue. The legacy score remains the default for old receipts.

| Exposed weave candidate | Exact programs | Exact answers |
| --- | ---: | ---: |
| Frozen parent | 19/48 | 21/48 |
| Conditional graph before categorical relation scores | 43/48 | 43/48 |
| Categorical relation scores | 47/48 | 47/48 |
| Corrected operation-pointer refit | 48/48 | 48/48 |
| Refit with coefficient lesion | 0/48 | 1/48 |
| Refit with hidden-token shuffle | 0/48 | 0/48 |

The final weave error was a wrong operation anchor: the detector chose "count"
inside an input name instead of the counting instruction. Correcting only the
gold operation anchors in a diagnostic recovered the correct graph. That
diagnostic was not counted as runtime success. The source-only refit subsequently
recovered the example through ordinary decoding.

## Full-source comparison

On all 500 original source-validation examples:

| Measure | Parent | Candidate | Paired gains | Paired regressions |
| --- | ---: | ---: | ---: | ---: |
| Exact program | 304 | 320 | 53 | 37 |
| Structurally equivalent program | 319 | 347 | 56 | 28 |
| Exact executed answer | 346 | 377 | 55 | 24 |

The candidate is not admitted as a replacement. The source-only calibrated
selector experiment is separate and must preserve its own disjoint tuning and
admission checks. The remaining source errors keep G03 open. G04-G12 are not
closed by 48 exposed examples, even with successful causal controls. Execution
uses the existing typed Python floor; these results do not establish that a
neural recurrent state performs the arithmetic itself.

## Reproduction and verification

Use `tools/refit_semantic_argument_proposals.py --objective operation_pointer`
with the frozen parent, its source compatibility report and all seven named
feature bundles. The CLI verifies the exact source cohort before fitting.

The frozen parent receipt is
`a79674be5460d12e985efe406939e181e0b51bec6163bd3c2b4ae41db9c55c59`.
The operation-pointer refit receipt is
`b557598e271ccf7d1569cf339f697a5d0f8be2c493b0a4995982eb6f33bd18c8`.
The complete development candidate receipt is
`d608bd08b8bc22f886f4fb089fe2eea078f559a9fe33e55bded6098cb24c3d1f`.

Retained rows, ablations, source-label audit, controls and hashes are under
`artifacts/rlc/semantic_program_clean_pointer_dev_20260912/`.
The coefficient artifact stays in the local evidence directory, with its file
hash and path recorded in the refit receipt. No serving package is changed.

The combined focused suite passed 53 tests. The subsequent CLI/source-contract
suite passed 17 tests, with overlap between these runs. Smoke passed 164 tests
with one skip after the detector repair. Compile, touched-file Ruff and writing
checks passed. This does not supersede the aggregate lint, governance and
layering failures recorded by the preceding checkpoint.
