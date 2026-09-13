# G03 operation views and register-link diagnosis

The compositional decoder can now use the semantic substrate's existing
operation feature views. The legacy artifact still uses `contextual_mean`.
Refitted candidates bind the selected views, coefficients, source identities,
validation measurements and selection objective in the training receipt.
Coefficient lesions remove every selected head.

Operation-chart calibration compared source-ordered predictions against
execution-ordered labels. Cataphoric programs can have different orders.
Calibration now compares both charts in source order; it does not change
the executable IR or grant gold spans to runtime decoding.

## Development results

All runs below use the same 500 exposed source-validation examples and 48
exposed weave tasks. The frozen parent scores 346/500 source answers and
21/48 weave answers. These are development comparisons, not fresh replication.

| Candidate | Source answers | Gains | Regressions | Weave answers |
| --- | ---: | ---: | ---: | ---: |
| Joint definitions plus pairwise argument ranking | 454/500 | 117 | 9 | 47/48 |
| Views selected on gold operation spans | 435/500 | 102 | 13 | 46/48 |
| Views selected on predicted operation charts | 454/500 | 113 | 5 | 47/48 |
| Predicted-chart views plus pairwise argument ranking | 458/500 | 120 | 8 | 47/48 |

The predicted-chart candidate selects `middle_mean` and `middle_last`, fitted
on 728 source-training examples. Validation selects the view combination and
length penalty using predicted operation spans and annotated input spans.
Its full-runtime source result is 415 exact programs and 438 structurally
equivalent programs. The combined ranker reaches 415 exact and 442 equivalent.
Neither satisfies the unchanged no-regression admission rule.

Coefficient lesion and hidden-token shuffle each score 0/48 programs and
answers for both predicted-chart candidates. These controls establish a
dependency on learned coefficients and hidden features in this development
run; they do not establish broad reasoning gain.

A paired-boundary pointer experiment adds an optional learned interaction
between start and end features. Legacy pointers retain their exact scoring
and serialized format. The source-trained experimental candidate produced an
uninformative validation chart; its full decode evaluation was interrupted.
Its coefficients and failed calibration receipt are retained. No runtime
success or promotion is claimed for that candidate.

## Root-cause follow-up

On all eleven remaining arithmetic program failures, a diagnostic supplying
gold operations found the correct argument phrases in the correct slots.
Their role/proposal margins favor the correct ordering. The chosen registers
are reversed. The next repair therefore concerns definition selection or
register linking, not operand-role recognition. Gold-operation diagnostics
are not counted as runtime successes.

## Verification and current activation

Focused groups passed 109, 22, 64 and 63 tests; these overlapping groups must
not be added into a unique-test count. Touched-file Ruff, compile, layering,
governance lint, writing and module-size checks passed.

The latest smoke run is red: 163 passed, one failed, one skipped. The live
surface alarm also fails in isolation with
`semantic_neural_activation_invalid:resident_manifest_drift`. No bound source
files drifted. The resident model path is unchanged; its signed migration
contract changed the steering component from deferred to qualified. The
activation binds the earlier manifest. Neither that activation nor the new
steering contract was altered to suppress the alarm. This configuration needs
its own requalification under G10/G11.

Full rows, hashes, controls and local candidate paths are recorded in
`artifacts/rlc/semantic_program_operation_views_dev_20260913/evidence-receipt.json`.
G03 remains open. G04-G12 are not closed by these measurements.
