# G03 function-coordinate development

The preceding component interventions attributed both validation gains and
losses to the relation head. Its factorization has a coordinate ambiguity:
reciprocal factor transforms preserve the relation function but change the
ordinary Euclidean update penalty. The opt-in
[function-coordinate metric](../G_RELATION_FUNCTION_GEOMETRY.md) removes that
ambiguity from the separate first-order factor contributions. It does not
prove that the metric improves transfer.

## Paired source measurements

Both attempts reused the same 16 source-training examples, 128 operation and
boundary retention examples, 5,063 inequalities, and 100 development-validation
examples as the scaled subspace trial. Validation examples did not enter fit.
The operation classifier remained frozen. Every stored proposal still had to
pass the original inequalities; no tolerance or retention floor was relaxed.

| Attempt | Training before/after | Validation before/after | Accepted updates |
| --- | --- | --- | --- |
| Function coordinates | 13/16, 13/16 | 91/100, 91/100 | 0 |
| Function coordinates with QR/SVD polish | 13/16, 13/16 | 91/100, 91/100 | 0 |

Both completed all 116 paired decodes. Each has zero measured gains and zero
measured regressions. These are failed repair attempts, not evidence that
function-coordinate learning preserves unseen answers.

Artifacts under `/Users/bryan/.aura/rlc-evidence/`:

- `semantic-functional-relation-20260919.json`, receipt
  `f15ececb53c5646282c417566b98e964351e16a4bb7a327142521c678f143cf6`.
- `semantic-functional-relation-qr-20260919.json`, receipt
  `4869f994bd12f3016c9fb94d5d6b8ca9d15ccb8a068cd08ff28e7e9210c796d2`.

## Numerical boundary

The affine projection failed independent numerical verification after a
physical-storage reserve. An analytic nearly opposed-constraint example
reproduced a Gram conditioning failure; solving active equalities through
QR/SVD repairs that example at the unchanged verification tolerance. It did
not repair this source problem. Passing the analytic test therefore does not
justify accepting the source update.

Failed fits with a checkpoint now retain a separate checksummed projection
diagnostic containing normals, requirements, anchor, fit identity and failed
receipt. The accepted numerical checkpoint remains separate. This permits
direct investigation without repeatedly mining and decoding the full cohort.
The diagnostic uses the existing file-write gateway; the ownership inventory
records that additional call explicitly, without granting serving authority.

G03 remains open. Neither candidate is promoted, and this record establishes
no fresh transfer, broad gain, fusion qualification or frontier comparison.

## Repaired active-set result

Direct replay found dependent active rows with incompatible equality targets
after storage reserves. Rebuilding the active set from zero resolves that
source projection without changing its verifier. A further null-space pivot
handles incompatible active equalities generally; 100 generated feasible
systems pass from both cold and warm starts. The latter pivot was tested
separately and was not loaded into the already-running source trial.

The completed cold-start source trial accepted three updates and satisfied
all 5,063 retained inequalities, with minimum stored margin
0.10000000760903127. Training improved from 13/16 to 16/16. Validation remained
91/100 with zero gains and zero regressions. This avoids the prior
coefficient-space trial's four regressions, but demonstrates no validation
accuracy gain. Broader source coverage and independent transfer remain needed.

Artifact: `semantic-functional-relation-cold-20260919.json` in the same evidence
directory. Receipt:
`bcb0e2ff0e0dd51f2889d3a2c8ffcd8f7c09eae6098eac2c20665ea031e3b9b2`.

Validation: 80 focused numerical/checkpoint tests pass. Smoke passes 164 tests
with one skip; lint, compile, governance, layering and writing gates pass.
No serving model changed.
