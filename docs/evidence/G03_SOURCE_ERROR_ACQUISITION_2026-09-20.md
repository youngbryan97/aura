# Source-wide error acquisition

The small iterative trial corrected its selected training error after a
single fit had satisfied stored constraints without changing the wrong
autonomous decode. That method now reaches the existing refit command through
`--acquire-training-errors CONTROLS`.

The command scans the entire parent-bound training cohort. Every result other
than verified semantic equivalence enters mining, including unresolved
verification. Controls are chosen by source hash and geometry. Validation and
test rows are not decoded during acquisition. All training rows remain in the
separate source-retention cohort; selecting errors does not discard those
constraints.

The exported candidate carries the acquisition receipt, observation identity,
source identities and measured outcomes. It checks that the trainer used
exactly the acquired cohort. Source changes during observation reject the
receipt. No acquisition result grants serving or transfer authority.

New refit invocations default to `source_anchors_v2` validation. Legacy
`register_indices_v1` remains an explicit replay option; historical artifacts
are unchanged. Comparing local register numbers had previously counted
equivalent source-grounded programs as different.

## Full experiment in progress

A frozen checkout at `997efa864` runs the same acquisition policy, before the
command integration. The complete incumbent training scan found 755 equivalent
programs out of 764, with nine witnessed differences. It selected those nine
and 16 controls for iterative mining, retaining all 764 source examples.

The experiment will independently replay the candidate on every training row
and both models on all 500 validation rows. The test split is untouched.
These scheduled evaluations are not results. G03 remains open.

Artifacts:

- `/Users/bryan/.aura/rlc-evidence/semantic-policy-source-expansion-20260920/`
- Parent: `c3817f75a0b9e8b72c601fbbd97c0494dca423bcb30eac9683b871831701d15a`.
- Frozen implementation: `5eca4e23c36b627d461852bce7ee115eccfe6d215c0257cfeda503fb37d38402`.

## Checks

`tests/test_semantic_training_acquisition.py` checks all unresolved statuses,
split exclusion, complete scan coverage, deterministic controls, source drift,
CLI admission, full-cohort retention and exported receipt binding. Its small
integration also exercises the actual decoder. These are mechanism checks,
not general-transfer measurements.
