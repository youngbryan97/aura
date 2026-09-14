# Outcome learning evidence

The common outcome ledger already distinguished measured outcomes from expired,
unobserved receipts in its calibration statistics and resolution observers. Its
credit-distribution function nevertheless sent both to `OutcomeLearner` and the
mattering model. An unobserved expiration therefore became a failed-action
training example and a measured-looking consequence through another path.

The production feed now requires `OutcomeReceipt.is_evidence` for these two
learning consumers. An expired receipt retains the existing accountability credit
policy, but does not train action correctness or mattering. Measured successes
and failures still reach both consumers. The outcome learner receives the
receipt identifier and observation kind with each new record.

Finite-number validation occurs before removing a pending receipt. A malformed
or non-finite observation cannot consume the receipt, poison calibration with
NaN, or prevent a later valid observation from resolving it. The shared evidence
predicate requires a resolved, measured, finite outcome in the supported range.

Validation: 61 focused tests passed across the outcome ledger, contextual action
values, action-value model, mattering, and the new feed regressions. Tests exercise
the production `open`, `resolve`, and `sweep` paths against isolated SQLite stores
with observable learning consumers.
The measured-evidence predicate is registered as an executable invariant.
Smoke: 163 passed, one skipped, one failed on the existing G10 manifest alarm.

This repair is not independent correctness adjudication. Callers still determine
what was observed. It does not retroactively repair prior learner records, change
the accountability-credit policy, establish cross-process exactly-once delivery,
or close the shared procedure-to-external-assessment loop. No broad reasoning
claim or G09 closure follows from these tests. Desktop validation is separate.
