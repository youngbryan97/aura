# Planning forecast attribution, 2026-09-18

This repairs the existing world-model calibration and screen-pursuit path.
It does not establish broad reasoning gain or live desktop qualification.

## Defects and changes

- Perfect zero calibration error was treated as a missing value. Signed
  aggregate errors could also cancel between bins. The shared curve now
  reports weighted absolute bin error and actual per-bin confidence means.
- Planning subtracted one global confidence error. It now uses observed
  accuracy in the requested confidence bin after the existing eight-example
  minimum. Unmeasured bins retain the original claim.
- The runtime counted alternative actions as prediction distance and read
  confidence at grading time. It now retains pre-action confidence and grades
  the forecast at the delivered action count.
- Grading cleared the action count before the rule learner checked it.
  Multi-action outcomes could therefore train a single-action rule. The count
  now survives grading; single-action learning still executes.
- Partial keyboard delivery retained the full planned forecast. The action
  helper now selects its precomputed delivered prefix before waiting for the
  world. Failed delivery clears the pending deliberation. No prefix prediction
  is reconstructed after the outcome.
- Model predictions and the no-change control now use the same existing
  correctness predicate. Missing observations do not become model failures.
- Nonfinite probabilities, invalid errors, and inconsistent saved bin
  statistics are rejected before mutation. Legacy bin means remain unmeasured.
  Legacy runtime planning statistics have no delivered-action provenance and
  are not imported into the corrected version-2 measurement.

## Verification

121 focused tests passed in 11.87 seconds. This includes the actual async
action helper with controlled delivery receipts of zero, one, two, and three
actions; hardware operations are substituted in these tests. It also includes
post-grading learner calls, same-contract baseline comparison, bin-error
cancellation, persistence, and invalid-input rejection.

Smoke: 164 passed, one skipped in 61.51 seconds. Lint, compile, governance,
and layering pass. Governance still reports its existing 1,919-call migration
debt; this change does not claim to remove it.

The desktop was not restarted for this checkpoint. G09 remains open, as do
the fresh controlled broad-domain measurements.
