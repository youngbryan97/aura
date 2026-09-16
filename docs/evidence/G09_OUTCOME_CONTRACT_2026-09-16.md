# Learning after an outcome rule changes

## Observed failure

The live neural stream reported episodic writes deferred until its 256-entry
queue shed pending observations. The admission reason was
`sync_approved|ontogeny:deferred`. This is relevant to G09: lost observations
cannot participate in retained knowledge or later correction.

The existing consequence repair (`b1330c0e6`) associates landed or shed writes
with the decision that deferred them. Its migration (`9bfae62ee`) revoked an
admission grant earned under the previous goal-class grader. The runtime log
then recorded authority granted again, including at 04:42:44 UTC on September
16. All these revisions preceded that runtime's launch snapshot.

The trainer still selected the old schema, and the head loader still accepted
the old checkpoint. A read-only database inspection found the same
`4bc2fb6f96be` schema on both 94 deferred successes from
`executive.goal_returned` and 67 deferred failures from
`executive.intent_complete` in the inspected September 15-16 decision window.
These are counts from the on-disk database observation, not a complete live
transaction export or a causal estimate of the loss rate.

## Repair

- `FeatureSchema.outcome_contract` binds the meaning of labels into the
  existing evidence identity. The default preserves undeclared legacy schemas;
  executive admission now declares owned-consequence version 2.
- Existing schema filtering retires incompatible training rows and head
  checkpoints. Historical records are not deleted or relabeled.
- The authority ledger persists the active contract. A changed or unbound
  grant returns to observation. Re-registering the same contract does not
  repeatedly revoke newly earned authority.
- Authority comparisons exclude incompatible rows, and a deciding grant with
  no ready head now demotes instead of returning early as still authoritative.
- Episode deduplication includes the evidence schema, so matching values under
  different outcome semantics remain distinct observations.
- Comparisons count each resolved episode once, regardless of collapsed
  attempt volume. A red test counted two observed outcomes as two million
  trials. Comparisons now ignore other control points and reject conflicting
  outcomes attached to one episode identity. Their receipts name the unit.
- The runtime invariant `ontogeny.authority_matches_outcome_contract` checks
  deciding grants against their declared current contract. Other control points
  with unchanged, undeclared contracts are not revoked by this migration.

## Verification and boundary

The initial regression tests failed on the missing contract. The first expanded
focused suites passed 189 tests, including persistent corpus retention,
checkpoint reload, authority re-registration, stale comparison rejection,
deduplication, positive and negative invariant controls, and existing deferred
write behavior.

This statistical repair does not make the historical comparison randomized
or remove dependence between different decisions in one workload. It removes
pseudoreplication of a single resolved outcome; controlled causal comparisons
remain a separate obligation.

This repairs reuse of evidence after its grading meaning changes. It does not
establish broad reasoning gain, make an imperfect outcome grader correct, or
prove that memory admission now drains on the live instance. Live deployment
and observation follow the checkpoint; G09 remains open.
