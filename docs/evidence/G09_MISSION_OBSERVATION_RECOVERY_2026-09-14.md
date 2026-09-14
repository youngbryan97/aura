# Mission Effects and Delayed Observation

The mission executor treated an unavailable post-action observer as a failed
action. Recovery could then repeat an action that had already succeeded.
The durable graph also reused its display summary: verification and recovery
arguments were omitted, and result values were stringified and truncated.

## Repair

- Verification distinguishes measured match, measured mismatch, unavailable
  observation, and an omitted check. Only measured matches count as successful
  observations. Unavailable and omitted checks do not enter the success rate.
- An executed action enters `awaiting_verification` and persists its receipt
  before calling the observer. Retry observes the retained effect. It does not
  repeat execution because a checker timed out, was cancelled, or failed.
- Each mission admits one advancement at a time. Cancellation releases that
  admission while retaining the outstanding observation.
- Durable graph records contain all task fields and typed result values.
  The existing display summary remains unchanged.
- SQLite errors propagate. Before execution, storage failure leaves the task
  pending; after execution, it leaves the receipt awaiting observation.
- Observation subprocesses require a successful read and an actual exit
  status. Timeout and cancellation use the existing process-group reaper.

## Evidence

`tests/test_mission_observation_resume.py` writes a real file, makes its
observer unavailable, closes and reopens SQLite, and verifies the file without
executing the write again. It also exercises measured mismatch, cancellation,
concurrent advancement, full record round-trip, failed process reads, omitted
checks, and storage failure before and after execution.

111 tests passed across that suite, task-planning runtime, executable mission
primitives, real-mission contracts, conclusive verification, shared procedure
execution, and goal-bound procedure planning.

Lint, compilation, governance, layering, and writing gates passed. Smoke
reported 163 passed, one skipped, and the existing installation alarm:
`semantic_neural_activation_invalid:resident_manifest_drift`, with no drifted
bound source files. This checkpoint does not re-seal that historical evidence.

The invariants `planning.task_record_roundtrip` and
`capabilities.verification_observation` are registered beside their mechanisms.

## Limits

This is not an exactly-once transaction with an external effect. A process
loss between the effect and its persisted receipt still needs effect-specific
reconciliation. SQLite access in the mission owner is still synchronous.
No broad live-task gain or G09 closure is claimed by these tests.
