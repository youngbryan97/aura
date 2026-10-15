# Unknown controls get independent attempts

The original run at 5024e69d1 repeatedly tried its first unknown control
after selecting a body. `try_controls` chose `unknown[0]` each time.
When that control's response could not be measured, the other current
unknown controls received no trial.

The fallback now retains issued counts for each control under the current
identification epoch, body number and birth. It selects a least-tried
current authorized unknown control. Each control has two issued attempts;
unavailable outcomes and failed dispatches still consume an attempt.
The retained bounds are 32 distinct controls per body and 64 body owners.
Reordering, removing and restoring controls cannot renew their allowance.

Existing active trials, their neutral observation phases and counter-motion
experiments keep precedence. An already-held ordinary key is released and
observed at rest before a new isolated trial starts. Exhaustion lets the
actor reach later calibration or planning even when its older global
`trying` count is below the discovery cap. Missing body custody yields
explicit unavailable status instead of an unproductive discovery loop.
The repair does not mark any response known.

The focused suite passed 81 tests in 11.12 seconds. It includes physical
input receipts and deadlines with every control response unavailable,
owner changes, key churn, failed dispatch, neutral-phase preservation and
the ordinary actor reaching its planning path after exhaustion. The new
invariant has a registered measured claim.

Original-game improvement and presentation readiness require a subsequent
run. Fair scheduling alone does not prove usable control or a win.
