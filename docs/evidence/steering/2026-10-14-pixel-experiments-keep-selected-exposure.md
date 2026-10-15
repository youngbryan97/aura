# Pixel experiments keep selected exposure — 14 October 2026

The original-game captures from `1b8a88bd3` showed short physical presses
with weak or absent matched movement. Inspection found that sleeping for a
selected settling interval did not constrain the frame returned afterward:
the collector could accept a queued capture from just 20 ms after input
acknowledgement, even when the selected interval was longer.

## Change

The collector requires the whole capture to start after the selected
neutral or settled input interval. A later delivery cannot make an early
capture qualify. Queued captures retain their timestamps and are skipped
within the existing bounded capture retry, rather than relabelled.

Before any pixel experiment, the actor selects one exposure using the
existing control-cadence estimator: recent upper-quartile frame intervals,
two observation intervals and a 20 ms response allowance. The minimum is
90 ms and the maximum is 500 ms. The selected value is fixed across both
keys and both cohorts before their outcomes are seen. Missing cadence or a
demand beyond the cap leaves the experiment unavailable.

The collection allowance remains finite: at most 30 seconds, bounded
further by the actor's remaining time. Physical acknowledgement and capture
waits, owner checks, frame-custody requirements, exposure matching,
independent confirmation and release cleanup retain their bounds. The
selected exposure is included in the actor's search record.

## Verification before the checkpoint

The related acquisition, custody, navigation, cadence and real Chromium
suites passed 142 tests in 30.17 seconds. The new regression supplies a
post-action capture from before the selected settling time, verifies that
it is skipped, and checks all 24 final capture bounds against the chosen
interval. Actor tests verify fixed measured-cadence configuration and
refusal when the requested sampling time cannot fit the cap. These tests
do not demonstrate original-game body identification or a win.

The initial test collection caught a missing invariant import; it was
corrected before the passing run. No source revision with that error was
committed or used for an original-game run.

The original game has not yet been measured from this change. Native Aura
prompt proof, successful control and presentation readiness remain open.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke passed. Smoke has 165 passing tests and one skip.
The existing aggregate module-size excess of 194 lines, response-generation
method excess and 123 governance regressions remain. No baseline was
changed. The gate log is
`/private/tmp/codex-steer/settled-exposure-gates-1014.log`.
