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

## Original-game run from 0dc75aa5c

The assigned original game 9 ran for 167.340228 seconds, exit 0, without a
wrapper timeout. Artifacts are in
`/private/tmp/codex-steer/settled-exposure-original9-1014`.

One fast menu stretch returned to navigation in 0.3 seconds with no key
downs. The game subsequently showed its level-completion screen, captured
in `game9_0073s.png`, and Level 2, captured in `game9_0083s.png`. The final
capture still showed Level 2's Play overlay. The campaign reported no whole
game win. This is observed level progress, not reliable planned control:
there were 524 decisions and zero assigned-body decisions. Median decision
picture age was 0.203451165 seconds.

The actor recorded one complete 24-press search with selected settling time
0.348795910 seconds, and a later 18-press search that timed out with selected
settling time 0.361587543 seconds. Both released their inputs. Neither
produced a qualified region or sampling hint.

The diagnostic observer reused ordinal filenames across search instances.
The later incomplete search overwrote the first 18 pairs of the complete
search. The root's 24 files therefore cannot form one valid replay. An
attempted combined replay is marked invalid in
`/private/tmp/codex-steer/motion-layer-replay-1014/invalidation.json`; its
aggregate metrics and region verdict are excluded. Per-pair bytes and
receipts remain genuine, but their grouping was lost. Existing artifacts
are preserved. Future observers create a distinct directory for each
discovery instance and record its schedule before physical collection.

Body identification, continuation through current menus, repeatability and
ordinary native-app presentation proof remain open.
