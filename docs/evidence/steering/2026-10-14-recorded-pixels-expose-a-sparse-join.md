# Recorded pixels expose a sparse joined outline

## Original run at 7f6a4102f

The bounded assigned-case-9 run in
`/private/tmp/codex-steer/current-texture-horizon-original9-1014` completed
in 166.2917037499 seconds. It made 648 choices, identified no body and did
not win. Median picture age was 0.2330577699 seconds. Its two pixel cohorts
ended with `unmatched_exposure` after 11 and 8 admitted presses. Physical
release was confirmed. Their ordinary observations continued for 58 and
44 frames respectively.

The diagnostic retained the browser's native encoded public screencast
frames and their source receipts. All 2,057 capture records matched a
source event; the first 1,200 encoded frames were retained. Re-decoding a
frame with its original crop reproduces its recorded RGB digest. Retention
cost had median 0.0004164161 seconds and maximum 0.064056917 seconds; this
instrumented run is not a clean latency benchmark.

## Exact offline reproduction

The first 95 actual observer inputs were replayed through ordinary
`WhatMoves` at their original capture times. On frozen 7f6a4102f, all 95
reproduced the recorded current entity numbers, boxes and landmark counts.
The baseline replay is retained separately in
`/private/tmp/codex-steer/replayed-material-baseline-1014`.

Track 27 had seven named points and an 11 by 10 box. After 0.10858 seconds,
its outline joined a thin painted line, grew to 18 by 29 and lost its
identity two observations later. This was within both the old 0.25-second
and current 0.35-second correspondence horizons.

Tracing the exact failing capture found an earlier test that skipped
texture separation unless the joined foreground contained more than twice
the prior support's pixels. A thin line greatly enlarges a box while adding
few pixels. Current marks and texture were present, but that preliminary
test prevented them from being examined.

The repair permits a bounded texture fit when either foreground count or
spatial extent has grown. Current RGB support, named-point ownership,
timestamps, birth, spread and support thresholds remain required. The
repaired replay retains the small track through observation 95. It does
not establish that this track is the player. Unassigned outline remainders
also remain available to the ordinary eyes.

Tests exercise both dense and sparse outlines, including changed texture,
stale or duplicate points, birth mismatch, insufficient spread, missing
motion and a matching-colour line without motion evidence. Actual generated
pixels also check retention of the prior moving track while crossing a
static line at two observation cadences.

## Paired exposure

In the first refused cohort, baseline-to-dispatch age varied by 0.08306
seconds, exceeding the existing 0.08-second comparability allowance. The
second cohort had 0.09043 seconds of that drift. Most drift occurred while
processing the neutral baseline before dispatch.

The collector now takes a fresh raw baseline after processing its neutral
observation, dispatches promptly, then supplies that exact baseline to the
ordinary tracker under the bounded physical lease. Its historical capture
time still attributes it to neutral input. This preserves intermediate
material history without adding observer service time to the paired
baseline-to-input interval. Existing exposure refusal thresholds remain.

These changes repair reproduced seams. The original-game causal chain and
successful ordinary live play remain unproven.
