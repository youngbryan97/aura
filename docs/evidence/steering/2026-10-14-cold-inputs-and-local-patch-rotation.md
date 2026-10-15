# Cold inputs and local patch rotation, 14 October 2026

The streaming checkpoint `92548cbbe96eb564cadc5864fa41aeb8442bd3d2` reached
actual independent input trials in assigned original case 9. The first run
completed in 106.263 seconds, with 510 decisions, zero body decisions and
511 capture-custody admissions. Its one experiment completed all 24 presses
and confirmed physical release. It found no region and applied no hint.
Median decision picture age was 0.217667 seconds.

A second run saved the actual admitted RGB pairs, input windows and motion
signatures. It completed in 105.064 seconds, with 462 decisions, zero body
decisions and median picture age 0.244065 seconds. All 24 presses completed.
Neither run demonstrated a win. Both ran the saved revision in the separate
frozen proof checkout; the second added bounded observational artifact writes.

The paired measurements expose the failed experiment. Twelve presses each
of left and right had zero correspondences moving more than 0.5 working
pixels. The largest absolute left feature was 0.006311; all right features
were zero. There were 92 cells with continuing angular-fit support and 106
with continuing translation support. Missing capture custody or wholesale
loss of point support did not cause this rejection. These presses supplied
no measured body response from the resting state.

Artifacts are in `/private/tmp/codex-steer/streamed-pixel-search-original9-1014/`
and `/private/tmp/codex-steer/paired-control-search-original9-1014/`.

## General changes

Cold experiment selection uses current declared nonzero translation
channels before turning channels. The source body model already separates
pace, push, lift, drive and turn. Obsolete keys and nonfinite or zero channel
values supply no translation priority. When a source model is incomplete,
its current control order is retained. No arrow-name preference remains.
Tests use unrelated opaque key names.

Sparse motion sampling retains distributed reference points and reserves
remaining capacity for actually changing pixels. Corners use local gradient
strength; a strongly textured backdrop cannot set the threshold for every
small changing patch. The reference estimate retains its actual population
count and is computed before these extra response samples. Point, RGB and
round-trip limits remain bounded.

The RGB matcher had already fitted each patch's rotation, then discarded
that value after its backward check. This checkpoint retains it as a
separate local-orientation search channel. The patch residual and the joint
translation/rotation normal matrix provide conditional fit uncertainty.
Flat, rank-deficient, nonfinite, oversized and out-of-range inputs remain
unavailable. Local texture orientation supplies no rigid-body yaw, facing,
physics response or control identity.

A small response can occupy one cell. Screen-wide maximum-statistic testing
and the separate confirmation cohort still apply. The measured response
bounds remain recorded. Point sampling can inspect one surrounding cell,
because matched windows and encoded edge pixels can respond outside the
body's centre. That larger sampling region does not change the tested
pattern or fabricate an object mask.

## Measurements

The new and related suites passed **148 tests in 28.33 seconds**. A final
range/budget guard recheck passed **22 tests in 16.14 seconds**.

Chromium fixtures exercise real keyboard events and the production reader,
collector, extractor and confirmation code. Both a 48-pixel and a 20-pixel
rotating patch produce a confirmed nearby sampling area. A no-effect scene
is rejected. Each uses 24 independent acknowledged presses and 48 physical
down/up events, and leaves body identity unassigned.

The smaller fixture initially failed despite completed acquisition. Added
corner capacity alone did not fix it. Retaining local patch rotation did.
Its strongest response can include nearby encoded boundary pixels; the
fixture verifies the sampling area rather than declaring that response's
rectangle to be the body. Material continuity and control identification
still require their existing independent evidence.

These are search and acquisition measurements. The new selection and patch
channel have not yet been measured on the original game. Native Aura prompt
performance, original-game body control, wins and presentation readiness
remain open. Holdouts and the exclusive model job are preserved.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke pass. Smoke has 165 passing tests and one skip in
41.52 seconds. The existing 194-line aggregate module-size debt,
response-generation method debt and 123 governance regressions remain.
No baseline was changed. The log is
`/private/tmp/codex-steer/cold-patch-gates-1014.log`.
