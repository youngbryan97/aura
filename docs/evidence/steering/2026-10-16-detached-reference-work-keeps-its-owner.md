# Detached reference work keeps its owner, 16 October 2026

Applying an old reference correspondence directly to a newer observer is
correctly refused by consumed-pixel custody. That also means a slow matcher
needs its own captured material state. Giving its copied state a new observer
identity would prevent a valid result from reaching the original live track.

`keep_material_frame` now copies the consumed RGB image, current masks and
point bases, together with the original observer's instance query receipts.
Its pixels and masks are read-only. It copies point state independently of
later live updates. It refuses changed captures, duplicate IDs, unsupported
masks and state beyond its budgets: 4096 published instances, 16 MiB of
support masks and 8192 material-mark entries, plus the existing 24-million-
sample RGB limit. It cannot retain only a convenient subset of competing
owners when a budget is exceeded.

`SourceImageCoordinates.on_track` accepts this captured frame. Attachment
still needs its exact pixels and unique visible ownership. A resulting point
can then reach the original live observer only through the same instance
birth and continuing measured material basis. Another observer, a restarted
basis or stale geometry remains unqualified. The source point still does not
replace the motion origin, establish control identity or supply a task role.

The replay tool can advance the live observer before delivering its result,
using `--deliver-after-frames`. On the existing case-18 recording, the frame-304
correspondence was delivered after frames 309, 312 and 315: 0.468009 seconds
of source time later. The saved frame qualified the original seven owned
correspondences, and the delivered point reached the current track at
(202.798521, 41.849774) working pixels, with estimated error 0.397133.

Thirteen of the remaining 17 frames retained a current point. The same four
losses from the earlier experiment remained explicit. Seven subsequent
frames had an independent fresh reference match; their largest difference
was 0.216533 working pixels. At the delivery frame itself, an independent
reference match was unavailable, so that comparison is unmeasured. The
replay took 8.424365 seconds. This is deliberately delayed recorded-data
delivery, not a benchmark of an asynchronous live matcher or a new game run.
Its receipt is
`artifacts/flash-campaign/2026-10-16-detached-material-owner/delayed-replay.json`.

Focused checks passed 247 tests in 35.04 seconds. They include delayed
attachment through the ordinary point-continuation helper, refusal by a
different observer or replacement basis, copied-state independence,
read-only masks/pixels and bounded retention. The canary is registered as
`detached_material_keeps_its_original_observer`.

An ordinary runtime reference provider and its worker lifecycle are still
needed. Source asset acquisition, destination occurrence binding and
source-derived contact conditions remain open. This checkpoint grants no
native-app readiness or original-game win. The G09 model job, resident
checkout/runtime, original game 9 and holdouts remain untouched.

Smoke passed 165 tests, one skipped, in 50.85 seconds. Lint, compile,
layering, DEPS consistency and writing passed. Bounded verification receipts
are in `artifacts/flash-campaign/2026-10-16-detached-material-owner/`.
