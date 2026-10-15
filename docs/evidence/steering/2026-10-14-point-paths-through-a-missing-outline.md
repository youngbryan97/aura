# Point paths through a missing outline

The original-case-9 run at 1c8bcd928 completed in 166.346061 seconds
without a win or a confirmed body. Both 24-press pixel cohorts completed;
they supplied 136 and 139 ordinary observations, respectively. Neither
qualified a sampling region. The run therefore did not test the preceding
facing-budget fix. Captures and receipts are retained in
`/private/tmp/codex-steer/cadence-kept-heading-original9-1014`.

A replay assembled from that run's candidate callbacks matched only 12 of
its first 200 observations. Those callbacks omit some pictures processed
by the vision worker. This replay cannot locate an original-game defect
reliably. The next diagnostic run will record every input at the ordinary
observation worker, including results that never reach the candidate
callback, with RGB digests, frame receipts, initial kinds and search hints.

The point tracker already kept named marks when no object outline was
found. Its `motion_of` method could only use the immediately preceding
picture. A returning outline therefore could not use marks continuously
observed since its last saved support. A synthetic pixel regression now
exercises that gap through `WhatMoves`, rather than only the point tracker.

The tracker now retains at most 32 pictures within the existing
0.35-second correspondence horizon. A bridge requires the same point name,
birth, consecutive picture counts, exact previous coordinates and matching
timestamps at every intervening picture. Its error includes every segment.
A missing or hidden mark breaks the bridge. Current rigid geometry and
material support still decide whether the resulting motion belongs to an
object. No control-identification threshold changes.

Saved support now uses the last observed outline in the object's path.
The public box can have been projected ahead while that support still
belongs to an older picture. Treating the projected box as the support's
location would manufacture correspondence. A mismatched path timestamp
now refuses the support. Two older test fixtures were corrected to record
the observed path when they advance `seen`.

Motion prediction also reuses bounded, read-only transition matrices and
computes only the mixture covariance diagonal used by its published error
estimates. An independent implementation of the previous full covariance
calculation matches means, weights and reported errors exactly across
irregular samples, three coordinate scales and owner/basis changes.

Before the point-history change, a profiled replay of the same 200 captured
inputs took 20.365 seconds with the old arithmetic and 18.791 seconds with
the replacement. Forecast cumulative time fell from 2.110 to 0.710 seconds.
Both versions matched all 200 recorded entity snapshots. This is one
offline profile under resident load, not a measured live speedup. Logs:
`/private/tmp/codex-steer/retained-heading-vision-profile-1014.log` and
`/private/tmp/codex-steer/retained-heading-vision-arithmetic-profile-1014.log`.

The point, material, pose, uncertainty and prediction regression suite
passed 318 tests in 39.81 seconds. These tests do not establish original-game
identity, steering or a win. The model's exclusive G09 job remains running;
the native Aura prompt path has not been demonstrated by these browser
actor diagnostics. Presentation readiness remains unproved.

Ruff, writing, changed-method size, layering, dependency graph and compile
checks passed. Smoke passed 165 tests with one skip in 46.40 seconds. The
module-size gate retains the previously recorded 194-line aggregate excess
and `ResponseGenerationPhase` method-count regression; neither changed.
