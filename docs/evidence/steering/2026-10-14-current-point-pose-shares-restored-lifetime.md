# Current point pose shares restored lifetime, October 14, 2026

The point-meaning layer can restore an object's earlier birth when a new
outline carries the continuing marks of an unmatched object. Its caller
updated `Thing.born` but retained the newer `PointPose.born`. The current
angle reader rejected the mismatched lifetime, and the next point path
started over.

A current pose now shares that restored lifetime only when its original
birth matched the object's original birth and its reference belongs to
that lifetime. Both the object and pose must be current in this capture.
Stale, unrelated or missing references remain unchanged and unqualified.
The angular reference remains its own measured basis. This change borrows
no old heading and establishes no control identity.

The actual point-meaning layer supplies the alias in the regression fixture.
Tests also exercise stale objects, stale poses, unrelated births and invalid
references. The scoped suites passed 122 tests in 18.00 seconds. A registered
canary protects current lifetime alignment without retagging stale geometry.

This repairs a data consistency defect. It has not yet improved an actual
browser result. Presentation readiness remains open, including reliable
body identity, sustained control and independently witnessed game completion.

Gate log: `/private/tmp/codex-steer/current-pose-birth-gates-1014.log`.

Ruff, writing, changed-method size, layering, dependency consistency,
compile, lint and smoke passed. Smoke passed 165 tests with one skip in
46.66 seconds. The inherited global size debt remains open. Governance
reports 123 existing regressions, with neither changed module in its
offender list. No baseline was changed. Governance log:
`/private/tmp/codex-steer/pose-birth-governance-1014.log`.
