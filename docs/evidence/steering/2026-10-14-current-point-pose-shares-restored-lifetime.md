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

## Browser measurement at 42e1db888

The clean frozen checkout at `42e1db888f3055d384de0b11822c51355858b768`
ran the assigned game 9 for 193.606 seconds. It reported a loss followed
by an unread run and no win. None of its 1,102 recorded decisions had a
body attribution or qualified optical heading. Median picture age was
0.18040 seconds. The lifetime fix therefore has no demonstrated game gain.

This run also retained material-point names, point-pose coordinates and
capture-clock bounds. A readout found 62 transitions where a new material
record carried at least three names and half the names of a disappearing
record. These are diagnostic transitions, not new control witnesses or
proof of physical actor identity. The capture bounds inspected during the
run spanned about 1.4 ms, so a broad clock interval does not explain that
inspected picture's age.

Next work should test geometric and temporal continuity across those
transitions and determine what prevents independent control trials from
accumulating. The point-meaning layer currently propagates a birth tag;
the actor can still reset its experiment when a visual number changes.
Any transfer of control history needs current, distinguishing evidence.

- Result: `/private/tmp/codex-steer/material-names-original9-1014/results.json`
- Material and pose records: `/private/tmp/codex-steer/material-names-original9-1014/candidate-measurements.jsonl`
- Diagnostic transitions: `/private/tmp/codex-steer/material-names-original9-1014/material-transfers-readout.json`
