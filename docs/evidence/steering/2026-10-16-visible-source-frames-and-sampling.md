# Visible source frames and their sampling grid

This checkpoint extends the ordinary reference worker without giving a
visual match control identity or a task role. It addresses two measured
coordinate errors and a point-capacity loss. The recorded composite still
has qualified geometry on only eight of 100 frames. Demo readiness remains
unproved.

## Coordinate corrections

The reference matcher reduced the captured picture with BOX sampling, then
mapped sample indices back by division alone. That omitted the sample-centre
offset. Independent raster-marker tests measured errors of 2.121473 and
0.707101 original pixels at the two reduction sizes. Composing the offset
reduces both errors below 0.25 pixels. These tests use independently drawn
marker footprints; agreement between two calls to the same matcher could
not detect the shared bias.

The earlier ordinary-worker fresh-match checks therefore did not establish
absolute pixel accuracy. They remain measurements of agreement between two
matches. This corrects that interpretation without changing the earlier
record.

Named point streams also need their actual sampling grid. Native point
coordinates are scaled to the working picture without a centre offset.
`SampleGrid` now retains original, sampled and working dimensions and maps
between working indices and original sample centres. It handles each axis
separately, including rounded heights. Independent BOX-footprint centroids
test this conversion. Object-bound source points use the same grid.

## Geometry before object grouping

`WhatMoves` publishes an immutable, bounded `MaterialMarks` stream before
grouping points into objects. It retains the consumed RGB receipt, epoch,
sequence, actual sampling grid and original named point births. Detached
and process-worker observations keep that stream's ownership.

An existing object-bound source point still requires unique material
ownership. When that cannot be established, a separate geometry-only
channel can locate a source frame through visible named marks. The initial
reference match still requires six verified correspondences. A rigid frame
requires at least three non-collinear named material marks; affine fallback
requires six. Both retain 70 percent of their original marks and the existing
one-working-pixel error ceiling. Missing marks cannot be replaced by newly
detected corners. The published frame has no object number, controlled
identity, semantic role or collision surface.

The ordinary wrapper publishes these frames separately from object-bound
points. Late worker results must qualify against current material before
replacing an older continuing binding. Different copies remain distinct.

## Capacity feedback

In the frame-304 pad replay, four original marks passed initial pixel
support after the coordinate corrections. Three captures later, two marks
were absent. Instrumenting the existing allocator established that all
four remained visually measured before allocation: capacity selection
dropped the two younger marks from 352 candidates while retaining 192.
That loss occurred after visual tracking, rather than from a failed patch
match at that step.

Active reference consumers can now request short retention leases for
currently visible named marks. Requests pass through the ordinary wrapper
and the separate-process observation protocol. They expire within one
second, use at most 64 marks and one third of the point budget, and cannot
restore a missing mark, a changed birth or a different stream epoch.

Current rigid support can also advance a bounded sampling proposal around
the reference. Existing validated exploration hints take precedence. The
proposal changes where new points are sampled; it supplies no object or
control authority. Missing, deformed, oversized and expired support refuses
the proposal. Tests exercise the normal consumer-to-`WhatMoves` connection.

## Recorded measurements

All runs below use 100 existing case-18 RGB captures, source-paced through
ordinary asynchronous observation. They start no model and send no game
input. Source bytes and operator-published reference bundles are supplied
to the private replay state root.

The retention run published object-bound points on 52 frames and the
composite's geometry-only frame on eight. Eight observations admitted three
retention requests. Its maximum event-loop pulse gap was 0.033320 seconds.

The sampling-proposal run published object-bound points on 53 frames and
the composite frame on the same eight frames. Its maximum loop pulse gap
was 0.029324 seconds. Sampling proposals did not improve composite-frame
availability in this recording. An original supporting mark was later lost
despite an admitted retention request; retention does not solve all losses.

Seven of the eight geometry-only frames could be checked by a fresh pixel
match. All seven agreed within one working pixel, with largest difference
0.375641. One remained unmeasured. This is fresh correspondence evidence,
not an independent game-state measurement or a gameplay pass.

Artifacts are under
`artifacts/flash-campaign/2026-10-16-visible-source-marks/`. They include the
failed initial carriers, coordinate tests, bounded replay receipts, ordinary
retention and sampling runs, fresh checks and final focused-test receipt.
The replay records `task_passed=None` and `presentation_qualified=False`.

## Remaining connection

The task controller still needs a source-derived contact predicate and a
verified binding of source instances to current observed occurrences.
Neither a matched source origin nor a box centre defines a landing surface.
Reading the case's bound movement method also exposes unsupported `Delete`
semantics and exhausted symbolic path budgets. No budget or control-identity
threshold was raised to conceal these gaps.

Fresh uncached asset export, native deployment, original-game wins, the
56-game campaign and independently witnessed native play remain open. G09,
the holdouts and the exclusive corpus job were not disturbed. No resident
runtime was started or restarted.

The final focused test set passed. Smoke passed 165 tests with one skipped;
lint, compile, layering, generated DEPS consistency and writing passed.
Governance retains 131 existing regressions. Module size retains the same
unrelated phantom-browser ceiling, response-phase method regressions and
197 excess lines. Neither baseline was expanded. The full suite was not run
beside the exclusive corpus/model job.
