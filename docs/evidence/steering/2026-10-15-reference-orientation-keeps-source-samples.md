# Reference orientation keeps source samples, 15 October 2026

The supplied-reference matcher searched scale and translation. Its local
patch fits could not recover a large initial rotation. It now accepts a bank
of clockwise-radian orientations. The combined scale/angle bank is limited to
32 entries; patch qualification still examines at most 24 distinct proposals.
The default remains the existing zero-angle search.

Each resized/rotated template carries a pixel-centre affine map from the
original reference. Final fits compose that map with the independently
measured patch transform. Matched source points return through its inverse.
This also accounts for the half-sample offset in a resized raster. A failed
orientation cannot suppress a different orientation's fit in the same region.
Repeated hypotheses with agreeing projected reference points coalesce;
different visible copies remain competing appearances.

No upstream source was copied. The implementation uses the installed Pillow
raster API and the existing original colour-patch and redundant-geometry
checks. The offline probe accepts repeated `--angle-degrees` arguments and
records its angle bank. It refuses a bank exceeding the combined budget.

The focused checks passed 88 tests in 11.34 seconds. Rotation tests compare
projected points with the weighted centre of a separately rasterized marker,
including resizing, both rotation directions, quarter turns and a half turn.
They also check duplicate visible material, failed-orientation suppression,
invalid banks and the registered `rotating_references_keep_their_source_samples`
claim. Existing source-coordinate and instance-name checks remain green.

An original case-18 static `ship` export, including source-coordinate
registration, completed in 3.27 seconds using the existing bounded external
renderer. The unchanged recorded gameplay frame 304 yielded one appearance
with seven points in 0.242031 seconds under the default scale bank and zero
angle. Its source is
`swf:a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
This reconfirms that reference correspondence on those pixels survived the
coordinate change. It is neither a new game run nor evidence of a rotating
original-game body being controlled.

Smoke passed 165 tests, one skipped, in 43.58 seconds. Lint, compile, layering,
DEPS consistency and writing passed. Receipts and the fresh ship export are
under `artifacts/flash-campaign/2026-10-15-oriented-references/` and
`artifacts/flash-campaign/2026-10-15-oriented-ship-reference/`. The prior
129 governance regressions were not changed. Ordinary reference acquisition,
track attachment and source-contact control remain open. No native runtime,
G09, original game 9 or holdout was operated.
