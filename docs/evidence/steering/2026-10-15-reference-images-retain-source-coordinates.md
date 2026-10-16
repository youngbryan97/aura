# Reference images retain source coordinates, 15 October 2026

The original case 18 code compares differences between the source-coordinate
origins of its objects, alongside speed and rotation guards. An exported
image's bounding-box centre cannot replace those origins.

A second diagnostic also found an instance-naming problem. Replaying twenty
captured frames through the ordinary pixel observer retained the platform as
several pieces. The largest piece owns seven of the 27 reference points.
It shares kind 0 with the controlled ship and another piece. The original
record confirms selected body 1, kind 0, at that capture. Labelling every
member of that visual kind would conflate distinct instances. This remains
a separate naming and goal-binding problem.

## Source-coordinate registration

The pure coordinate module admits a bounded, static exported-frame SVG with
an explicit source transform. Unsupported structures, external resources,
active content, unresolved transforms and missing geometry refuse. A missing
registration never defaults to an image centre. Fractional viewport padding
keeps source pixel units.

The exporter gains `--with-registration` for one sprite or button frame.
It exports the same source frame as SVG, then rasterizes that document in
an owned Chromium software browser. Existing redundant appearance matching
must align this second raster with the reference PNG and reconcile the
whole frame extent. Only then is its source transform carried into PNG
coordinates. The packet keeps both byte identities and measured fitting
error; constructor inputs are copied into immutable geometry.

Projection into an observed picture requires the expected source identity,
exact reference pixels and exact picture pixels of the fitted appearance.
This supplies a coordinate point, without collision geometry, a task role,
temporal identity, a controlled body or permission to send input.

The adapter follows the external renderer's frame-export convention, checked
against its [pinned 26.3.0 frame exporter](https://github.com/jindrapetrik/jpexs-decompiler/blob/version26.3.0/libsrc/ffdec_lib/src/com/jpexs/decompiler/flash/exporters/FrameExporter.java).
The renderer remains an external tool. No renderer implementation was copied.

## Measurements

The original asset export, SVG export, independent software raster and
registration completed in 4.38 seconds. The recovered source origin in the
129 by 131 PNG is `(58.500010, 22.500032)`, with calibration error
`0.001177` pixels. The independent RGBA raster is saved with file and raw-byte
hashes alongside the SVG, PNG, source, declaration and child receipts.

Recalibration from those retained bytes and a fresh appearance fit into the
recorded RGB picture projects source point `(0, 0)` to
`(682.155503, 562.588475)`. The visual-box centre is
`(693.780713, 644.855595)`, 83.084443 pixels away. The current appearance fit
has error 1.693988 pixels. The correspondence step took 0.400 seconds.
This is recorded-pixel proof, not ordinary live gameplay.

Evidence:
`artifacts/flash-campaign/2026-10-15-registered-reference-raster/`, including
`coordinate-raster.png` and `captured-source-point.json`. The instance
ownership diagnostic is in
`artifacts/flash-campaign/2026-10-15-reference-material-ownership/`.

Final focused tests passed 75 in 11.76 seconds. They cover changed source and
picture bytes, reflected and skewed coordinate frames, invalid XML contracts,
unmatched rasters and mutable constructor inputs. An independent browser
raster places a source-coordinate marker at the projected point. Smoke passed
165, one skipped, in 53.24 seconds. Lint and compile passed after the last
changes; layering, dependencies and writing passed before the final immutable
copy and retained-raster additions. Governance lint remains at the same 129
pre-existing baseline regressions, with no baseline change.

The normal actor does not yet acquire and bind these registered instances.
Source-origin contact constraints, body registration and instance naming still
need to reach its controller. No landing, game win, campaign qualification or
native-app presentation proof follows from this result. G09, original game 9,
holdouts and the resident runtime remain untouched.
