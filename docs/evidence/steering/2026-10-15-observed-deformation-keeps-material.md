# Observed deformation keeps material — 15 October 2026

The body selected during fresh CPU play at `4ab5de94b` was lost while it
remained visible. Seven continuing named marks formed incompatible rigid
motions. A bounded affine fit could explain six of them, with redundant
correspondence and leave-one-out prediction. The previous rigid-only path
discarded the motion and then the identity.

A separate affine observation now carries limited image scale and shear.
It requires at least six pairs, at least 70 percent retained, noncollinear
support, positive determinant, bounded singular values, a residual bound,
leave-one-out prediction and an independently fitted reverse check. Its
target uncertainty grows away from the measured marks and carries the
existing reference precision budget. Missing or nonfinite target geometry
returns infinite uncertainty. No correspondence or control threshold is
created by this fit.

This uses ordinary affine alignment geometry, as described in
[Baker and Matthews, section 2](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf).
The implementation is a sparse, bounded least-trimmed point fit, not their
image-alignment optimizer. Source points still come from the existing
forward/reverse RGB patch tracker. The new transform reaches point motion,
material association, visible-piece reassembly, texture reconciliation and
the retained coordinate origin. An explicitly deformed pose cannot supply
a rigid angular phase or establish that an input turns the body.

On the first 90 frames of the retained case-18 play, body 2 now retains
its identity for 88 observations, with 86 accepted motion observations.
The preceding implementation retained it for 70 observations, with 68
accepted motion observations. The replay artifacts are
`artifacts/flash-campaign/2026-10-15-case18-parts/replay.json` and
`replay-affine.json`. That is perception replay; the changed controller's
actions and the game's outcome have not yet been measured.

Validation includes three generated RGB scenes in which the source tracker
extracts its own points from a deforming textured object. No exact point
coordinates are injected into those tests. The same test file checks
reflections, collapse, collinearity, unrelated points, insufficient samples,
inverse mapping, target uncertainty and the full point-to-object connection.
All 21 tests passed. Earlier focused material and causal regressions passed
155 tests, and another 39 reference/birth/affine tests passed. Smoke passed
165 tests with one skipped; Ruff, layering, dependencies and writing passed.
The final combined regression artifact is retained locally.

The preceding navigation-only fresh run at `39b41cd50` still did not land.
It retained 565 frames without recording drops, and its terminal words
reported zero landings and two crashes. Its two fast intervals were 34.0
and 27.7 seconds at 7.1 pictures a second. Its self-counted seven is not a
game score or a success. This affine repair still needs fresh CPU play;
native-app proof and the 56-game campaign remain open. G09 and the protected
original cases remain untouched by the probes.
