# Projective texture presentation, 14 October 2026

`texture_coordinates_to_a_surface.py` maps a source image region through
measured smooth texture coordinates and homogeneous destination positions.
Source barycentric weights combine homogeneous positions before division.
The result follows perspective-correct interpolation rather than assuming
an affine screen rectangle. Texture-triangle boundaries, homogeneous planes,
viewport, target bounds and scissor all constrain the search region.

The existing canvas route accepts this texture projection between recorded
framebuffer copies. Every edge must retain the current context, epoch, image
and content version. Texture interpolation and base-level nearest sampling
must be supplied explicitly; both default to unknown/refused. Degenerate
coordinates, unsupported ranges and exhausted budgets decline the geometry.
Accepted hints retain `coverage="unverified"`; source alpha, blending and
visible pixel support remain the observer/reconciliation stage's work.

The seven geometry component suites pass 159 tests in 18.38 seconds, including
14 projective mapping tests. Three independent WebGL2 probes sample a marked
texture through affine, projective and partly clipped triangles, including
scissor. Each compares more than 4,000 pixel centres away from a 1.5-pixel
sampling/raster boundary band, with zero mismatches. This measures admitted
search-region mapping, not exact software rasterization.

An initial canary demanded bitwise equality after floating-point projection.
The GPU comparisons already passed; the canary was corrected to a 1e-9
coordinate comparison for its independently known identity fixture. This
does not change any tracking, body-control or visibility threshold.

Other cases verify unknown interpolation/sampling, singular UV coordinates,
out-of-range indices and epochs, measured empty regions and the combined
texture-copy/blit route to a current canvas. The invariant is registered with
its validating test. There is no renderer, game or character dispatch.

Production public capture, shader-output/resource binding and same-frame
pixel reconciliation remain open. No original-game run was added for this
component; no heldout asset, mechanic or frame was inspected. The latest
original-game diagnostic still has no body attribution and no win. Ordinary
native Aura prompt-driven play remains required. Presentation is not ready.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke pass. Smoke has 165 passing tests and one skip in
44.58 seconds. The inherited 194-line module-size debt, response-generation
method debt and 123 governance regressions remain. No owned new module
appears in those regression lists and no baseline was changed. Gate output
is `/private/tmp/codex-steer/texture-projection-gates-1014.log`.
