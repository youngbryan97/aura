# Public framebuffer presentation, 14 October 2026

## Why this work exists

The browser run at code checkpoint `642ce99e0775e461f1963b743edc8c44ebb0d00c`
made 938 decisions and identified no controlled body. The unit and angular
coverage repairs therefore have no demonstrated gain on assigned original
case 9. The earlier renderer audit recovered bounds for 189 of 190 draws,
but framebuffer bounds alone did not identify the captured canvas pixels.

This work adds coordinate and provenance machinery. It supplies geometric
search hints. Pixel support, object correspondence, causal control and a
game outcome remain separate measurements. There is no game dispatch,
character name, private VM state, renderer-specific uniform name or inferred
vertical flip in the new module.

## Public audit

The diagnostic collector was extended in `/private/tmp/codex-steer/` to read
the public READ/DRAW framebuffer attachments, sampler texture units, raster
state, clears and `blitFramebuffer` calls. It restores the active texture
unit after bounded diagnostic queries. This collector is not installed in
production and its synchronous reads are not a throughput design.

Run: `presentation-state-public-graphics-original9-1014` under that private
directory. The frozen proof checkout was at `642ce99e0`. Its owned subprocess
completed in 105.117106 seconds, exit 0, without timeout. No resident model
or runtime was stopped, restarted or loaded. G09 induction remained active.

One saved capture contains 23 draws, three copy/clear events and no recorded
collector exception. Its JSON is 1,111,110 bytes. This is an intro-frame
presentation measurement; it does not establish a full active-play capture.

The captured public route is:

1. Framebuffer 3 draws into renderbuffer image 1.
2. A nearest color blit copies `(0,0,2560,1522)` to texture image 48 through
   framebuffer 49. The recorded scissor covers the full destination.
3. A textured draw samples image 48 and writes texture image 52. Framebuffer
   3 has changed attachments; its name alone no longer identifies image 1.
4. A nearest color blit from framebuffer 53 copies source rectangle
   `(0,1522,2560,0)` into the default framebuffer `(0,0,2560,1522)`.

The flip is thus a public operation, rather than a guessed screen offset.
The shader presentation-copy edge, sampler overrides, immutable image content
versions and a paired pixel capture still need production implementation.
The recorded texture object filters are not sufficient when a sampler is bound.
The existing diagnostic cannot supply the new module's current-canvas contract.

No heldout program, mechanic, frame or run was inspected. The program IDs and
image numbers above are evidence labels and are absent from production code.

## Implemented coordinate contract

`core/perception/graphics_presented_on_canvas.py` clips admitted triangles
against all six homogeneous planes before perspective division and viewport
mapping. It follows explicit nearest color copy rectangles, preserving flips,
scales, source clipping and destination scissor. The final coordinate change
converts bottom-left framebuffer units to top-left canvas units.

Frozen image versions include context, epoch, attachment image, content
generation and dimensions. Copy routes must be ordered, bounded and unique,
and must reach exactly the captured image version. Missing routes, stale
contents, ambiguous routes, unknown copy semantics, singular projection or
budget overflow return unknown. A fully clipped region is a measured empty
result. Every accepted region retains `coverage="unverified"`.

The current module is an offline geometry foundation. It is not yet an
observation provider, shader evaluator, pixel segmentation replacement or
new source of control authority.

## Independent verification

The new test suite passed 26 tests in 4.57 seconds. One test creates its own
WebGL2 context, renders a homogeneous triangle partly outside the clip volume,
then copies it across all four axis-flip combinations, with and without a
scaled destination and scissor. GPU pixels are compared with mapped polygons.
It checks more than 80,000 pixel centres away from a 1.25-pixel raster and
nearest-resampling boundary band, with zero mismatches. Boundary pixels are
excluded explicitly; this is not a claim of exact software rasterization.

Other tests cover two-copy routes, immutable hints, attachment/content reuse,
clock epochs, unmeasured receipt clocks, unsupported filters, malformed
rectangles, resource limits and the registered invariant. These are synthetic
measurements, not evidence of improved game play or presentation readiness.

The implementation follows coordinate transformations, primitive clipping
and framebuffer copy semantics in the
[OpenGL ES 3.0.6 specification](https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf),
sections 2.13, 2.18 and 4.3.3, and the
[WebGL 2.0 specification](https://registry.khronos.org/webgl/specs/2.0.0/).
No upstream implementation was copied.

## Checkpoint gates

After the import-style correction, Ruff passes on all owned changed Python
files. Smoke passes with 165 passed and one skipped in 45.02 seconds. Writing,
changed-method size, layering, dependency checks, compile and repository lint
pass. The new module has 244 lines and no class approaches the method cap.

Aggregate module size retains the pre-existing `ResponseGenerationPhase`
22-method regression against a baseline of 20, and 194 lines of total
oversize debt. Governance retains 123 pre-existing regressions; the new
module is absent from that list. Neither baseline was relaxed. Gate output
is `/private/tmp/codex-steer/graphics-map-gates-1014.log`.

Ordinary Aura prompt submission, current visible-body discovery and a
witnessed original-game success remain unproved. Presentation is not ready.
