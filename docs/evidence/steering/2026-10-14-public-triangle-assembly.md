# Public triangle assembly, 14 October 2026

`triangles_from_public_draws.py` assembles public triangles, triangle strips
and triangle fans. It preserves the first vertex for nonindexed draws,
unsigned element widths, byte order, offsets, winding and fixed restart.
Restart resets strip parity and fan anchors; partial and degenerate triangles
produce no geometry. Empty admitted draws differ from unknown draws.

Element bytes retain context, epoch, buffer identity and content generation.
Stale buffers, unavailable ranges, unsupported modes and exhausted budgets
return unknown. The result contains vertex indices, not object identities.
No game, renderer or character dispatch is present.

The six component suites pass 145 tests in 15.11 seconds, including 20 new
assembly tests. Three independent WebGL2 probes render the original indexed
draw and its assembled triangles with back-face culling and fixed restart.
All returned RGBA pixels agree for triangles, strips and fans. Other tests
cover all three unsigned widths, explicit byte order, reuse, missing bytes,
partial draws, degeneracy and bounded work. The invariant is registered with
its validating test.

The pipeline now has tested source arithmetic, uniform storage, vertex
addressing, topology, homogeneous clipping and framebuffer-copy coordinates.
These remain offline components. A texture-copy presentation form is also
recognized, but its coordinate edge and current resource/raster custody are
not yet assembled by a production observer. Pixel-supported object continuity
and ordinary native Aura play remain separate requirements.

No original-game run was added for assembly, and no heldout asset, mechanic
or frame was inspected. The latest original-game diagnostic identified no
body and showed no win. Presentation is not ready.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke pass. Smoke has 165 passing tests and one skip in
42.88 seconds. The inherited 194-line size debt, response-generation method
debt and 123 governance regressions remain. No owned new module appears in
those regression lists and no baseline was changed. Gate output is
`/private/tmp/codex-steer/triangle-assembly-gates-1014.log`.
