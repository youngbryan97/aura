# Reflected uniform-buffer positions, 14 October 2026

## Implemented

`shader_uniforms_from_buffer_bytes.py` decodes uniform fields through reflected
types, byte offsets, array strides and matrix strides. Buffer and binding
records must match context, epoch, buffer identity and content generation.
The bound range and the captured range both constrain every scalar read.
Byte order is explicit; unknown byte order, nonfinite values, missing bytes
and unsupported layouts remain unknown.

GLSL declarations provide structure fields and array extents. Named block
reflection can use a block name instead of its source instance name. The
parser now records that declared relationship. The decoder resolves it from
the declaration and public block descriptor, rather than from known renderer
names. Anonymous blocks and direct instance-qualified names are covered too.

Decoded values feed `position_from_program` directly. Unused color fields
may remain absent while a measured position dependency has all its bytes.
This code has no game dispatch or private application state. It reads no
files, starts no model and issues no input.

These are offline arithmetic and resource contracts. Production collection
must still record all resource writes, successful links, draw inputs, image
generations, capture clocks and the complete presentation route. The earlier
private diagnostic lacks some of those records and cannot be promoted by
inventing them. This decoder is not yet connected to Aura's live eyes.

## Independent measurement

The three component suites pass 83 tests in 8.78 seconds: 27 uniform-buffer
tests, 30 position-arithmetic tests and 26 presentation tests.

Two new probes compile independent WebGL2 shaders with uniform blocks in
column-major and row-major layouts. They read actual public reflection,
upload randomized matrices into a bound buffer range with a nonzero aligned
start, and execute two instances through GPU transform feedback. The decoder
and position evaluator agree with those GPU outputs and an independent
numeric matrix product within `rtol=2e-6, atol=2e-6`.

Other cases cover explicit big/little byte order, padded matrix strides,
array strides, names resolved through block declarations, missing unused
fields, buffer and content reuse, ranges, mutable input bytes, duplicate
fields, integer and boolean storage, malformed names and unsupported types.
The resource invariant is registered with its validating test.

Public types and reflection semantics follow the
[WebGL 2.0 specification](https://registry.khronos.org/webgl/specs/2.0.0/)
and [GLSL ES 3.00](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf).
The implementation is newly written. No upstream shader or tracker source
was copied.

No original-game run was added for this component. No heldout asset, mechanic
or frame was inspected. Controlled-body discovery and ordinary native Aura
play remain unproved; presentation is not ready.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke pass. Smoke has 165 passing tests and one skip.
The new decoder has 241 lines. Aggregate module size retains the previously
measured 194-line debt and `ResponseGenerationPhase` method regression;
governance retains 123 regressions. No owned new module appears in those
regression lists, and no baseline was changed. Gate output is
`/private/tmp/codex-steer/shader-uniform-gates-1014.log`.

Correction to the preceding shader arithmetic record: the parser had eleven
methods, including its constructor, rather than twelve. This does not change
the method-cap result.
