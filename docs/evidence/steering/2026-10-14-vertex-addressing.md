# Vertex addressing, 14 October 2026

`shader_attributes_from_buffer_bytes.py` supplies a declared scalar or vector
input from its own current captured buffer version. It follows byte offset,
stride, stored component type and instance divisor. Signed normalization
clamps the minimum value to minus one; unsigned normalization uses its full
range. Missing vector components follow the API's zero/zero/zero/one defaults.
Only components the declared input consumes are read from the captured range.

Unobserved resource generations, unsupported packed formats, invalid alignment,
wrong shader conversion, missing bytes and nonfinite values return unknown.
Matrix attribute slots, disabled constant attributes and index/primitive
assembly remain separate work. These inputs feed the existing typed position
evaluator; they confer no identity, control or visible support.

The five geometry component suites pass 125 tests in 12.62 seconds, including
23 new addressing tests. Two new independent WebGL2 transform-feedback probes
measure signed-byte normalized attributes and unsigned-short integer
attributes. Both use a nonzero offset, padded stride, divisor two and default
components. Four emitted instances agree with the decoder and evaluator
within `rtol=2e-6,atol=2e-6`.

Other cases cover half floats, signed and unsigned normalization, explicit
byte order, vertex versus instance indexing, captured subranges, resource
reuse, missing unused components and refusal cases. The invariant is
registered with its test. The implementation uses public API semantics and
newly written code; no external source was copied.

These are offline draw-input components. The production observer, ordered
presentation graph and same-frame pixel support remain open. The paired
active-frame diagnostic is recorded in `2026-10-14-public-framebuffer-presentation.md`.
It still identified no controlled body and showed no win. No heldout frame,
asset or mechanic was inspected. Presentation is not ready.

## Checkpoint gates

After making the vec2 test fixture a valid constructor, the addressing suite
passes 23 tests in 3.94 seconds. Ruff, writing, changed-method size, layering,
dependency checks, compile, repository lint and smoke pass. Smoke has 165
passing tests and one skip. The new decoder has 118 lines.

The aggregate size and governance gates retain the same 194-line debt,
response-generation method debt and 123 governance regressions. No new module
appears in the regression lists and no baseline was changed. The gate record
is `/private/tmp/codex-steer/shader-attributes-gates-1014.log`.
