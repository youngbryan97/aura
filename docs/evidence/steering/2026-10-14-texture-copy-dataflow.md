# Texture-copy dataflow, 14 October 2026

`a_texture_copied_by_its_shader.py` recognizes a bounded dataflow form: one
unchanged `texture(sampler2D, vec2)` result reaches one vec4 fragment output.
Structure wrappers, aliases and identity constructors are resolved through
immutable declared expressions. The sampler and UV input names come from the
source. There are no known renderer names or program numbers in the code.

Color changes, coordinate changes, sample bias, component writes, flat or
centroid interpolation, uniform coordinates and multiple outputs are refused.
A vertex position that depends on an unobserved texture stays unknown. The
parser records output expressions and input qualifiers; successful-link,
sampler, raster, image-version and frame custody still belong to collection.
Recognizing the form alone supplies no presentation edge or visible object.

The four component suites pass 102 tests in 10.58 seconds, including 19 new
copy-form tests. One independent WebGL2 probe compiles a structured copy
shader and copies a randomized 16 by 8 RGBA image under explicit nearest
sampling. All returned GPU pixel bytes match the input. This verifies the
admitted probe, not unknown filters, samplers or clipping states.

An offline replay of the assigned intro audit parsed all three vertex shaders
and two of three fragment shaders. Its presentation fragment is recognized
as the copy form. Another fragment is refused for an unsupported token;
its coverage is not inferred or replaced. The private replay record is
`/private/tmp/codex-steer/texture-copy-form-replay-1014.json`. This is source
coverage evidence, not another game run or a promoted current-frame receipt.

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke pass. Smoke has 165 passing tests and one skip.
The inherited 194-line module-size debt, response-generation method debt and
123 governance regressions remain. No baseline was changed. Bounded serial
gate output is `/private/tmp/codex-steer/texture-copy-gates-1014.log`.

No heldout asset, mechanic or frame was inspected. No resident model or
runtime was restarted. The observer-to-pixel connection and controlled-body
discovery remain open. Ordinary native Aura play and a witnessed win are
still required; presentation is not ready.
