# Declared shader position arithmetic, 14 October 2026

## Implemented

`shader_position_source.py` parses a bounded straight-line GLSL ES 3.00 subset
into immutable expression data. `shader_position_math.py` evaluates the final
position dependencies. Source identifiers resolve through declarations; no
renderer name, game name, program number or compiled game formula is used.
Shader text never executes as Python or host JavaScript.

The subset includes numeric scalars, vectors and matrices, structure fields,
fixed arrays, declared inputs, constructors, index selection, arithmetic and
ordered swizzle reads/writes. Uniform storage is supplied as typed values;
reflected byte-layout decoding is the next component. Unknown calls, dynamic
control flow, unsupported preprocessing, lower precision, missing values,
invalid dimensions, nonfinite numbers and exhausted budgets return unknown.

Structure arguments retain separate dependencies. Reading a position field
does not read an unrelated color field, including color values absent from
the observation. Calls that could mutate an argument are not admitted. Local
references keep their earlier immutable values across subsequent writes.

Position arithmetic uses float32. It is a geometric proposal and does not
claim a formal GPU rounding bound. Fragment coverage, stencil, depth, textures,
clipping, presentation and pixel support remain separate. The source subset
is not a complete GLSL compiler or proof that a program linked successfully;
the eventual observer must supply successful-link and resource-generation
custody. These modules are not yet installed in the live observation path.

The implementation uses the published semantics in the
[GLSL ES 3.00 specification](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf).
All parser and evaluator code here is newly written; no tracker or shader
compiler source was copied.

## Measurement

The combined arithmetic and presentation suites pass 56 tests in 6.13 seconds:
30 shader tests and 26 presentation tests. The shader GPU test sends twelve
randomized highp matrix instances through an independently compiled WebGL2
program. Transform feedback records its output, which agrees with host
arithmetic within `rtol=2e-6, atol=2e-6`. This tolerance measures this probe;
it does not qualify all shaders or hardware.

Other cases measure renamed symbols, matrix multiplication order, column-major
matrix constructors and indexing, uniform-block fields, instance and vertex
indices, simultaneous swizzle writes, missing unused color values, fixed array
extents and explicit refusal cases. The invariant is registered with its
validation test and a narrow organism dependency edge.

The first GPU probe failed because a Python string put a literal newline
inside a JavaScript quoted shader string. It reached no GPU comparison and
was not scored as passing. The test uses a raw string and now executes the
independent comparison.

No new original-game run was performed for this component. The most recent
ordinary browser entry still has zero body discoveries, and ordinary native
Aura prompt submission and a witnessed win remain unproved. Presentation is
not ready. No heldout program, asset, mechanic or frame was inspected.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke pass. Smoke has 165 passing tests and one skip.
Source parsing has 298 lines; arithmetic has 356 lines. The largest new class
has twelve methods. Aggregate module size retains the previously measured
194-line debt and `ResponseGenerationPhase` method regression. Governance
retains 123 existing regressions. No new source module appears in either
regression list, and no baseline was changed. The serial bounded gate record
is `/private/tmp/codex-steer/shader-position-gates-1014.log`.
