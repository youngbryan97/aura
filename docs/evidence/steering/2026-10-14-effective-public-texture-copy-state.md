# Effective public texture-copy state

The saved public graphics captures show a bound sampler whose nearest
filter and clamp parameters override conflicting texture-object values.
They also show blending enabled with `ONE / ZERO` factors and `FUNC_ADD`
for RGB and alpha separately. That equation replaces the destination;
it needs neither an opaque-source assumption nor a known-clear assumption.
The semantics follow the [WebGL 2 specification](https://registry.khronos.org/webgl/specs/2.0.0/)
and its OpenGL ES sampler and blending definitions.

`a_texture_copy_with_public_state.py` now admits the existing
`TextureProjection` from a complete immutable public draw snapshot. It
checks the current program link and fragment source digest, unchanged
sample dataflow, reflected fragment-output attachment, effective sampler,
base and maximum texture levels, full channel writes, and replacement
raster state. A bound sampler with missing parameters cannot fall back
to texture-object parameters. Unknown blend or raster state is refused.
The geometry remains bounded and its fragment coverage remains unverified.

Source and destination attachment identities retain context, epoch and
content generation. A single destination write must advance its content
generation exactly once, and the sampled source must match the recorded
version. This prevents an adapter from silently using an earlier version
of a texture that was written twice in one frame.

Tests cover the recorded numeric sampler/blend values, conflicting state
in both directions, changed program links, wrong fragment attachments,
missing state, stale image generations and extreme numeric values.
An admitted copy is passed through the existing presentation mapper and
keeps `coverage="unverified"`. The registered invariant does not establish
material identity or control.

The saved diagnostic captures lack complete storage/content-generation
and fragment-output receipts. They cannot honestly instantiate every
field in the new contract. A production WebGL collector and an exact
presentation-to-RGB pairing remain required. This checkpoint supplies
admission semantics, not a connected production geometry provider or a
resolution of the original game's control failure.
