"""Where a steering vector goes when it enters the residual stream.

Adding ``alpha * v`` to a hidden state does two things at once and only one of
them is wanted. It turns the state toward ``v``, which is the intervention. It
also makes the state longer, and at every layer the injection is applied that
extra length is carried into the next block's input.

Measured on Aura-Qwen3.8-27B on 2026-09-12: sixteen layers at alpha 0.2
returned eight characters. The direction was right and the model had stopped
writing sentences. A strength that breaks the writer cannot be compared against
a prompt, so every route toward the text control ran out of coherence before it
ran out of alpha -- 0.833 at the strongest setting that still wrote, against a
text control at 2.000.

Rotation separates the two effects. Add the vector, then put the state back on
the sphere it came from. The state is turned by the same angle and the block
reading it next sees a vector the size it expects::

    translate   h' = h + a*s*v
    rotate      h' = (h + a*s*v) * ||h|| / ||h + a*s*v||
    clip        rotate, but only once the growth exceeds CLIP_GROWTH_CEILING
    project     h' = h + (a*s - <h,v>) * v
    lift        project, but only upward

``a*s`` is identical in all of them: alpha as a fraction of the stream's own
norm, so one setting means one thing across model sizes.

The first three all compound, and that is what breaks them. Each layer acts on
what the layer before it did, so sixteen layers of translation bury the content
under the vector and sixteen of rotation scale the content away -- cos(21 deg)
to the sixteenth is 0.31, and at alpha 1.6 it is one in ten thousand. Measured
on the 27B: rotation over all sixteen layers scored zero at every alpha tried,
with word variety at 0.97, which is a reply where almost no word repeats
because none of it is a sentence.

Projection does not compound. It SETS the component along the vector instead of
adding to it, so a layer that runs after another arrives at the same place
rather than twice as far, and the part of the stream orthogonal to the vector
is not touched at all. That is what makes sixteen layers usable, including the
twelve carrying recurrent state.

The mode is a measurement, not a preference. `AURA_STEERING_INJECTION` names
it; the default is whatever the last campaign found holds.
"""

from __future__ import annotations

import os
from typing import Any

TRANSLATE = "translate"
ROTATE = "rotate"
CLIP = "clip"
PROJECT = "project"
LIFT = "lift"

INJECTION_MODES = (TRANSLATE, ROTATE, CLIP, PROJECT, LIFT)

#: How much longer than it started ``clip`` lets the state get before it
#: rescales. 1.0 is ``rotate``; a very large value is ``translate``.
CLIP_GROWTH_CEILING = 1.25

#: Below this a norm is noise, and dividing by it would amplify the noise into
#: the stream of every later block.
_NORM_FLOOR = 1e-6

# Projection, because that is what was measured and qualified. Translation
# lengthens the stream at every layer it touches, which is survivable at four
# sites and not at sixteen: rotation over all sixteen scored zero at every
# alpha tried, and translation reached two characters at 0.2. Projection over
# the same sixteen reached +1.417 with replies at 888 characters against a
# baseline of 954, and it is the geometry the passing campaign ran.
_DEFAULT_MODE = PROJECT

_ENV_VAR = "AURA_STEERING_INJECTION"


def injection_mode() -> str:
    """The geometry in force, from the environment or the measured default."""
    named = os.getenv(_ENV_VAR, "").strip().lower()
    return named if named in INJECTION_MODES else _DEFAULT_MODE


def _rescaled(h: Any, moved: Any, ceiling: float | None) -> Any:
    """``moved``, shortened back toward ``h``'s per-token length.

    Per token, not per tensor. One sequence position may be twice as long as
    another, and a single ratio would steer the short ones harder than the
    long ones for no reason anyone chose.
    """
    import mlx.core as mx

    # float32 for the ratio. In bfloat16 the division carries about three
    # decimal digits, and this factor multiplies every component of the
    # residual stream at every steered block.
    before = mx.linalg.norm(mx.astype(h, mx.float32), axis=-1, keepdims=True)
    after = mx.linalg.norm(mx.astype(moved, mx.float32), axis=-1, keepdims=True)
    ratio = before / mx.maximum(after, _NORM_FLOOR)
    if ceiling is not None:
        # Let it grow up to the ceiling and only pull back past that, so a
        # small nudge keeps the extra magnitude that made it a nudge.
        ratio = mx.minimum(ratio * float(ceiling), mx.array(1.0, dtype=ratio.dtype))
    return moved * mx.astype(ratio, moved.dtype)


def _projected(h: Any, delta: Any, *, upward_only: bool) -> Any:
    """``h`` with its component along ``delta`` set to ``delta``'s own length.

    ``delta`` is alpha times the stream reference times a unit composite, so
    its length IS the target the component is being set to and its direction is
    the composite. Recovering both from the one argument keeps the caller --
    the hook, inside the forward pass of every block -- unchanged.
    """
    import mlx.core as mx

    target = mx.linalg.norm(mx.astype(delta, mx.float32))
    unit = mx.astype(delta, mx.float32) / mx.maximum(target, _NORM_FLOOR)
    component = mx.sum(mx.astype(h, mx.float32) * unit, axis=-1, keepdims=True)
    change = target - component
    if upward_only:
        # A reply that already leans the way the vector points is left alone.
        # Setting the component would otherwise pull it BACK to the target, so
        # the strongest samples are the ones steering damps.
        change = mx.maximum(change, mx.array(0.0, dtype=change.dtype))
    return h + mx.astype(change * unit, h.dtype)


def inject(h: Any, delta: Any, *, mask: Any | None = None, mode: str | None = None) -> Any:
    """Return ``h`` with ``delta`` applied under the named geometry.

    ``delta`` arrives already scaled -- alpha times the stream reference times
    the unit composite. ``mask`` selects which token positions receive it;
    ``None`` means all of them.
    """
    mode = mode or injection_mode()
    if mode == TRANSLATE:
        return h + (mask * delta if mask is not None else delta)

    if mode in (PROJECT, LIFT):
        moved = _projected(h, delta, upward_only=mode == LIFT)
    else:
        moved = _rescaled(
            h, h + delta, CLIP_GROWTH_CEILING if mode == CLIP else None
        )
    if mask is None:
        return moved
    # The mask chooses positions, so it interpolates between the state that was
    # there and the state rotation produced. Multiplying the ROTATED state by
    # the mask would zero every unsteered position instead of leaving it alone.
    return h + mask * (moved - h)


__all__ = [
    "CLIP",
    "CLIP_GROWTH_CEILING",
    "INJECTION_MODES",
    "LIFT",
    "PROJECT",
    "ROTATE",
    "TRANSLATE",
    "inject",
    "injection_mode",
]
