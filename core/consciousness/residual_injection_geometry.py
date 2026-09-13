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

``a*s`` is identical in all three: alpha as a fraction of the stream's own
norm, so one setting means one thing across model sizes.

The mode is a measurement, not a preference. `AURA_STEERING_INJECTION` names
it; the default is whatever the last campaign found holds.
"""

from __future__ import annotations

import os
from typing import Any

TRANSLATE = "translate"
ROTATE = "rotate"
CLIP = "clip"

INJECTION_MODES = (TRANSLATE, ROTATE, CLIP)

#: How much longer than it started ``clip`` lets the state get before it
#: rescales. 1.0 is ``rotate``; a very large value is ``translate``.
CLIP_GROWTH_CEILING = 1.25

#: Below this a norm is noise, and dividing by it would amplify the noise into
#: the stream of every later block.
_NORM_FLOOR = 1e-6

_DEFAULT_MODE = TRANSLATE

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


def inject(h: Any, delta: Any, *, mask: Any | None = None, mode: str | None = None) -> Any:
    """Return ``h`` with ``delta`` applied under the named geometry.

    ``delta`` arrives already scaled -- alpha times the stream reference times
    the unit composite. ``mask`` selects which token positions receive it;
    ``None`` means all of them.
    """
    mode = mode or injection_mode()
    if mode == TRANSLATE:
        return h + (mask * delta if mask is not None else delta)

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
    "ROTATE",
    "TRANSLATE",
    "inject",
    "injection_mode",
]
