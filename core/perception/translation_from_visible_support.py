"""Distinguish a translated object from a changing detection box.

Offline 2026-10-12 changing pieces of scenery acquired control responses.
The centre of a proposal can move while its texture stays still. A measured
translation needs agreeing points or stable extent and matching local pixels.
This does not replace body velocity with the motion of animated texture.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.boundary_support_for_translation import boundaries_agree, visible_boundary
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.verify import invariant

PIXEL_UNCERTAINTY = 2.0


def translation_status(thing: Any, blob: dict, at: float, points: Any = None) -> str:
    """Separate missing evidence from an observed disagreement with the proposed movement."""
    if not 0 < at - thing.seen <= .35:
        return "unavailable"
    previous = thing.path[-1][1:] if thing.path else (thing.x, thing.y)
    dx, dy = blob["x"] - previous[0], blob["y"] - previous[1]
    if not all(math.isfinite(v) for v in (dx, dy, thing.w, thing.h, blob["w"], blob["h"])):
        return "unavailable"
    followed = points is not None and points.before == thing.seen and points.at == at
    if followed and (points.support < 3 or not points.inside(blob)):
        return "unavailable"
    if followed and math.hypot(dx - points.dx, dy - points.dy) > PIXEL_UNCERTAINTY:
        return "contradicted"
    # Opposed edges must share some displacement. Growing only one edge
    # changes the centre without independently locating a translation.
    if (math.hypot(dx, dy) > 1e-6 and abs(dx) <= abs(blob["w"] - thing.w) / 2
            and abs(dy) <= abs(blob["h"] - thing.h) / 2):
        return "unavailable"
    if followed:
        if (math.hypot(dx, dy) > 1e-6
                and math.hypot(points.dx, points.dy) <= PIXEL_UNCERTAINTY
                and not matching_support(thing, blob)):
            return "unavailable"
        return "measured"
    if matching_support(thing, blob):
        return "measured"
    # Its look changed only where both looks were there to compare: pixels not had are not pixels changed.
    seen_both = (_sample(getattr(thing, "patch", None), getattr(thing, "support", None)) is not None
                 and _sample(blob.get("patch"), blob.get("support")) is not None)
    return LOOK_CHANGED if seen_both else "unavailable"


#: A box gone whole from one place to another, both edges with it, whose pixels are not the ones it had: a thing seen
#: going while its look changes, as anything drawn in turns does (a walk, a flap, a spin). Its going is seen, not
#: proved by its outline: offline 2026-10-12, with only a matching outline taken for going, 96 of every 100 things seen
#: crossing a food fight's picture were taken to be standing still.
LOOK_CHANGED = "its look changed"


def seen_going(status: str) -> bool:
    """Whether a status is a thing seen going: its outline carried along, or its look changing as it went."""
    return status in ("measured", LOOK_CHANGED)


def translation_is_observed(thing: Any, blob: dict, at: float, points: Any = None) -> bool:
    """Whether current observed support agrees with this box displacement."""
    return translation_status(thing, blob, at, points) == "measured"


def matching_support(thing: Any, blob: dict) -> bool:
    """Compare the visible extent and owned pixels without extending a missing interval."""
    if not boundaries_agree(getattr(thing, "boundary", None), blob.get("boundary")):
        return False
    if max(abs(blob["w"] - thing.w), abs(blob["h"] - thing.h)) > PIXEL_UNCERTAINTY:
        return False
    before = _sample(getattr(thing, "patch", None), getattr(thing, "support", None))
    after = _sample(blob.get("patch"), blob.get("support"))
    if before is None or after is None:
        return False
    old, old_mask = before
    new, new_mask = after
    shared = old_mask & new_mask
    if shared.sum() < 9 or shared.sum() < .7 * (old_mask | new_mask).sum():
        return False
    alike = np.abs(old.astype(np.int16) - new.astype(np.int16)).max(axis=2) <= 40
    return float(alike[shared].mean()) >= .8


def _sample(patch: Any, support: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if (not isinstance(patch, np.ndarray) or patch.ndim != 3 or patch.shape[2] < 3
            or not isinstance(support, np.ndarray) or support.shape != patch.shape[:2]
            or min(patch.shape[:2]) < 1 or support.sum() < 3):
        return None
    ys = np.rint(np.linspace(0, patch.shape[0] - 1, 9)).astype(int)
    xs = np.rint(np.linspace(0, patch.shape[1] - 1, 9)).astype(int)
    core = ndimage.binary_erosion(support.astype(bool), border_value=0)
    return patch[ys[:, None], xs, :3], core[ys[:, None], xs]


#: How a thing's going is known where nothing tracked it from pixels: it was told where it is, as a world that says
#: where its things are does, and that is its going measured.
DIRECT = "direct"


def told_directly(thing: Any) -> bool:
    """Whether a thing's going was told, not tracked from pixels."""
    default = 'unavailable' if isinstance(getattr(thing, 'pixels', None), PixelsConsumed) else DIRECT
    return getattr(thing, "motion_evidence", default) == DIRECT


def motion_was_measured(thing: Any, at: float) -> bool:
    """Pixel tracks expose their motion timestamp; things told directly, and direct state observations, use their own
    timestamp. Offline 2026-10-12 every thing a world told directly was taken for one whose going was never measured,
    and nothing it did under any key was learned from."""
    if getattr(thing, 'seen', at) != at:
        return False
    if told_directly(thing):
        return True
    capture = getattr(thing, 'pixels', None)
    visual = isinstance(capture, PixelsConsumed)
    return (getattr(thing, 'motion_measured_at', None if visual else at) == at
            and (not visual or capture.at == at))


def motion_status(thing: Any, at: float) -> str:
    if motion_was_measured(thing, at):
        return "measured"
    evidence = getattr(thing, 'motion_evidence', 'unavailable') if getattr(thing, 'seen', at) == at else 'unavailable'
    return 'unavailable' if evidence in (DIRECT, 'measured') else evidence


@invariant('perception.visual_capture_cannot_default_to_measured_motion', scope='perception',
           owner='core/perception/translation_from_visible_support.py', observational=False)
def _captured_motion_authority_invariant() -> tuple:
    from types import SimpleNamespace

    item = SimpleNamespace(seen=1., pixels=PixelsConsumed(1., (32, 32, 3), '0'*64))
    assert not told_directly(item) and motion_status(item, 1.) == 'unavailable'
    item.motion_evidence = 'measured'
    assert not motion_was_measured(item, 1.) and motion_status(item, 1.) == 'unavailable'
    item.motion_measured_at = 1.
    assert motion_was_measured(item, 1.)
    assert motion_was_measured(SimpleNamespace(seen=1.), 1.)
    return ()


@invariant("perception.control_motion_needs_agreeing_visible_support", scope="perception",
           owner="core/perception/translation_from_visible_support.py", observational=False)
def _translation_support_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.points_that_keep_their_motion import PointMotion

    picture = np.full((48, 64, 3), 30, np.uint8)
    picture[14:26, 14:26] = 200
    boundary = visible_boundary(picture, 14, 14, 26, 26, np.ones((12, 12), bool))
    thing = SimpleNamespace(x=20., y=20., w=12., h=12., seen=1., path=[], boundary=boundary,
                            patch=np.full((12, 12, 3), 200, np.uint8), support=np.ones((12, 12), bool))
    blob = dict(x=25., y=20., w=12., h=12., patch=thing.patch, support=thing.support, boundary=boundary)
    assert translation_is_observed(thing, blob, 1.1)
    points = PointMotion(0., 0., .01, 4, 1., 1.1)
    assert not translation_is_observed(thing, blob, 1.1, points)
    blob.update(x=20., w=24., patch=np.full((12, 24, 3), 200, np.uint8), support=np.ones((12, 24), bool))
    assert not translation_is_observed(thing, blob, 1.1)
    return ()
