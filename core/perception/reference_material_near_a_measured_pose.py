"""Use previous geometry to propose fresh correspondence, without temporal identity."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from core.perception.a_visual_reference_in_the_picture import (
    ReferenceAppearance, _SceneCorrelations, _appearance_for_fit, _digest,
    _picture_view, _pictures_valid, _supported_fit,
)
from core.verify import invariant


def _near_places(pixels: np.ndarray, template: np.ndarray, place: np.ndarray, *, whole_picture: bool = False) -> tuple:
    """Find at most three coarse peaks before the ordinary fine patch solver starts."""
    height, width = template.shape[:2]
    radius = max(4, min(24, math.ceil(min(height, width)*.15)))
    left, top = max(0, math.floor(place[0])-radius), max(0, math.floor(place[1])-radius)
    right, bottom = min(pixels.shape[1], math.ceil(place[0])+width+radius), min(pixels.shape[0], math.ceil(place[1])+height+radius)
    if whole_picture:
        left, top, right, bottom = 0, 0, pixels.shape[1], pixels.shape[0]
    if right-left < width or bottom-top < height:
        return ()
    mask = ndimage.binary_erosion(template[..., 3] >= 250)
    if mask.sum() < 36:
        return ()
    rgb = template[..., :3].astype(np.float32)/255
    if float(np.std(rgb[mask], axis=0).max()) < .03:
        return ()
    scene = pixels[top:bottom, left:right].astype(np.float32)/255
    cost = _SceneCorrelations(scene, (height, width)).cost(rgb, mask.astype(np.float32))
    found = []
    for _ in range(3):
        y, x = np.unravel_index(cost.argmin(), cost.shape)
        if not math.isfinite(float(cost[y, x])):
            break
        found.append((left+int(x), top+int(y)))
        dx, dy = max(4, width//2), max(4, height//2)
        cost[max(0, y-dy):y+dy+1, max(0, x-dx):x+dx+1] = math.inf
    return tuple(found)


def _positions_for_fit(places: tuple, position: int | None) -> tuple:
    """An admitted fine probe selects one ranked peak from its search phase."""
    return places if position is None else places[position % 3:position % 3+1]


def _template_at_measured_shape(reference: np.ndarray, linear: np.ndarray, offset: np.ndarray,
                                shape: tuple) -> tuple | None:
    """Keep the measured affine shape, clipping only at the captured image edge."""
    height, width = reference.shape[:2]
    corners = np.array(((0., 0.), (width-1., 0.), (0., height-1.), (width-1., height-1.))) @ linear.T+offset
    low = np.maximum(0, np.floor(corners.min(axis=0))).astype(int)
    high = np.minimum(np.asarray(shape[::-1])-1, np.ceil(corners.max(axis=0))).astype(int)
    size = high-low+1
    if min(size) < 12:
        return None
    template_offset = offset-low
    ys, xs = np.mgrid[:size[1], :size[0]]
    places = (np.stack((xs, ys), axis=-1)-template_offset) @ np.linalg.inv(linear).T
    template = np.stack([ndimage.map_coordinates(reference[..., c], [places[..., 1], places[..., 0]],
        order=1, mode='constant', cval=0.) for c in range(4)], axis=-1)
    return template, linear, template_offset, low


def near_reference_appearances(picture: np.ndarray, reference: np.ndarray,
        proposals: tuple[ReferenceAppearance, ...], *, edge: int = 480,
        position: int | None = None) -> tuple[ReferenceAppearance, ...]:
    """Refit at most four poses against current pixels with the ordinary patch gates.

    A failed local fit permits three wider candidate positions at the same
    proposed shape. At most 24 positions reach the patch solver. Six current
    correspondences, reverse patch journeys and a redundant fit must qualify
    again. Unsearched copies remain possible; results confer no object
    identity or task role. An explicit position selects one of three local
    or three wide peaks for one pose, with at most one fine fit.
    """
    if (not _pictures_valid(picture, reference, edge) or not isinstance(proposals, tuple)
            or not 1 <= len(proposals) <= 4
            or position is not None and (type(position) is not int or not 0 <= position < 6 or len(proposals) != 1)):
        return ()
    reference_hash, picture_hash = _digest(reference), _digest(picture)
    pixels, scale, origin, _ratio = _picture_view(picture, edge)
    found = []
    for proposal in proposals:
        if not isinstance(proposal, ReferenceAppearance) or proposal.reference_sha256 != reference_hash:
            continue
        try:
            linear, offset = np.asarray(proposal.linear, float), np.asarray(proposal.offset, float)
            if (linear.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(linear).all()
                    or not np.isfinite(offset).all() or np.linalg.det(linear) <= 1e-8
                    or np.linalg.cond(linear) > 100 or np.max(abs(offset)) > 8192):
                continue
            viewed = linear*scale[:, None]
            placed_offset = scale*(offset-origin)
            measured_template = _template_at_measured_shape(reference, viewed, placed_offset, pixels.shape[:2])
            if measured_template is None:
                continue
            template, template_linear, template_offset, place = measured_template
            tried = set()
            for whole_picture in ((False, True) if position is None else (bool(position//3),)):
                measured = False
                places = _near_places(pixels, template, place, whole_picture=whole_picture)
                for x, y in _positions_for_fit(places, position):
                    if (x, y) in tried:
                        continue
                    tried.add((x, y))
                    fit = _supported_fit(pixels, template, x, y)
                    if fit is None:
                        continue
                    measured = True
                    candidate = _appearance_for_fit(reference, reference_hash, picture_hash, fit,
                                                   template_linear, template_offset, scale, origin)
                    probes = ((0., 0.), (reference.shape[1]-1., 0.), (0., reference.shape[0]-1.))
                    if not any(max(math.dist(candidate.project(p), old.project(p)) for p in probes) < 1. for old in found):
                        found.append(candidate)
                    if len(found) == 8:
                        return tuple(found)
                if measured:
                    break
        except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            continue
    return tuple(found)


@invariant('perception.optional_reference_refits_one_position', scope='perception',
           owner='core/perception/reference_material_near_a_measured_pose.py', observational=False)
def _single_position_refit_invariant() -> tuple:
    places = ((1, 2), (3, 4), (5, 6))
    assert [_positions_for_fit(places, n) for n in range(6)] == [(p,) for p in places]*2
    assert _positions_for_fit((), 0) == ()
    assert _positions_for_fit(places[:1], 2) == ()
    assert _positions_for_fit(places, None) == places
    return ()


@invariant('perception.near_reference_proposals_need_current_material', scope='perception',
           owner='core/perception/reference_material_near_a_measured_pose.py', observational=False)
def _near_reference_material_invariant() -> tuple:
    from core.perception.a_visual_reference_in_the_picture import reference_appearances

    ref = np.random.default_rng(101).integers(0, 255, (40, 48, 4), dtype=np.uint8)
    ref[..., 3] = 255
    first = np.zeros((96, 144, 3), np.uint8)
    first[20:60, 30:78] = ref[..., :3]
    proposals = reference_appearances(first, ref, scales=(1.,))
    assert len(proposals) == 1
    assert near_reference_appearances(np.zeros_like(first), ref, proposals) == ()
    current = near_reference_appearances(first, ref, proposals)
    assert len(current) == 1 and len(current[0].source_points) >= 6
    distant = np.zeros((96, 240, 3), np.uint8)
    distant[30:70, 150:198] = ref[..., :3]
    recovered = near_reference_appearances(distant, ref, proposals)
    assert len(recovered) == 1 and math.dist(recovered[0].project((11., 13.)), (161., 43.)) < .5
    return ()
