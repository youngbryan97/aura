"""Use previous geometry to propose fresh correspondence, without temporal identity."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage

from core.perception.a_visual_reference_in_the_picture import (
    ReferenceAppearance, _SceneCorrelations, _appearance_for_fit, _digest, _oriented_template,
    _picture_view, _pictures_valid, _supported_fit,
)
from core.verify import invariant


def _near_places(pixels: np.ndarray, template: np.ndarray, place: np.ndarray) -> tuple:
    """Find coarse local peaks before the ordinary fine patch solver starts."""
    height, width = template.shape[:2]
    radius = max(4, min(24, math.ceil(min(height, width)*.15)))
    left, top = max(0, math.floor(place[0])-radius), max(0, math.floor(place[1])-radius)
    right, bottom = min(pixels.shape[1], math.ceil(place[0])+width+radius), min(pixels.shape[0], math.ceil(place[1])+height+radius)
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


def near_reference_appearances(picture: np.ndarray, reference: np.ndarray,
        proposals: tuple[ReferenceAppearance, ...], *, edge: int = 480) -> tuple[ReferenceAppearance, ...]:
    """Refit at most four poses against current pixels with the ordinary patch gates.

    A proposal supplies geometry only. Six current correspondences, reverse
    patch journeys and a redundant fit must qualify again. Unsearched copies
    remain possible; results confer no object identity or task role.
    """
    if (not _pictures_valid(picture, reference, edge) or not isinstance(proposals, tuple)
            or not 1 <= len(proposals) <= 4):
        return ()
    reference_hash, picture_hash = _digest(reference), _digest(picture)
    pixels, scale, origin, _ratio = _picture_view(picture, edge)
    centre = (np.asarray(reference.shape[1::-1])-1)/2
    source, found = Image.fromarray(reference), []
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
            size = tuple(int(round(v)) for v in np.linalg.norm(viewed, axis=0)*np.asarray(source.size))
            if min(size) < 12 or size[0] > pixels.shape[1] or size[1] > pixels.shape[0]:
                continue
            angle = math.atan2(viewed[1, 0], viewed[0, 0])
            template, template_linear, template_offset = _oriented_template(source, size, angle)
            if template.shape[0] > pixels.shape[0] or template.shape[1] > pixels.shape[1]:
                continue
            place = viewed @ centre+scale*(offset-origin)-(np.asarray(template.shape[1::-1])-1)/2
            for x, y in _near_places(pixels, template, place):
                fit = _supported_fit(pixels, template, x, y)
                if fit is None:
                    continue
                candidate = _appearance_for_fit(reference, reference_hash, picture_hash, fit,
                                               template_linear, template_offset, scale, origin)
                probes = ((0., 0.), (reference.shape[1]-1., 0.), (0., reference.shape[0]-1.))
                if not any(max(math.dist(candidate.project(p), old.project(p)) for p in probes) < 1. for old in found):
                    found.append(candidate)
                if len(found) == 8:
                    return tuple(found)
        except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            continue
    return tuple(found)


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
    return ()
