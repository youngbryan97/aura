"""Refine one existing correspondence against owned current photometric evidence.

Training and held-out source pixels test a bounded affine correction. The
original bidirectional patch observations remain unchanged; their new actual
residual is reported. Existing material precision and whole-pixel gates still
decide admission. A finer pose alone establishes no object or control role.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage, optimize

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, _digest
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.verify import invariant


@dataclass(frozen=True)
class ReferencePoseRefinement:
    pose: ReferenceAppearance | None
    reason: str
    training_before: float | None = None
    training_after: float | None = None
    heldout_before: float | None = None
    heldout_after: float | None = None
    raw_heldout_before: float | None = None
    raw_heldout_after: float | None = None
    maximum_shift: float | None = None
    patch_residual: float | None = None
    evaluations: int | None = None

    def receipt(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != 'pose'}


def refine_reference_pose(picture: np.ndarray, reference: np.ndarray, pose: ReferenceAppearance,
                          *, max_evaluations: int = 24) -> ReferencePoseRefinement:
    refused = ReferencePoseRefinement
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 30:
        raise ValueError('pixel refinement needs a bounded solver evaluation count')
    if (not isinstance(pose, ReferenceAppearance) or material_samples(pose) is None
            or not isinstance(picture, np.ndarray) or picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3
            or not 16 <= min(picture.shape[:2]) <= max(picture.shape[:2]) <= 4096
            or picture.size > 24_000_000 or not isinstance(reference, np.ndarray) or reference.dtype != np.uint8
            or reference.ndim != 3 or reference.shape[2] != 4 or not 16 <= min(reference.shape[:2]) <= max(reference.shape[:2]) <= 512
            or pose.reference_sha256 != _digest(reference) or pose.picture_sha256 != _digest(picture)):
        return refused(None, 'current_reference_correspondence_unavailable')
    mask = ndimage.binary_erosion(reference[..., 3] >= 250, iterations=2)
    yy, xx = np.nonzero(mask)
    if len(xx) < 128 or float(np.std(reference[yy, xx, :3].astype(float), axis=0).max()) < 4.:
        return refused(None, 'photometric_texture_unavailable')
    linear, offset = np.asarray(pose.linear), np.asarray(pose.offset)
    singular = np.linalg.svd(linear, compute_uv=False)
    if singular[0] > 8. or singular[-1] < .05 or singular[0]/singular[-1] > 100:
        return refused(None, 'photometric_scale_unavailable')
    height, width = reference.shape[:2]
    centre, size = np.array((width, height))/2, max(width, height)
    unit = (np.column_stack((xx, yy))-centre)/size
    initial = np.r_[linear.reshape(-1)*size, linear @ centre+offset]
    here = unit @ initial[:4].reshape(2, 2).T+initial[4:]
    inside = ((here[:, 0] >= 2) & (here[:, 1] >= 2)
              & (here[:, 0] < picture.shape[1]-2) & (here[:, 1] < picture.shape[0]-2))
    # Separate spatial tiles rather than fitting and checking the same pixels.
    training = np.flatnonzero(inside & (((xx//4+yy//4) % 2) == 0))
    heldout = np.flatnonzero(inside & (((xx//4+yy//4) % 2) == 1))
    if min(len(training), len(heldout)) < 64:
        return refused(None, 'photometric_holdout_unavailable')
    training = training[np.linspace(0, len(training)-1, min(2048, len(training)), dtype=int)]
    heldout = heldout[np.linspace(0, len(heldout)-1, min(2048, len(heldout)), dtype=int)]
    sigma = .5*float(np.linalg.norm(linear, ord=2))
    margin = math.ceil(8+3*sigma)
    visible = here[inside]
    low = np.maximum(0, np.floor(visible.min(axis=0)-margin)).astype(int)
    high = np.minimum(picture.shape[1::-1], np.ceil(visible.max(axis=0)+margin)).astype(int)
    if int(np.prod(high-low)) > 1_048_576:
        return refused(None, 'photometric_image_budget_exceeded')
    current = picture[low[1]:high[1], low[0]:high[0]].astype(float)/255.
    rgb = reference[..., :3].astype(float)/255.
    target = ndimage.gaussian_filter(rgb, (.5, .5, 0.))[yy, xx]
    image = ndimage.gaussian_filter(current, (sigma, sigma, 0.))

    def residual(parameters, indices, *, raw=False):
        positions = unit[indices] @ parameters[:4].reshape(2, 2).T+parameters[4:]-low
        pixels = current if raw else image
        observed = np.stack([ndimage.map_coordinates(pixels[..., c], (positions[:, 1], positions[:, 0]),
            order=1, mode='constant', cval=-1., prefilter=False) for c in range(3)], axis=-1)
        expected = rgb[yy[indices], xx[indices]] if raw else target[indices]
        return (observed-expected).reshape(-1)

    fit = optimize.least_squares(lambda parameters: residual(parameters, training), initial,
        bounds=(initial-4., initial+4.), loss='soft_l1', f_scale=.04,
        max_nfev=max_evaluations, x_scale='jac')
    if not fit.success or np.linalg.matrix_rank(fit.jac) < 6:
        return refused(None, 'photometric_fit_unresolved', evaluations=int(fit.nfev))
    parameters = fit.x
    measured = dict(training_before=float(abs(residual(initial, training)).mean()),
        training_after=float(abs(residual(parameters, training)).mean()),
        heldout_before=float(abs(residual(initial, heldout)).mean()),
        heldout_after=float(abs(residual(parameters, heldout)).mean()),
        raw_heldout_before=float(abs(residual(initial, heldout, raw=True)).mean()),
        raw_heldout_after=float(abs(residual(parameters, heldout, raw=True)).mean()), evaluations=int(fit.nfev))
    if (measured['training_before']-measured['training_after'] < .001
            or measured['heldout_before']-measured['heldout_after'] < .001
            or measured['raw_heldout_before']-measured['raw_heldout_after'] < .001):
        return refused(None, 'photometric_holdout_did_not_improve', **measured)
    refined_linear = parameters[:4].reshape(2, 2)/size
    refined_offset = parameters[4:]-refined_linear @ centre
    corners = np.array(((0., 0.), (width, 0.), (0., height), (width, height)))
    shifted = corners @ refined_linear.T+refined_offset
    maximum_shift = float(np.linalg.norm(shifted-(corners @ linear.T+offset), axis=1).max())
    if (not np.isfinite(parameters).all() or np.linalg.det(refined_linear) <= 1e-8
            or np.linalg.cond(refined_linear) > 100 or maximum_shift > 4.):
        return refused(None, 'photometric_correction_outside_window', maximum_shift=maximum_shift, **measured)
    starts, ends = np.asarray(pose.source_points), np.asarray(pose.observed_points)
    error = float(np.linalg.norm(starts @ refined_linear.T+refined_offset-ends, axis=1).max())
    extent_low, extent_high = shifted.min(axis=0), shifted.max(axis=0)
    refined = replace(pose, linear=tuple(map(tuple, refined_linear)), offset=tuple(refined_offset),
        box=tuple(map(float, (*extent_low, *(extent_high-extent_low)))), error=error)
    if material_samples(refined) is None:
        return refused(None, 'refined_patch_correspondence_unavailable', **measured)
    return refused(refined, 'current_photometric_correction', maximum_shift=maximum_shift, patch_residual=error, **measured)


@invariant('perception.pixel_pose_refinement_cannot_discard_its_patch_uncertainty', scope='perception',
           owner='core/perception/a_reference_pose_refined_in_current_pixels.py', observational=False)
def _pixel_pose_refinement_invariant() -> tuple:
    pixels = np.zeros((32, 32, 4), np.uint8)
    assert refine_reference_pose(pixels[..., :3], pixels, None).pose is None
    assert ReferencePoseRefinement(None, 'unmeasured').heldout_after is None
    return ()
