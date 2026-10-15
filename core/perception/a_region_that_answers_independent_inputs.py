"""Test press-level visual contrasts in disjoint prespecified cohorts.

Maximum-statistic permutations cover every measured cell. Confirmation uses
the fixed discovered pattern without selecting a second region. P-values
depend on randomized exchangeable trials; they do not identify a player.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label

from core.verify import invariant


@dataclass(frozen=True)
class ResponsiveRegion:
    bounds: tuple[int, int, int, int]
    discovery_p: float
    confirmation_p: float
    effect: float
    status: str = "search_hint_only"
    reference: str = "screen_unreferenced"


def _contrast(data: np.ndarray, flags: np.ndarray):
    a, b = data[flags], data[~flags]
    difference = a.mean(axis=0) - b.mean(axis=0)
    variance = a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)
    return np.sum(difference**2 / (variance + .2**2), axis=-1), difference


def _camera_answers(camera, flags, rng, permutations):
    difference = float(np.linalg.norm(camera[flags].mean(axis=0) - camera[~flags].mean(axis=0)))
    if difference <= 1.5:
        return False
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(flags)
        exceed += np.linalg.norm(camera[order].mean(axis=0) - camera[~order].mean(axis=0)) >= difference
    return (1 + exceed) / (1 + permutations) <= .01


def region_from_press_features(features: np.ndarray, counts: np.ndarray, camera: np.ndarray | None,
                               keys: tuple[str, ...], cohorts: tuple[str, ...], *,
                               first_key: str, second_key: str, tile: int = 12,
                               permutations: int = 499, seed: int = 41,
                               available: np.ndarray | None = None) -> ResponsiveRegion | None:
    """Input rows are distinct admitted presses, never frames from one press."""
    if (not isinstance(features, np.ndarray) or features.ndim != 4 or features.shape[-1] != 3
            or not 20 <= len(features) <= 40 or min(features.shape[1:3]) < 2
            or features.shape[1] * features.shape[2] > 600 or first_key == second_key
            or type(tile) is not int or not 6 <= tile <= 32
            or type(permutations) is not int or not 199 <= permutations <= 999
            or type(seed) is not int or seed < 0
            or not isinstance(counts, np.ndarray) or counts.shape != features.shape[:-1]
            or camera is not None and (not isinstance(camera, np.ndarray) or camera.shape != (len(features), 2)
                                       or camera.dtype.kind not in "fiu" or not np.isfinite(camera).all())
            or type(keys) is not tuple or type(cohorts) is not tuple
            or len(keys) != len(features) or len(cohorts) != len(features)
            or counts.dtype.kind not in "iu"
            or features.dtype.kind not in "fiu"
            or not np.isfinite(features).all() or not np.isfinite(counts).all() or np.any(counts < 0)):
        return None
    if available is None:
        available = np.repeat((counts >= 3)[..., None], 3, axis=-1)
    if (not isinstance(available, np.ndarray) or available.shape != features.shape
            or available.dtype != bool):
        return None
    flags = np.asarray([k == first_key for k in keys])
    if any(k not in {first_key, second_key} for k in keys) or any(c not in {"discovery", "confirmation"} for c in cohorts):
        return None
    split = np.asarray([c == "discovery" for c in cohorts])
    # Cohort membership is created before outcomes or action labels are inspected.
    fit, held = features[split], features[~split]
    fit_flags, held_flags = flags[split], flags[~split]
    if any(np.count_nonzero(f == value) < 5 for f in (fit_flags, held_flags) for value in (False, True)):
        return None
    rng = np.random.default_rng(seed)
    if camera is not None and _camera_answers(camera[split], fit_flags, rng, permutations):
        return None
    channels = np.all(available[split], axis=0)
    usable = np.any(channels, axis=-1)
    if not np.any(usable):
        return None
    fit = np.where(channels, fit, 0.)
    statistic, difference = _contrast(fit, fit_flags)
    effect = np.linalg.norm(difference, axis=-1)
    exceed = np.zeros(usable.shape, np.int32)
    for _ in range(permutations):
        null, _ = _contrast(fit, rng.permutation(fit_flags))
        exceed += np.max(null[usable]) >= statistic
    probability = (1 + exceed) / (1 + permutations)
    eligible = usable & (probability <= .01) & (effect >= .6)
    if np.count_nonzero(eligible) < 2 or eligible.mean() > .2:
        return None
    groups, size = label(binary_dilation(eligible, iterations=1))
    components = []
    for number in range(1, size + 1):
        support = eligible & (groups == number)
        if np.count_nonzero(support) >= 2:
            components.append((float(effect[support].sum()), support))
    components.sort(key=lambda entry: entry[0], reverse=True)
    if not components or len(components) > 1 and components[1][0] >= .4 * components[0][0]:
        return None
    support = components[0][1]
    # Confirmation's values cannot change the region or its direction.
    if not np.all(available[~split][:, support][:, channels[support]]):
        return None
    pattern = np.where(support[..., None], difference, 0.)
    norm = float(np.linalg.norm(pattern))
    if norm <= 1e-6:
        return None
    pattern /= norm
    held_values = np.einsum("nhwc,hwc->n", np.where(channels, held, 0.), pattern)
    contrast = float(held_values[held_flags].mean() - held_values[~held_flags].mean())
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(held_flags)
        exceed += held_values[order].mean() - held_values[~order].mean() >= contrast
    confirm_p = (1 + exceed) / (1 + permutations)
    if (contrast <= 0 or confirm_p > .01
            or camera is not None and _camera_answers(camera[~split], held_flags, rng, permutations)):
        return None
    yy, xx = np.nonzero(support)
    return ResponsiveRegion((int(xx.min()) * tile, int(yy.min()) * tile,
                             (int(xx.max()) + 1) * tile, (int(yy.max()) + 1) * tile),
                            float(probability[support].max()), float(confirm_p), float(effect[support].max()),
                            reference="screen_unreferenced" if camera is None else "distributed_translation")


@invariant("perception.pixel_response_uses_distinct_confirmation_cohorts", scope="perception",
           owner="core/perception/a_region_that_answers_independent_inputs.py", observational=False)
def _pixel_cohort_invariant() -> tuple:
    data = np.zeros((24, 4, 5, 3), np.float32)
    counts = np.full((24, 4, 5), 4)
    camera = np.zeros((24, 2))
    keys = ("a", "b") * 12
    cohorts = ("discovery",) * 12 + ("confirmation",) * 12
    assert region_from_press_features(data, counts, camera, keys, cohorts, first_key="a", second_key="b") is None
    return ()
