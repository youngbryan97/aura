"""An unrelated difficult patch cannot keep a converged point in the solver."""
import numpy as np
import pytest
from scipy import ndimage

from core.perception import points_before_the_picture_is_reduced as source

pytestmark = pytest.mark.unit


def _images(seed=37):
    before = ndimage.gaussian_filter(np.random.default_rng(seed).uniform(.05, .95, (96, 96, 3)), (.6, .6, 0.))
    after = ndimage.shift(before, (.4, -.7, 0.), order=1, mode='constant')
    after[:35, :35] = before[:35, :35]
    points = np.array(((20., 20.), (50., 50.), (75., 75.)))
    initial = points + np.array(((0., 0.), (6., 4.), (-.65, .36)))
    return before, after, points, initial


@pytest.mark.parametrize('seed', [37, 52, 89])
def test_each_patch_has_the_same_solution_alone_or_among_difficult_neighbours(seed):
    before, after, points, initial = _images(seed)
    grouped = source._refine(before, after, points, initial)
    for index in range(len(points)):
        alone = source._refine(before, after, points[index:index+1], initial[index:index+1])
        for result, one in zip(grouped, alone, strict=True):
            np.testing.assert_allclose(result[index:index+1], one, rtol=0., atol=1e-10)


def test_converged_patches_stop_sampling_while_another_patch_uses_its_budget(monkeypatch):
    before, after, points, initial = _images()
    original, sampled = source._samples, []

    def samples(picture, places):
        sampled.append(len(places))
        return original(picture, places)

    monkeypatch.setattr(source, '_samples', samples)
    source._refine(before, after, points, initial)
    # Templates and the final residual still measure every patch. The
    # iterations between them must let completed patches leave the batch.
    assert min(sampled[3:-3]) < len(points)


def test_unobservable_patches_are_retired_after_the_first_fit(monkeypatch):
    image = np.full((32, 32, 3), .5)
    calls, original = [], source._samples

    def samples(picture, places):
        calls.append(len(places))
        return original(picture, places)

    monkeypatch.setattr(source, '_samples', samples)
    ends, errors, angles, useful = source._refine(image, image, np.array(((10., 10.),)), np.array(((10., 10.),)))
    assert not useful.any() and len(calls) == 15
    assert np.isfinite(ends).all() and np.isfinite(errors).all() and np.isfinite(angles).all()


def test_independent_patch_fits_have_a_registered_numerical_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert source._independent_patch_fit_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'local_patch_fits_finish_independently')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
