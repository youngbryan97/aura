"""Dense static-raster agreement can locate artwork without thick opaque patches."""
import numpy as np
import pytest

from core.perception.coordinates_in_a_visual_reference import calibrated_source_coordinates

pytestmark = pytest.mark.unit


def image():
    pixels = np.zeros((40, 96, 4), np.uint8)
    for x in range(3, 90):
        y = 4+round(x*.2)
        pixels[y:y+2, x] = (210, 180+x//3, 70, 255)
        if x % 13 == 0:
            pixels[y:y+10, x:x+2] = (30+x, 70, 120, 255)
    pixels[30:32, 8:25] = (240, 80, 120, 255)
    return pixels


def svg():
    return b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:f="https://www.free-decompiler.com/flash" f:objectType="frame" width="96px" height="40px"><g transform="matrix(1,0,0,1,47.5,-5.5)"/></svg>'


def test_thin_current_independent_rasters_can_establish_the_declared_pixel_frame():
    pixels = image()
    coordinates = calibrated_source_coordinates('source', svg(), pixels, pixels)
    assert coordinates is not None
    assert coordinates.linear == ((1., 0.), (0., 1.))
    assert coordinates.offset == (47.5, -5.5) and coordinates.calibration_error == 0.


@pytest.mark.parametrize('damage', ['shift', 'colour', 'alpha', 'uniform', 'rows', 'columns'])
def test_dense_static_alignment_needs_both_axes_and_all_visible_bytes(damage):
    pixels, reference = image(), image()
    if damage == 'shift':
        reference = np.roll(reference, 1, axis=1)
    elif damage == 'colour':
        reference[30:32, 8:25, :3] = 255
    elif damage == 'alpha':
        reference[30:32, 8:25, 3] = 0
    elif damage == 'uniform':
        pixels[:] = reference[:] = (60, 80, 30, 255)
    elif damage == 'rows':
        pixels[:] = reference[:] = 0
        pixels[20:22] = reference[20:22] = (60, 80, 30, 255)
    else:
        pixels[:] = reference[:] = 0
        pixels[:, 20:22] = reference[:, 20:22] = (60, 80, 30, 255)
    assert calibrated_source_coordinates('source', svg(), pixels, reference) is None


def test_transparent_rgb_is_immaterial_but_hidden_opaque_damage_is_not():
    from core.perception.two_rasters_in_one_declared_pixel_frame import same_declared_pixel_frame

    first, second = image(), image()
    second[first[..., 3] == 0, :3] = 255
    assert same_declared_pixel_frame(first, second)
    second[30, 8, 0] += 3
    assert not same_declared_pixel_frame(first, second)


def test_the_dense_route_cannot_replace_an_existing_measured_calibration(monkeypatch):
    from core.perception import coordinates_in_a_visual_reference as work

    pixels = image()
    pose = type('Pose', (), dict(linear=((1., 0.), (0., 1.)), offset=(.1, .2), error=.3,
        project=lambda _self, p: (p[0]+.1, p[1]+.2)))()
    monkeypatch.setattr(work, 'reference_appearances', lambda *_a, **_kw: (pose,))
    measured = work.calibrated_source_coordinates('source', svg(), pixels, pixels)
    assert measured.offset == (47.6, -5.3) and measured.calibration_error == .3


def test_ambiguous_correspondences_remain_ambiguous_even_when_rasters_equal(monkeypatch):
    from core.perception import coordinates_in_a_visual_reference as work

    monkeypatch.setattr(work, 'reference_appearances', lambda *_a, **_kw: (object(), object()))
    assert work.calibrated_source_coordinates('source', svg(), image(), image()) is None


def test_dense_static_alignment_has_a_measured_canary():
    from core.perception.two_rasters_in_one_declared_pixel_frame import _dense_static_frame_invariant

    assert _dense_static_frame_invariant() == ()
