"""Reference registration needs an independent raster and exact pixel custody."""
from __future__ import annotations

import base64
import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from core.perception.coordinates_in_a_visual_reference import (
    _source_image_coordinates_invariant,
    calibrated_source_coordinates,
    exported_frame_coordinates,
)
from core.perception.a_visual_reference_in_the_picture import reference_appearances


def _svg(matrix='1,0,0,1,13,17', extra='', size='64'):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:f="https://www.free-decompiler.com/flash" '
        f'f:objectType="frame" width="{size}px" height="64px"><g transform="matrix({matrix})">{extra}</g></svg>').encode()


def _texture():
    pixels = np.random.default_rng(71).integers(0, 256, (64, 64, 4), np.uint8)
    pixels[..., 3] = 255
    return pixels


@pytest.mark.parametrize('matrix,origin', [('1,0,0,1,13,17', (13., 17.)),
    ('-1,0,0,1,90,-12', (90., -12.)), ('1,.2,.1,1,-10,31', (-10., 31.))])
def test_source_origins_follow_the_declared_coordinate_frame_instead_of_the_image_centre(matrix, origin):
    pixels = _texture()
    coordinates = calibrated_source_coordinates('source-A', _svg(matrix), pixels, pixels)
    assert coordinates is not None and coordinates.offset == pytest.approx(origin)
    picture = np.full((128, 128, 3), 31, np.uint8)
    picture[23:87, 29:93] = pixels[..., :3]
    poses = reference_appearances(picture, pixels, scales=(1.,))
    assert len(poses) == 1
    point = coordinates.project((0., 0.), poses[0], picture, pixels, source='source-A')
    assert point == pytest.approx((origin[0]+29, origin[1]+23), abs=.5)


@pytest.mark.parametrize('change', ['source', 'picture', 'reference', 'point', 'boolean', 'pose'])
def test_changed_sources_or_pixels_cannot_carry_the_point(change):
    pixels = _texture()
    coordinates = calibrated_source_coordinates('source-A', _svg(), pixels, pixels)
    picture = pixels[..., :3].copy()
    pose = reference_appearances(picture, pixels, scales=(1.,))[0]
    source, point = 'source-A', (0., 0.)
    if change == 'source':
        source = 'source-B'
    elif change == 'picture':
        picture[0, 0] = 0
    elif change == 'reference':
        pixels = pixels.copy()
        pixels[0, 0] = 0
    elif change == 'point':
        point = (float('nan'), 0.)
    elif change == 'boolean':
        point = (True, 0.)
    else:
        pose = replace(pose, reference_sha256='0'*64)
    assert coordinates.project(point, pose, picture, pixels, source=source) is None


@pytest.mark.parametrize('damage', ['unknown', 'viewBox', 'missing-transform', 'singular', 'nonfinite', 'external',
                                  'active', 'animation', 'entity', 'oversize', 'multiple-groups'])
def test_unsupported_coordinate_contracts_do_not_default_to_an_origin(damage):
    svg = _svg()
    if damage == 'unknown':
        svg = svg.replace(b'objectType="frame"', b'objectType="other"')
    elif damage == 'viewBox':
        svg = svg.replace(b'width="64px"', b'viewBox="0 0 10 10" width="64px"')
    elif damage == 'missing-transform':
        svg = svg.replace(b' transform="matrix(1,0,0,1,13,17)"', b'')
    elif damage == 'singular':
        svg = _svg('0,0,0,0,0,0')
    elif damage == 'nonfinite':
        svg = _svg('1e999,0,0,1,0,0')
    elif damage == 'external':
        svg = _svg(extra='<image href="https://example.org/image.png"/>')
    elif damage == 'active':
        svg = _svg(extra='<script>anything()</script>')
    elif damage == 'animation':
        svg = _svg(extra='<animateTransform attributeName="transform"/>')
    elif damage == 'entity':
        svg = b'<!DOCTYPE svg [<!ENTITY x "any">]>'+svg
    elif damage == 'oversize':
        svg = _svg(size='513')
    else:
        svg = svg.replace(b'</svg>', b'<g transform="matrix(1,0,0,1,0,0)"/></svg>')
    with pytest.raises(ValueError):
        exported_frame_coordinates(svg)


def test_unmatched_rasters_cannot_supply_registration_and_fractional_viewports_keep_pixel_units():
    pixels = _texture()
    assert calibrated_source_coordinates('source', _svg(), pixels, np.flipud(pixels).copy()) is None
    assert exported_frame_coordinates(_svg(size='64.25'))[0] == (65, 64)
    assert _source_image_coordinates_invariant() == ()


def test_source_coordinate_custody_has_a_registered_check():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'source_reference_points_need_measured_raster_alignment')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


def test_coordinate_geometry_is_owned_after_construction_and_invalid_geometry_is_refused():
    pixels = _texture()
    coordinates = calibrated_source_coordinates('source', _svg(), pixels, pixels)
    linear, offset = [[1., 0.], [0., 1.]], [13., 17.]
    owned = replace(coordinates, linear=linear, offset=offset)
    linear[0][0], offset[0] = 99., 999.
    assert owned.linear == ((1., 0.), (0., 1.)) and owned.offset == (13., 17.)
    for changes in ({'linear': ((0., 0.), (0., 0.))}, {'offset': (float('nan'), 0.)},
                    {'reference_sha256': 'missing'}, {'calibration_error': True}):
        with pytest.raises(ValueError):
            replace(coordinates, **changes)


@pytest.mark.asyncio
async def test_independent_software_rasterization_places_a_source_marker_at_the_registered_point():
    from tools.reference_frame_registration import rasterized_frame

    pixels = _texture()
    png = io.BytesIO()
    Image.fromarray(pixels).save(png, format='PNG')
    encoded = base64.b64encode(png.getvalue()).decode()
    image = f'<image x="-13" y="-17" width="64" height="64" href="data:image/png;base64,{encoded}"/>'
    svg = _svg(extra=image)
    rendered = await rasterized_frame(svg)
    coordinates = calibrated_source_coordinates('source', svg, rendered, pixels)
    assert coordinates is not None and coordinates.offset == pytest.approx((13., 17.), abs=.5)
    marked = await rasterized_frame(_svg(extra=image+'<circle cx="0" cy="0" r="2" fill="#e13a00"/>'))
    assert tuple(marked[17, 13, :3]) == (225, 58, 0)
