"""Offline reference probes cannot silently substitute recordings or change their pixels."""
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools.measure_source_points_on_recorded_material import _contained, _frames, _pixels, _reference

pytestmark = pytest.mark.unit


def test_a_bounded_prefix_and_future_frames_keep_the_recorded_frame_boundary(tmp_path):
    events = [dict(sequence=n, kind='frame' if n % 2 == 0 else 'input') for n in range(20)]
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    frames, following = _frames(tmp_path, 10, 3, follow=2)
    assert [e['sequence'] for e in frames] == [6, 8, 10]
    assert [e['sequence'] for e in following] == [12, 14]
    with pytest.raises(ValueError, match='recorded frame'):
        _frames(tmp_path, 11, 3, follow=2)


def test_out_of_order_events_cannot_become_a_valid_prefix(tmp_path):
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(dict(sequence=n, kind='frame')) for n in (0, 2, 1, 3)))
    with pytest.raises(ValueError, match='increase'):
        _frames(tmp_path, 3, 3)


@pytest.mark.parametrize('name', ['/tmp/elsewhere.png', '../elsewhere.png'])
def test_recording_paths_cannot_escape_their_bundle(tmp_path, name):
    with pytest.raises(ValueError):
        _contained(tmp_path, name)


def test_decoded_rgb_bytes_must_agree_with_the_recording(tmp_path):
    picture = np.random.default_rng(31).integers(0, 255, (32, 48, 3), np.uint8)
    path = tmp_path/'frame.png'
    Image.fromarray(picture).save(path)
    event = dict(file=path.name, shape=list(picture.shape), dtype='uint8',
                 raw_sha256=hashlib.sha256(picture.tobytes()).hexdigest())
    assert np.array_equal(_pixels(tmp_path, event), picture)
    picture[0, 0] = 0
    Image.fromarray(picture).save(path)
    with pytest.raises(ValueError, match='bytes changed'):
        _pixels(tmp_path, event)


@pytest.mark.parametrize('change', [None, 'coordinate_source', 'coordinate_pixels', 'declaration', 'schema'])
def test_reference_replay_checks_the_coordinate_namespace_and_rendered_declaration(tmp_path, monkeypatch, change):
    from core.perception import visual_assets_in_a_flash_file as declarations
    from core.perception.a_visual_reference_in_the_picture import _digest
    from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
    from tools.export_program_visual_reference import _inventory

    asset = SimpleNamespace(character_id=1, definition_offset=10)
    table = SimpleNamespace(source='source-A', named=lambda name: asset if name == 'example' else None)
    monkeypatch.setattr(declarations, 'read_flash_visual_assets', lambda raw: table)
    pixels = np.full((32, 48, 4), 127, np.uint8)
    root = tmp_path/'rendered'
    folder = root/'DefineSprite_1_example'
    folder.mkdir(parents=True)
    Image.fromarray(pixels).save(folder/'1.png')
    (tmp_path/'source.swf').write_bytes(b'supplied source parser fixture')
    svg = b'<svg/>'
    (tmp_path/'frame.svg').write_bytes(svg)
    coordinates = SourceImageCoordinates('source-A', _digest(pixels), hashlib.sha256(svg).hexdigest(),
        (48, 32), ((1., 0.), (0., 1.)), (0., 0.), .01)
    record = dict(schema='aura.program_visual_reference.v1', source='source-A', status='rendered',
        images=_inventory(root, 1), declaration=dict(name='example', character_id=1, definition_offset=10),
        registration=dict(coordinates=asdict(coordinates), svg='frame.svg'))
    if change == 'coordinate_source':
        record['registration']['coordinates']['source'] = 'source-B'
    elif change == 'coordinate_pixels':
        record['registration']['coordinates']['reference_sha256'] = '0'*64
    elif change == 'declaration':
        record['declaration']['character_id'] = 2
    elif change == 'schema':
        record['schema'] = 'other'
    (tmp_path/'receipt.json').write_text(json.dumps(record))
    if change is None:
        measured, image = _reference(tmp_path)
        assert measured == coordinates and np.array_equal(image, pixels)
    else:
        with pytest.raises(ValueError):
            _reference(tmp_path)
