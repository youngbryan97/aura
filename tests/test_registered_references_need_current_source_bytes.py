"""Local reference acquisition retains its source and raster evidence."""
import hashlib
import io
import json
import struct
from dataclasses import asdict, replace

import numpy as np
import pytest
from PIL import Image

from core.perception.coordinates_in_a_visual_reference import calibrated_source_coordinates
from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
from core.perception.visual_references_for_current_sources import (
    _current_reference_source_invariant, acquire_visual_references,
    current_visual_sources, registered_reference,
)
from tools.publish_program_visual_reference import publish_reference_bundle

pytestmark = pytest.mark.unit


@pytest.fixture
def bundle(tmp_path):
    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    stage = int(rect, 2).to_bytes(len(rect)//8, 'big')+struct.pack('<HH', 24*256, 1)
    definition = struct.pack('<H', (39 << 6) | 6)+struct.pack('<HHH', 7, 1, 0)
    export = struct.pack('<H', (56 << 6) | 9)+struct.pack('<HH', 1, 7)+b'icon\0'
    body = stage+definition+export+b'\0\0'
    raw = b'FWS\x06'+struct.pack('<I', len(body)+8)+body
    table = read_flash_visual_assets(raw)
    root = tmp_path/'bundle'
    folder = root/'rendered'/'DefineSprite_7_icon'
    folder.mkdir(parents=True)
    (root/'source.swf').write_bytes(raw)
    pixels = np.random.default_rng(719).integers(10, 230, (64, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format='PNG')
    png = encoded.getvalue()
    (folder/'1.png').write_bytes(png)
    (root/'coordinate-raster.png').write_bytes(png)
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:f="https://www.free-decompiler.com/flash" '
           b'f:objectType="frame" width="64px" height="64px"><g transform="matrix(1,0,0,1,12,8)"/></svg>')
    (root/'frame.svg').write_bytes(svg)
    coordinates = calibrated_source_coordinates(table.source, svg, pixels, pixels)
    record = dict(schema='aura.program_visual_reference.v1', source=table.source, status='rendered',
        declaration=asdict(table.named('icon')),
        images=[dict(path='DefineSprite_7_icon/1.png', file_sha256=hashlib.sha256(png).hexdigest(), size=[64, 64])],
        registration=dict(coordinates=asdict(coordinates), svg='frame.svg', raster='coordinate-raster.png',
                          raster_file_sha256=hashlib.sha256(png).hexdigest(),
                          raster_raw_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
    (root/'receipt.json').write_text(json.dumps(record))
    return root, table, pixels


def test_a_registered_bundle_reaches_only_its_current_source(bundle, tmp_path):
    root, table, pixels = bundle
    index = tmp_path/'index'
    published = publish_reference_bundle(root, root=index)
    assert published.is_file()
    result = acquire_visual_references((table,), root=index)
    reference, = result.references
    assert not result.unavailable and np.array_equal(reference.pixels, pixels)
    assert not reference.pixels.flags.writeable
    assert reference.declaration == table.named('icon')
    assert reference.coordinates.offset == pytest.approx((12., 8.))
    other = replace(table, source='swf:'+'2'*64)
    absent = acquire_visual_references((other,), root=index)
    assert not absent.references and 'no published references' in absent.unavailable[0]
    assert not hasattr(reference, 'controlled') and not hasattr(reference, 'role')


def test_runtime_acquisition_has_no_publication_effect(bundle, tmp_path, monkeypatch):
    from core.runtime import file_write_gateway
    from core.perception import visual_references_for_current_sources as catalog

    root, table, _pixels = bundle
    index = tmp_path/'index'
    published = publish_reference_bundle(root, root=index)
    before = published.read_bytes()
    monkeypatch.setattr(file_write_gateway, 'get_file_write_gateway',
                        lambda: pytest.fail('ordinary acquisition cannot request a writer'))
    assert acquire_visual_references((table,), root=index).references
    assert published.read_bytes() == before and not hasattr(catalog, 'publish_reference_bundle')


@pytest.mark.parametrize('change', ['program', 'declaration', 'reference', 'raster', 'svg', 'receipt', 'escape', 'link'])
def test_changed_bundle_evidence_is_unavailable_instead_of_reused(bundle, tmp_path, change):
    root, table, _pixels = bundle
    index = tmp_path/'index'
    publish_reference_bundle(root, root=index)
    receipt = root/'receipt.json'
    record = json.loads(receipt.read_text())
    if change == 'program':
        (root/'source.swf').write_bytes(b'not the current program')
    elif change == 'declaration':
        record['declaration']['definition_tag'] = 83
    elif change == 'reference':
        (root/'rendered'/'DefineSprite_7_icon'/'1.png').write_bytes(b'changed')
    elif change == 'raster':
        (root/'coordinate-raster.png').write_bytes(b'changed')
    elif change == 'svg':
        (root/'frame.svg').write_bytes(b'<svg/>')
    elif change == 'receipt':
        record['registration']['coordinates']['offset'] = [90., 8.]
    elif change == 'escape':
        record['images'][0]['path'] = '../../../elsewhere.png'
    else:
        (root/'frame.svg').unlink()
        (root/'frame.svg').symlink_to(root/'coordinate-raster.png')
    if change in ('declaration', 'receipt', 'escape'):
        receipt.write_text(json.dumps(record))
    result = acquire_visual_references((table,), root=index)
    assert not result.references and result.unavailable


def test_a_receipt_cannot_invent_a_coordinate_calibration(bundle):
    root, _table, _pixels = bundle
    path = root/'receipt.json'
    record = json.loads(path.read_text())
    record['registration']['coordinates']['offset'] = [90., 8.]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='measured raster alignment'):
        registered_reference(root)


def test_a_current_source_must_still_contain_the_exact_declaration(bundle, tmp_path):
    root, table, _pixels = bundle
    index = tmp_path/'index'
    publish_reference_bundle(root, root=index)
    changed = replace(table, assets=(replace(table.assets[0], export_offsets=(999,)),))
    result = acquire_visual_references((changed,), root=index)
    assert not result.references and 'current source' in result.unavailable[0]


def test_an_empty_or_incomplete_read_cannot_select_a_catalog(bundle):
    _, table, _ = bundle
    assert current_visual_sources((replace(table, issues=('incomplete',)),)) == ()
    assert current_visual_sources([table]) == ()
    assert _current_reference_source_invariant() == ()


def test_guide_current_source_replacement_does_not_keep_historical_artwork(bundle):
    from core.cognition.a_guide_to_a_place import Guide
    from core.perception.reading_a_program import ProgramRead

    _, table, _ = bundle
    guide = Guide(place='arbitrary')
    guide.take_in_program(ProgramRead(visual_assets=(table,)))
    assert guide.program_visual_assets == (table,)
    second = replace(table, source='swf:'+'3'*64)
    guide.take_in_program(ProgramRead(visual_assets=(second,)))
    assert guide.program_visual_assets == (second,)
    guide.take_in_program(ProgramRead())
    assert guide.program_visual_assets == ()
