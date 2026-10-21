"""Reference frame selection belongs to complete current source boundaries."""
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, replace

import pytest

from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
from core.perception.visual_frames_in_a_source_declaration import read_declared_visual_frames
from core.perception.visual_references_for_current_sources import registered_reference
from tests.test_registered_references_need_current_source_bytes import bundle  # noqa: F401
from tools.export_program_visual_reference import _render_command

pytestmark = pytest.mark.unit


def tag(code, data=b''):
    if len(data) < 63:
        return struct.pack('<H',code*64+len(data))+data
    return struct.pack('<HI',code*64+63,len(data))+data


def source(*, count=2, control=None, compressed=False):
    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    stage = int(rect,2).to_bytes(len(rect)//8,'big')+struct.pack('<HH',24*256,1)
    if control is None:
        control = tag(43,b'idle\0')+tag(1)+tag(43,b'changed\0\x01')+tag(1)+tag(0)
    body = stage+tag(39,struct.pack('<HH',7,count)+control)+tag(56,struct.pack('<HH',1,7)+b'icon\0')+tag(0)
    raw = (b'CWS' if compressed else b'FWS')+b'\x06'+struct.pack('<I',len(body)+8)
    return raw+(zlib.compress(body) if compressed else body)


@pytest.mark.parametrize('compressed',[False,True])
def test_every_source_frame_has_its_own_index_and_its_labels_are_only_data(compressed):
    raw = source(compressed=compressed)
    asset = read_flash_visual_assets(raw).named('icon')
    timeline = read_declared_visual_frames(raw,asset)
    assert timeline.complete and timeline.source == 'swf:'+hashlib.sha256(raw).hexdigest()
    assert timeline.frame_count == 2
    assert [(f.number,f.labels) for f in timeline.frames] == [(1,('idle',)),(2,('changed',))]
    assert timeline.frames[0].show_offset < timeline.frames[1].show_offset
    assert not hasattr(timeline,'role') and not hasattr(timeline,'current_frame')


@pytest.mark.parametrize('raw',[source(count=3),source(count=1),
    source(control=tag(1)+tag(1)), source(control=tag(1,b'bad')+tag(1)+tag(0)),
    source(control=tag(43,b'bad\0\x02')+tag(1)+tag(1)+tag(0)),
    source(control=tag(1)+tag(1)+tag(43,b'late\0')+tag(0)),
    source(control=tag(1)+tag(1)+b'\xff\xff'),source(count=4097)])
def test_incomplete_and_malformed_frame_enumerations_supply_no_selected_frame(raw):
    asset = read_flash_visual_assets(raw).named('icon')
    result = read_declared_visual_frames(raw,asset)
    assert not result.complete and not result.frames and result.issues


def test_a_timeline_cannot_borrow_an_address_from_a_different_source_declaration():
    raw = source()
    asset = read_flash_visual_assets(raw).named('icon')
    for changed in (replace(asset,definition_offset=99),replace(asset,export_offsets=(99,)),
                    replace(asset,character_id=8),replace(asset,definition_offset=False)):
        assert not read_declared_visual_frames(raw,changed).complete


def test_the_export_command_selects_a_supplied_variant_without_an_alias(tmp_path):
    asset = read_flash_visual_assets(source()).named('icon')
    args = _render_command(tmp_path/'java',tmp_path/'tool.jar',tmp_path/'source.swf',
        tmp_path/'out',tmp_path/'home',asset,frame=2)
    assert args[args.index('-select')+1] == '7:2'
    assert args[args.index('-sublength')+1] == '1' and asset.name not in args
    for bad in (0,-1,True,4097):
        with pytest.raises(ValueError):
            _render_command(tmp_path/'java',tmp_path/'tool.jar',tmp_path/'source.swf',
                tmp_path/'out',tmp_path/'home',asset,frame=bad)


def test_registered_variant_rechecks_its_exact_current_source_frame(bundle):
    root,_,_ = bundle
    raw=source()
    (root/'source.swf').write_bytes(raw)
    record=json.loads((root/'receipt.json').read_text())
    record.update(source='swf:'+hashlib.sha256(raw).hexdigest(),
                  declaration=asdict(read_flash_visual_assets(raw).named('icon')),
                  frame_number=2,declared_frame_count=2)
    record['registration']['coordinates']['source']=record['source']
    old=root/'rendered'/'DefineSprite_7_icon'/'1.png'
    old.rename(old.with_name('2.png'))
    record['images'][0]['path']='DefineSprite_7_icon/2.png'
    (root/'receipt.json').write_text(json.dumps(record))
    reference=registered_reference(root)
    assert reference.frame_number == 2
    for key,value in (('frame_number',3),('frame_number',True),('frame_number',1),('declared_frame_count',1)):
        changed={**record,key:value}
        (root/'receipt.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='frame'):
            registered_reference(root)


def test_missing_variants_are_enumerated_once_and_coverage_keeps_unknowns(bundle):
    from core.perception.visual_references_for_current_sources import ReferencesAcquired, reference_variant_coverage
    from tools.prepare_program_visual_variants import missing_variant_jobs

    root,_,_ = bundle
    original=registered_reference(root)
    raw=source()
    asset=read_flash_visual_assets(raw).named('icon')
    key='swf:'+hashlib.sha256(raw).hexdigest()
    first=replace(original,source=key,declaration=asset,coordinates=replace(original.coordinates,source=key),
                  declared_frame_count=2)
    catalog=ReferencesAcquired((first,first))
    assert missing_variant_jobs(raw,catalog) == ((asset,2),)
    coverage,=reference_variant_coverage(catalog.references)
    assert coverage.available_frames == (1,) and coverage.missing_frames == (2,)
    second=replace(first,frame_number=2)
    assert missing_variant_jobs(raw,ReferencesAcquired((first,second))) == ()
    assert reference_variant_coverage((first,second))[0].missing_frames == ()
    assert reference_variant_coverage((original,))[0].missing_frames is None
    with pytest.raises(ValueError,match='current'):
        missing_variant_jobs(source(count=3),catalog)
    raw=source(count=3,control=tag(1)+tag(1)+tag(1)+tag(0))
    first=replace(first,source='swf:'+hashlib.sha256(raw).hexdigest(),
                  declaration=read_flash_visual_assets(raw).named('icon'))
    with pytest.raises(ValueError,match='job count'):
        missing_variant_jobs(raw,ReferencesAcquired((first,)),limit=1)


def test_source_frame_claim_has_a_registered_measured_check():
    from core.perception.visual_frames_in_a_source_declaration import _visual_frame_custody_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _visual_frame_custody_invariant() == ()
    suite=ValidationSuite()
    install_realtime_control_claims(suite)
    test=next(t for t in suite.tests() if t.name == 'visual_reference_variants_require_current_source_frames')
    assert test.predict(None) is True and any(c.test == test.name for c in suite.claims())


def test_equal_raster_bytes_cannot_erase_distinct_source_coordinate_variants(bundle,tmp_path):
    from core.perception.visual_references_for_current_sources import acquire_visual_references
    from tools.publish_program_visual_reference import publish_reference_bundle

    root,_,_=bundle
    raw=source()
    (root/'source.swf').write_bytes(raw)
    record=json.loads((root/'receipt.json').read_text())
    record.update(source='swf:'+hashlib.sha256(raw).hexdigest(),
        declaration=asdict(read_flash_visual_assets(raw).named('icon')),frame_number=1,declared_frame_count=2)
    record['registration']['coordinates']['source']=record['source']
    (root/'receipt.json').write_text(json.dumps(record))
    index=tmp_path/'index'
    first=publish_reference_bundle(root,root=index)
    other=tmp_path/'second'
    import shutil
    shutil.copytree(root,other)
    record.update(frame_number=2)
    image=other/'rendered'/'DefineSprite_7_icon'/'1.png'
    image.rename(image.with_name('2.png'))
    record['images'][0]['path']='DefineSprite_7_icon/2.png'
    from PIL import Image
    import numpy as np
    from core.perception.coordinates_in_a_visual_reference import calibrated_source_coordinates
    svg=(other/'frame.svg').read_bytes().replace(b'12,8',b'20,8')
    (other/'frame.svg').write_bytes(svg)
    with Image.open(image.with_name('2.png')) as rendered:
        pixels=np.array(rendered.convert('RGBA'))
    record['registration']['coordinates']=asdict(calibrated_source_coordinates(record['source'],svg,pixels,pixels))
    (other/'receipt.json').write_text(json.dumps(record))
    second=publish_reference_bundle(other,root=index)
    table=read_flash_visual_assets(raw)
    catalog=acquire_visual_references((table,),root=index)
    assert first != second and len(catalog.references) == 2
    a,b=catalog.references
    assert a.key == b.key and a.variant_key != b.variant_key
    assert {a.frame_number,b.frame_number} == {1,2}
    assert a.coordinates.offset != b.coordinates.offset


def test_reference_scheduling_visits_other_assets_before_more_variants(bundle):
    from core.perception.visual_references_for_current_sources import interleaved_reference_variants

    root,_,_=bundle
    first=registered_reference(root)
    a=tuple(replace(first,frame_number=n) for n in (1,2,3,4))
    b=replace(first,declaration=replace(first.declaration,character_id=8))
    c=replace(first,declaration=replace(first.declaration,character_id=9))
    ordered=interleaved_reference_variants((*a,b,c))
    assert [(r.declaration.character_id,r.frame_number) for r in ordered] == [(7,1),(8,1),(9,1),(7,2),(7,3),(7,4)]
    reordered=interleaved_reference_variants((a[2],b,a[0],a[1],c,a[3]))
    assert reordered == ordered
