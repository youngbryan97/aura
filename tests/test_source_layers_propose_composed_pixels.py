"""Source layers propose appearances; current pixels still decide admission."""
import struct
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances
from core.perception.composed_source_artwork_for_pixel_search import (
    ComposedVisualReference, compose_source_layers, with_source_layer_proposals,
)
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.guarded_effects_in_flash_methods import read_flash_effect_graph
from core.perception.source_layers_with_copied_positions import SourceLayerProposal, layers_in_source_graph
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, FlashVisualAssets
from core.perception.visual_references_for_current_sources import RegisteredVisualReference, reference_variant_coverage
from core.perception.whole_referents_from_source_material import WholeReferentsInView
from tests.test_guarded_source_effects_do_not_enumerate_paths import _action, _binding, _push
from tests.test_source_geometry_precedes_object_grouping import attach, scene  # noqa: F401

pytestmark = pytest.mark.unit
SOURCE = 'swf:'+'1'*64
BASE = DeclaredVisualAsset('drawing-A', 17, 'sprite', 0, 39, (10,))
OVER = DeclaredVisualAsset('drawing-B', 19, 'sprite', 20, 39, (30,))
TABLE = FlashVisualAssets(SOURCE, (BASE, OVER))


def attachment(symbol, instance, depth, container='holder'):
    return _push(depth, instance, symbol, 3, container)+b'\x1c'+_push('attachMovie')+b'\x52\x17'


def copied(axis, delta=0, base='a', over='b'):
    code = _push(over)+b'\x1c'+_push(axis, base)+b'\x1c'+_push(axis)+b'\x4e'
    return code+(_push(delta)+b'\x47' if delta else b'')+b'\x4f'


def graph(extra=b'', *, second_container='holder', second_depth=40, delta=(0, 0)):
    body = attachment(BASE.name, 'a', 10)+attachment(OVER.name, 'b', second_depth, second_container)
    body += copied('_x', delta[0])+copied('_y', delta[1])+extra+b'\0'
    return replace(read_flash_effect_graph(_binding(body)), source=SOURCE+'/function')


def proposal(delta=(0., 0.)):
    return SourceLayerProposal(SOURCE, 'layout', SOURCE+'/function', 17, 19, delta, (10., 40.), (1, 2, 3, 4))


def registered(asset, pixels, origin, *, frame=1):
    coordinates = SourceImageCoordinates(SOURCE, _digest(pixels), '0'*64, pixels.shape[1::-1],
                                        ((1., 0.), (0., 1.)), origin, 0.)
    return RegisteredVisualReference(SOURCE, asset, coordinates, pixels, frame, 2)


@pytest.fixture
def references():
    base = np.random.default_rng(743).integers(10, 230, (64, 64, 4), dtype=np.uint8)
    base[..., 3] = 255
    over = np.zeros((24, 24, 4), np.uint8)
    over[2:22, 3:21] = (30, 80, 170, 200)
    over[10:12] = (220, 200, 80, 255)
    return registered(BASE, base, (20., 20.)), registered(OVER, over, (8., 8.))


def test_owned_unconditional_effects_propose_layer_order_and_copied_coordinates():
    reading = graph(delta=(-3, 5))
    found, = layers_in_source_graph(reading, TABLE)
    assert found.base_character == 17 and found.over_character == 19
    assert found.source_delta == (-3., 5.) and found.depths == (10., 40.)
    assert all(offset in reading.lifted for offset in found.offsets)
    assert 'current_pixels' in found.uncertainties and 'receiver_aliases' in found.uncertainties
    assert not layers_in_source_graph(reading, replace(TABLE, source='swf:'+'2'*64))


@pytest.mark.parametrize('damage', ['parent', 'depth', 'later-position', 'transform', 'call', 'duplicate', 'conditional'])
def test_incompatible_placement_effects_cannot_supply_a_layer_proposal(damage):
    extra, options = b'', {}
    if damage == 'parent':
        options['second_container'] = 'different'
    elif damage == 'depth':
        options['second_depth'] = 5
    elif damage == 'later-position':
        extra = _push('a')+b'\x1c'+_push('_x', 90)+b'\x4f'
    elif damage == 'transform':
        extra = _push('b')+b'\x1c'+_push('_rotation', 20)+b'\x4f'
    elif damage == 'call':
        extra = _push(0, 'b')+b'\x1c'+_push('removeMovieClip')+b'\x52\x17'
    elif damage == 'duplicate':
        extra = attachment(OVER.name, 'b', 50)
    reading = graph(extra, **options)
    if damage == 'conditional':
        effect = next(e for e in reading.effects if e.operation == 'set_member')
        reading = replace(reading, effects=tuple(replace(e, guard=0) if e is effect else e for e in reading.effects))
    assert not layers_in_source_graph(reading, TABLE)


def test_an_owned_prefix_can_propose_a_picture_without_certifying_later_execution():
    reading = graph()
    partial = replace(reading, issues=('later source loop requires a summary',))
    found, = layers_in_source_graph(partial, TABLE)
    assert 'execution' in found.uncertainties and not partial.complete


def test_composition_uses_registered_origins_and_premultiplied_alpha(references):
    base, over = references
    result = compose_source_layers(base, over, proposal((-3., 5.)))
    expected = Image.fromarray(base.pixels)
    expected.alpha_composite(Image.fromarray(over.pixels), (9, 17))
    assert np.max(abs(result.pixels.astype(int)-np.asarray(expected).astype(int))) <= 1
    assert result.coordinates.offset == base.coordinates.offset
    assert result.components == references and result.placement.uncertainties
    assert result.variant_key[-1].startswith('composition:') and not result.pixels.flags.writeable
    assert reference_variant_coverage((base, over, result)) == reference_variant_coverage(references)


def test_a_layer_outside_the_base_expands_the_search_picture_and_preserves_its_source_origin(references):
    base, over = references
    result = compose_source_layers(base, over, proposal((-30., 0.)))
    assert result.pixels.shape[1] == 82 and result.coordinates.offset == (38., 20.)
    assert result.coordinates.linear == base.coordinates.linear
    assert compose_source_layers(base, over, proposal((4096., 0.))) is None


def test_component_frames_and_origins_remain_distinct_even_when_pixels_match(references):
    base, over = references
    other = replace(over, frame_number=2, coordinates=replace(over.coordinates, offset=(9., 8.)))
    a, b = compose_source_layers(base, over, proposal()), compose_source_layers(base, other, proposal())
    assert a.variant_key != b.variant_key
    same_pixels = replace(over, frame_number=2)
    c = compose_source_layers(base, same_pixels, proposal())
    assert np.array_equal(a.pixels, c.pixels) and a.variant_key != c.variant_key


def test_missing_or_changed_components_remain_unavailable_without_dropping_raw_references(references):
    combined, refused = with_source_layer_proposals(references[:1], (proposal(),))
    assert combined == references[:1] and refused == ('source_layer_reference_unavailable',)
    changed = replace(references[1], pixels=np.zeros_like(references[1].pixels))
    assert compose_source_layers(references[0], changed, proposal()) is None
    combined, refused = with_source_layer_proposals(references, (proposal(),), capacity=2)
    assert combined == references and refused == ('composed_reference_capacity_unavailable',)


def test_current_pixels_qualify_the_composition_while_the_single_asset_is_contradicted(scene):
    _coordinates, _pose, picture, rgba, observer = scene
    # Reuse the ordinary owned mark fixture, without assigning a body or goal.
    coordinates = replace(scene[0], source=SOURCE)
    base = RegisteredVisualReference(SOURCE, BASE, coordinates, rgba)
    over_pixels = np.zeros((24, 24, 4), np.uint8)
    over_pixels[1:23, 1:23] = (10, 180, 240, 255)
    over = registered(OVER, over_pixels, (0., 0.))
    composed = compose_source_layers(base, over, proposal())
    assert isinstance(composed, ComposedVisualReference)
    picture = picture.copy()
    picture[30:94, 40:104] = composed.pixels[..., :3]
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels

    observer.pixels_consumed = consumed_pixels(picture, 4.)
    observer.material_marks = replace(observer.material_marks, pixels=observer.pixels_consumed)
    pose, = reference_appearances(picture, composed.pixels, scales=(1.,))
    from core.perception.a_source_point_on_visible_marks import attach_source_point_to_visible_marks

    binding = attach_source_point_to_visible_marks(composed.coordinates, (0., 0.), pose, picture, composed.pixels,
        observer, 4., source=SOURCE).binding
    assert binding is not None
    point = binding.current(observer, 4., source=SOURCE)
    whole, = WholeReferentsInView().observe(observer, picture, 4., ((composed, binding, point),)).values()
    assert whole.w == pytest.approx(64., abs=.1) and whole.contact_verified is None and whole.role is None
    assert not observer.things
    # The uncomposed artwork has many matching patches but fails the full
    # current material check at the independently drawn overlaid region.
    original_pose, = reference_appearances(picture, rgba, scales=(1.,))
    original = attach_source_point_to_visible_marks(base.coordinates, (0., 0.), original_pose, picture, rgba,
        observer, 4., source=SOURCE).binding
    assert original is not None
    registry = WholeReferentsInView()
    assert not registry.observe(observer, picture, 4., ((base, original, original.current(observer, 4., source=SOURCE)),))
    assert registry.readings[0]['reason'] == 'whole_pixel_support_unresolved'


def test_a_guide_carries_source_proposals_only_for_the_current_read(references):
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception.reading_a_program import ProgramRead
    from core.perception.source_references_while_play_advances import source_layers_in_play

    guide = Guide()
    guide.take_in_program(ProgramRead(visual_assets=(TABLE,), visual_layers=(proposal(),)))
    token = THE_GUIDE.set(guide)
    try:
        assert source_layers_in_play((TABLE,)) == (proposal(),)
        assert not source_layers_in_play((replace(TABLE, source='swf:'+'2'*64),))
        guide.take_in_program(ProgramRead())
        assert not source_layers_in_play((TABLE,))
    finally:
        THE_GUIDE.reset(token)


def test_actual_program_read_derives_the_relationship_from_owned_binary_instructions():
    from core.perception.reading_a_program import read_program

    body = attachment(BASE.name, 'a', 10)+attachment(OVER.name, 'b', 40)+copied('_x')+copied('_y')+b'\0'
    script = _push('assemble')+_action(0x9B, b'\0\0\0'+struct.pack('<H', len(body)))+body+b'\x1d\0'
    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    stage = int(rect, 2).to_bytes(len(rect)//8, 'big')+struct.pack('<HH', 24*256, 1)

    def tag(code, payload):
        return struct.pack('<H', code << 6 | (len(payload) if len(payload) < 63 else 63)) \
            +(struct.pack('<I', len(payload)) if len(payload) >= 63 else b'')+payload

    definitions = b''.join(tag(39, struct.pack('<HHH', asset.character_id, 1, 0)) for asset in (BASE, OVER))
    exports = struct.pack('<H', 2)+b''.join(struct.pack('<H', asset.character_id)+asset.name.encode()+b'\0' for asset in (BASE, OVER))
    content = stage+definitions+tag(56, exports)+tag(12, script)+b'\0\0'
    raw = b'FWS\x06'+struct.pack('<I', len(content)+8)+content
    read = read_program(raw)
    found, = read.visual_layers
    assert found.base_character == 17 and found.over_character == 19 and found.method == 'assemble'
    assert found.source == read.action_sources.source and found.instruction_source.startswith(found.source+'/')
    assert found.source_delta == (0., 0.)


def test_ordinary_reference_catalog_adds_compositions_without_inventing_frame_coverage(references, monkeypatch):
    from concurrent.futures import Future

    from core.perception import source_references_while_play_advances as work
    from core.perception.visual_references_for_current_sources import ReferencesAcquired

    monkeypatch.setattr(work, 'acquire_visual_references', lambda *_a, **_kw: ReferencesAcquired(references))
    catalog = work._acquired_for_play((TABLE,), (proposal(),), None)
    assert len(catalog.references) == 3 and any(isinstance(r, ComposedVisualReference) for r in catalog.references)
    owner = work.SourceReferencesInPlay()
    finished = Future()
    finished.set_result(catalog)
    owner._work = work._Work(finished, owner._generation, 'catalog', 1.)
    owner._consume()
    assert owner.variant_coverage == reference_variant_coverage(references)
    record, = owner.compositions
    assert [c['character_id'] for c in record['components']] == [17, 19]
    assert record['execution_verified'] is False and record['geometry_authority'] is False
    owner.retire()
    assert not owner.compositions and not owner.layers


def test_recorded_compositions_keep_their_source_and_instruction_owners(references):
    from core.perception.composed_source_artwork_for_pixel_search import composition_record
    from core.perception.recording_the_frames_and_inputs import _source_reference_snapshot
    from types import SimpleNamespace

    composed = compose_source_layers(*references, proposal())
    observer = SimpleNamespace(source_reference_compositions=(composition_record(composed),))
    record, = _source_reference_snapshot(observer)['composed_search_references']
    assert record['source'] == SOURCE and record['offsets'] == (1, 2, 3, 4)
    assert record['uncertainties'] and record['reference_sha256'] == _digest(composed.pixels)


def test_a_composed_appearance_is_not_present_merely_because_source_proposes_it(scene):
    coordinates, _pose, picture, rgba, observer = scene
    base = RegisteredVisualReference(SOURCE, BASE, replace(coordinates, source=SOURCE), rgba)
    other = np.full((24, 24, 4), (40, 180, 230, 255), np.uint8)
    composed = compose_source_layers(base, registered(OVER, other, (0., 0.)), proposal())
    poses = reference_appearances(picture, composed.pixels, scales=(1.,))
    from core.perception.a_source_point_on_visible_marks import attach_source_point_to_visible_marks

    registry, seen = WholeReferentsInView(), {}
    for pose in poses:
        binding = attach_source_point_to_visible_marks(composed.coordinates, (0., 0.), pose, picture, composed.pixels,
            observer, 4., source=SOURCE).binding
        if binding is not None:
            seen.update(registry.observe(observer, picture, 4., ((composed, binding,
                binding.current(observer, 4., source=SOURCE)),)))
    assert not seen and not observer.things


def test_the_live_reference_owner_keeps_a_fresh_composition_and_its_visible_marks(scene):
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception import source_references_while_play_advances as work
    from core.perception.a_source_point_on_visible_marks import attach_source_point_to_visible_marks
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels

    coordinates, _pose, picture, rgba, observer = scene
    base = RegisteredVisualReference(SOURCE, BASE, replace(coordinates, source=SOURCE), rgba)
    over_pixels = np.full((24, 24, 4), (20, 170, 210, 255), np.uint8)
    composed = compose_source_layers(base, registered(OVER, over_pixels, (0., 0.)), proposal())
    picture = picture.copy()
    picture[30:94, 40:104] = composed.pixels[..., :3]
    observer.pixels_consumed = consumed_pixels(picture, 4.)
    observer.material_marks = replace(observer.material_marks, pixels=observer.pixels_consumed)
    pose, = reference_appearances(picture, composed.pixels, scales=(1.,))
    binding = attach_source_point_to_visible_marks(composed.coordinates, (0., 0.), pose, picture, composed.pixels,
        observer, 4., source=SOURCE).binding
    assert binding is not None
    observer.fresh_source_material_frames = ((composed.variant_key, binding),)
    owner = work.SourceReferencesInPlay()
    owner.tables, owner.layers, owner.catalog = (TABLE,), (proposal(),), (composed,)
    owner._loaded, owner._next = True, float('inf')
    token = THE_GUIDE.set(Guide(program_visual_assets=(TABLE,), program_visual_layers=(proposal(),)))
    try:
        owner.advance(observer, picture, 4., (TABLE,), now=10.)
        assert owner.frames == ((composed, binding),)
        assert len(owner.source_frames) == 1 and owner.retention is not None
        assert len(owner.retention.names) >= 3 and owner.retention.epoch == observer.material_marks.epoch
        assert not observer.things
    finally:
        owner.retire()
        THE_GUIDE.reset(token)


@pytest.mark.parametrize('damage', ['marks', 'stale', 'source', 'bytes', 'item', 'capacity'])
def test_retention_cannot_keep_unmeasured_or_unowned_source_marks(scene, damage):
    from core.perception.source_references_while_play_advances import retention_for_source_frames

    reference = registered(BASE, scene[3], scene[0].offset)
    observer = scene[-1]
    # This binding's source is intentionally independent of the reference.
    original = attach(scene).binding
    frames = ((reference, original),)
    if damage == 'marks':
        observer.material_marks = None
    elif damage == 'stale':
        observer.pixels_consumed = replace(observer.pixels_consumed, at=3.9)
    elif damage == 'source':
        pass
    elif damage == 'bytes':
        frames = ((replace(reference, source='source'), original),)
        frames = ((replace(frames[0][0], coordinates=replace(reference.coordinates, reference_sha256='f'*64)), original),)
    elif damage == 'item':
        frames = (None,)
    else:
        frames = frames*65
    assert retention_for_source_frames(observer, 4., frames) is None
