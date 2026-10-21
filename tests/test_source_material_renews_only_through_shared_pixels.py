"""Gradual point replacement must cross verified old material and current pixels."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.points_that_keep_their_motion import SeenPoint
from core.perception.source_material_renewed_through_shared_marks import renew_source_material
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, FlashVisualAssets
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from tests.test_source_geometry_precedes_object_grouping import scene, attach, moved  # noqa: F401

pytestmark = pytest.mark.unit


def reference(scene):
    return RegisteredVisualReference('source', DeclaredVisualAsset('artwork', 17, 'sprite', 0, 39, (10,)),
                                     scene[0], scene[3])


def advance(scene, *, remove=0, fresh=0):
    observer = scene[-1]
    old = observer.material_marks
    at = old.pixels.at+.1
    picture = np.roll(scene[2], (2*(old.sequence-19), 3*(old.sequence-19)), axis=(0, 1))
    points = [replace(p, x=p.x+3., y=p.y+2., old_x=p.x, old_y=p.y, before=p.at, at=at,
                      pictures=p.pictures+1) for p in old.points[remove:]]
    for index in range(fresh):
        x = 53.+index*4.+3.*(old.sequence-19)
        y = 52.+(index%2)*15.+2.*(old.sequence-19)
        points.append(SeenPoint(1000+old.sequence*10+index, x, y, x-3., y-2., at-.1, at-.1, at, .001, 2))
    receipt = consumed_pixels(picture, at)
    observer.material_marks = replace(old, sequence=old.sequence+1, pixels=receipt, points=tuple(points))
    observer.pixels_consumed = receipt
    return picture


def test_a_rolling_majority_preserves_the_original_source_origin_after_gradual_replacement(scene):
    original = attach(scene).binding
    binding = original
    roots = {(a[0], a[1]) for a in original.anchors}
    for index in range(6):
        picture = advance(scene, remove=2, fresh=3)
        outcome = renew_source_material(reference(scene), binding, scene[-1], picture, scene[-1].pixels_consumed.at)
        assert outcome.binding is not None, outcome
        assert outcome.added <= outcome.retained//2
        binding = outcome.binding
        assert binding.registration_error >= original.registration_error
        assert binding.origin == pytest.approx((52.+3.*(index+1), 38.+2.*(index+1)), abs=.05)
        assert len(binding.renewal_lineage) == index+1
    assert len(roots & {(a[0], a[1]) for a in binding.anchors}) < 3
    assert original.inspect(scene[-1], binding.at, source='source').current is None
    assert binding.inspect(scene[-1], binding.at, source='source').current is not None
    assert not scene[-1].things


@pytest.mark.parametrize('change', ['no_shared', 'wrong_pixels', 'wrong_reference', 'foreign_source',
                                   'wrong_epoch', 'gap', 'bad_patch'])
def test_renewal_cannot_recover_from_lost_or_contradicted_material(scene, change):
    binding, ref = attach(scene).binding, reference(scene)
    picture = advance(scene, remove=2, fresh=3)
    observer, at = scene[-1], scene[-1].pixels_consumed.at
    if change == 'no_shared':
        observer.material_marks = replace(observer.material_marks,
            points=tuple(replace(p, number=p.number+10000) for p in observer.material_marks.points))
    elif change == 'wrong_pixels':
        picture = np.zeros_like(picture)
    elif change == 'wrong_reference':
        ref = replace(ref, pixels=np.zeros_like(ref.pixels))
    elif change == 'foreign_source':
        ref = replace(ref, source='other')
    elif change == 'wrong_epoch':
        observer.material_marks = replace(observer.material_marks, epoch='2'*32)
    elif change == 'gap':
        observer.material_marks = replace(observer.material_marks, sequence=observer.material_marks.sequence+1)
    else:
        picture[:] = 0
        receipt = consumed_pixels(picture, at)
        observer.material_marks = replace(observer.material_marks, pixels=receipt)
        observer.pixels_consumed = receipt
    assert renew_source_material(ref, binding, observer, picture, at).binding is None


def test_repeated_captures_cannot_renew_or_reduce_existing_uncertainty(scene):
    binding = attach(scene).binding
    assert renew_source_material(reference(scene), binding, scene[-1], scene[2], 4.).reason == 'renewal_interval_unavailable'


def test_unchanged_supported_marks_do_not_spend_lineage_capacity(scene):
    binding = attach(scene).binding
    picture = advance(scene)
    result = renew_source_material(reference(scene), binding, scene[-1], picture, 4.1)
    assert result.reason == 'renewal_not_needed' and result.binding is None


def test_renewal_refuses_exhausted_lineage_instead_of_discarding_its_history(scene):
    binding = attach(scene).binding
    binding = replace(binding, sequence=100,
        renewal_lineage=tuple((i+1, 1.+i*.05, '0'*64) for i in range(32)))
    observer = scene[-1]
    observer.material_marks = replace(observer.material_marks, sequence=100)
    picture = np.roll(scene[2], (2, 3), axis=(0, 1))
    marks = observer.material_marks
    receipt = consumed_pixels(picture, 4.1)
    points = tuple(replace(p, x=p.x+3., y=p.y+2., old_x=p.x, old_y=p.y, before=p.at, at=4.1,
                           pictures=p.pictures+1) for p in marks.points)
    observer.material_marks = replace(marks, sequence=101, pixels=receipt, points=points)
    observer.pixels_consumed = receipt
    assert renew_source_material(reference(scene), binding, observer, picture, 4.1).reason == 'renewal_lineage_capacity_unavailable'


def test_current_worker_renewal_reaches_the_live_reference_owner_and_same_whole_identity(scene):
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    from core.perception.play_observation_without_holding_the_loop import _seen
    from core.perception.source_references_while_play_advances import SourceReferencesInPlay
    from core.perception.whole_referents_from_source_material import WholeReferentsInView

    source = 'swf:'+'3'*64
    ref = reference(scene)
    ref = replace(ref, source=source, coordinates=replace(ref.coordinates, source=source))
    binding = replace(attach(scene).binding, source=source)
    table = FlashVisualAssets(source, (ref.declaration,))
    wholes = WholeReferentsInView()
    before, = wholes.observe(scene[-1], scene[2], 4., ((ref, binding, binding.current(scene[-1], 4., source=source)),)).values()
    picture = advance(scene, remove=2, fresh=3)

    class Eyes:
        def __init__(self):
            self.__dict__.update(vars(scene[-1]))

        def see(self, _picture, _at):
            return []

    _, state = _seen(Eyes(), picture, 4.1, reference_frames=((ref, binding),), reference_tables=(table,), whole_referents=wholes)
    assert state['source_material_renewal_readings'][0]['reason'] == 'renewed'
    key, renewed = state['fresh_source_material_frames'][0]
    assert key == ref.variant_key and renewed.renewal_lineage == ((binding.sequence, binding.at, binding.picture_sha256),)
    after, = state['source_referents'].values()
    assert after.number == before.number and after.born == before.born
    assert after.contact_verified is None and after.role is None
    owner = SourceReferencesInPlay()
    owner.tables, owner.catalog, owner.frames = (table,), (ref,), ((ref, binding),)
    owner._loaded, owner._next = True, float('inf')
    token = THE_GUIDE.set(Guide(program_visual_assets=(table,)))
    try:
        owner.advance(SimpleNamespace(**state), picture, 4.1, (table,), now=10.)
        assert owner.frames == ((ref, renewed),)
        assert owner.retention is not None and set(owner.retention.names) == {(a[0], a[1]) for a in renewed.anchors}
    finally:
        owner.retire()
        THE_GUIDE.reset(token)


def test_renewal_work_is_bounded_and_current_source_scoped(scene):
    from core.perception.source_material_renewed_through_shared_marks import renew_current_source_frames

    source = 'swf:'+'3'*64
    ref = reference(scene)
    ref = replace(ref, source=source, coordinates=replace(ref.coordinates, source=source))
    binding = replace(attach(scene).binding, source=source)
    picture = advance(scene, remove=2, fresh=3)
    table = FlashVisualAssets(source, (ref.declaration,))
    frames = ((ref, binding),)*64
    offered, readings = renew_current_source_frames(frames, scene[-1], picture, 4.1, (table,))
    assert len(offered) == len(readings) == 4
    assert renew_current_source_frames(frames, scene[-1], picture, 4.1, ())[0] == ()
    assert renew_current_source_frames(frames+frames, scene[-1], picture, 4.1, (table,))[0] == ()


def test_renewal_contract_has_a_measured_registered_canary():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.source_material_renewed_through_shared_marks import _source_material_renewal_invariant

    assert _source_material_renewal_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'source_material_renewal_cannot_start_without_shared_pixels')
    assert check.predict(None) is True
