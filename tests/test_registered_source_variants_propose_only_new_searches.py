"""A changed source frame can be searched without pretending it was observed."""
from dataclasses import replace

import numpy as np
import pytest

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance, SourceFramePoseHint
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.reference_material_near_a_measured_pose import near_reference_appearances
from core.perception.search_hints_for_source_frame_variants import hint_for_source_variant, with_source_variant_hints
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from tests.test_source_geometry_precedes_object_grouping import scene  # noqa: F401

pytestmark = pytest.mark.unit


def variants(scene):
    coordinates, pose, picture, rgba, _observer = scene
    asset = DeclaredVisualAsset('artwork', 17, 'sprite', 0, 39, (10,))
    donor = RegisteredVisualReference('source', asset, coordinates, rgba, 1, 2)
    other = rgba.copy()
    other[24:40, 24:40, :3] = (230, 40, 150)
    target = RegisteredVisualReference('source', asset, replace(coordinates,
        reference_sha256=_digest(other), offset=(4., 14.)), other, 2, 2)
    proposal = SourceReferenceProposal(donor, consumed_pixels(picture, 4.), (pose,), 1)
    return donor, target, proposal


def test_source_coordinates_transfer_the_hint_but_require_new_current_patch_evidence(scene):
    donor, target, proposal = variants(scene)
    hint = hint_for_source_variant(donor, target, proposal.appearances[0])
    assert isinstance(hint, SourceFramePoseHint) and material_samples(hint) is None
    assert hint.project(target.coordinates.offset) == pytest.approx(proposal.appearances[0].project(donor.coordinates.offset))
    picture = np.zeros_like(scene[2])
    picture[26:90, 51:115] = target.pixels[..., :3]
    found = near_reference_appearances(picture, target.pixels, (hint,))
    assert len(found) == 1 and isinstance(found[0], ReferenceAppearance) and material_samples(found[0]) is not None
    assert target.coordinates.project((0., 0.), found[0], picture, target.pixels, source='source') == pytest.approx((55., 40.), abs=.1)
    assert target.coordinates.project((0., 0.), hint, picture, target.pixels, source='source') is None
    assert near_reference_appearances(np.zeros_like(picture), target.pixels, (hint,)) == ()


@pytest.mark.parametrize('change', ['source', 'asset', 'bytes', 'coordinates', 'unknown_correspondence'])
def test_unowned_or_unmeasured_material_cannot_seed_another_variant(scene, change):
    donor, target, proposal = variants(scene)
    pose = proposal.appearances[0]
    if change == 'source':
        target = replace(target, source='other')
    elif change == 'asset':
        target = replace(target, declaration=replace(target.declaration, character_id=99))
    elif change == 'bytes':
        target = replace(target, pixels=np.zeros_like(target.pixels))
    elif change == 'coordinates':
        target = replace(target, coordinates=replace(target.coordinates, source='other'))
    else:
        pose = replace(pose, source_points=(), observed_points=())
    assert hint_for_source_variant(donor, target, pose) is None


def test_hint_generation_keeps_real_observations_and_never_chains_an_unmeasured_hint(scene):
    donor, target, original = variants(scene)
    result = with_source_variant_hints((original,), (donor, target))
    assert result[0] is original and len(result) == 2
    derived = result[1]
    assert derived.reference is target and isinstance(derived.appearances[0], SourceFramePoseHint)
    assert derived.capture == original.capture and derived.revision == original.revision
    assert with_source_variant_hints((derived,), (donor, target)) == ()
    assert with_source_variant_hints(result, (donor, target)) == result


def test_a_foreign_family_or_capture_hint_cannot_enter_a_reference_proposal(scene):
    donor, target, original = variants(scene)
    hint = hint_for_source_variant(donor, target, original.appearances[0])
    with pytest.raises(ValueError):
        SourceReferenceProposal(replace(target, declaration=replace(target.declaration, character_id=99)), original.capture, (hint,), 1)
    with pytest.raises(ValueError):
        SourceReferenceProposal(target, consumed_pixels(np.zeros_like(scene[2]), 4.), (hint,), 1)


def test_affine_frame_registration_uses_both_source_coordinate_maps(scene):
    donor, target, original = variants(scene)
    donor = replace(donor, coordinates=replace(donor.coordinates, linear=((1.1, .2), (.1, .8))))
    target = replace(target, coordinates=replace(target.coordinates, linear=((.8, -.1), (.2, 1.2))))
    hint = hint_for_source_variant(donor, target, original.appearances[0])
    for point in ((0., 0.), (8., 11.), (-4., 29.)):
        a = np.asarray(donor.coordinates.linear)@point+donor.coordinates.offset
        b = np.asarray(target.coordinates.linear)@point+target.coordinates.offset
        assert hint.project(b) == pytest.approx(original.appearances[0].project(a))


def test_unmeasured_searches_cannot_take_more_than_their_admitted_share(scene):
    from core.perception.source_geometry_in_current_pixels import _reference_search_share, _reference_to_measure

    donor, target, original = variants(scene)
    hint = hint_for_source_variant(donor, target, original.appearances[0])
    records = {donor.variant_key: (donor, original.appearances, 1), target.variant_key: (target, (hint,), 1)}
    choices = []
    for cursor in range(16):
        assert _reference_search_share(records, cursor) == ((donor.variant_key,), cursor)
        keys, turn = _reference_search_share(records, cursor, frozenset({donor.key[:2]}))
        choices.append(_reference_to_measure(keys, set(), turn))
    assert choices.count(donor.variant_key) == 12 and choices.count(target.variant_key) == 4
    records[target.variant_key] = target, original.appearances, 1  # Scheduling checks pose type only.
    assert _reference_search_share(records, 7) == (tuple(records), 7)


def test_failed_search_positions_are_required_before_a_variant_can_spend_a_slot(scene, monkeypatch):
    from core.perception import source_geometry_in_current_pixels as module
    from core.perception.visual_assets_in_a_flash_file import FlashVisualAssets

    donor, target, original = variants(scene)
    source = 'swf:'+'4'*64
    donor = replace(donor, source=source, coordinates=replace(donor.coordinates, source=source))
    target = replace(target, source=source, coordinates=replace(target.coordinates, source=source))
    original = replace(original, reference=donor)
    proposals = with_source_variant_hints((original,), (donor, target))
    table = FlashVisualAssets(source, (donor.declaration,))
    attempted = []

    def absent(_picture, _reference, poses, *, position):
        attempted.append((isinstance(poses[0], SourceFramePoseHint), position))
        return ()

    monkeypatch.setattr(module, 'near_reference_appearances', absent)
    worker = module.SourceGeometryInCurrentPixels()
    for i in range(8):
        worker._fits(scene[2], 4.+i*.1, proposals, (table,), (), one_position=True)
    assert attempted[:6] == [(False, i) for i in range(6)]
    assert attempted[7][0] is True
    assert not any(is_hint for is_hint, _position in attempted[:7])


def test_generation_is_bounded_and_an_actual_target_match_keeps_its_authority(scene):
    donor, target, original = variants(scene)
    hinted = with_source_variant_hints((original,), (donor, target))[1]
    picture = np.zeros_like(scene[2])
    picture[26:90, 51:115] = target.pixels[..., :3]
    found = near_reference_appearances(picture, target.pixels, hinted.appearances)
    assert len(found) == 1
    actual = SourceReferenceProposal(target, consumed_pixels(picture, 4.1), found, 7)
    result = with_source_variant_hints((original, actual, hinted), (donor, target))
    assert result == (original, actual)
    assert with_source_variant_hints((original,), (target,)*33) == (original,)


def test_variant_hints_have_a_registered_boundary_canary():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.search_hints_for_source_frame_variants import _source_variant_hint_invariant

    assert _source_variant_hint_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'source_frame_hints_cannot_be_material_measurements')
    assert check.predict(None) is True


def test_a_hint_owns_its_numeric_input_without_mutable_aliases():
    linear, offset = np.eye(2), np.array((3., 4.))
    hint = SourceFramePoseHint('0'*64, '1'*64, linear, offset, ('source', 7, '2'*64, 1))
    linear[0, 0], offset[0] = 90., 999.
    assert hint.project((5., 6.)) == (8., 10.)


def test_ordinary_reference_owner_supplies_sibling_hints_to_the_next_observation(scene):
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception.source_references_while_play_advances import SourceReferencesInPlay
    from core.perception.visual_assets_in_a_flash_file import FlashVisualAssets

    donor, target, proposal = variants(scene)
    source = 'swf:'+'4'*64
    donor = replace(donor, source=source, coordinates=replace(donor.coordinates, source=source))
    target = replace(target, source=source, coordinates=replace(target.coordinates, source=source))
    proposal = replace(proposal, reference=donor)
    table = FlashVisualAssets(source, (donor.declaration,))
    owner = SourceReferencesInPlay()
    owner.tables, owner.catalog, owner.proposals = (table,), (donor, target), (proposal,)
    owner._loaded, owner._next = True, float('inf')
    token = THE_GUIDE.set(Guide(program_visual_assets=(table,)))
    try:
        owner.advance(scene[-1], scene[2], 4., (table,), now=10.)
        assert len(owner.proposals) == 2
        assert isinstance(owner.proposals[1].appearances[0], SourceFramePoseHint)
        assert owner.frames == owner.points == ()
        owner.advance(scene[-1], scene[2], 4.1, (), now=10.1)
        assert not owner.proposals
    finally:
        owner.retire()
        THE_GUIDE.reset(token)
