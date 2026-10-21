"""Clipping at the captured boundary is distinct from missing material within it."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.agency.a_requested_relation_in_play import position_for_relation
from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.play_observation_without_holding_the_loop import _seen
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.source_geometry_in_current_pixels import SourceGeometryInCurrentPixels
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.visual_assets_in_a_flash_file import FlashVisualAssets
from core.perception.what_moves_in_the_picture import WhatMoves
from core.perception.whole_referents_from_source_material import WholeReferentsInView
from tests.test_pixel_search_hints_reach_owned_sampling import captured
from tests.test_source_geometry_precedes_object_grouping import scene  # noqa: F401
from tests.test_whole_referents_require_current_source_material import registered

pytestmark = pytest.mark.unit


def partial(scene, crop):
    reference = registered(scene)
    source = 'swf:'+'6'*64
    reference = replace(reference, source=source, coordinates=replace(reference.coordinates, source=source))
    top, left, bottom, right = crop
    picture = scene[2][top:bottom, left:right].copy()
    # This is a search proposal. The current refitter still has to find six
    # bidirectional pixel correspondences before the observer can use it.
    old = scene[1]
    pose = replace(old, offset=(old.offset[0]-left, old.offset[1]-top),
                   picture_sha256=_digest(picture))
    proposal = SourceReferenceProposal(reference, consumed_pixels(picture, 1.), (pose,), 1)
    table = FlashVisualAssets(reference.source, (reference.declaration,))
    worker, eyes, registry = SourceGeometryInCurrentPixels(), WhatMoves(), WholeReferentsInView()
    worker._essential_seconds = .3
    _, state = _seen(eyes, picture, captured(2.), fresh_references=worker,
        reference_proposals=(proposal,), reference_tables=(table,), whole_referents=registry)
    return picture, eyes, worker, registry, proposal, table, state


@pytest.mark.parametrize('crop,clipped', [
    ((0, 0, 80, 160), (False, False, False, True)),
    ((0, 0, 128, 90), (False, False, True, False)),
    ((45, 0, 128, 160), (False, True, False, False)),
    ((0, 55, 128, 160), (True, False, False, False)),
])
def test_source_assembly_keeps_only_its_current_visible_extent(scene, crop, clipped):
    picture, eyes, worker, registry, _, _, state = partial(scene, crop)
    assert worker.material_frames
    assert state['source_referents'], registry.readings
    item, = state['source_referents'].values()
    assert item.clipped_edges == clipped and .5 < item.visible_fraction < 1.
    left, top, right, bottom = item.box()
    assert -.51 <= left < right <= picture.shape[1]-.49
    assert -.51 <= top < bottom <= picture.shape[0]-.49
    assert item.role is None and item.contact_verified is None and not eyes.things
    assert registry.readings[0]['reason'] == 'qualified_visible_part'


def test_hidden_material_inside_a_clipped_capture_still_refuses_an_assembly(scene):
    picture, eyes, worker, registry, proposal, table, state = partial(scene, (0, 0, 128, 90))
    assert state['source_referents']
    bindings = worker.material_frames
    changed = picture.copy()
    changed[30:94, -1] = 0
    worker._essential_seconds = .3
    _, state = _seen(eyes, changed, captured(2.1, sequence=5), fresh_references=worker,
        reference_proposals=(proposal,), reference_tables=(table,), whole_referents=registry,
        reference_frames=bindings)
    assert not state['source_referents']
    assert any(reading['reason'] == 'whole_edge_support_unresolved' for reading in registry.readings)


def test_a_clipped_image_boundary_cannot_be_used_as_the_missing_object_edge(scene):
    _, _, _, _, _, _, state = partial(scene, (0, 0, 80, 160))
    target, = state['source_referents'].values()
    mine = SimpleNamespace(x=20., y=20., w=8., h=8.)
    assert position_for_relation('on', mine, target) is not None
    assert position_for_relation('below', mine, target) is None
    assert position_for_relation('inside', mine, target) is None
    _, _, _, _, _, _, state = partial(scene, (45, 0, 128, 160))
    target, = state['source_referents'].values()
    assert position_for_relation('on', mine, target) is None
    assert position_for_relation('above', mine, target) is None


def test_clipped_relation_regions_cannot_bypass_the_destination_guard(scene):
    from core.agency.evidence_for_a_requested_relation import observed_relation_region

    _, _, _, _, _, _, state = partial(scene, (45, 0, 128, 160))
    target, = state['source_referents'].values()
    mine = SimpleNamespace(number=1, born=1., kind=1, seen=2., x=20., y=20., w=8., h=8.)
    assert observed_relation_region('on', mine, target, 2.) is None


def test_clipped_source_claim_is_registered_and_executable():
    from core.agency.a_requested_relation_in_play import _clipped_relation_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _clipped_relation_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    test = next(item for item in suite.tests() if item.name == 'clipped_source_observations_keep_unseen_edges_unknown')
    assert test.predict(None) is True and any(claim.test == test.name for claim in suite.claims())
