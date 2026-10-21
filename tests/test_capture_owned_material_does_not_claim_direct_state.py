"""A visual capture identifies its pixels; it does not certify measured motion."""
from types import SimpleNamespace

import pytest

from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.source_material_groups_for_visual_attention import SourceMaterialGroupsInView
from core.perception.translation_from_visible_support import motion_status, motion_was_measured, told_directly
from tests.test_source_geometry_precedes_object_grouping import scene, attach  # noqa: F401
from tests.test_source_material_renews_only_through_shared_pixels import reference

pytestmark = pytest.mark.unit


def test_an_ordinary_partial_material_question_is_not_direct_state_or_measured_motion(scene):
    binding, ref, observer = attach(scene).binding, reference(scene), scene[-1]
    point = binding.current(observer, 4., source=ref.source)
    group, = SourceMaterialGroupsInView().observe(observer, scene[2], 4., ((ref, binding, point),)).values()
    assert group.pixels == observer.pixels_consumed and not group.geometry_qualified
    assert not told_directly(group)
    assert motion_status(group, 4.) == 'unavailable' and not motion_was_measured(group, 4.)


@pytest.mark.parametrize('evidence', [None, 'measured'])
def test_current_pixels_without_a_motion_timestamp_cannot_receive_motion_credit(evidence):
    view = SimpleNamespace(seen=4., pixels=PixelsConsumed(4., (32, 32, 3), '0'*64))
    if evidence is not None:
        view.motion_evidence = evidence
    assert not told_directly(view) and not motion_was_measured(view, 4.)
    assert motion_status(view, 4.) == 'unavailable'


@pytest.mark.parametrize('capture_at,motion_at,expected', [(4., 4., True), (4., 3.9, False),
                                                        (3.9, 4., False), (4., None, False)])
def test_tracked_visual_motion_keeps_its_actual_capture_and_measurement_time(capture_at, motion_at, expected):
    view = SimpleNamespace(seen=4., pixels=PixelsConsumed(capture_at, (32, 32, 3), '0'*64),
                           motion_evidence='measured', motion_measured_at=motion_at)
    assert motion_was_measured(view, 4.) is expected
    assert motion_status(view, 4.) == ('measured' if expected else 'unavailable')


def test_declared_direct_state_and_existing_exact_state_ports_keep_their_authority():
    legacy = SimpleNamespace(seen=4.)
    declared = SimpleNamespace(seen=4., pixels=PixelsConsumed(3.9, (32, 32, 3), '0'*64), motion_evidence='direct')
    for state in (legacy, declared):
        assert told_directly(state) and motion_was_measured(state, 4.)
        assert not motion_was_measured(state, 4.1)


def test_visual_motion_authority_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.translation_from_visible_support import _captured_motion_authority_invariant

    assert _captured_motion_authority_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'visual_capture_cannot_default_to_measured_motion')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
