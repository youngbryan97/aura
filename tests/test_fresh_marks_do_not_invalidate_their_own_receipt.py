"""The provider and retained pose share a budget that keeps the observed core decisive."""
import pytest

from core.perception.continuing_material_before_outlines import _continuing_marks
from core.perception.points_that_keep_their_motion import FollowingPoints, SeenPoint
from core.perception.position_from_the_same_points import new_mark_capacity
from tests.test_continuing_material_survives_a_changed_outline import scene

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('fresh_count', [0, 1, 4, 12, 40])
def test_extra_coherent_points_cannot_make_an_existing_material_receipt_unmatchable(fresh_count):
    eyes, body, blob, _ = scene()
    prior = body.point_pose
    retained = [SeenPoint(n, x+3., y, x, y, .5, 1., 1.1, .01, 8)
                for n, (x, y) in prior.visible_marks.items()]
    # Fresh marks share this frame's displacement but lie in an attached part
    # that is absent from the current body's proposal.
    extra = [SeenPoint(100+n, 18.+(n % 4)*2., 5.+(n//4), 15.+(n % 4)*2., 5.+(n//4),
                       1., 1., 1.1, .01, 2) for n in range(fresh_count)]
    points = FollowingPoints()
    points.points = tuple(retained+extra)
    motion = points.motion_of(body, 1.1)
    assert motion is not None and motion.support <= len(retained)+new_mark_capacity(len(retained))
    assert set(prior.anchors) <= set(motion.points)
    assert _continuing_marks(body, motion, 1.1) is not None
    eyes._point_motion = {body.number: motion}
    assert motion.inside(blob) and eyes._match([blob], 1.1) == ({body.number: 0}, [])
    eyes._moved(body, blob, 1.1)
    assert body.motion_position == pytest.approx((23., 20.))
    assert body.point_pose.reference_at == .8 and body.born == .5


def test_the_new_mark_budget_cannot_displace_the_old_consensus():
    for retained in range(3, 257):
        count = new_mark_capacity(retained)
        assert retained+count <= 256 and retained/(retained+count) >= 2./3.


def test_the_admission_contract_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.position_from_the_same_points import _fresh_mark_capacity_invariant

    assert _fresh_mark_capacity_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'fresh_marks_cannot_outvote_retained_material')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
