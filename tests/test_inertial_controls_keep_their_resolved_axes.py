"""A learned slow acceleration cannot disappear from the planner's motion axes."""
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers, _inertial_axis_resolution_invariant

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('force', [-39., -35., -33., 33., 35., 39.])
def test_two_measured_slow_pulses_preserve_their_resolved_axis(axis, force):
    hers = WhichIsHers()
    hers.kind, hers.carried = 1, True
    for key, began in [('q', 1.), ('', 2.), ('q', 3.), ('', 4.)]:
        for dt in (.25, .35, .45, .55):
            push = [0., 0.]
            push[axis] = force if key else 0.
            hers._hers.add(key, *(10. + value * dt for value in push), press=began, when=began + dt)
    assert hers.push_of('q')[axis] == pytest.approx(force)
    assert hers.keys_known(['q'])
    assert hers.keys_that_move_her(['q'])['q'][axis] == pytest.approx(force * .5)
    assert hers.axes(['q']) == ((True, False) if axis == 0 else (False, True))
    assert not hers.identification.receipts and hers.number is None


@pytest.mark.parametrize('force', [-32., -19., -4., 0., 4., 19., 32.])
def test_subresolution_effects_do_not_supply_an_axis(force):
    hers = WhichIsHers()
    hers.kind, hers.carried = 1, True
    hers.push_of = lambda key: (force, 20.) if key else (0., 20.)
    assert not hers.keys_that_move_her(['q'])
    assert hers.axes(['q']) == (False, False)


def test_missing_neutral_response_cannot_turn_environmental_acceleration_into_a_control():
    hers = WhichIsHers()
    hers.kind, hers.carried = 1, True
    hers.push_of = lambda key: (80., 30.) if key else None
    assert not hers.keys_known(['q'])
    assert not hers.keys_that_move_her(['q']) and hers.axes(['q']) == (None, None)


def test_inertial_axis_resolution_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _inertial_axis_resolution_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'inertial_control_axes_use_displacement_resolution')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
