"""Moving turns need continuing pictures and input contrasts, rather than a curved trace alone."""
import copy
import math

import pytest

from core.agency.turning_while_travelling import TurningWhileTravelling
from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def _observer():
    return TurningWhileTravelling(moving=20., settle_s=.2, gap_s=.35, turns=.5)


def _interval(turns, key, began, rate, *, measured=True, speed=80., missing=None, gap=False, offset=0.):
    delays = (0., .21, .3, .4, .5, .6) if not gap else (0., .21, .3, .8, .9, 1.)
    for index, after in enumerate(delays):
        angle = offset + rate * after
        turns.saw(key, began, began + after, speed * math.cos(angle), speed * math.sin(angle),
                  measured=measured and index != missing)


def _learn(turns, rates=(0., 1.7, 0., 1.7, 0.), **kwargs):
    for index, (key, rate) in enumerate(zip(('', 'bend', '', 'bend', ''), rates, strict=True)):
        _interval(turns, key, 1.2 * index, rate, offset=2.9 + .3 * index, **kwargs)


def test_two_repeated_curves_contrast_with_two_observed_neutral_intervals():
    turns = _observer()
    for index, (key, rate) in enumerate((('', 0.), ('bend', 1.7), ('', 0.), ('bend', 1.7), ('', 0.))):
        _interval(turns, key, .8 * index, rate, offset=2.9 + .3 * index)
        if index < 4:
            assert not turns.rates()
    assert turns.rates() == {'bend': pytest.approx(1.7)}
    assert len({trial.began for trial in turns.trials if trial.key == 'bend'}) == 2


def test_an_autonomous_curve_is_not_a_control_response():
    turns = _observer()
    _learn(turns, rates=(1.7,) * 5)
    assert not turns.rates()


@pytest.mark.parametrize('bad', ['predicted', 'gap', 'missing', 'stopped'])
def test_missing_or_discontinuous_travel_cannot_supply_a_control_interval(bad):
    turns = _observer()
    arguments = {'measured': False} if bad == 'predicted' else {'gap': True} if bad == 'gap' else {'missing': 3} if bad == 'missing' else {'speed': 0.}
    _learn(turns, **arguments)
    assert not turns.rates()


def test_one_continuing_press_cannot_become_two_independent_intervals():
    turns = _observer()
    _interval(turns, '', 0., 0.)
    _interval(turns, 'bend', .8, 1.7)
    for step in range(15, 30):
        at = .8 + step * .1
        turns.saw('bend', .8, at, 80 * math.cos(1.7 * at), 80 * math.sin(1.7 * at), measured=True)
    _interval(turns, '', 4., 0.)
    assert not turns.rates() and sum(trial.key == 'bend' for trial in turns.trials) <= 1


def test_contradictory_repeated_curves_retire_the_model():
    turns = _observer()
    _learn(turns)
    assert turns.rates()
    _interval(turns, 'bend', 6., 0.)
    _interval(turns, '', 6.8, 0.)
    assert not turns.rates()


@pytest.mark.parametrize('at', [float('nan'), float('inf'), .2, .3])
def test_invalid_or_old_times_cannot_advance_the_interval(at):
    turns = _observer()
    turns.saw('bend', 0., .3, 80., 0., measured=True)
    before = copy.deepcopy(turns)
    turns.saw('else', 0., at, 0., 80., measured=True)
    assert turns == before


def test_a_travel_rate_does_not_predict_a_stationary_turn_and_requests_motion_first():
    faces = WhichWaySheFaces(facing=0.)
    _learn(faces.travel_turns)
    faces.along['drive'] = [100., 100.]
    faces.travel_turns.travelling = False
    faces.held, faces.held_since, faces.now = 'bend', 6., 6.4
    assert faces.turns_her() and faces.facing_now() == 0.
    assert faces.key_toward(0., 0., 0., 100.) == 'drive'
    faces.travel_turns.travelling = True
    assert faces.key_toward(0., 0., 0., 100.) == 'bend'


def test_heading_changes_from_one_unstable_arc_remain_unmodeled():
    turns = _observer()
    for index, key in enumerate(('', 'bend', '', 'bend', '')):
        for after in (0., .21, .3, .4, .5, .6):
            angle = (8 * after ** 2) if key else 0.
            turns.saw(key, index * .8, index * .8 + after,
                      80 * math.cos(angle), 80 * math.sin(angle), measured=True)
    assert not turns.rates()


@pytest.mark.parametrize('samples,span', [(2,.12),(3,.23),(4,.10)])
def test_an_incomplete_arc_cannot_remove_the_driver_and_block_further_calibration(samples,span):
    from core.agency.turning_while_travelling import TravelTurn

    faces = WhichWaySheFaces(facing=0.)
    faces.pressed.update(('drive','bend-a','bend-b'))
    faces.along['drive'] = [40.,42.]
    faces.ways['drive'] = [0.,.01]
    faces.travel_turns.trials.append(TravelTurn('drive',1.,1.6,.75,samples=samples,span=span))
    assert not faces.travel_turns.curved('drive')
    assert faces.ahead_key() == 'drive' and not faces.turning_keys()
    assert faces.trial(['drive','bend-a','bend-b']) == 'drive'
    assert faces.trial(['drive','bend-a','bend-b']) == 'bend-a'


def test_one_complete_curve_still_requests_confirmation_without_lending_a_turn_rate():
    from core.agency.turning_while_travelling import TravelTurn

    turns = _observer()
    turns.trials.append(TravelTurn('bend',1.,1.6,.75,samples=4,span=.2))
    assert turns.curved('bend') and turns.repeats('bend') == 1
    assert not turns.rates()


def test_the_travel_turn_boundary_has_a_registered_measured_claim():
    from core.agency.turning_while_travelling import _travel_turns_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _travel_turns_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'travel_turns_require_repeated_observed_input_contrasts')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
