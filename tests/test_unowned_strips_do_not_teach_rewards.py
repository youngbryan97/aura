"""Measured strips before interaction cannot train a reward or resource model."""
from types import SimpleNamespace

import pytest

from core.agency.playing_as_it_happens import _Run
from core.agency.reading_while_playing import _read_the_bars, _unowned_bar_reward_invariant
from core.agency.what_meeting_things_does import WhatMeetingDoes
from tests.test_a_bar_that_empties_is_what_she_has_left import _picture

pytestmark = pytest.mark.unit


def _observed(*, body=False, pointer=False, response=False, named=False):
    run = _Run(keys=['left'], began=0., last_moving=0.)
    run.regions_read = [{'text': 'HEALTH', 'center_x': .03, 'center_y': .07}] if named else []
    run.presses = SimpleNamespace(moves_anything=lambda at: response)
    hers = SimpleNamespace(kind=1 if body else None, follows_pointer=pointer, thing=lambda moves: None)
    moves = SimpleNamespace(shape=(240, 320), things={}, share=lambda x, y: (x/320, y/240))
    meeting, at = WhatMeetingDoes(), 0.
    for level in [116, 116, 100, 100, 70, 70, 40, 40, 20, 20, 40, 40, 60, 60]:
        for _ in range(6):
            at += .1
            _read_the_bars(run, meeting, hers, moves, _picture(level, 10), at, None)
    return run, meeting


def test_unnamed_changes_are_retained_without_granting_reward_or_scarcity():
    run, meeting = _observed()
    assert run.bars.bars() and any(bar.full < .6 for bar in run.bars.bars())
    assert run.gains == run.losses == 0 and not meeting.verdicts
    assert run.vitals == meeting.vitals == 1.


@pytest.mark.parametrize('reference', ['body', 'pointer', 'response'])
def test_an_interactive_reference_cannot_supply_an_unknown_meter_role(reference):
    run, meeting = _observed(**{reference: True})
    assert run.bars.bars() and any(bar.full < .6 for bar in run.bars.bars())
    assert run.gains == run.losses == 0 and not meeting.verdicts
    assert run.vitals == meeting.vitals == 1.


@pytest.mark.parametrize('label,meaning', [('HEALTH', 'down is bad'), ('YOU', 'down is bad'),
    ('ENEMY', 'down is good'), ('SCORE', 'up is good'), ('TIME', 'neither'),
    ('PLAY', 'neither'), ('cloud', ''), ('', '')])
def test_a_strip_needs_semantics_beyond_a_legible_nearby_word(label, meaning):
    from core.agency.what_she_has_left import what_a_bar_measures
    from core.perception.how_full_a_bar_is import Bar

    bar = Bar(row=10, thick=6, colour=(200, 0, 0), start=40, end=130)
    regions = [{'text': label, 'center_x': .06, 'center_y': .05}]
    assert what_a_bar_measures(bar, regions, 320, 200)[1] == meaning


def test_unknown_meter_roles_have_a_registered_measured_claim():
    from core.agency.what_she_has_left import _bar_role_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _bar_role_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'measured_strips_require_a_known_role')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_named_health_remains_available_before_body_discovery():
    run, meeting = _observed(named=True)
    assert run.gains > 0 and run.losses > 0 and meeting.verdicts
    assert run.vitals < .6


def test_unowned_meter_feedback_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _unowned_bar_reward_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'unowned_strips_cannot_teach_reward')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('side', [None, .1])
def test_a_single_unlabelled_number_remains_measured_without_reward(side):
    from core.agency.what_meeting_things_does import Readouts

    counters, verdicts = Readouts(), []
    for at, value in enumerate([2, 2, 7, 7, 3, 3]):
        verdicts += counters.read([{'text': str(value), 'center_x': .1, 'center_y': .1}], float(at), side)
    assert not verdicts and list(counters.current.values()) == [3]


def test_named_and_opposed_counters_keep_their_feedback():
    from core.agency.what_meeting_things_does import Readouts

    counters, verdicts = Readouts(), []
    for at, value in enumerate([2, 2, 7, 7]):
        verdicts += counters.read([{'text': 'Score ' + str(value), 'center_x': .1, 'center_y': .1}], float(at), None)
    assert len(verdicts) == 1 and verdicts[0]['what'] == 'gain'
    opposed, verdicts = Readouts(), []
    for at, value in enumerate([2, 2, 7, 7]):
        verdicts += opposed.read([{'text': str(value), 'center_x': x, 'center_y': .1} for x in [.1, .9]], float(at), .1)
    assert sorted(v['what'] for v in verdicts) == ['gain', 'loss']
