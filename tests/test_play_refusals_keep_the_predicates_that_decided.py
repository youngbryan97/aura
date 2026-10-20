"""Diagnosis observes the actual branches without rerunning a selector."""
from types import SimpleNamespace

import pytest

from core.agency.a_requested_relation_in_play import SpatialRequest, bind_destination, read_destination
from core.agency.why_a_motion_history_is_called_carried import (
    _motion_family_reading_invariant, read_motion_family,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('statistic,effect,pace,stops,reason', [
    (0., 40., 100., False, 'push_statistic_unresolved'),
    (30., 10., 100., False, 'push_effect_unresolved'),
    (30., 40., 50., False, 'pace_statistic_wins'),
    (30., 40., 20., True, 'stops_on_release'),
    (30., 40., 20., False, 'push_statistic_wins'),
    (20., 20., 20., False, 'pace_statistic_wins'),
    (20., 20., 19., False, 'push_statistic_wins'),
])
def test_motion_readings_preserve_the_existing_ordered_classifier(statistic, effect, pace, stops, reason):
    calls = []
    def measured(name, value):
        calls.append(name)
        return value
    speeds = SimpleNamespace(by_push=lambda: measured('push', (statistic, effect)),
                             _by_pace_alone=lambda: measured('pace', pace),
                             _stops_when_let_go=lambda: measured('release', stops))
    reading = read_motion_family(speeds, statistic_over=20., effect_over=20.)
    assert reading.accelerated == (statistic >= 20. and effect >= 20. and statistic > pace and not stops)
    assert reading.reason == reason
    assert calls == (['push'] if statistic < 20. or effect < 20. else
                     ['push', 'pace'] if statistic <= pace else ['push', 'pace', 'release'])
    assert (reading.pace_statistic is None) == ('pace' not in calls)
    assert (reading.stops_on_release is None) == ('release' not in calls)


def _body(number, *, kind=1, width=20., seen=5.):
    return SimpleNamespace(number=number, kind=kind, x=50., y=60., w=width, h=10., seen=seen)


@pytest.mark.parametrize('case,status', [
    ('no_name', 'no_named_referent'), ('partial', 'incomplete_name'),
    ('duplicate', 'ambiguous_named_referent'), ('stale', 'referent_not_current'),
    ('narrow', 'relation_geometry_unavailable'), ('bound', 'bound'),
])
def test_destination_diagnostics_and_the_existing_api_share_one_result(case, status):
    body, target = _body(1), _body(2, kind=2)
    things, names = [body, target], {2: 'blue platform'}
    if case == 'no_name':
        names = {2: 'base'}
    elif case == 'partial':
        names = {2: 'platform'}
    elif case == 'duplicate':
        things.append(_body(3, kind=2))
    elif case == 'stale':
        target.seen = 4.
    elif case == 'narrow':
        target.w = 10.
    request = SpatialRequest('on', 'blue platform', 'instruction')
    reading = read_destination(request, body, things, names, 5.)
    assert reading.status == status
    assert bind_destination(request, body, things, names, 5.) == reading.binding
    assert (reading.binding is not None) == (case == 'bound')
    assert reading.named and len(reading.candidates) == reading.candidate_count


def test_a_truncated_diagnostic_cannot_remove_a_competing_referent():
    body = _body(1)
    rivals = [_body(number, kind=2) for number in range(2, 82)]
    reading = read_destination(SpatialRequest('on', 'platform', ''), body, rivals, {2: 'platform'}, 5.)
    assert reading.binding is None and reading.status == 'ambiguous_named_referent'
    assert len(reading.named) == len(reading.candidates) == 32
    assert reading.named_count == reading.candidate_count == 80


def test_predicate_reading_has_an_executable_invariant():
    assert _motion_family_reading_invariant() == ()
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    test = next(t for t in suite.tests() if t.name == 'play_refusals_keep_the_predicates_that_decided')
    assert test.predict(None) is True and any(c.test == test.name for c in suite.claims())


@pytest.mark.asyncio
async def test_recorded_model_and_refusal_are_the_original_decisions(tmp_path, monkeypatch):
    import json
    from core.agency.which_one_answers_to_her import WhichIsHers, _Speeds
    from core.perception.recording_the_frames_and_inputs import (
        record_control_observation, record_frames_and_inputs, record_play_choice,
    )

    speeds = _Speeds()
    speeds.by_push = lambda: (30., 40.)
    speeds._by_pace_alone = lambda: 50.
    assert not speeds.carried()
    hers = WhichIsHers()
    hers.motion_family_reading = (4., 1, speeds.last_family_reading)
    run = SimpleNamespace(held_trying=False, held='', held_since=None, keys=[])
    body = _body(1)
    body.born, body.moved, body.vx, body.vy = 0., True, 1., 2.
    request = SpatialRequest('on', 'platform', 'instruction')
    reading = read_destination(request, body, [_body(2)], {1: 'base'}, 5.)
    choosing = SimpleNamespace(mine=body, last_target=((50., 60.), 'wait', None), ways={}, rest=(0., 0.),
        response_s=.1, next_picture_s=.1, meeting=SimpleNamespace(spatial_request=request),
        goal_reading=reading, goal_binding=None, target=lambda: pytest.fail('selector rerun'))
    directory = tmp_path / 'recording'
    async with record_frames_and_inputs(directory) as recorder:
        record_control_observation(run, SimpleNamespace(things={}), hers, 5.)
        record_play_choice(choosing, '', 'wait', None, 5.)
        await recorder.queue.join()
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    observation, choice = events
    assert observation['motion_family']['at'] == 4.
    assert observation['motion_family']['reason'] == 'pace_statistic_wins'
    assert observation['motion_family']['stops_on_release'] is None
    assert choice['requested']['binding_status'] == 'no_named_referent'
    assert choice['requested']['named_referents'] == [[2, 'base']]
