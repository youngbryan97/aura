"""The ordinary binder's region guides motion without certifying contact."""
from dataclasses import replace
import math
import pytest

from core.agency.a_requested_relation_in_play import take_motion_directions
from core.agency.choosing_where_to_be import _Choosing
from core.agency.imagining_ahead import Plan, _run, _the_scene, best_plan
from core.agency.a_requested_region_in_a_plan import _planned_relation_invariant, current_requested_region
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from tests.test_requested_relations_reach_the_actual_target import _scene

pytestmark = pytest.mark.unit


def _choosing(relation, *, position=(90., 150.), carried=False):
    body, target, _, moves, hers, meeting, guide = _scene({1: 'platform'})
    body.x, body.y = position
    target.x, target.y, target.w, target.h = 140., 150., 100., 80.
    moves.scale = 1.
    pushes = {'': (0., 0.), 'left': (-80., 0.), 'right': (80., 0.),
              'up': (0., -80.), 'down': (0., 80.)}
    hers.way_of = {'left': (-60., 0.), 'right': (60., 0.), 'up': (0., -60.), 'down': (0., 60.)}.get
    if carried:
        hers.carried = True
        hers.push_of = pushes.get
    take_motion_directions(meeting, f'Move {relation} the platform.')
    token = THE_GUIDE.set(guide)
    try:
        choosing = _Choosing(moves, hers, meeting, list(pushes)[1:], next_picture_s=.1)
        choosing.at = body.seen
        goal, why, aim = choosing.target()
        assert why == 'meet' and aim is target
    finally:
        THE_GUIDE.reset(token)
    return choosing, goal


@pytest.mark.parametrize('relation,position', [('inside', (90., 150.)), ('on', (140., 105.))])
def test_partial_overlap_with_a_requested_referent_earns_no_predicted_contact(relation, position):
    choosing, goal = _choosing(relation, position=position)
    scene = _the_scene(choosing, 1.2)
    result = _run(scene, Plan(((0., ''),)), goal, '', 1.)
    assert result.gets == 0
    assert result.costs == 0


def test_an_already_satisfied_containment_region_does_not_pull_to_the_objects_centre():
    choosing, goal = _choosing('inside', position=(100., 150.))
    result = best_plan(choosing, '', goal)
    assert result is not None and result.decides
    assert result.plan.first == ''
    assert result.ends_at == (100., 150.)
    assert result.gets == 0


def test_a_still_body_above_the_requested_support_has_a_region_driven_approach():
    choosing, goal = _choosing('on', position=(140., 90.))
    result = best_plan(choosing, '', goal)
    assert result is not None and result.decides
    assert result.plan.first == 'down'
    assert result.gets == 0


def test_other_earned_contacts_keep_their_ordinary_reward():
    choosing, goal = _choosing('inside', position=(90., 150.))
    choosing.meeting.spatial_request = None
    choosing.goal_binding = None
    choosing.meeting.stance = lambda *_args, **_kwargs: 'meet'
    choosing.meeting.known = lambda _kind: True
    scene = _the_scene(choosing, 1.2)
    assert _run(scene, Plan(((0., ''),)), goal, '', 1.).gets == 1


@pytest.mark.parametrize('change', ['body_birth', 'body_seen', 'target_birth', 'target_seen',
    'target_gone', 'target_extent', 'bounds', 'basis', 'epoch', 'request', 'goal', 'stance'])
def test_stale_or_differently_owned_regions_cannot_reach_the_plan(change):
    choosing, goal = _choosing('inside')
    binding = choosing.goal_binding
    if change == 'body_birth':
        choosing.mine.born += .1
    elif change == 'body_seen':
        choosing.mine.seen -= .1
    elif change == 'target_birth':
        binding.thing.born += .1
    elif change == 'target_seen':
        binding.thing.seen -= .1
    elif change == 'target_gone':
        choosing.moves.things.pop(binding.thing.number)
    elif change == 'target_extent':
        binding.thing.w += 1.
    elif change == 'bounds':
        choosing.goal_binding = replace(binding, region=replace(binding.region, bounds=(0., 0., 200., 200.)))
    elif change == 'basis':
        choosing.goal_binding = replace(binding, region=replace(binding.region, basis=('another-observer',)))
    elif change == 'epoch':
        choosing.moves._instance_name_epoch = 'another-epoch'
    elif change == 'request':
        take_motion_directions(choosing.meeting, 'Move on the platform.')
    elif change == 'goal':
        goal = goal[0]+1., goal[1]
    else:
        choosing.goal_stance = 'avoid'
    assert current_requested_region(choosing, goal) is None
    assert _the_scene(choosing, 1.2, goal=goal).relation is None


def test_optional_fields_cannot_promote_box_geometry_into_physical_acceptance():
    choosing, goal = _choosing('inside')
    binding = choosing.goal_binding
    choosing.goal_binding = replace(binding, region=replace(binding.region, contact_verified=True,
                                                           maximum_speed=999., maximum_angle=90., unknowns=()))
    region = current_requested_region(choosing, goal)
    assert region.contact_verified is None and region.maximum_speed is None and region.maximum_angle is None
    assert 'observed_contact' in region.unknowns


def test_settled_geometry_is_not_an_observed_contact_or_success():
    choosing, goal = _choosing('inside', position=(100., 150.))
    result = best_plan(choosing, '', goal)
    forecast = result.relation
    assert forecast.status == 'predicted_geometry' and forecast.terminal_error == 0.
    assert forecast.path_error == 0. and forecast.terminal_relative_speed == 0.
    assert forecast.observed_contact is None and forecast.observed_success is None and result.gets == 0


def test_the_region_moves_on_the_world_forecast_and_delivery_clock():
    choosing, goal = _choosing('inside')
    target = choosing.goal_binding.thing
    target.vx, target.vy, target.moved = 10., -3., True
    choosing.response_s = .2
    scene = _the_scene(choosing, 1.2, goal=goal)
    assert scene.relation_velocity == pytest.approx((10., -3.))
    first = scene.relation_bounds[1, 0]
    delta = .2+scene.times[0]
    assert first == pytest.approx(tuple(v+d for v, d in zip(choosing.goal_binding.region.bounds,
                                                        (10.*delta, -3.*delta, 10.*delta, -3.*delta), strict=True)))


def test_a_known_harmful_contact_is_not_removed_by_the_relation_objective():
    choosing, goal = _choosing('inside', position=(100., 150.))
    target = choosing.goal_binding.thing
    choosing.meeting.told[target.kind] = 'avoid'
    choosing.goal_stance = 'avoid'
    scene = _the_scene(choosing, 1.2, goal=goal)
    result = _run(scene, Plan(((0., ''),)), goal, '', 1.)
    assert result.costs == 1 and result.gets == 0


def test_a_region_with_no_future_target_support_is_not_extrapolated():
    choosing, goal = _choosing('inside')
    choosing.goal_binding.thing.where_at = lambda after: None if after > .2 else (140., 150.)
    scene = _the_scene(choosing, 1.2, goal=goal)
    assert scene.relation is None


def test_the_new_geometric_contract_is_registered_with_its_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _planned_relation_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    for name in ('requested_regions_keep_current_owners_and_acceptance_unknown', 'braking_pulses_keep_the_executable_control_cadence'):
        check = next(t for t in suite.tests() if t.name == name)
        assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_short_braking_candidates_use_opposing_measured_pushes_and_executable_times():
    from core.agency.arriving_by_her_pushes import _braking_pulse_invariant, braking_pulse_holds

    pushes = {'': (0., 0.), 'a': (-80., 0.), 'b': (80., 0.), 'c': (0., 80.)}
    assert braking_pulse_holds(pushes, .1, 1.2) == (
        ((0., 'a'), (.1, 'b'), (.2, '')), ((0., 'b'), (.1, 'a'), (.2, '')))
    for missing in ('a', 'b'):
        assert not braking_pulse_holds({k: v for k, v in pushes.items() if k != missing}, .1, 1.2)
    assert not braking_pulse_holds(pushes, .1, .2)
    assert not braking_pulse_holds(pushes, float('nan'), 1.2)
    assert not braking_pulse_holds({'a': (-80., 0.), 'b': (80., 0.)}, .1, 1.2)
    assert _braking_pulse_invariant() == ()


@pytest.mark.parametrize('velocity', [-1e-13, -5.551115123125783e-16, 5.551115123125783e-16, 1e-13])
def test_velocity_roundoff_cannot_trigger_repeated_actuation_inside_a_satisfied_region(velocity):
    choosing, goal = _choosing('inside', position=(100., 150.), carried=True)
    choosing.mine.vx = velocity
    result = best_plan(choosing, '', goal, arrive_at=choosing.goal_binding.thing)
    assert result.plan.held == ((0., ''),)


@pytest.mark.parametrize('carried', [False, True])
@pytest.mark.parametrize('position', [(70., 150.), (190., 150.), (140., 80.), (140., 195.),
                                     (95.4, 150.), (184.6, 150.), (140., 115.4), (140., 184.6)])
def test_actual_chooser_settles_inside_the_region_with_supplied_measured_controls(carried, position):
    # This tests ordinary binding/planning with explicit names and motion
    # measurements. It supplies no vision, discovery or native-play result.
    initial, _goal = _choosing('inside', position=position, carried=carried)
    body, target = initial.mine, initial.goal_binding.thing
    guide = Guide()
    guide.seen.saw(target.kind, 'platform')
    token = THE_GUIDE.set(guide)
    held, settled = '', 0
    try:
        for _ in range(100):
            choosing = _Choosing(initial.moves, initial.hers, initial.meeting,
                ['left', 'right', 'up', 'down'], next_picture_s=.1, response_s=.05)
            choosing.at = body.seen
            chosen, why, aim = choosing.key(held)
            assert why == 'meet' and aim is target and choosing.planned.gets == 0
            # The old input persists during delivery. Integrate the fixture
            # independently with exact constant-acceleration substeps.
            for control in (held, chosen):
                for _ in range(10):
                    if carried:
                        ax, ay = initial.hers.push_of(control)
                        body.x += body.vx*.005+.5*ax*.005**2
                        body.y += body.vy*.005+.5*ay*.005**2
                        body.vx, body.vy = body.vx+ax*.005, body.vy+ay*.005
                    else:
                        body.vx, body.vy = initial.hers.way_of(control) or (0., 0.)
                        body.x += body.vx*.005
                        body.y += body.vy*.005
                    body.seen += .005
            target.seen = body.seen
            held = chosen
            inside = 96.-.5 <= body.x <= 184.+.5 and 116.-.5 <= body.y <= 184.+.5
            settled = settled+1 if inside and math.hypot(body.vx, body.vy) <= 2. else 0
            if settled >= 8:
                return
        pytest.fail(f'not settled after the bounded fixture: position={(body.x, body.y)}, speed={(body.vx, body.vy)}')
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_the_made_geometric_forecast_reaches_the_actual_choice_record(tmp_path):
    import json

    from core.perception.recording_the_frames_and_inputs import record_frames_and_inputs, record_play_choice

    choosing, goal = _choosing('inside', position=(100., 150.))
    choosing.planned = best_plan(choosing, '', goal)
    choosing.target = lambda: pytest.fail('recording selected a new target')
    async with record_frames_and_inputs(tmp_path/'trace') as recorder:
        record_play_choice(choosing, '', 'meet', choosing.goal_binding.thing, choosing.at)
        await recorder.queue.join()
    event = json.loads((tmp_path/'trace/events.jsonl').read_text().splitlines()[0])
    forecast = event['planned']['relation']
    assert forecast['status'] == 'predicted_geometry' and forecast['terminal_error'] == 0.
    assert forecast['region']['target_owner'] == list(choosing.goal_binding.region.target_owner)
    assert forecast['observed_success'] is None and forecast['observed_contact'] is None
    assert event['planned']['gets'] == 0
