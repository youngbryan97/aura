"""A missing referent requests observation, without assigning its role by geometry."""
from dataclasses import replace

import pytest

from core.agency.a_requested_relation_in_play import SpatialRequest, read_destination
from core.agency.looking_for_an_unbound_referent import (
    _referent_attention_invariant, referents_to_observe,
)
from core.perception.names_for_observed_instances import NamesInScene, query_for
from core.perception.what_things_look_like import LookingAtThings
from tests.test_visual_names_stay_with_the_boxed_instance import _scene, _thing

pytestmark = pytest.mark.unit


def setup():
    body = _thing()
    target = _thing(2, kind=1, x=140., y=150., w=40., h=10.)
    other = _thing(3, kind=2, x=220.)
    body.moved, target.moved, other.moved = True, False, True
    return body, target, other, _scene(body, target, other)


def test_one_query_slot_serves_the_goal_before_the_default_player_priority():
    body, target, other, scene = setup()
    request = SpatialRequest('on', 'blue platform', '')
    looking = LookingAtThings()
    before = looking._to_look_at(scene, 1, set(), at=4.)
    sought = referents_to_observe(request, scene, body, {}, 4.)
    after = looking._to_look_at(scene, 1, set(), at=4., sought=sought)
    assert before[0].number == body.number
    assert after[0].number == target.number and after[1].number == body.number
    assert set(q.key for q in before) == set(q.key for q in after)
    assert not read_destination(request, body, [target], {}, 4.).binding


def test_a_request_can_seek_the_referent_while_body_discovery_continues():
    body, target, _, scene = setup()
    sought = referents_to_observe(SpatialRequest('on', 'platform', ''), scene, None, {}, 4.)
    assert sought[0] == query_for(target, scene, 4.).key
    assert not referents_to_observe(None, scene, body, {}, 4.)


def test_geometry_only_ranks_questions_and_cannot_discard_a_fragment_to_be_inspected():
    body, target, _, scene = setup()
    target.w = 5.
    request = SpatialRequest('inside', 'blue compartment', '')
    sought = referents_to_observe(request, scene, body, {}, 4.)
    assert query_for(target, scene, 4.).key in sought
    assert read_destination(request, body, [target], {}, 4.).binding is None


def test_a_bound_current_goal_needs_no_extra_query_and_stale_instances_gain_no_priority():
    body, target, _, scene = setup()
    request = SpatialRequest('on', 'platform', '')
    assert not referents_to_observe(request, scene, body, {1: 'platform'}, 4.)
    target.seen = 3.
    assert not any(key[2] == target.number for key in referents_to_observe(request, scene, body, {}, 4.))


def test_scoped_names_do_not_lend_a_label_to_an_unnamed_peer():
    body, target, other, scene = setup()
    other.kind = target.kind
    names = NamesInScene(True, ((other.number, other.born, other.kind, 'platform'),))
    # The named peer is not current, so it cannot bind. The current unnamed
    # target still needs observation even though their visual kind agrees.
    other.seen = 3.
    sought = referents_to_observe(SpatialRequest('on', 'platform', ''), scene, body, names, 4.)
    assert sought == (query_for(target, scene, 4.).key,)


def test_foreign_epoch_requests_and_already_asked_instances_cannot_bypass_query_custody():
    body, target, other, scene = setup()
    looking = LookingAtThings()
    wanted = query_for(target, scene, 4.)
    foreign = replace(wanted, epoch='0'*32)
    chosen = looking._to_look_at(scene, 1, set(), at=4., sought=(foreign.key,))
    assert chosen[0].number == body.number
    looking.asked_instances.add(wanted.key)
    chosen = looking._to_look_at(scene, 1, set(), at=4., sought=(wanted.key,))
    assert target.number not in {query.number for query in chosen}
    assert chosen[0].number == body.number


def test_the_task_request_preserves_existing_query_size_and_budget_gates(monkeypatch):
    body, target, _, scene = setup()
    looking = LookingAtThings()
    wanted = query_for(target, scene, 4.)
    target.w = 290.
    assert wanted.number not in {q.number for q in looking._to_look_at(scene, 1, set(), at=4., sought=(wanted.key,))}
    monkeypatch.setattr('core.perception.what_things_look_like.MOST_A_GAME', 1)
    looking.asked_instances.add(query_for(body, scene, 4.).key)
    assert not looking._to_look_at(scene, 1, set(), at=4., sought=(wanted.key,))


def test_missing_referent_attention_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _referent_attention_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'unbound_referents_request_current_instance_observation')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
