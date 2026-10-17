"""A control that reached the next screen still has a use after its discovery."""
from types import SimpleNamespace

import pytest

from core.agency.looking_ahead import worth_finding_out
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
from core.skills.screen_pursuit_bearings import screen_options
from core.skills.screen_pursuit_decision_lesson import _first_what_goes_on, _never_tried_here

pytestmark = pytest.mark.unit


def _returned_to_menu(label='Start', *, reached='Instructions Continue', answered=True):
    control = a_click_on(label)
    here = WhatWorksHere(told=('left', 'right'))
    labels, title = [control], 'Welcome to the activity'
    here.looked_at(labels, title)
    before = dict(clickable=labels, text=title)
    here.looked_at([a_click_on('Continue')], reached)
    here.tried(control, answered, before=before, after=dict(clickable=[a_click_on('Continue')], text=reached))
    here.looked_at(labels, title)
    return here, control


@pytest.mark.parametrize('label', ['Start', 'Next', 'Continue', 'Resume', 'BTART', '(BLAY)'])
def test_the_ordinary_choice_path_keeps_a_known_way_forward_with_untried_keys(label):
    here, control = _returned_to_menu(label)
    options = screen_options(here.available())
    fresh = _never_tried_here(options, SimpleNamespace(tried=set()), here)
    assert control not in fresh and 'left' in fresh
    novelty = worth_finding_out(SimpleNamespace(), None, [option.name for option in options], never_tried=fresh)
    assert control not in novelty, 'instrumental use was counted as new information'
    valued = _first_what_goes_on(here, novelty)
    assert control in valued
    assert valued[control] > max(valued['left'], valued['right'])


@pytest.mark.parametrize('label', ['Score', 'Launch Meter', 'Start Over', 'Reset', 'Restart', 'New Game', 'Retry'])
def test_readouts_and_reset_controls_receive_no_new_route_value(label):
    here, control = _returned_to_menu(label)
    assert control not in _first_what_goes_on(here, {'left': 1.})


def test_a_control_that_never_answered_has_no_known_forward_use():
    here, control = _returned_to_menu(answered=False)
    assert control not in _first_what_goes_on(here, {'left': 1.})


def test_a_forward_route_from_another_screen_is_not_a_current_route():
    here, control = _returned_to_menu()
    here.looked_at([control], 'Edit settings for audio display')
    assert control not in _first_what_goes_on(here, {'left': 1.})


def test_missing_current_control_cannot_be_reintroduced_from_route_memory():
    here, control = _returned_to_menu()
    here.on_screen = ()
    assert control not in _first_what_goes_on(here, {'left': 1.})


def test_the_actual_available_action_set_limits_retained_route_use():
    here, control = _returned_to_menu()
    valued = _first_what_goes_on(here, {'left': 1.}, available=['left'])
    assert control not in valued


@pytest.mark.parametrize('damage', ['self_loop', 'unknown_destination', 'contradiction', 'false_count', 'missing_count', 'unknown_screen'])
def test_missing_or_conflicting_route_evidence_gives_no_reuse_priority(damage):
    here, control = _returned_to_menu()
    graph = here.leads
    receipt = f'{graph._here}|{control}'
    if damage == 'self_loop':
        graph.to[receipt] = graph._here
    elif damage == 'unknown_destination':
        graph.to[receipt] = 800
    elif damage == 'contradiction':
        graph.went[receipt]['nothing'] = 1
    elif damage == 'false_count':
        graph.went[receipt]['on'] = True
    elif damage == 'missing_count':
        graph.went.pop(receipt)
    else:
        graph._here = None
    assert control not in _first_what_goes_on(here, {'left': 1.})


def test_the_new_route_keeps_the_existing_lesson_and_debate_checks(monkeypatch):
    from core.skills import screen_pursuit_decision_lesson as choosing

    here, control = _returned_to_menu()
    received = []

    def checked(_here, valued):
        received.append(dict(valued))
        return {name: 0. if name == control else value for name, value in valued.items()}

    monkeypatch.setattr(choosing, '_as_checked', checked)
    result = choosing._first_what_goes_on(here, {'left': 1.})
    assert received and control in received[0]
    assert result[control] == 0. and result['left'] > 0.


def test_known_navigation_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.skills.known_navigation_after_discovery import _known_navigation_use_invariant

    assert _known_navigation_use_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'navigation_use_is_separate_from_discovery')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
