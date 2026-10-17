"""A control read earlier cannot rename new writing at its old position."""
from types import SimpleNamespace
import time

import pytest

from core.skills.screen_pursuit_as_it_happens import _ways_on_read_right_before

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('old,current', [('Start', '0%'), ('Play', '00:30'),
                                        ('Continue', '+'), ('Resume', 'FTHY ABAIN')])
@pytest.mark.parametrize('offset', [0., .015])
def test_old_control_coordinates_cannot_replace_a_current_counter_or_unread_label(old, current, offset):
    reflex = SimpleNamespace(keep={'ways_on_read': [(old, .5, .65, time.monotonic())]})
    observation = dict(text=current, layout=[dict(text=current, center_x=.5+offset, center_y=.65)])
    _ways_on_read_right_before(reflex, observation)
    assert observation['layout'][0]['text'] == current
    assert observation['text'] == current
    assert not reflex.keep.get('ways_on_read')


def test_the_actual_read_hook_does_not_offer_a_counter_as_an_old_start_button():
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

    reflex = PlayingAsItHappens(page=None, band=(0., 0., 1., 1.), goal='play', ends_at=time.monotonic()+60.)
    reflex.keep['ways_on_read'] = [('Start', .5, .65, time.monotonic())]
    observation = dict(text='0%', layout=[dict(text='0%', center_x=.5, center_y=.65)])
    reflex.read(observation)
    assert observation['layout'][0]['text'] == '0%' and not reflex.way_on_shown
    assert reflex.labels_seen == {'0%'} and not reflex.keep.get('ways_on_read')


def test_the_coordinate_label_boundary_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.skills.screen_pursuit_as_it_happens import _coordinate_label_invariant

    assert _coordinate_label_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'coordinates_cannot_rename_current_words')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
