"""A fixed physical trial must admit its measurement under observed capture jitter."""
import pytest

from core.agency.time_to_observe_a_control import observed_response_seconds
from core.agency.which_one_answers_to_her import PUSH_PICTURES, _Speeds

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('slow', [.18, .25, .3, .35])
def test_a_recent_slow_capture_cannot_leave_the_reserved_push_unmeasurable(slow):
    options = dict(least_s=.45, settle_s=.2, samples=PUSH_PICTURES, span_s=.15, longest_s=1.8)
    duration = observed_response_seconds([.1]*11+[slow], **options)
    assert duration is not None and duration <= options['longest_s']
    measured = _Speeds()
    at = 0.
    while at+slow < duration-1e-9:
        at += slow
        if at >= options['settle_s']:
            measured.add('engine', 0., -60.*at, press=0., when=at)
    assert len(measured.pushes()['engine']) == 1
    assert measured.pushes()['engine'][0] == pytest.approx((0., -60.))


def test_an_observed_gap_cannot_make_the_finite_physical_budget_unbounded():
    options = dict(least_s=.45, settle_s=.2, samples=4, span_s=.15, longest_s=.8)
    assert observed_response_seconds([.1]*11+[.3], **options) is None
