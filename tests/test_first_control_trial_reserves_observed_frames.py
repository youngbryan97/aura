"""Cold discovery cannot discard the capture intervals it already measured."""
from types import SimpleNamespace

import pytest

from core.agency.time_to_observe_a_control import observed_response_seconds
from core.agency.which_one_answers_to_her import PUSH_PICTURES, SETTLE_S, WhichIsHers, _Speeds

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('intervals', [[.27], [.24, .33]])
def test_initial_measured_intervals_reserve_a_complete_acceleration_measurement(intervals):
    duration = observed_response_seconds(intervals, least_s=.45, settle_s=SETTLE_S,
                                         samples=PUSH_PICTURES, span_s=.15, longest_s=1.8)
    assert duration is not None
    cadence = max(intervals)
    record = _Speeds()
    at = cadence
    while at < duration-1e-9:
        if at >= SETTLE_S:
            record.add('opaque', 0., -60.*at, press=0., when=at)
        at += cadence
    assert len(record.pushes()['opaque']) == 1
    assert record.pushes()['opaque'][0] == pytest.approx((0., -60.))


@pytest.mark.asyncio
async def test_the_actual_first_delivery_uses_its_two_captured_intervals(monkeypatch):
    from core.agency import trying_controls_with_a_view_of_their_response as trying
    from core.agency.playing_as_it_happens import _Run

    run = _Run(keys=['opaque-a', 'opaque-b'], began=1.)
    run.intervals.extend([.24, .33])
    hers = WhichIsHers()
    leases = []
    monkeypatch.setattr(trying, 'lease_control_trial', lambda *_a, **_kw: leases.append(run.body_trial))

    async def acknowledged(_hands, current, observer, key, at, *, trying):
        current.held, current.held_since, current.held_trying = key, at+.02, trying
        observer.holding(key, current.held_since, trying=trying)

    await trying.try_controls(None, run, hers, 2., SimpleNamespace(things={}), hold=acknowledged, seconds=.45)
    assert len(leases) == 1 and run.body_trial.began == 2.02
    assert run.body_trial.until-run.body_trial.began >= SETTLE_S+PUSH_PICTURES*.33-1e-9
    assert run.body_trial.until-run.body_trial.began <= 1.8
    assert hers.number is None and not hers.identification.receipts
