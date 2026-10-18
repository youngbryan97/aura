"""A physical probe counts when another discovery strategy takes over."""
import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from core.agency.a_trial_has_a_physical_end import cancel_trial_deadline
from core.agency.inputs_while_measuring_control import hold_control
from core.agency.playing_as_it_happens import _Run
from core.agency.trying_controls_with_a_view_of_their_response import try_controls
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize('ordinary_first', [False, True])
async def test_discovery_uses_an_untried_key_after_two_delivered_probes_in_either_order(ordinary_first):
    keys, delivered = ['opaque-a', 'opaque-b', 'opaque-c'], []
    run = _Run(keys=list(keys), began=time.monotonic(), tried_at=-math.inf)
    planned = iter(([(None, False)] if ordinary_first else [])+
                   [('opaque-a', False), ('', False), ('opaque-a', False), ('', False), (None, True)])

    class Counter:
        key, done, phase_seconds = 'opaque-a', False, .01
        def choose(self, *_args, **_kwargs):
            if self.done:
                return None
            choice, self.done = next(planned, (None, True))
            return choice
        def released_at_deadline(self, *_args, **_kwargs):
            return False

    class Hands:
        async def down(self, key):
            delivered.append(('down', key))
        async def up(self, key):
            delivered.append(('up', key))

    run.counter_trial = Counter()
    hers, moves, hands = WhichIsHers(), SimpleNamespace(things={}, shape=(200, 200)), Hands()
    try:
        for _ in range(8):
            await try_controls(hands, run, hers, time.monotonic(), moves, hold=hold_control, seconds=.01)
            deadline = getattr(run, '_trial_input_deadline', None)
            if deadline is not None:
                await asyncio.wait_for(asyncio.shield(deadline), 1.)
            trial = run.body_trial
            await asyncio.sleep(max(0., trial.until-time.monotonic())+.002)
            if len([key for kind, key in delivered if kind == 'down']) >= 3:
                break
        assert [key for kind, key in delivered if kind == 'down'] == ['opaque-a', 'opaque-a', 'opaque-b']
        assert hers.number is None and not hers.identification.receipts
    finally:
        cancel_trial_deadline(run)
        if run.held:
            await hold_control(hands, run, hers, '', time.monotonic())


@pytest.mark.parametrize('damage', ['not_independent', 'other_hold', 'not_trying', 'old_time', 'nonfinite', 'foreign_key'])
def test_an_unacknowledged_or_ineligible_probe_cannot_spend_an_exposure(damage):
    from core.agency.control_trials_keep_their_delivered_counts import next_discovery_control, record_discovery_delivery

    run = SimpleNamespace(keys=['a', 'b'], began=1., held='a', held_since=2., held_trying=True)
    hers = SimpleNamespace(identification=SimpleNamespace(epoch=0))
    key, at, independent = 'a', 2., True
    if damage == 'not_independent':
        independent = False
    elif damage == 'other_hold':
        run.held = 'b'
    elif damage == 'not_trying':
        run.held_trying = False
    elif damage == 'old_time':
        at = .5
    elif damage == 'nonfinite':
        at = math.nan
    elif damage == 'foreign_key':
        key = 'not-authorized'
    assert not record_discovery_delivery(run, hers, key, at, independent=independent)
    assert next_discovery_control(run, hers) == 'a'


def test_duplicate_receipts_and_reordered_controls_keep_their_delivered_history():
    from core.agency.control_trials_keep_their_delivered_counts import next_discovery_control, record_discovery_delivery

    run = SimpleNamespace(keys=['a', 'b', 'c'], began=0., held='a', held_since=1., held_trying=True)
    hers = SimpleNamespace(identification=SimpleNamespace(epoch=0))
    assert record_discovery_delivery(run, hers, 'a', 1., independent=True)
    assert not record_discovery_delivery(run, hers, 'a', 1., independent=True)
    run.keys.reverse()
    assert next_discovery_control(run, hers) == 'c'
    run.held, run.held_since = 'c', 2.
    assert record_discovery_delivery(run, hers, 'c', 2., independent=True)
    assert next_discovery_control(run, hers) == 'b'
    hers.identification.epoch += 1
    assert next_discovery_control(run, hers) == 'c'
    assert not record_discovery_delivery(run, hers, 'c', 2., independent=True)
    run.held_since = 3.
    assert record_discovery_delivery(run, hers, 'c', 3., independent=True)
    hers.identification.epoch -= 1
    assert next_discovery_control(run, hers) is None


def test_new_controls_cannot_grow_the_retained_count_table_without_a_bound():
    from core.agency.calibration_for_a_current_body import MAX_CALIBRATION_CONTROLS
    from core.agency.control_trials_keep_their_delivered_counts import next_discovery_control, record_discovery_delivery

    run = SimpleNamespace(keys=[], began=0., held='', held_since=None, held_trying=True)
    hers = SimpleNamespace(identification=SimpleNamespace(epoch=0))
    for n in range(MAX_CALIBRATION_CONTROLS):
        run.keys, run.held, run.held_since = [str(n)], str(n), float(n+1)
        assert record_discovery_delivery(run, hers, str(n), float(n+1), independent=True)
    run.keys, run.held, run.held_since = ['extra'], 'extra', 100.
    assert not record_discovery_delivery(run, hers, 'extra', 100., independent=True)
    assert next_discovery_control(run, hers) is None
    assert len(run._delivered_body_probes.counts) == MAX_CALIBRATION_CONTROLS


def test_delivered_discovery_history_is_a_registered_measured_claim():
    from core.agency.control_trials_keep_their_delivered_counts import _delivered_probe_count_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _delivered_probe_count_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'discovery_strategies_share_delivered_probe_counts')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
