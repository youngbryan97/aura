"""A previous focus result cannot identify the receiver of a later key."""
import json
from types import SimpleNamespace

import pytest

from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens
from core.skills.screen_pursuit_on_a_page import OnAPage

pytestmark = pytest.mark.unit


class ChangingFocusPage:
    def __init__(self):
        self.focused, self.checks, self.delivered = False, 0, []
        self.reply = True
        self.keyboard = SimpleNamespace(down=self.down, up=self.up, press=self.press)

    async def evaluate(self, script):
        assert 'focus()' in script
        self.checks += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        self.focused = self.reply is True
        return self.reply

    async def down(self, key):
        self.delivered.append(('down', key, self.focused))

    async def up(self, key):
        self.delivered.append(('up', key, self.focused))

    async def press(self, key, **kwargs):
        self.delivered.append(('press', key, self.focused))


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['normal', 'held', 'tapped'])
async def test_each_actual_input_consumer_rechecks_after_another_control_takes_focus(kind):
    page = ChangingFocusPage()
    hands = OnAPage(page, 'owned fixture') if kind == 'normal' else PlayingAsItHappens(page, (0., 0., 1., 1.), 'test', 0.)
    action = hands.press if kind == 'normal' else hands.down if kind == 'held' else hands.tap
    await action('left')
    page.focused = False
    await action('right')
    delivered = [event for event in page.delivered if event[0] != 'up']
    assert len(delivered) == 2 and all(event[2] for event in delivered)
    assert page.checks == 2
    if kind != 'normal':
        await hands.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['normal', 'held', 'tapped'])
async def test_a_pointer_focus_flag_cannot_replace_the_first_keyboard_target_query(kind):
    page = ChangingFocusPage()
    hands = OnAPage(page, 'owned fixture') if kind == 'normal' else PlayingAsItHappens(page, (0., 0., 1., 1.), 'test', 0.)
    hands._focused = True
    action = hands.press if kind == 'normal' else hands.down if kind == 'held' else hands.tap
    await action('left')
    assert page.checks == 1 and page.delivered[0][2]
    if kind != 'normal':
        await hands.close()


@pytest.mark.asyncio
async def test_release_is_delivered_without_waiting_to_focus_an_input_target():
    page = ChangingFocusPage()
    hands = PlayingAsItHappens(page, (0., 0., 1., 1.), 'test', 0.)
    await hands.down('left')
    page.focused = False
    page.reply = RuntimeError('target unavailable')
    await hands.up('left')
    assert page.delivered[-1] == ('up', 'ArrowLeft', False)
    assert page.checks == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('reply,status', [(True, 'focused'), (False, 'not_focused'),
                                         ({'focused': True}, 'unmeasured'), (RuntimeError('closed'), 'unavailable')])
async def test_a_target_receipt_does_not_become_transport_completion(tmp_path, reply, status):
    from core.perception.recording_the_frames_and_inputs import record_frames_and_inputs, recorded_play_ports

    page = ChangingFocusPage()
    page.reply = reply
    hands = PlayingAsItHappens(page, (0., 0., 1., 1.), 'test', 0.)
    directory = tmp_path / 'focus-receipts'
    async with record_frames_and_inputs(directory):
        _, ports = recorded_play_ports(None, hands)
        await ports.down('left')
        await ports.up('left')
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    target, = [event for event in events if event['kind'] == 'keyboard_target']
    assert target['status'] == status and target['surface'] == f'page:{id(page)}'
    assert target['began'] <= target['ended'] <= target['received_at']
    assert not {'accepted_by_game', 'controlled', 'task_passed'} & target.keys()
    assert len([event for event in events if event['kind'] == 'input' and event['phase'] == 'completed']) == 2


def test_current_keyboard_target_receipts_have_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.recording_the_frames_and_inputs import _keyboard_target_receipt_invariant

    assert _keyboard_target_receipt_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'keyboard_target_receipts_do_not_claim_acceptance')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
