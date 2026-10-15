"""Campaign recovery uses settled attempts and never grants itself a gameplay pass."""
from __future__ import annotations

import asyncio
import copy
import json

import pytest

from tools.play_campaign_checkpoint import (
    campaign_lease,
    campaign_plan,
    resume_campaign,
    run_campaign,
)

pytestmark = pytest.mark.unit


def _plan():
    return campaign_plan([2, 5, 8], excluded={5}, identity={'commit': 'one', 'assets': 'original'}, seconds=30.)


@pytest.mark.asyncio
async def test_each_attempt_is_saved_before_play_and_a_win_claim_remains_unqualified():
    saved, called = [], []
    record = _plan()

    async def save(value):
        saved.append(copy.deepcopy(value))

    async def play(number):
        called.append(number)
        assert next(c for c in saved[-1]['cases'] if c['case'] == number)['state'] == 'running'
        return {'game': number, 'won': True}

    results = await run_campaign(record, play=play, save=save)
    assert called == [2, 8] and len(results) == 2
    assert record['status'] == 'finished_requires_independent_review'
    assert not record['presentation_qualified']
    assert all(c['task_passed'] is None and c['qualification'] == 'unmeasured' for c in record['cases'])
    assert saved[0]['cases'][2]['state'] == 'pending'


@pytest.mark.asyncio
async def test_failure_retains_the_unattempted_denominator_and_resume_does_not_replay_settled_work():
    record, called = _plan(), []

    async def save(value):
        pass

    async def fail(number):
        return {'game': number, 'error': 'input delivery unavailable'}

    await run_campaign(record, play=fail, save=save)
    assert [c['state'] for c in record['cases']] == ['recorded', 'excluded', 'pending']
    assert record['status'] == 'stopped_on_error'
    resumed = resume_campaign(record, _plan())

    async def play(number):
        called.append(number)
        return {'game': number, 'won': False}

    results = await run_campaign(resumed, play=play, save=save)
    assert called == [8] and [r['game'] for r in results] == [2, 8]
    assert results[0]['error'] == 'input delivery unavailable'


@pytest.mark.asyncio
async def test_cancellation_retains_an_unfinished_case_and_prevents_blind_resume():
    record, saved = _plan(), []
    entered = asyncio.Event()

    async def save(value):
        saved.append(copy.deepcopy(value))

    async def play(number):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run_campaign(record, play=play, save=save))
    await asyncio.wait_for(entered.wait(), 1.)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert saved[-1]['status'] == 'interrupted_requires_review'
    assert saved[-1]['cases'][0]['state'] == 'interrupted'
    with pytest.raises(ValueError, match='ownership'):
        resume_campaign(saved[-1], _plan())


@pytest.mark.parametrize('change', ['commit', 'assets', 'budget', 'selection', 'exclusions', 'running'])
def test_resume_refuses_changed_identity_or_unresolved_ownership(change):
    record = _plan()
    if change in ('commit', 'assets'):
        record['identity'][change] = 'changed'
    elif change == 'budget':
        record['seconds'] = 31.
    elif change == 'selection':
        record['cases'][0]['case'] = 3
    elif change == 'exclusions':
        record['cases'][1]['state'] = 'pending'
    else:
        record['cases'][0]['state'] = 'running'
    with pytest.raises(ValueError):
        resume_campaign(record, _plan())


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong', [None, {'game': 8, 'won': True}, 'won'])
async def test_foreign_or_missing_results_stop_the_campaign(wrong):
    record = _plan()

    async def save(value):
        pass

    async def play(number):
        return wrong

    results = await run_campaign(record, play=play, save=save)
    assert record['status'] == 'stopped_on_error'
    assert results[0]['game'] == 2 and 'no result for this case' in results[0]['error']
    assert record['cases'][2]['state'] == 'pending'


@pytest.mark.parametrize('cases,seconds', [([], 30.), ([1, 1], 30.), ([True], 30.), ([1], float('nan')),
                                          ([1], float('inf')), ([1], 0.), ([1], 3601.)])
def test_invalid_plans_never_launch_work(cases, seconds):
    with pytest.raises(ValueError):
        campaign_plan(cases, excluded=set(), identity={}, seconds=seconds)


def test_selective_asset_validation_never_opens_excluded_programs(tmp_path, monkeypatch):
    from pathlib import Path

    from tests.test_offline_flash_bundle import _bundle
    from tools.serve_offline_flash_bundle import check_bundle

    manifest = _bundle(tmp_path)
    manifest['cases'].append({'id': 5, 'status': 'asset_verified', 'program': {'sha256': 'unopened'}})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    original = Path.read_bytes
    opened = []

    def read(path):
        opened.append(path)
        assert path.name != '5.swf'
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    assert len(check_bundle(tmp_path, case_ids={0})['cases']) == 3
    assert any(p.name == '0.swf' for p in opened)


@pytest.mark.asyncio
async def test_no_case_runs_when_the_pre_attempt_checkpoint_cannot_be_saved():
    record = _plan()

    async def save(value):
        if value['cases'][0]['state'] == 'running':
            raise OSError('storage unavailable')

    async def play(number):
        pytest.fail('an unrecorded case must not launch')

    with pytest.raises(OSError, match='storage'):
        await run_campaign(record, play=play, save=save)


@pytest.mark.asyncio
async def test_a_second_supervisor_cannot_own_the_same_checkpoint(tmp_path):
    checkpoint = tmp_path / 'case.json'
    async with campaign_lease(checkpoint):
        with pytest.raises(RuntimeError, match='another supervisor'):
            async with campaign_lease(checkpoint):
                pytest.fail('a second campaign entered')
    async with campaign_lease(checkpoint):
        pass


@pytest.mark.asyncio
async def test_the_cli_excludes_protected_cases_before_asset_reading_and_resumes_without_replaying(tmp_path, monkeypatch):
    from tools import measure_on_archived_games as runner

    bundle, called, validated = tmp_path / 'bundle', [], []
    bundle.mkdir()
    (bundle / 'manifest.json').write_text(json.dumps({'cases': [
        {'id': n, 'status': 'asset_verified', 'url': f'{n}.html'} for n in [2, 5, 8]]}))
    held = tmp_path / 'held.json'
    held.write_text(json.dumps({'held_out': [5]}))
    out = tmp_path / 'pass.json'

    def served(path, active):
        assert path == bundle
        validated.append(active)
        return {n: f'http://local/{n}' for n in active}

    async def play(number, *args, **kwargs):
        called.append(number)
        saved = json.loads((tmp_path / 'pass.json.state.json').read_text())
        assert next(c for c in saved['cases'] if c['case'] == number)['state'] == 'running'
        return {'game': number, 'won': True}

    monkeypatch.setattr(runner, '_served', served)
    monkeypatch.setattr(runner, '_one', play)
    argv = ['--bundle', str(bundle), '--all', '--exclude', '8', '--held-out', str(held), '--out', str(out)]
    assert len(await runner.main(argv)) == 1
    assert called == [2] and validated == [{2}]
    record = json.loads((tmp_path / 'pass.json.state.json').read_text())
    assert [c['state'] for c in record['cases']] == ['recorded', 'excluded', 'excluded']
    assert not record['presentation_qualified'] and record['cases'][0]['task_passed'] is None
    assert len(await runner.main([*argv, '--resume'])) == 1
    assert called == [2] and validated == [{2}]
    with pytest.raises(SystemExit):
        await runner.main(argv)


@pytest.mark.asyncio
async def test_a_missing_case_is_a_retained_failed_attempt(tmp_path, monkeypatch):
    from tools import measure_on_archived_games as runner

    out = tmp_path / 'pass.json'
    monkeypatch.setattr(runner, 'ARCHIVED', {})
    result = await runner.main(['--games', '8', '--held-out', str(tmp_path / 'no-held'), '--out', str(out)])
    assert result == [{'game': 8, 'error': 'no copy that plays here'}]
    assert json.loads(out.read_text()) == result
    assert json.loads((tmp_path / 'pass.json.state.json').read_text())['status'] == 'stopped_on_error'


@pytest.mark.asyncio
async def test_plan_only_starts_no_browser_asset_reader_or_model_connection(tmp_path, monkeypatch):
    from tools import measure_on_archived_games as runner

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text(json.dumps({'cases': [{'id': 2}, {'id': 5}]}))
    held = tmp_path / 'held.json'
    held.write_text(json.dumps({'held_out': [5]}))

    def forbidden(*args, **kwargs):
        pytest.fail('plan-only work must not open assets, a browser or a model')

    monkeypatch.setattr(runner, '_served', forbidden)
    monkeypatch.setattr(runner, '_one', forbidden)
    monkeypatch.setattr(runner, '_her_model_from', forbidden)
    out = tmp_path / 'pass.json'
    assert await runner.main(['--bundle', str(bundle), '--all', '--held-out', str(held), '--plan-only',
                              '--her-model', 'http://uncontacted', '--out', str(out)]) == []
    record = json.loads((tmp_path / 'pass.json.state.json').read_text())
    assert [c['state'] for c in record['cases']] == ['pending', 'excluded']


@pytest.mark.asyncio
@pytest.mark.parametrize('fails', [False, True])
async def test_short_case_budgets_reach_play_and_both_globals_are_restored(monkeypatch, fails):
    from types import SimpleNamespace

    from core.capabilities import phantom_browser
    from core.skills import screen_pursuit as pursuit
    from core.skills import sovereign_browser_drawing as drawing
    from tools import measure_on_archived_games as runner

    async def nothing(*args, **kwargs):
        pass

    class Browser:
        page = SimpleNamespace(goto=nothing)

        async def ensure_ready(self):
            return True

        def get_status(self):
            return {'graphics': {'renderer': 'test driver'}}

        close = staticmethod(nothing)

    async def play(*args):
        assert drawing.PLAY_UNTIL_WON_S == 30.
        assert drawing.LEAST_SHARE_S == 30.
        if fails:
            raise RuntimeError('play failed')
        return {'won': True}

    original_tell = pursuit._tell
    monkeypatch.setattr(drawing, 'PLAY_UNTIL_WON_S', 1200.)
    monkeypatch.setattr(drawing, 'LEAST_SHARE_S', 360.)
    monkeypatch.setattr(drawing, 'played_on_the_drawing', play)
    monkeypatch.setattr(phantom_browser, 'PhantomBrowser', lambda **kwargs: Browser())
    monkeypatch.setattr(runner.asyncio, 'sleep', nothing)
    result = await runner._one(2, 'local-test', .5, False)
    assert bool(result['error']) == fails
    assert drawing.PLAY_UNTIL_WON_S == 1200. and drawing.LEAST_SHARE_S == 360.
    assert pursuit._tell is original_tell
