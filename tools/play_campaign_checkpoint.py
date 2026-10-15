"""Keep each attempted environment durable without certifying its own result.

A process may finish while its task remains unresolved. Each case retains
that distinction, including cases not reached before a failure or interruption.
The caller supplies the environments and the ordinary play function.
"""
from __future__ import annotations

import asyncio
import copy
import fcntl
import math
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


@asynccontextmanager
async def campaign_lease(checkpoint: Path):
    """Refuse a second supervisor of the same checkpoint; a crash releases the lease."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    lock_path = checkpoint.with_name(checkpoint.name + '.lock')
    with local_internal_governed_scope('play_campaign_checkpoint'):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(lock_path.parent, source='play_campaign_checkpoint')
        await gateway.write_bytes_if_absent_async(lock_path, b'', source='play_campaign_checkpoint')
    fd = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another supervisor owns this campaign checkpoint') from error
        yield
    finally:
        os.close(fd)


def campaign_plan(cases: list[int], *, excluded: set[int], identity: dict[str, Any],
                  seconds: float) -> dict[str, Any]:
    if (not cases or any(type(n) is not int or n < 0 for n in cases) or len(set(cases)) != len(cases)
            or not math.isfinite(seconds) or not 0 < seconds <= 3600):
        raise ValueError('a campaign needs unique case identities and a finite case budget of at most one hour')
    return {'schema': 'aura.play_campaign.v1', 'identity': copy.deepcopy(identity), 'seconds': seconds,
            'created_at': time.time(), 'status': 'prepared', 'presentation_qualified': False,
            'cases': [{'case': n, 'state': 'excluded' if n in excluded else 'pending',
                       'task_passed': None, 'qualification': 'unmeasured', 'result': None} for n in cases]}


def resume_campaign(record: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Resume settled attempts only, with the same code, assets, selection and budget."""
    if (record.get('schema') != expected['schema'] or record.get('identity') != expected['identity']
            or record.get('seconds') != expected['seconds']
            or [(c.get('case'), c.get('state') == 'excluded') for c in record.get('cases', [])]
            != [(c['case'], c['state'] == 'excluded') for c in expected['cases']]):
        raise ValueError('the retained campaign does not match this code, assets, selection or budget')
    allowed = {'excluded', 'pending', 'recorded'}
    if any(c.get('state') not in allowed for c in record['cases']):
        raise ValueError('an unfinished case needs ownership and input-release review before another attempt')
    if any(c['state'] == 'recorded' and (not isinstance(c.get('result'), dict)
            or c['result'].get('game') != c['case']) for c in record['cases']):
        raise ValueError('a recorded attempt lacks its own result')
    return copy.deepcopy(record)


async def run_campaign(record: dict[str, Any], *, play: Callable[[int], Awaitable[dict]],
                       save: Callable[[dict], Awaitable[None]]) -> list[dict]:
    """One case at a time; stop on an error and preserve all pending cases."""
    record['status'] = 'running'
    await save(record)
    for case in record['cases']:
        if case['state'] != 'pending':
            continue
        case.update(state='running', started_at=time.time())
        await save(record)
        try:
            result = await play(case['case'])
            if not isinstance(result, dict) or result.get('game') != case['case']:
                raise ValueError('play returned no result for this case')
        except asyncio.CancelledError:
            case.update(state='interrupted', finished_at=time.time())
            record['status'] = 'interrupted_requires_review'
            await asyncio.shield(save(record))
            raise
        except Exception as error:  # noqa: BLE001 - the case remains a failed attempt
            result = {'game': case['case'], 'error': f'{type(error).__name__}: {error}'}
        case.update(state='recorded', result=result, finished_at=time.time())
        if result.get('error'):
            record['status'] = 'stopped_on_error'
            await save(record)
            break
        await save(record)
    else:
        record['status'] = 'finished_requires_independent_review'
        await save(record)
    return [c['result'] for c in record['cases'] if c['state'] == 'recorded']
