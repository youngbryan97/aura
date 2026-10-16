"""A revisit reacquires source evidence instead of substituting remembered controls."""
from __future__ import annotations

import base64
import re
import struct
from pathlib import Path

import pytest

from core.skills import sovereign_browser_the_program as programs


@pytest.fixture(autouse=True)
def private_summary_store(monkeypatch, tmp_path):
    monkeypatch.setattr('core.runtime.what_she_learned._KEPT_IN', tmp_path/'summaries')


def _swf(name):
    def tag(code, body):
        return struct.pack('<H', code << 6 | len(body))+body

    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    sprite = tag(39, struct.pack('<HHH', 7, 1, 0))
    export = tag(56, struct.pack('<HH', 1, 7)+name.encode()+b'\0')
    body = int(rect, 2).to_bytes(len(rect)//8, 'big')+struct.pack('<HH', 24*256, 1)+sprite+export+b'\0\0'
    return b'FWS\x06'+struct.pack('<I', len(body)+8)+body


class _Page:
    url = 'https://example.org/current-drawing'
    program_url = 'https://example.org/current-program.swf'

    def __init__(self, data):
        self.data, self.fetches, self.inline = data, 0, []
        self.inline_unavailable = False

    async def evaluate(self, script, arg=None):
        if script == programs._WHAT_IT_LOADED:
            return [[self.program_url, 'other', 0]] if self.program_url else []
        if script == programs._FETCH_IT:
            self.fetches += 1
            if isinstance(self.data, Exception):
                raise self.data
            return base64.b64encode(self.data).decode() if isinstance(self.data, bytes) else self.data
        if script == programs._ITS_OWN_SCRIPTS:
            if self.inline_unavailable:
                raise RuntimeError('current inline source is unavailable')
            return {'scripts': self.inline, 'stage': [480, 320]}
        raise AssertionError('unexpected browser request')


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [False, True])
async def test_a_revisit_keeps_exact_instruction_and_asset_custody_for_the_acquired_file(changed):
    from core.runtime.what_she_learned import named, recall

    page = _Page(_swf('first icon'))
    first = await programs.read_what_the_page_runs(page)
    assert first.action_sources is not None and first.action_sources.complete
    assert first.visual_assets[0].named('first icon') is not None
    summary = recall(named('a program read', page.program_url))
    assert summary['reader'] == programs.READER
    assert programs._from_kept(summary['read']).action_sources is None
    if changed:
        page.data = _swf('different icon')
    current = await programs.read_what_the_page_runs(page)
    assert page.fetches == 2
    assert current.action_sources is not None and current.action_sources.complete
    assert current.visual_assets[0].source == current.action_sources.source
    if changed:
        assert current.action_sources.source != first.action_sources.source
        assert current.visual_assets[0].named('first icon') is None
        assert current.visual_assets[0].named('different icon') is not None
    else:
        assert current.action_sources == first.action_sources
        assert current.visual_assets == first.visual_assets


@pytest.mark.asyncio
@pytest.mark.parametrize('refused', [None, '!not base64!', True, RuntimeError('fetch refused')])
async def test_failed_acquisition_cannot_fall_back_to_a_saved_program(refused):
    page = _Page(_swf('icon'))
    assert await programs.read_what_the_page_runs(page)
    page.data = refused
    current = await programs.read_what_the_page_runs(page)
    assert not current and current.action_sources is None and current.visual_assets == ()
    assert page.fetches == 2


@pytest.mark.asyncio
async def test_a_new_script_at_the_same_address_replaces_the_old_read_controls():
    page = _Page(b'if(e.key === "ArrowLeft") actor.x -= 2;')
    page.program_url = 'https://example.org/current-program.js'
    first = await programs.read_what_the_page_runs(page)
    assert set(first.keys) == {'left'}
    page.data = b'if(e.key === "ArrowRight") actor.x += 5;'
    current = await programs.read_what_the_page_runs(page)
    assert set(current.keys) == {'right'} and page.fetches == 2


@pytest.mark.asyncio
async def test_an_inline_world_is_read_from_current_scripts_even_at_a_visited_address():
    folder = Path(__file__).parent/'fixtures/worlds_that_move'
    source = re.findall(r'<script>([\s\S]*?)</script>', (folder/'shooter.html').read_text())[0]
    engine = (folder/'world.js').read_text()
    page = _Page(None)
    page.program_url = ''
    page.inline = [engine, source]
    first = await programs.read_what_the_page_runs(page)
    assert first.world.hers.paces['right'] == (240., 0.)
    page.inline = [engine, source.replace('240 * dt', '135 * dt')]
    current = await programs.read_what_the_page_runs(page)
    assert current.world.hers.paces['right'] == (135., 0.)
    page.inline_unavailable = True
    assert not await programs.read_what_the_page_runs(page)


def test_byte_admission_is_strict_bounded_and_registered(monkeypatch):
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert programs._current_program_bytes_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'current_programs_need_current_acquired_bytes')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
    monkeypatch.setattr(programs, 'MOST_BYTES', 4)
    assert programs._program_bytes(base64.b64encode(b'abcd').decode()) == b'abcd'
    assert programs._program_bytes(base64.b64encode(b'abcde').decode()) is None
    assert programs._program_bytes('AAAA\n') is None
    assert programs._program_bytes('☺') is None
