"""A software renderer request never retries with host graphics."""
from types import SimpleNamespace

import pytest

from core.capabilities.browser_graphics_policy import (
    _software_graphics_invariant,
    graphics_arguments,
    software_graphics_verified,
)
from core.capabilities.phantom_browser import PhantomBrowser

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('platform', ['darwin', 'linux', 'win32'])
def test_software_policy_selects_the_cpu_driver_on_each_host(platform):
    assert graphics_arguments('software', platform) == ['--use-gl=angle', '--use-angle=swiftshader']
    assert '--enable-unsafe-swiftshader' not in graphics_arguments('software', platform)


@pytest.mark.parametrize('graphics', [None, {}, {'renderer': None}, {'renderer': 'ANGLE Metal Renderer: Apple'}])
def test_missing_or_host_renderer_reports_do_not_prove_software(graphics):
    assert not software_graphics_verified(graphics)


@pytest.mark.parametrize('backend,engine', [('unknown', 'chromium'), ('software', 'firefox'), ('software', 'webkit')])
def test_unsupported_policies_fail_before_starting_a_driver(backend, engine):
    with pytest.raises(ValueError):
        PhantomBrowser(graphics_backend=backend, browser_type=engine)


class _Session:
    def __init__(self, renderer):
        self.renderer, self.detached = renderer, False

    async def send(self, method):
        assert method == 'SystemInfo.getInfo'
        return {'gpu': {'auxAttributes': {'glRenderer': self.renderer}, 'featureStatus': {}}}

    async def detach(self):
        self.detached = True


class _Browser:
    def __init__(self, renderer):
        self.session, self.closed = _Session(renderer), False

    async def new_browser_cdp_session(self):
        return self.session

    async def close(self):
        self.closed = True


def _launcher(monkeypatch, renderer='ANGLE (Google, SwiftShader Device (Subzero))', fail=False):
    launched, opened = [], []

    async def launch(**kwargs):
        launched.append(kwargs)
        if fail:
            raise RuntimeError('driver unavailable')
        opened.append(_Browser(renderer))
        return opened[-1]

    browser = PhantomBrowser(graphics_backend='software')
    browser.playwright = SimpleNamespace(chromium=SimpleNamespace(launch=launch, executable_path='retained-browser'))
    monkeypatch.setattr(browser, '_chromium_launch_candidates', lambda: [('bundled', None)])
    return browser, launched, opened


@pytest.mark.asyncio
async def test_requested_driver_is_measured_before_the_browser_is_handed_back(monkeypatch):
    browser, launched, opened = _launcher(monkeypatch)
    loaded, label = await browser._launch_chromium()
    assert loaded is opened[0] and label == 'bundled'
    assert opened[0].session.detached and not opened[0].closed
    assert software_graphics_verified(browser.get_status()['graphics'])
    assert browser.get_status()['graphics_backend_requested'] == 'software'
    assert '--use-angle=swiftshader' in launched[0]['args']
    assert '--use-angle=metal' not in launched[0]['args']


@pytest.mark.asyncio
@pytest.mark.parametrize('renderer', [None, 'ANGLE Metal Renderer: Apple'])
async def test_unverified_driver_is_closed_without_retrying_host_graphics(monkeypatch, renderer):
    browser, launched, opened = _launcher(monkeypatch, renderer=renderer)
    with pytest.raises(RuntimeError, match='not verified'):
        await browser._launch_chromium()
    assert len(launched) == 1 and opened[0].closed
    assert '--use-angle=swiftshader' in launched[0]['args']


@pytest.mark.asyncio
async def test_failed_software_launch_never_gets_a_default_renderer_attempt(monkeypatch):
    browser, launched, _ = _launcher(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='all Chromium'):
        await browser._launch_chromium()
    assert len(launched) == 1 and '--use-angle=swiftshader' in launched[0]['args']


def test_the_software_policy_invariant_checks_the_report_and_arguments():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _software_graphics_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'software_graphics_need_a_measured_cpu_renderer')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.external
@pytest.mark.asyncio
async def test_opted_in_cpu_browser_delivers_keys_to_captured_pixels(tmp_path):
    import asyncio
    import io
    import json
    import os
    from pathlib import Path

    import numpy as np
    from PIL import Image

    if os.environ.get('AURA_RUN_CPU_BROWSER') != '1':
        pytest.skip('explicit CPU browser test was not requested')
    browser = PhantomBrowser(graphics_backend='software', principal='owner')
    try:
        assert await browser.ensure_ready(), browser.get_status()
        await browser.page.set_viewport_size({'width': 160, 'height': 96})
        webgl = await browser.page.evaluate('''() => {
          const c=document.createElement('canvas'), g=c.getContext('webgl2');
          if(!g)return {available:false};g.clearColor(.125,.25,.5,1);g.clear(g.COLOR_BUFFER_BIT);
          const p=new Uint8Array(4);g.readPixels(0,0,1,1,g.RGBA,g.UNSIGNED_BYTE,p);
          return {available:true,rgba:Array.from(p)};}''')
        assert webgl == {'available': True, 'rgba': [32, 64, 128, 255]}
        await browser.page.set_content('''<style>body{margin:0}</style><canvas width="160" height="96"></canvas>
        <script>const c=document.querySelector('canvas'),g=c.getContext('2d');let x=20,held=false;
        onkeydown=e=>{if(e.key==='ArrowRight')held=true};onkeyup=e=>{if(e.key==='ArrowRight')held=false};
        function frame(){if(held)x=Math.min(140,x+1);g.fillStyle='black';g.fillRect(0,0,160,96);
          g.fillStyle='#f02030';g.fillRect(x,40,8,8);requestAnimationFrame(frame)}frame();</script>''')

        async def picture():
            data = await asyncio.wait_for(browser.page.screenshot(), 3.)
            pixels = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            ys, xs = np.nonzero((pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] < 80))
            assert len(xs) == 64
            return float(xs.mean()), data

        before, initial = await picture()
        await browser.page.keyboard.down('ArrowRight')
        await asyncio.sleep(.15)
        await browser.page.keyboard.up('ArrowRight')
        after, final = await picture()
        await asyncio.sleep(.1)
        stopped, _ = await picture()
        assert after > before + 2 and stopped == after
        record = {'scope': 'synthetic Canvas keyboard and screenshot, not original gameplay',
                  'graphics': browser.get_status()['graphics'], 'webgl2': webgl,
                  'before_x': before, 'after_x': after, 'stopped_x': stopped}
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        folder = Path(os.environ['AURA_CPU_BROWSER_EVIDENCE']) if os.environ.get('AURA_CPU_BROWSER_EVIDENCE') else tmp_path
        with local_internal_governed_scope('cpu_browser_verification'):
            gateway = get_file_write_gateway()
            await gateway.ensure_directory_async(folder, source='cpu_browser_verification')
            await gateway.write_text_async(folder / 'pixels.json', json.dumps(record, indent=2), source='cpu_browser_verification')
            await gateway.write_bytes_async(folder / 'before.png', initial, source='cpu_browser_verification')
            await gateway.write_bytes_async(folder / 'after.png', final, source='cpu_browser_verification')
    finally:
        await asyncio.wait_for(browser.close(), 15.)
