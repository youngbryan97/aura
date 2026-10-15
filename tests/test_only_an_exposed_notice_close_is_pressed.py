"""Notice controls need intact JavaScript words and the current pointer hit target."""
import os

import pytest

from core.skills.sovereign_browser_drawing import _CLOSES_OVER_IT

pytestmark = pytest.mark.unit


def test_javascript_word_boundaries_reach_the_browser_as_regex_boundaries():
    assert '\x08' not in _CLOSES_OVER_IT
    assert r'/\b(close|dismiss)\b/' in _CLOSES_OVER_IT
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.skills.sovereign_browser_drawing import _notice_word_boundaries_invariant

    assert _notice_word_boundaries_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'notice_controls_keep_javascript_word_boundaries')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('label', ['BTART', 'P1AY', 'Nex'])
def test_a_current_navigation_reading_also_reaches_the_fast_play_menu_gate(label):
    from core.skills.screen_pursuit_as_it_happens import WAYS_ON_TRIED, PlayingAsItHappens

    playing = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal='', ends_at=0.)
    playing.read({'text': f'GENERIC TITLE {label}', 'layout': [{'text': label}]})
    assert playing.way_on_shown
    assert all(playing._a_menu_first() for _ in range(WAYS_ON_TRIED))
    assert not playing._a_menu_first()  # retain the existing finite navigation allowance


@pytest.mark.external
@pytest.mark.asyncio
async def test_opted_in_cpu_browser_closes_only_the_current_exposed_notice(tmp_path):
    if os.environ.get('AURA_RUN_CPU_BROWSER') != '1':
        pytest.skip('explicit CPU browser test was not requested')
    import asyncio
    import json

    from core.capabilities.phantom_browser import PhantomBrowser
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    browser = PhantomBrowser(graphics_backend='software', principal='owner')
    try:
        assert await browser.ensure_ready(), browser.get_status()
        await browser.page.set_viewport_size({'width': 320, 'height': 200})
        await browser.page.set_content('''<style>body{margin:0}#host{display:block;width:320px;height:200px}</style>
        <div id="host"></div><script>
        const s=document.querySelector('#host').attachShadow({mode:'open'});
        s.innerHTML=`<style>canvas{position:absolute;inset:0}button{position:absolute;left:260px;top:10px;width:40px;height:40px}
          #block{position:absolute;inset:0;background:black;z-index:99}</style>
          <canvas width="320" height="200"></canvas><button aria-label="Dismiss notice"><svg width="20" height="20"></svg></button>`;
        s.querySelector('button').onclick=e=>{window.dismissed=(window.dismissed||0)+1;e.currentTarget.remove()};
        </script>''')
        query = _CLOSES_OVER_IT % (0., 0., 320., 200.)
        baseline = query.replace(r'\b', '\x08')
        assert json.loads(await browser.page.evaluate(baseline)) == []
        exposed = json.loads(await browser.page.evaluate(query))
        assert exposed == [[280., 30.]]
        await browser.page.evaluate('''() => {const s=document.querySelector('#host').shadowRoot;
          const block=document.createElement('div');block.id='block';s.appendChild(block)}''')
        assert json.loads(await browser.page.evaluate(query)) == []
        await browser.page.evaluate("document.querySelector('#host').shadowRoot.querySelector('#block').remove()")
        await browser.page.mouse.click(*exposed[0])
        assert await browser.page.evaluate('window.dismissed') == 1
        assert json.loads(await browser.page.evaluate(query)) == []
        with local_internal_governed_scope('exposed_notice_verification'):
            gateway = get_file_write_gateway()
            await gateway.write_text_async(tmp_path / 'notice.json', json.dumps(dict(
                scope='generated DOM notice in CPU browser; no original-game outcome',
                graphics=browser.get_status()['graphics'], original_boundary_result=[], exposed=exposed,
                covered_result=[], dismissed=1), indent=2), source='exposed_notice_verification')
    finally:
        await asyncio.wait_for(browser.close(), 15.)
