"""A hover can expose a new notice between the hit test and the physical click."""
import os
import time

import pytest

from core.skills import sovereign_browser_drawing as drawing


def test_unmeasured_or_untyped_delivery_cannot_become_a_receiver_receipt():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert drawing._cover_press_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'cover_press_keeps_its_actual_receiver')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.external
@pytest.mark.asyncio
@pytest.mark.parametrize('notice_on_hover', [False, True])
async def test_normal_startup_rechecks_an_intercepted_press_without_pressing_the_started_canvas(monkeypatch, notice_on_hover):
    if os.environ.get('AURA_RUN_CPU_BROWSER') != '1':
        pytest.skip('explicit CPU browser test was not requested')
    from core.capabilities.phantom_browser import PhantomBrowser

    browser = PhantomBrowser(graphics_backend='software', principal='owner')
    monkeypatch.setattr(drawing, '_tell', lambda _said: None)
    try:
        assert await browser.ensure_ready(), browser.get_status()
        page = browser.page
        await page.set_viewport_size({'width': 320, 'height': 200})
        await page.set_content('''<style>body{margin:0}#host{display:block;width:320px;height:200px}</style>
        <div id="host"></div><script>
        const s=document.querySelector('#host').attachShadow({mode:'open'});
        s.innerHTML=`<style>canvas{position:absolute;inset:0}button{position:absolute;inset:0;cursor:pointer}
          #notice{position:absolute;inset:0;background:white;z-index:9}</style>
          <canvas width="320" height="200"></canvas><button aria-label="Start">Start</button>`;
        window.startPresses=0;window.gamePresses=0;window.noticePresses=0;window.trusted=[];
        s.querySelector('canvas').onclick=()=>window.gamePresses++;
        s.querySelector('button').onclick=e=>{window.startPresses++;window.trusted.push(e.isTrusted);
          setTimeout(()=>e.target.remove(),350)};
        </script>''')
        if notice_on_hover:
            await page.evaluate('''()=>{
              const s=document.querySelector('#host').shadowRoot;
              s.querySelector('button').addEventListener('pointerenter',()=>{
                const notice=document.createElement('div');notice.id='notice';s.appendChild(notice);
                notice.onclick=e=>{window.noticePresses++;window.trusted.push(e.isTrusted);e.target.remove()};
              },{once:true});
            }''')
        began = time.monotonic()
        band = await drawing._start_what_is_covered(page, (0., 0., 1., 1.))
        elapsed = time.monotonic()-began
        assert band == (0., 0., 1., 1.)
        # The first actual receiver is the notice, followed by exactly one
        # start control. The delayed response must never receive a second press.
        assert await page.evaluate('[startPresses,gamePresses,noticePresses]') == [1, 0, int(notice_on_hover)]
        assert all(await page.evaluate('trusted'))
        assert elapsed < 6., elapsed
    finally:
        await browser.close()


@pytest.mark.external
@pytest.mark.asyncio
async def test_synthetic_events_cannot_claim_that_the_physical_press_reached_the_control():
    if os.environ.get('AURA_RUN_CPU_BROWSER') != '1':
        pytest.skip('explicit CPU browser test was not requested')
    from core.capabilities.phantom_browser import PhantomBrowser

    browser = PhantomBrowser(graphics_backend='software', principal='owner')
    try:
        assert await browser.ensure_ready(), browser.get_status()
        page = browser.page
        await page.set_content('<button style="width:200px;height:100px">Start</button>')
        state = await page.evaluate(drawing._COVER_PRESS % (50., 50., 'true'))
        assert state['description'] and state['received'] is None
        await page.evaluate('''()=>{const button=document.querySelector('button');
          for(const kind of ['mousedown','click'])button.dispatchEvent(new MouseEvent(kind,
            {bubbles:true,composed:true,button:0,clientX:50,clientY:50}));}''')
        assert (await page.evaluate(drawing._COVER_PRESS % (50., 50., 'false')))['received'] is None
        await page.mouse.click(50., 50.)
        assert (await page.evaluate(drawing._COVER_PRESS % (50., 50., 'false')))['received'] is True
    finally:
        await browser.close()
