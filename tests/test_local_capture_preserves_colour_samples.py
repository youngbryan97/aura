"""Local capture must preserve fine colour marks before correspondence sees them."""
import numpy as np
import pytest

from core.perception.frames_as_they_are_drawn import (
    CanvasFrames,
    PageFrames,
    _lossless_frame_invariant,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize('reader', [PageFrames, CanvasFrames])
async def test_real_browser_capture_keeps_fine_colour_marks_and_the_capture_owner(reader):
    api = pytest.importorskip('playwright.async_api')
    async with api.async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        frames = None
        try:
            page = await browser.new_page(viewport={'width': 200, 'height': 128}, device_scale_factor=1)
            await page.set_content('<body style="margin:0"><canvas width="200" height="128"></canvas></body>')
            await page.evaluate('''() => {
                const ctx = document.querySelector('canvas').getContext('2d');
                const image = ctx.createImageData(200,128);
                for (let y=0;y<128;y++) for (let x=0;x<200;x++) {
                    const i=4*(y*200+x);
                    image.data[i]=x%2 ? 255 : 0;
                    image.data[i+1]=(y*13+x*7)%256;
                    image.data[i+2]=y%2 ? 240 : 17;
                    image.data[i+3]=255;
                }
                ctx.putImageData(image,0,0);
            }''')
            frames = reader(page)
            captured = await frames.look(dict(x=0,y=0,width=200,height=128))
            assert captured is not None
            picture, at = captured
            y, x = np.mgrid[:128,:200]
            expected = np.stack((np.where(x%2,255,0), (y*13+x*7)%256, np.where(y%2,240,17)),axis=-1).astype(np.uint8)
            np.testing.assert_array_equal(np.asarray(picture),expected)
            assert at.lower <= at.upper <= at.arrived and at.surface and at.epoch > 0 and at.sequence > 0
            if isinstance(frames, PageFrames):
                assert frames._options['format'] == 'png'
        finally:
            if frames is not None:
                await frames.close()
            await browser.close()


def test_lossless_crop_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _lossless_frame_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'local_frame_encoding_keeps_colour_samples')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
