"""Public copy provenance maps search geometry without inventing visibility."""
from dataclasses import FrozenInstanceError, replace

import pytest

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.perception.graphics_presented_on_canvas import (
    FramebufferCopy,
    ImageVersion,
    PresentedCanvas,
    _presentation_custody_invariant,
    presented_hints,
    project_triangles,
)


def _frame(epoch=2):
    return MeasuredFrameTime(3., lower=3., upper=3.01, arrived=3.02,
                             source="browser-monotonic", source_at=30., epoch=epoch)


def _image(image, content=1, width=100, height=80):
    return ImageVersion(1, 2, image, content, width, height)


POLYGON = (((10., 20.), (20., 20.), (10., 30.)),)


def test_direct_canvas_coordinates_flip_only_the_framebuffer_origin():
    image = _image(0)
    result = presented_hints(POLYGON, image, 2, (), PresentedCanvas(image, 2, _frame()))
    assert result.reason is None
    assert result.hints.polygons == (((10., 60.), (20., 60.), (10., 50.)),)
    assert result.hints.coverage == "unverified"
    with pytest.raises(FrozenInstanceError):
        result.hints.coverage = "visible"


@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (False, True), (True, False), (True, True)])
def test_public_copy_endpoint_order_determines_each_flip(flip_x, flip_y):
    source, target = _image(7), _image(0)
    rect = (100. if flip_x else 0., 80. if flip_y else 0.,
            0. if flip_x else 100., 0. if flip_y else 80.)
    copy = FramebufferCopy(source, target, rect, (0., 0., 100., 80.), 3)
    result = presented_hints(POLYGON, source, 2, (copy,), PresentedCanvas(target, 4, _frame()))
    expected = tuple(tuple((100 - x if flip_x else x, y if flip_y else 80 - y) for x, y in p) for p in POLYGON)
    assert result.hints.polygons == expected
    assert result.hints.copies == (3,)


def test_two_copies_retain_scaling_crop_and_scissor():
    source, middle, target = _image(7), _image(8, width=200, height=160), _image(0)
    first = FramebufferCopy(source, middle, (0., 0., 100., 80.), (0., 0., 200., 160.), 3)
    second = FramebufferCopy(middle, target, (0., 160., 200., 0.), (0., 0., 100., 80.), 5,
                             scissor=(10., 50., 15., 60.))
    # The flip maps original y20..30 to destination y60..50. Scissor cuts x15.
    result = presented_hints(POLYGON, source, 2, (first, second), PresentedCanvas(target, 6, _frame()))
    assert result.reason is None and result.hints.copies == (3, 5)
    points = result.hints.polygons[0]
    assert all(10 <= x <= 15 and 20 <= y <= 30 for x, y in points)
    assert (15., 20.) in points and (15., 25.) in points


@pytest.mark.parametrize("mutation", [
    {"color_only": False}, {"nearest": False}, {"sequence": 2}, {"sequence": 11},
    {"source_rect": (0., 0., 0., 80.)}, {"destination_rect": (0., 0., float("nan"), 80.)},
    {"source": _image(7, content=2)}, {"source": replace(_image(7), epoch=3)},
    {"source": replace(_image(7), context=2)}, {"destination": _image(0, content=2)},
])
def test_missing_stale_or_unsupported_copy_cannot_promote_geometry(mutation):
    source, target = _image(7), _image(0)
    copy = FramebufferCopy(source, target, (0., 0., 100., 80.), (0., 0., 100., 80.), 3)
    result = presented_hints(POLYGON, source, 2, (replace(copy, **mutation),), PresentedCanvas(target, 10, _frame()))
    assert result.hints is None


def test_attachment_reuse_and_multiple_routes_remain_unknown():
    source, target = _image(7), _image(0)
    copy = FramebufferCopy(source, target, (0., 0., 100., 80.), (0., 0., 100., 80.), 3)
    newer = PresentedCanvas(replace(target, content=2), 5, _frame())
    assert presented_hints(POLYGON, source, 2, (copy,), newer).hints is None
    duplicate = replace(copy, sequence=4)
    assert presented_hints(POLYGON, source, 2, (copy, duplicate), PresentedCanvas(target, 5, _frame())).hints is None


@pytest.mark.parametrize("clock", [
    3., _frame(epoch=1),
    MeasuredFrameTime(3., lower=None, upper=3., arrived=3.02, source="receipt", source_at=None, epoch=2),
    MeasuredFrameTime(3., lower=3., upper=3.1, arrived=3.02, source="browser", source_at=30., epoch=2),
])
def test_receipt_or_wrong_epoch_cannot_link_geometry_to_current_pixels(clock):
    source = _image(0)
    assert presented_hints(POLYGON, source, 2, (), PresentedCanvas(source, 2, clock)).hints is None


def test_homogeneous_clipping_precedes_division_and_viewport_mapping():
    vertices = ((-4., -2., 0., 2.), (2., -2., 0., 2.), (2., 2., 0., 2.))
    [polygon] = project_triangles(vertices, (0, 1, 2), (10., 20., 100., 80.))
    assert len(polygon) == 4
    assert all(10 <= x <= 110 and 20 <= y <= 100 for x, y in polygon)
    assert (10., 20.) in polygon and (110., 100.) in polygon
    scaled = tuple(tuple(v * 3 for v in p) for p in vertices)
    assert project_triangles(scaled, (0, 1, 2), (10., 20., 100., 80.)) == (polygon,)


def test_fully_clipped_geometry_is_measured_empty_while_singular_or_invalid_is_unknown():
    behind = ((-1., -1., 0., -2.), (1., -1., 0., -2.), (0., 1., 0., -2.))
    assert project_triangles(behind, (0, 1, 2), (0., 0., 100., 80.)) == ()
    singular = ((0., 0., 0., 0.), (1., 0., 0., 1.), (0., 1., 0., 1.))
    assert project_triangles(singular, (0, 1, 2), (0., 0., 100., 80.)) is None
    assert project_triangles(singular, (0, 1, 3), (0., 0., 100., 80.)) is None
    assert project_triangles(singular, (0, 1, 2), (0., 0., -100., 80.)) is None


def test_resource_and_hop_budgets_abstain():
    source, target = _image(7), _image(0)
    copy = FramebufferCopy(source, target, (0., 0., 100., 80.), (0., 0., 100., 80.), 3)
    assert presented_hints(POLYGON, source, 2, (copy,) * 65, PresentedCanvas(target, 99, _frame())).reason == "copy_budget"
    images = tuple(_image(i) for i in range(18))
    edges = tuple(FramebufferCopy(a, b, (0., 0., 100., 80.), (0., 0., 100., 80.), i + 2)
                  for i, (a, b) in enumerate(zip(images[:-1], images[1:], strict=True)))
    result = presented_hints(POLYGON, images[0], 1, edges, PresentedCanvas(images[-1], 30, _frame()))
    assert result.reason == "presentation_hop_budget"


def test_presentation_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _presentation_custody_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "graphics_hints_require_current_presentation" for c in suite.claims())


_GPU_COPY = """({vertices,sourceRect,destinationRect,scissor}) => {
 const canvas=document.querySelector('canvas');
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
 if(!gl)return null;
 const shader=(type,source)=>{
  const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));
  return s;
 };
 const p=gl.createProgram();
 gl.attachShader(p,shader(gl.VERTEX_SHADER,`#version 300 es
 layout(location=0) in vec4 position;void main(){gl_Position=position;}`));
 gl.attachShader(p,shader(gl.FRAGMENT_SHADER,`#version 300 es
 precision highp float;out vec4 pixel;void main(){pixel=vec4(1,0,0,1);}`));
 gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));
 const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
 gl.texStorage2D(gl.TEXTURE_2D,1,gl.RGBA8,128,96);
 const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);
 gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,texture,0);
 if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('incomplete');
 gl.disable(gl.SCISSOR_TEST);gl.viewport(0,0,128,96);gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT);
 gl.useProgram(p);const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
 gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices.flat()),gl.STATIC_DRAW);
 gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,4,gl.FLOAT,false,0,0);
 gl.drawArrays(gl.TRIANGLES,0,3);
 gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);gl.clear(gl.COLOR_BUFFER_BIT);
 if(scissor){gl.enable(gl.SCISSOR_TEST);gl.scissor(scissor[0],scissor[1],scissor[2]-scissor[0],scissor[3]-scissor[1]);}
 gl.blitFramebuffer(...sourceRect,...destinationRect,gl.COLOR_BUFFER_BIT,gl.NEAREST);
 gl.bindFramebuffer(gl.READ_FRAMEBUFFER,null);
 const data=new Uint8Array(128*96*4);gl.readPixels(0,0,128,96,gl.RGBA,gl.UNSIGNED_BYTE,data);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic graphics operation failed');
 const red=Array.from({length:128*96},(_,i)=>data[i*4]);
 gl.disable(gl.SCISSOR_TEST);gl.deleteBuffer(b);gl.deleteFramebuffer(f);gl.deleteTexture(texture);gl.deleteProgram(p);
 return red;
}"""


def _inside_and_distance(point, polygons):
    import math

    inside, distance = False, math.inf
    for polygon in polygons:
        signs = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            dx, dy = b[0] - a[0], b[1] - a[1]
            signs.append(dx * (point[1] - a[1]) - dy * (point[0] - a[0]))
            norm = dx * dx + dy * dy
            t = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / norm)) if norm else 0.
            distance = min(distance, math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy))
        inside |= all(v >= 0 for v in signs) or all(v <= 0 for v in signs)
    return inside, distance


@pytest.mark.asyncio
async def test_copy_mapping_matches_independent_gpu_pixels_across_flips_crop_and_clipping():
    from playwright.async_api import async_playwright

    vertices = ((-2.4, -1.6, 0., 2.), (1.6, -1.2, 0., 2.), (.4, 1.8, 0., 2.))
    polygons = project_triangles(vertices, (0, 1, 2), (0., 0., 128., 96.))
    source, target = _image(7, width=128, height=96), _image(0, width=128, height=96)
    checked, mismatches = 0, 0
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content('<canvas width="128" height="96"></canvas>')
            for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
                for crop in (False, True):
                    rect = (128. if flip_x else 0., 96. if flip_y else 0.,
                            0. if flip_x else 128., 0. if flip_y else 96.)
                    destination = (8., 12., 112., 88.) if crop else (0., 0., 128., 96.)
                    scissor = (24., 16., 104., 80.) if crop else None
                    copy = FramebufferCopy(source, target, rect, destination, 3, scissor=scissor)
                    hints = presented_hints(polygons, source, 2, (copy,), PresentedCanvas(target, 4, _frame())).hints
                    red = await page.evaluate(_GPU_COPY, dict(vertices=vertices, sourceRect=rect,
                                                              destinationRect=destination, scissor=scissor))
                    assert red is not None, "WebGL2 probe unavailable; mapping remains unmeasured"
                    for y in range(96):
                        for x in range(128):
                            inside, distance = _inside_and_distance((x + .5, y + .5), hints.polygons)
                            if distance <= 1.25:
                                continue  # nearest resampling and raster edge ownership span a pixel
                            checked += 1
                            mismatches += inside != (red[(95 - y) * 128 + x] > 0)
        finally:
            await browser.close()
    assert checked > 80_000 and mismatches == 0, (checked, mismatches)
