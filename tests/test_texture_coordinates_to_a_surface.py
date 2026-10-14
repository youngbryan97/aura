"""Perspective texture-coordinate mapping is compared with GPU sampling."""
from dataclasses import replace

import numpy as np
import pytest

from core.perception.graphics_presented_on_canvas import ImageVersion
from core.perception.texture_coordinates_to_a_surface import (
    TextureProjection,
    _texture_projection_invariant,
)


def _projection():
    return TextureProjection(ImageVersion(1, 2, 3, 4, 64, 48), ImageVersion(1, 2, 5, 6, 80, 64),
                             ((-1., -1., 0., 1.), (1., -1., 0., 1.), (-1., 1., 0., 1.)),
                             ((0., 0.), (1., 0.), (0., 1.)), (0, 1, 2), (0., 0., 80., 64.), 7,
                             smooth=True, base_level_nearest=True)


REGION = (((10., 8.), (30., 8.), (30., 25.), (10., 25.)),)


@pytest.mark.parametrize("change", [
    {"smooth": False}, {"base_level_nearest": False}, {"indices": (0, 1, 3)},
    {"positions": ((0., 0., 0., float("nan")),) * 3}, {"coordinates": ((0., 0.),) * 3},
    {"coordinates": ((0., 0.), (2., 0.), (0., 1.))}, {"viewport": (0., 0., 0., 64.)},
    {"source": ImageVersion(1, 3, 3, 4, 64, 48)},
])
def test_missing_interpolation_sampler_or_geometry_provenance_stays_unknown(change):
    assert replace(_projection(), **change).mapped(REGION) is None


def test_a_source_region_outside_the_texture_triangle_is_measured_empty():
    assert _projection().mapped((((60., 45.), (63., 45.), (60., 47.)),)) == ()
    assert _texture_projection_invariant() == ()


def test_texture_copy_then_blit_reaches_the_current_canvas_with_unverified_coverage():
    from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
    from core.perception.graphics_presented_on_canvas import (
        FramebufferCopy,
        PresentedCanvas,
        presented_hints,
    )

    projection = _projection()
    target = ImageVersion(1, 2, 0, 8, 80, 64)
    blit = FramebufferCopy(projection.destination, target, (0., 64., 80., 0.), (0., 0., 80., 64.), 9)
    time = MeasuredFrameTime(1., lower=1., upper=1., arrived=1.01, source="browser", source_at=10., epoch=2)
    result = presented_hints(REGION, projection.source, 6, (projection, blit), PresentedCanvas(target, 10, time))
    assert result.reason is None and result.hints.coverage == "unverified"
    assert result.hints.copies == (7, 9)


_GPU_PROJECTION = r"""({positions,coordinates,pixels,scissor}) => {
 const canvas=document.createElement('canvas');canvas.width=80;canvas.height=64;
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});if(!gl)return null;
 const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
 const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,'#version 300 es\nin vec4 position;in vec2 coordinate;out vec2 where;void main(){gl_Position=position;where=coordinate;}'));
 gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;uniform sampler2D image;in vec2 where;out vec4 color;void main(){color=texture(image,where);}'));
 gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
 for(const [name,values,size] of [['position',positions,4],['coordinate',coordinates,2]]){
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values.flat()),gl.STATIC_DRAW);
  const location=gl.getAttribLocation(p,name);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,0,0);
 }
 const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,64,48,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(pixels));
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
 gl.uniform1i(gl.getUniformLocation(p,'image'),0);gl.viewport(0,0,80,64);gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT);
 if(scissor){gl.enable(gl.SCISSOR_TEST);gl.scissor(scissor[0],scissor[1],scissor[2]-scissor[0],scissor[3]-scissor[1]);}
 gl.drawArrays(gl.TRIANGLES,0,3);const result=new Uint8Array(80*64*4);gl.readPixels(0,0,80,64,gl.RGBA,gl.UNSIGNED_BYTE,result);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic texture projection failed');return Array.from(result);
}"""


@pytest.mark.asyncio
@pytest.mark.parametrize("positions,scissor", [
    (((-1., -1., 0., 1.), (1., -1., 0., 1.), (-1., 1., 0., 1.)), None),
    (((-1., -1., 0., 1.), (2., -2., 0., 2.), (-3., 3., 0., 3.)), None),
    (((-1.5, -1.2, 0., 1.), (2., -2., 0., 2.), (-3., 3., 0., 3.)), (3., 6., 70., 53.)),
])
async def test_projective_regions_match_gpu_pixels_through_clipping_and_scissor(positions, scissor):
    from playwright.async_api import async_playwright

    from tests.test_graphics_presented_on_canvas import _inside_and_distance

    projection = replace(_projection(), positions=positions, scissor=scissor)
    polygons = projection.mapped(REGION)
    assert polygons is not None
    texture = np.zeros((48, 64, 4), np.uint8)
    texture[:, :, 3] = 255
    texture[8:25, 10:30, 0] = 255
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            measured = await page.evaluate(_GPU_PROJECTION, dict(positions=positions, coordinates=projection.coordinates,
                                                                 pixels=texture.reshape(-1).tolist(), scissor=scissor))
            assert measured is not None, "WebGL2 probe unavailable; projective mapping remains unmeasured"
        finally:
            await browser.close()
    checked, mismatches = 0, 0
    for y in range(64):
        for x in range(80):
            inside, distance = _inside_and_distance((x + .5, y + .5), polygons)
            if distance <= 1.5:
                continue
            checked += 1
            mismatches += inside != (measured[(y * 80 + x) * 4] > 0)
    assert checked > 4000 and mismatches == 0, (checked, mismatches)


def test_projection_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "texture_regions_use_measured_projective_coordinates" for c in suite.claims())
