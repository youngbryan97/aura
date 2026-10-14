"""Primitive assembly retains winding, restart, ranges and resource custody."""
import struct
from dataclasses import replace

import pytest

from core.perception.shader_uniforms_from_buffer_bytes import BufferRange
from core.perception.triangles_from_public_draws import (
    ElementBinding,
    _triangle_assembly_invariant,
    triangle_draw,
)


@pytest.mark.parametrize("mode,expected", [
    (4, (10, 11, 12)), (5, (10, 11, 12, 12, 11, 13)), (6, (10, 11, 12, 10, 12, 13)),
])
def test_array_primitives_keep_first_and_winding(mode, expected):
    assert triangle_draw(mode, 4, first=10).indices == expected


@pytest.mark.parametrize("code,width,component_type", [("B", 1, 5121), ("H", 2, 5123), ("I", 4, 5125)])
@pytest.mark.parametrize("order", ["little", "big"])
def test_fixed_restart_resets_strip_winding_and_discards_partial_triangles(code, width, component_type, order):
    marker = 2 ** (8 * width) - 1
    raw = struct.pack(("<" if order == "little" else ">") + code * 8, 0, 1, 2, 3, marker, 4, 5, 6)
    data = BufferRange(1, 2, 3, 4, 16, raw, order)
    binding = ElementBinding(1, 2, 3, 4, component_type, 16)
    assert triangle_draw(5, 8, binding=binding, data=data).indices == (0, 1, 2, 2, 1, 3, 4, 5, 6)
    assert triangle_draw(6, 8, binding=binding, data=data).indices == (0, 1, 2, 0, 2, 3, 4, 5, 6)
    assert triangle_draw(4, 8, binding=binding, data=data).indices == (0, 1, 2, 4, 5, 6)
    assert marker in triangle_draw(5, 8, binding=binding, data=data, fixed_restart=False).indices


@pytest.mark.parametrize("change", [{"context": 7}, {"epoch": 3}, {"buffer": 8}, {"content": 5},
                                     {"component_type": 5122}, {"offset": -1}])
def test_stale_and_unsupported_element_bindings_stay_unknown(change):
    data = BufferRange(1, 2, 3, 4, 0, bytes((0, 1, 2)), "little")
    binding = ElementBinding(1, 2, 3, 4, 5121, 0)
    assert triangle_draw(4, 3, binding=replace(binding, **change), data=data).indices is None


def test_empty_degenerate_incomplete_and_uncaptured_draws_are_distinguished():
    assert triangle_draw(4, 0).indices == ()
    assert triangle_draw(4, 2).indices == ()
    data = BufferRange(1, 2, 3, 4, 0, bytes((0, 0, 1)), "little")
    binding = ElementBinding(1, 2, 3, 4, 5121, 0)
    assert triangle_draw(4, 3, binding=binding, data=data).indices == ()
    assert triangle_draw(4, 4, binding=binding, data=data).reason == "uncaptured_element_bytes"
    assert triangle_draw(4, 4097).reason == "draw_mode_or_budget"
    assert triangle_draw(1, 3).indices is None
    assert triangle_draw(4, True).indices is None


_GPU_ASSEMBLY = r"""({mode,indices,triangles}) => {
 const canvas=document.createElement('canvas');canvas.width=32;canvas.height=32;
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});if(!gl)return null;
 const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);return s;};
 const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,'#version 300 es\nin vec2 position;void main(){gl_Position=vec4(position,0.0,1.0);}'));
 gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1.0);}'));
 gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
 const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-0.8,-0.8,0.8,-0.8,-0.8,0.8,0.8,0.8,-0.4,-0.4,0.4,-0.4,0.0,0.4]),gl.STATIC_DRAW);
 const location=gl.getAttribLocation(p,'position');gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,2,gl.FLOAT,false,0,0);
 gl.enable(gl.CULL_FACE);gl.cullFace(gl.BACK);gl.frontFace(gl.CCW);gl.viewport(0,0,32,32);gl.clearColor(0,0,0,1);
 const elements=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,elements);
 const draw=(kind,ids)=>{gl.clear(gl.COLOR_BUFFER_BIT);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint8Array(ids),gl.STATIC_DRAW);
  gl.drawElements(kind,ids.length,gl.UNSIGNED_BYTE,0);const image=new Uint8Array(32*32*4);gl.readPixels(0,0,32,32,gl.RGBA,gl.UNSIGNED_BYTE,image);return Array.from(image);};
 const actual=draw(mode,indices),expanded=draw(gl.TRIANGLES,triangles);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic assembly operation failed');return {actual,expanded};
}"""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [4, 5, 6])
async def test_assembled_primitives_match_independent_gpu_rasterization_with_culling_and_restart(mode):
    from playwright.async_api import async_playwright

    indices = (0, 1, 2, 3, 255, 4, 5, 6)
    data = BufferRange(1, 2, 3, 4, 0, bytes(indices), "little")
    binding = ElementBinding(1, 2, 3, 4, 5121, 0)
    triangles = triangle_draw(mode, len(indices), binding=binding, data=data).indices
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            result = await page.evaluate(_GPU_ASSEMBLY, dict(mode=mode, indices=indices, triangles=triangles))
            assert result is not None, "WebGL2 probe unavailable; primitive assembly remains unmeasured"
        finally:
            await browser.close()
    assert result["actual"] == result["expanded"]
    assert any(result["actual"][i] == 255 for i in range(0, len(result["actual"]), 4))


def test_assembly_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _triangle_assembly_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "draw_triangles_keep_assembly_and_versions" for c in suite.claims())
