"""A texture-copy form cannot borrow names or hide color and UV changes."""
import numpy as np
import pytest

from core.perception.a_texture_copied_by_its_shader import (
    TextureCopyForm,
    _texture_copy_invariant,
    texture_copy_form,
)
from core.perception.shader_position_math import position_from_source
from core.perception.shader_position_source import parse_positions


def _source(body="pixel=texture(picture,where);", declarations=""):
    return """#version 300 es
    precision highp float;uniform highp sampler2D picture;in vec2 where;out vec4 pixel;
    DECLARATIONS void main(){BODY}""".replace("BODY", body).replace("DECLARATIONS", declarations)


@pytest.mark.parametrize("name", ["picture", "transform", "protagonist", "opaque_157"])
def test_names_are_read_from_declared_dataflow(name):
    source = _source().replace("picture", name)
    assert texture_copy_form(parse_positions(source)) == TextureCopyForm(name, "where", "pixel")


def test_structure_wrappers_and_identity_constructors_preserve_the_sample():
    source = _source("""Inputs visible=Inputs(gl_FragCoord,where);
     vec4 answer=texture(picture,vec2(visible.coordinates));pixel=vec4(answer);""",
     "struct Inputs{vec4 clip;vec2 coordinates;};")
    assert texture_copy_form(parse_positions(source)) == TextureCopyForm("picture", "where", "pixel")


@pytest.mark.parametrize("body", [
    "pixel=texture(picture,where)*vec4(0.5);",
    "pixel=texture(picture,where+vec2(0.1));",
    "pixel=vec4(texture(picture,where).rgb,1.0);",
    "pixel=clamp(texture(picture,where),vec4(0.0),vec4(1.0));",
    "pixel=texture(picture,where,0.0);",
    "pixel=vec4(where,0.0,1.0);",
    "pixel=texture(picture,where);pixel.yz=vec2(0.0);",
])
def test_color_coordinate_sample_or_later_component_changes_remain_unknown(body):
    assert texture_copy_form(parse_positions(_source(body))) is None


@pytest.mark.parametrize("replacement", ["flat in vec2", "centroid in vec2", "uniform vec2"])
def test_coordinate_interpolation_and_origin_are_not_assumed(replacement):
    assert texture_copy_form(parse_positions(_source().replace("in vec2", replacement))) is None


def test_multiple_fragment_outputs_are_not_assumed_to_be_the_presented_attachment():
    source = _source("pixel=texture(picture,where);second=pixel;", "out vec4 second;")
    assert texture_copy_form(parse_positions(source)) is None


def test_an_unmeasured_vertex_texture_dependency_still_cannot_supply_position():
    source = _source("gl_Position=texture(picture,where);pixel=vec4(1.0);")
    result = position_from_source(source, {"where": (.5, .5), "picture": 0})
    assert result.position is None and result.reason


_GPU_TEXTURE = r"""({source,sampler,coordinates,bytes}) => {
 const canvas=document.createElement('canvas');canvas.width=16;canvas.height=8;
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});if(!gl)return null;
 const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
 const vertex=`#version 300 es
 in vec2 vertex;out vec2 ${coordinates};void main(){gl_Position=vec4(vertex,0.0,1.0);${coordinates}=(vertex+vec2(1.0))*vec2(0.5);}`;
 const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,source));
 gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
 const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
 const location=gl.getAttribLocation(p,'vertex');gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,2,gl.FLOAT,false,0,0);
 const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,16,8,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(bytes));
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
 gl.uniform1i(gl.getUniformLocation(p,sampler),0);gl.viewport(0,0,16,8);gl.drawArrays(gl.TRIANGLES,0,6);
 const result=new Uint8Array(16*8*4);gl.readPixels(0,0,16,8,gl.RGBA,gl.UNSIGNED_BYTE,result);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic texture-copy operation failed');return Array.from(result);
}"""


@pytest.mark.asyncio
async def test_recognized_structured_texture_copy_preserves_independent_gpu_pixels():
    from playwright.async_api import async_playwright

    source = _source("Inputs visible=Inputs(gl_FragCoord,where);pixel=texture(picture,vec2(visible.coordinates));",
                     "struct Inputs{vec4 position;vec2 coordinates;};")
    form = texture_copy_form(parse_positions(source))
    assert form is not None
    pixels = np.random.default_rng(708).integers(0, 256, (8, 16, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            measured = await page.evaluate(_GPU_TEXTURE, dict(source=source, sampler=form.sampler,
                                                               coordinates=form.coordinates, bytes=pixels.reshape(-1).tolist()))
            assert measured is not None, "WebGL2 probe unavailable; texture-copy form remains unmeasured"
        finally:
            await browser.close()
    assert measured == pixels.reshape(-1).tolist()


def test_texture_copy_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _texture_copy_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "texture_copy_requires_unchanged_dataflow" for c in suite.claims())
