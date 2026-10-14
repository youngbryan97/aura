"""Vertex addressing is measured separately from position arithmetic."""
import struct
from dataclasses import replace

import numpy as np
import pytest

from core.perception.shader_attributes_from_buffer_bytes import (
    AttributeBinding,
    _attribute_addressing_invariant,
    attribute_from_bytes,
)
from core.perception.shader_position_math import position_from_program
from core.perception.shader_position_source import parse_positions
from core.perception.shader_uniforms_from_buffer_bytes import BufferRange


def _program(kind="vec4"):
    constructor = "vec4(point,0.0,1.0)" if kind == "vec2" else "vec4(point)"
    return parse_positions(f"#version 300 es\nin {kind} point;out vec4 witness;void main(){{gl_Position={constructor};witness=gl_Position;}}")


def test_stride_and_offset_select_vertices_without_reading_unused_components():
    program = _program("vec2")
    data = BufferRange(1, 2, 3, 4, 8, struct.pack("<6f", 1., 2., 90., 91., 3., 4.), "little")
    binding = AttributeBinding("point", 1, 2, 3, 4, 4, 5126, offset=8, stride=16)
    assert attribute_from_bytes(program, binding, data, vertex=1).value == (3., 4.)
    assert attribute_from_bytes(program, binding, data, vertex=2).reason == "uncaptured_attribute_bytes"


@pytest.mark.parametrize("component_type,raw,expected", [
    (5120, struct.pack("2b", -128, 127), (-1., 1., 0., 1.)),
    (5121, bytes((0, 255)), (0., 1., 0., 1.)),
    (5122, struct.pack("<2h", -32768, 32767), (-1., 1., 0., 1.)),
    (5123, struct.pack("<2H", 32768, 65535), (32768 / 65535, 1., 0., 1.)),
    (5131, struct.pack("<2e", .25, .75), (.25, .75, 0., 1.)),
])
def test_normalization_and_missing_component_defaults_follow_stored_type(component_type, raw, expected):
    binding = AttributeBinding("point", 1, 2, 3, 4, 2, component_type, normalized=True)
    data = BufferRange(1, 2, 3, 4, 0, raw, "little")
    assert attribute_from_bytes(_program(), binding, data, vertex=0).value == expected


def test_divisor_uses_instance_while_zero_divisor_uses_vertex_and_integer_inputs_remain_integral():
    data = BufferRange(1, 2, 3, 4, 0, struct.pack(">4H", 1, 2, 3, 4), "big")
    binding = AttributeBinding("point", 1, 2, 3, 4, 2, 5123, divisor=2, integer=True)
    result = attribute_from_bytes(_program("uvec4"), binding, data, vertex=99, instance=3)
    assert result.value == (3, 4, 0, 1)
    assert position_from_program(_program("uvec4"), {"point": result.value}).position == (3., 4., 0., 1.)
    vertex_binding = replace(binding, divisor=0)
    assert attribute_from_bytes(_program("uvec4"), vertex_binding, data, vertex=0, instance=99).value == (1, 2, 0, 1)


@pytest.mark.parametrize("change", [
    {"context": 8}, {"epoch": 3}, {"buffer": 4}, {"content": 5},
    {"component_type": 33640}, {"components": 5}, {"offset": 1}, {"stride": 3},
    {"divisor": -1}, {"normalized": 1}, {"integer": True}, {"name": "hero"},
])
def test_unsupported_or_stale_attributes_remain_unknown(change):
    data = BufferRange(1, 2, 3, 4, 0, struct.pack("<4f", 1., 2., 3., 4.), "little")
    binding = AttributeBinding("point", 1, 2, 3, 4, 4, 5126)
    assert attribute_from_bytes(_program(), replace(binding, **change), data, vertex=0).value is None


def test_nonfinite_mutable_or_unmeasured_bytes_and_invalid_indices_are_refused():
    binding = AttributeBinding("point", 1, 2, 3, 4, 4, 5126)
    data = BufferRange(1, 2, 3, 4, 0, struct.pack("<4f", 1., 2., np.inf, 4.), "little")
    assert attribute_from_bytes(_program(), binding, data, vertex=0).reason == "nonfinite_attribute"
    assert attribute_from_bytes(_program(), binding, replace(data, data=bytearray(data.data)), vertex=0).value is None
    assert attribute_from_bytes(_program(), binding, replace(data, byte_order="guess"), vertex=0).value is None
    assert attribute_from_bytes(_program(), binding, data, vertex=True).value is None


_GPU_ATTRIBUTE = r"""({source,bytes,componentType,integer}) => {
 const gl=document.createElement('canvas').getContext('webgl2');if(!gl)return null;
 const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
 const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,source));
 gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1.0);}'));
 gl.transformFeedbackVaryings(p,['witness'],gl.INTERLEAVED_ATTRIBS);gl.linkProgram(p);
 if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
 const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(bytes),gl.STATIC_DRAW);
 const location=gl.getAttribLocation(p,'point');gl.enableVertexAttribArray(location);
 const stride=componentType===gl.BYTE?6:8,offset=2;
 if(integer)gl.vertexAttribIPointer(location,2,componentType,stride,offset);
 else gl.vertexAttribPointer(location,2,componentType,true,stride,offset);
 gl.vertexAttribDivisor(location,2);
 const out=gl.createBuffer();gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,out);gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER,64,gl.STREAM_READ);
 gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,out);gl.enable(gl.RASTERIZER_DISCARD);
 gl.beginTransformFeedback(gl.POINTS);gl.drawArraysInstanced(gl.POINTS,0,1,4);gl.endTransformFeedback();
 const result=new Float32Array(16);gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,result);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic attribute operation failed');return Array.from(result);
}"""


@pytest.mark.asyncio
@pytest.mark.parametrize("integer", [False, True])
async def test_strided_divisor_and_defaults_match_independent_gpu_inputs(integer):
    from playwright.async_api import async_playwright

    source = "#version 300 es\nin TYPE point;out vec4 witness;void main(){gl_Position=vec4(point);witness=gl_Position;}"
    source = source.replace("TYPE", "uvec4" if integer else "vec4")
    raw = b"zz" + struct.pack("<2H", 7, 900) + b"xxxx" + struct.pack("<2H", 3, 400) if integer else b"zz" + struct.pack("2b", -128, 127) + b"xxxx" + struct.pack("2b", 0, -64)
    binding = AttributeBinding("point", 1, 2, 3, 4, 2, 5123 if integer else 5120,
                               offset=2, stride=8 if integer else 6, divisor=2, normalized=not integer, integer=integer)
    data = BufferRange(1, 2, 3, 4, 0, raw, "little")
    program = parse_positions(source)
    outputs = [position_from_program(program, {"point": attribute_from_bytes(program, binding, data, vertex=0, instance=i).value}).position
               for i in range(4)]
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            measured = await page.evaluate(_GPU_ATTRIBUTE, dict(source=source, bytes=list(raw), componentType=binding.component_type, integer=integer))
            assert measured is not None, "WebGL2 probe unavailable; vertex addressing remains unmeasured"
        finally:
            await browser.close()
    np.testing.assert_allclose(np.asarray(outputs).reshape(-1), measured, rtol=2e-6, atol=2e-6)


def test_vertex_addressing_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _attribute_addressing_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "vertex_bytes_keep_draw_addressing" for c in suite.claims())
