"""Reflection controls byte layout; source declarations control names and extents."""
import struct
from dataclasses import replace

import numpy as np
import pytest

from core.perception.shader_position_math import position_from_program
from core.perception.shader_position_source import parse_positions
from core.perception.shader_uniforms_from_buffer_bytes import (
    BufferRange,
    UniformBinding,
    UniformMember,
    _uniform_resource_invariant,
    decode_uniform_block,
)


def _case(order="little", row_major=False):
    source = """#version 300 es
    struct Placement {mat4 pose;vec4 tint;};
    layout(std140) uniform World {Placement items[2];} group;
    in vec2 point;void main(){gl_Position=group.items[gl_InstanceID].pose*vec4(point,0.0,1.0);}"""
    program = parse_positions(source)
    matrices = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    raw = bytearray(256)
    prefix, stride, array_stride = 32, 20, 96
    for index, matrix in enumerate(matrices):
        for r in range(4):
            for c in range(4):
                offset = prefix + index * array_stride + (r * stride + c * 4 if row_major else c * stride + r * 4)
                struct.pack_into(("<" if order == "little" else ">") + "f", raw, offset, matrix[r, c])
    data = BufferRange(1, 2, 3, 4, 0, bytes(raw), order)
    binding = UniformBinding(1, 2, 3, 4, prefix, 200, "World")
    members = (UniformMember("World.items[0].pose", 35676, 0, size=2,
                             array_stride=array_stride, matrix_stride=stride, row_major=row_major),)
    return program, binding, data, members, matrices


@pytest.mark.parametrize("order,row_major", [("little", False), ("little", True), ("big", False), ("big", True)])
def test_bound_offsets_strides_and_endianness_feed_the_actual_position_evaluator(order, row_major):
    program, binding, data, members, matrices = _case(order, row_major)
    decoded = decode_uniform_block(program, binding, data, members)
    assert decoded.reason is None
    values = {**decoded.inputs, "point": (2., 3.)}
    for i in range(2):
        measured = position_from_program(program, values, instance=i)
        np.testing.assert_allclose(measured.position, matrices[i] @ (2., 3., 0., 1.))
    assert "tint" not in decoded.inputs["group"]["items"][0]
    assert data.data == _case(order, row_major)[2].data


def test_uniform_block_name_resolves_a_declared_instance_and_an_unnamed_block():
    program, binding, data, members, _ = _case()
    direct = replace(members[0], name="group.items[0].pose")
    assert decode_uniform_block(program, binding, data, (direct,)).inputs is not None
    assert decode_uniform_block(program, replace(binding, block="GuessedWorld"), data, members).inputs is None
    source = "#version 300 es\nlayout(std140) uniform Data {vec4 location;};void main(){gl_Position=location;}"
    program = parse_positions(source)
    raw = BufferRange(1, 2, 3, 4, 0, struct.pack("<4f", 1., 2., 3., 4.), "little")
    bound = UniformBinding(1, 2, 3, 4, 0, 16, "Data")
    assert decode_uniform_block(program, bound, raw, (UniformMember("Data.location", 35666, 0),)).inputs == {"location": (1., 2., 3., 4.)}


@pytest.mark.parametrize("field,value", [("context", 9), ("epoch", 3), ("buffer", 7), ("content", 5), ("content", 0)])
def test_a_buffer_name_cannot_borrow_another_resource_or_content_version(field, value):
    program, binding, data, members, _ = _case()
    assert decode_uniform_block(program, replace(binding, **{field: value}), data, members).inputs is None


@pytest.mark.parametrize("change", [
    {"offset": 2}, {"matrix_stride": 12}, {"matrix_stride": 0}, {"row_major": 1},
    {"array_stride": 0}, {"size": 3}, {"size": 257}, {"gl_type": 35666},
    {"name": "World.items[2].pose"}, {"name": "World.items[0].facing"},
    {"name": "private.actor"}, {"name": "World.items[-1].pose"},
])
def test_malformed_or_undeclared_layouts_remain_unknown(change):
    program, binding, data, members, _ = _case()
    assert decode_uniform_block(program, binding, data, (replace(members[0], **change),)).inputs is None


def test_uncaptured_out_of_range_nonfinite_and_duplicate_bytes_are_rejected():
    program, binding, data, members, _ = _case()
    assert decode_uniform_block(program, binding, replace(data, data=data.data[:60]), members).reason == "uncaptured_uniform_bytes"
    assert decode_uniform_block(program, replace(binding, size=100), data, members).reason == "outside_bound_range"
    assert decode_uniform_block(program, binding, replace(data, byte_order="unknown"), members).inputs is None
    assert decode_uniform_block(program, binding, replace(data, data=bytearray(data.data)), members).inputs is None
    raw = bytearray(data.data)
    struct.pack_into("<f", raw, binding.start, float("nan"))
    assert decode_uniform_block(program, binding, replace(data, data=bytes(raw)), members).reason == "nonfinite_uniform"
    assert decode_uniform_block(program, binding, data, members + members).reason == "duplicate_uniform_bytes"


def test_integer_and_boolean_uniform_storage_is_typed():
    source = "#version 300 es\nuniform uvec2 ids;uniform bvec2 flags;void main(){gl_Position=vec4(float(ids.x),float(flags.y),0.0,1.0);}"
    program = parse_positions(source)
    data = BufferRange(1, 2, 3, 4, 0, struct.pack("<2I2i", 7, 8, 0, -3), "little")
    binding = UniformBinding(1, 2, 3, 4, 0, 16)
    layouts = (UniformMember("ids", 36294, 0), UniformMember("flags", 35671, 8))
    values = decode_uniform_block(program, binding, data, layouts)
    assert values.inputs == {"ids": (7, 8), "flags": (False, True)}
    assert position_from_program(program, values.inputs).position == (7., 1., 0., 1.)


def test_uniform_resource_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _uniform_resource_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "uniform_bytes_keep_resource_versions" for c in suite.claims())


_GPU_BLOCK = r"""({source,matrices,point}) => {
 const gl=document.createElement('canvas').getContext('webgl2');if(!gl)return null;
 const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
 const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,source));
 gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1);}'));
 gl.transformFeedbackVaryings(p,['witness'],gl.INTERLEAVED_ATTRIBS);gl.linkProgram(p);
 if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
 const block=gl.getUniformBlockIndex(p,'World'),size=gl.getActiveUniformBlockParameter(p,block,gl.UNIFORM_BLOCK_DATA_SIZE);
 const indices=Array.from(gl.getActiveUniformBlockParameter(p,block,gl.UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES));
 const members=indices.map(i=>{const u=gl.getActiveUniform(p,i);return {name:u.name,gl_type:u.type,size:u.size};});
 for(const [name,code] of [['offset',gl.UNIFORM_OFFSET],['array_stride',gl.UNIFORM_ARRAY_STRIDE],['matrix_stride',gl.UNIFORM_MATRIX_STRIDE],['row_major',gl.UNIFORM_IS_ROW_MAJOR]]){
  const values=gl.getActiveUniforms(p,indices,code);members.forEach((m,i)=>m[name]=name==='row_major'?Boolean(values[i]):values[i]);
 }
 const start=gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT),raw=new Uint8Array(start+size),view=new DataView(raw.buffer);
 const little=new Uint8Array(new Uint32Array([0x01020304]).buffer)[0]===4;
 for(const m of members){if(m.gl_type!==gl.FLOAT_MAT4)continue;
  const base=Number(m.name.match(/items\[(\d+)\]/)[1]);
  for(let i=0;i<m.size;i++)for(let r=0;r<4;r++)for(let c=0;c<4;c++){
   const offset=start+m.offset+i*m.array_stride+(m.row_major?r*m.matrix_stride+c*4:c*m.matrix_stride+r*4);
   view.setFloat32(offset,matrices[base+i][r][c],little);
  }
 }
 const buffer=gl.createBuffer();gl.bindBuffer(gl.UNIFORM_BUFFER,buffer);gl.bufferData(gl.UNIFORM_BUFFER,raw,gl.STATIC_DRAW);
 gl.uniformBlockBinding(p,block,0);gl.bindBufferRange(gl.UNIFORM_BUFFER,0,buffer,start,size);
 const location=gl.getAttribLocation(p,'point');gl.disableVertexAttribArray(location);gl.vertexAttrib2f(location,...point);
 const out=gl.createBuffer();gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,out);gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER,32,gl.STREAM_READ);
 gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,out);gl.enable(gl.RASTERIZER_DISCARD);
 gl.beginTransformFeedback(gl.POINTS);gl.drawArraysInstanced(gl.POINTS,0,1,2);gl.endTransformFeedback();
 const values=new Float32Array(8);gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,values);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic block operation failed');
 return {members,start,size,bytes:Array.from(raw),byte_order:little?'little':'big',positions:Array.from(values)};
}"""


@pytest.mark.asyncio
@pytest.mark.parametrize("row_major", [False, True])
async def test_reflected_gpu_buffer_layout_reaches_the_same_positions(row_major):
    from playwright.async_api import async_playwright

    source = """#version 300 es
    struct Placement {mat4 pose;vec4 tint;};
    layout(std140,ORDER) uniform World {Placement items[2];} group;
    in vec2 point;out vec4 witness;
    void main(){gl_Position=group.items[gl_InstanceID].pose*vec4(point,0.0,1.0);witness=gl_Position;}"""
    source = source.replace("ORDER", "row_major" if row_major else "column_major")
    matrices = np.random.default_rng(151).uniform(-2, 2, (2, 4, 4)).astype(np.float32)
    point = (.75, -.25)
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            measured = await page.evaluate(_GPU_BLOCK, dict(source=source, matrices=matrices.tolist(), point=point))
            assert measured is not None, "WebGL2 probe unavailable; reflected layout remains unmeasured"
        finally:
            await browser.close()
    program = parse_positions(source)
    data = BufferRange(1, 2, 3, 4, 0, bytes(measured["bytes"]), measured["byte_order"])
    binding = UniformBinding(1, 2, 3, 4, measured["start"], measured["size"], "World")
    layouts = tuple(UniformMember(**item) for item in measured["members"])
    values = decode_uniform_block(program, binding, data, layouts)
    assert values.reason is None
    outputs = [position_from_program(program, {**values.inputs, "point": point}, instance=i).position for i in range(2)]
    expected = np.asarray([m @ np.asarray((*point, 0., 1.), dtype=np.float32) for m in matrices])
    np.testing.assert_allclose(outputs, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(outputs, np.asarray(measured["positions"]).reshape(2, 4), rtol=2e-6, atol=2e-6)
