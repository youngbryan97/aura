"""Position arithmetic is checked against independent numbers and GPU execution."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from core.perception.shader_position_math import (
    _shader_position_invariant,
    position_from_program,
    position_from_source,
)
from core.perception.shader_position_source import UnsupportedPositionError, parse_positions


def _shader(body, declarations=""):
    return f"#version 300 es\nprecision highp float; precision highp int;\n{declarations}\nvoid main(){{{body}}}"


@pytest.mark.parametrize("name", ["transform", "displacement", "color", "a_91"])
def test_position_is_calculated_by_operations_instead_of_identifier_spelling(name):
    source = _shader(f"gl_Position={name}*vec4(vertex,0.0,1.0);", f"uniform mat4 {name}; in vec2 vertex;")
    matrix = np.array(((2., 0., 0., 3.), (0., -3., 0., 5.), (0., 0., 1., 0.), (0., 0., 0., 2.)))
    assert position_from_source(source, {name: matrix, "vertex": (4., 2.)}).position == (11., -1., 0., 2.)


def test_matrix_multiplication_order_is_preserved_and_vectors_are_not_matrices():
    declarations = "uniform mat4 a; uniform mat4 b; in vec2 vertex;"
    a, b = np.eye(4), np.eye(4)
    a[0, 0], b[0, 3] = 2, 3
    inputs = {"a": a, "b": b, "vertex": (1., 0.)}
    assert position_from_source(_shader("gl_Position=a*(b*vec4(vertex,0.0,1.0));", declarations), inputs).position[0] == 8
    assert position_from_source(_shader("gl_Position=b*(a*vec4(vertex,0.0,1.0));", declarations), inputs).position[0] == 5
    assert position_from_source(_shader("gl_Position=vec4(1.0,2.0,3.0,4.0)*a;", declarations), inputs).position == (2., 2., 3., 4.)


def test_structure_fields_ignore_unrelated_unavailable_color_dependencies():
    declarations = """struct Shape {mat4 pose;vec4 tint;};
      struct Output {vec4 clip;vec4 color;};uniform Shape placements[3];in vec2 vertex;"""
    source = _shader("""uint i=uint(gl_InstanceID);mat4 a=placements[i].pose;
      vec4 unused=clamp(placements[i].tint,vec4(0.0),vec4(1.0));
      Output result=Output(a*vec4(vertex,0.0,1.0),unused);gl_Position=result.clip;""", declarations)
    placements = [{"pose": np.eye(4)} for _ in range(3)]
    placements[2]["pose"][0, 3] = 7
    inputs = {"placements": placements, "vertex": (1., 2.)}
    assert position_from_source(source, inputs, instance=2).position == (8., 2., 0., 1.)
    assert position_from_source(source, inputs, instance=3).reason == "array_index_or_extent"
    assert position_from_source(source, {**inputs, "placements": placements[:2]}).position is None


def test_uniform_block_structure_and_unnamed_block_fields_are_declared_generically():
    source = _shader("gl_Position=group.offset*vec4(vertex,0.0,1.0);",
                     "layout(std140) uniform Container { mat4 offset; } group; in vec2 vertex;")
    assert position_from_source(source, {"group": {"offset": np.eye(4)}, "vertex": (2., 3.)}).position == (2., 3., 0., 1.)
    source = _shader("gl_Position=offset*vec4(vertex,0.0,1.0);",
                     "layout(std140) uniform Container { mat4 offset; }; in vec2 vertex;")
    assert position_from_source(source, {"offset": np.eye(4), "vertex": (2., 3.)}).position == (2., 3., 0., 1.)


def test_writes_capture_the_previous_values_and_read_swizzles_may_repeat():
    source = _shader("""vec4 a=vec4(1.0,2.0,3.0,4.0);vec4 b=a;
      a.xy=a.yx;b.zw=a.xx;gl_Position=b;gl_Position.yz=vec2(-a.y,gl_Position.z*2.0-gl_Position.w);""")
    assert position_from_source(source, {}).position == (1., -1., 2., 2.)
    program = parse_positions(source)
    with pytest.raises(FrozenInstanceError):
        program.position = None


def test_matrix_constructors_and_indexing_use_columns():
    source = _shader("mat2x3 a=mat2x3(1.0,2.0,3.0,4.0,5.0,6.0);gl_Position=vec4(a[1],1.0);")
    assert position_from_source(source, {}).position == (4., 5., 6., 1.)
    source = _shader("mat4 a=mat4(2.0);gl_Position=a*vec4(1.0,2.0,3.0,4.0);")
    assert position_from_source(source, {}).position == (2., 4., 6., 8.)
    source = _shader("vec2 a=vec2(2.0,3.0);gl_Position=vec4(a*a,0.0,1.0);")
    assert position_from_source(source, {}).position == (4., 9., 0., 1.)


def test_integer_instance_and_vertex_math_has_explicit_ranges():
    source = _shader("uint i=uint(gl_InstanceID)+base;gl_Position=vec4(float(i),float(gl_VertexID),0.0,1.0);", "uniform uint base;")
    assert position_from_source(source, {"base": 9}, instance=2, vertex=7).position == (11., 7., 0., 1.)
    assert position_from_source(source, {"base": 2**32 - 1}, instance=1).reason == "integer_range"
    assert position_from_source(source, {"base": 3.5}).reason == "integer_type"
    assert position_from_source(source, {"base": 3}, instance=True).reason == "draw_index"


@pytest.mark.parametrize("body", [
    "gl_Position=secret(vec4(1.0));", "for(int i=0;i<3;i++){gl_Position=vec4(1.0);}",
    "if(true){gl_Position=vec4(1.0);}", "gl_Position=vec4(1.0);gl_Position.xx=vec2(0.0);",
    "vec4 a=vec4(1.0);vec4 a=vec4(2.0);gl_Position=a;", "gl_Position=vec4(1.0/0.0);",
    "gl_Position=vec4(sqrt(-1.0));", "gl_Position=vec4(1e39);", "gl_Position=gl_Position;",
    "vec2 a=vec2(1.0);gl_Position=vec4(a.z,0.0,0.0,1.0);",
    "vec2 a=vec2(1.0);gl_Position=vec4(a[2],0.0,0.0,1.0);",
    "gl_Position=vec4(1.0);gl_Position.xyz=vec2(1.0);",
    "vec4 a=vec4(1.0);gl_Position=vec4(a.xg,0.0,1.0);",
])
def test_unsupported_undefined_or_ambiguous_position_returns_unknown(body):
    result = position_from_source(_shader(body), {})
    assert result.position is None and result.reason


@pytest.mark.parametrize("change", [
    lambda s: s.replace("300 es", "100"),
    lambda s: s.replace("highp float", "mediump float"),
    lambda s: s + "\n#define gl_Position result\n",
    lambda s: s + " " * 32768,
])
def test_versions_precision_and_source_budgets_are_not_guessed(change):
    assert position_from_source(change(_shader("gl_Position=vec4(1.0);")), {}).position is None


def test_depth_budgets_and_missing_inputs_return_unknown():
    source = _shader("gl_Position=vec4(" + "(" * 60 + "1.0" + ")" * 60 + ");")
    assert position_from_source(source, {}).reason == "expression_depth"
    with pytest.raises(UnsupportedPositionError):
        parse_positions(_shader("gl_Position=vec4(1.0);", "uniform vec4 values[257];"))
    program = parse_positions(_shader("gl_Position=point;", "in vec4 point;"))
    assert position_from_program(program, {}).reason == "missing_input"
    assert position_from_program(program, {"point": (1., 2., 3.)}).reason == "input_shape"
    assert position_from_program(program, {"point": (1., 2., np.nan, 4.)}).reason == "nonfinite"


_GPU_POSITION = r"""({source,point,matrices,count}) => {
 const gl=document.createElement('canvas').getContext('webgl2',{antialias:false});if(!gl)return null;
 const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
 const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,source));
 gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1);}'));
 gl.transformFeedbackVaryings(p,['witness'],gl.INTERLEAVED_ATTRIBS);gl.linkProgram(p);
 if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));gl.useProgram(p);
 const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(point),gl.STATIC_DRAW);
 const location=gl.getAttribLocation(p,'vertex');gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,2,gl.FLOAT,false,0,0);
 for(let i=0;i<matrices.length;i++)gl.uniformMatrix4fv(gl.getUniformLocation(p,`placements[${i}].pose`),false,new Float32Array(matrices[i]));
 const out=gl.createBuffer();gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,out);gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER,count*16,gl.STREAM_READ);
 gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,out);gl.enable(gl.RASTERIZER_DISCARD);
 gl.beginTransformFeedback(gl.POINTS);gl.drawArraysInstanced(gl.POINTS,0,1,count);gl.endTransformFeedback();
 const values=new Float32Array(count*4);gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,values);
 if(gl.getError()!==gl.NO_ERROR)throw Error('synthetic position operation failed');
 return Array.from(values);
}"""


@pytest.mark.asyncio
async def test_renamed_structured_position_math_matches_gpu_transform_feedback():
    from playwright.async_api import async_playwright

    random = np.random.default_rng(911)
    matrices = random.uniform(-2., 2., (12, 4, 4)).astype(np.float32)
    point = random.uniform(-1., 1., 2).astype(np.float32)
    declarations = "struct Placement {mat4 pose;vec4 tint;}; uniform Placement placements[12];in vec2 vertex;out vec4 witness;"
    body = """uint slot=uint(gl_InstanceID);vec4 p=placements[slot].pose*vec4(vertex,0.0,1.0);
      gl_Position=p;gl_Position.yz=vec2(-p.y,p.z*2.0-p.w);witness=gl_Position;"""
    source = _shader(body, declarations)
    program = parse_positions(source)
    inputs = {"placements": [{"pose": m} for m in matrices], "vertex": point}
    predicted = [position_from_program(program, inputs, instance=i).position for i in range(12)]
    assert all(v is not None for v in predicted)
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            measured = await page.evaluate(_GPU_POSITION, dict(source=source, point=point.tolist(), count=12,
                                                               matrices=[m.flatten(order="F").tolist() for m in matrices]))
            assert measured is not None, "WebGL2 probe unavailable; shader math remains unmeasured"
        finally:
            await browser.close()
    np.testing.assert_allclose(np.asarray(predicted).reshape(-1), measured, rtol=2e-6, atol=2e-6)


def test_shader_position_claim_runs_its_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _shader_position_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(c.test == "shader_positions_use_declared_operations" for c in suite.claims())
