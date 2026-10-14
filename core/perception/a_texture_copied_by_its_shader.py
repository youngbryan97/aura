"""Recognize a declared single texture sample copied to one fragment output.

This proves a dataflow form, not its presentation or visible pixels. Sampler
bindings, interpolation, raster state and current image generations must be
observed before a renderer can use the form as a presentation edge.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.perception.shader_position_source import (
    Expression,
    PositionProgram,
    ShaderType,
    UnsupportedPositionError,
    parse_positions,
)
from core.verify import invariant


@dataclass(frozen=True)
class TextureCopyForm:
    sampler: str
    coordinates: str
    output: str


class _Dataflow:
    def __init__(self, program: PositionProgram) -> None:
        self.structures = dict(program.structures)
        self.work = 0

    def resolved(self, node: Expression, depth: int = 0) -> Expression:
        self.work += 1
        if depth > 64 or self.work > 4096:
            raise UnsupportedPositionError("copy_dependency_budget")
        if node.operation == "typed":
            return self.resolved(node.parts[1], depth + 1)
        if node.operation == "field":
            parent = self.resolved(node.parts[0], depth + 1)
            if parent.operation != "call" or parent.parts[0] not in self.structures:
                raise UnsupportedPositionError("copy_field")
            fields = [name for name, _ in self.structures[parent.parts[0]]]
            if node.parts[1] not in fields or len(fields) != len(parent.parts[1]):
                raise UnsupportedPositionError("copy_structure")
            return self.resolved(parent.parts[1][fields.index(node.parts[1])], depth + 1)
        return node

    def identity(self, node: Expression, kind: str) -> Expression:
        for _ in range(32):
            node = self.resolved(node)
            if node.operation != "call" or node.parts[0] != kind or len(node.parts[1]) != 1:
                return node
            node = node.parts[1][0]
        raise UnsupportedPositionError("identity_constructor_budget")


def texture_copy_form(program: PositionProgram) -> TextureCopyForm | None:
    """Only direct smooth coordinates and an unchanged vec4 sample are admitted."""
    if not isinstance(program, PositionProgram) or len(program.outputs) != 1:
        return None
    try:
        output, expression = program.outputs[0]
        if expression.operation != "typed" or expression.parts[0] != ShaderType("vec4"):
            return None
        flow = _Dataflow(program)
        sample = flow.identity(expression, "vec4")
        if sample.operation != "call" or sample.parts[0] != "texture" or len(sample.parts[1]) != 2:
            return None
        sampler = flow.resolved(sample.parts[1][0])
        coordinates = flow.identity(sample.parts[1][1], "vec2")
        if sampler.operation != "input" or coordinates.operation != "input":
            return None
        sampler_name, sampler_type = sampler.parts
        coordinates_name, coordinates_type = coordinates.parts
        if sampler_type != ShaderType("sampler2D") or coordinates_type != ShaderType("vec2"):
            return None
        qualifiers = dict(program.input_qualifiers)
        uv_mode = qualifiers.get(coordinates_name, ())
        if ("uniform" not in qualifiers.get(sampler_name, ()) or "in" not in uv_mode
                or any(mode in uv_mode for mode in ("flat", "centroid"))):
            return None
        return TextureCopyForm(sampler_name, coordinates_name, output)
    except (UnsupportedPositionError, KeyError, ValueError, TypeError):
        return None


@invariant("perception.texture_copy_requires_unchanged_dataflow", scope="perception",
           owner="core/perception/a_texture_copied_by_its_shader.py", observational=False)
def _texture_copy_invariant() -> tuple:
    source = """#version 300 es
    precision highp float;uniform sampler2D picture;in vec2 where;out vec4 pixel;
    void main(){pixel=texture(picture,where);}"""
    assert texture_copy_form(parse_positions(source)) == TextureCopyForm("picture", "where", "pixel")
    changed = source.replace("texture(picture,where)", "texture(picture,where)*vec4(0.5)")
    assert texture_copy_form(parse_positions(changed)) is None
    assert texture_copy_form(parse_positions(source.replace("in vec2", "flat in vec2"))) is None
    return ()
