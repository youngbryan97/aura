"""Evaluate declared position dependencies without running shader source.

The admitted arithmetic uses finite highp float32 values and bounded integers.
It is a geometric proposal, not a rasterizer or a GPU rounding bound. Structure
fields are lazy so unrelated color arithmetic cannot become position evidence.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.shader_position_source import (
    Expression,
    PositionProgram,
    ShaderType,
    UnsupportedPositionError,
    parse_positions,
)
from core.verify import invariant


@dataclass(frozen=True)
class PositionResult:
    position: tuple[float, float, float, float] | None
    reason: str | None = None


@dataclass(frozen=True)
class _Value:
    base: str
    data: np.ndarray


@dataclass(frozen=True)
class _External:
    kind: ShaderType
    data: Any


@dataclass(frozen=True)
class _Structure:
    kind: str
    arguments: tuple[Expression, ...]


def _shape(name: str) -> tuple[str, tuple[int, ...]]:
    if name in {"float", "int", "uint", "bool"}:
        return name, ()
    vector = re.fullmatch(r"([iub]?)vec([234])", name)
    if vector:
        return {"": "float", "i": "int", "u": "uint", "b": "bool"}[vector[1]], (int(vector[2]),)
    matrix = re.fullmatch(r"mat([234])(?:x([234]))?", name)
    if matrix:
        return "float", (int(matrix[2] or matrix[1]), int(matrix[1]))
    raise UnsupportedPositionError("numeric_type")


def _value(base: str, data: Any, *, cast: bool = False) -> _Value:
    array = np.asarray(data)
    if array.size > 16 or array.ndim > 2 or array.dtype.kind not in "biuf":
        raise UnsupportedPositionError("numeric_input")
    if not np.all(np.isfinite(array)):
        raise UnsupportedPositionError("nonfinite")
    if base == "bool":
        if not cast and array.dtype.kind != "b":
            raise UnsupportedPositionError("boolean_type")
        return _Value(base, array.astype(np.bool_, copy=True))
    if array.dtype.kind == "b" and not cast:
        raise UnsupportedPositionError("numeric_type")
    if base in {"int", "uint"}:
        if not cast and array.dtype.kind not in "iu":
            raise UnsupportedPositionError("integer_type")
        lower, upper = (0, 2**32 - 1) if base == "uint" else (-2**31, 2**31 - 1)
        values = np.trunc(array) if array.dtype.kind == "f" else array
        if np.any(values < lower) or np.any(values > upper):
            raise UnsupportedPositionError("integer_range")
        return _Value(base, values.astype(np.int64, copy=True))
    if base != "float":
        raise UnsupportedPositionError("numeric_type")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise UnsupportedPositionError("float_range")
    return _Value(base, result)


def _swizzle(field: str, size: int, *, write: bool = False) -> tuple[int, ...]:
    alphabet = next((letters for letters in ("xyzw", "rgba", "stpq") if all(c in letters for c in field)), None)
    if alphabet is None or not 1 <= len(field) <= 4:
        raise UnsupportedPositionError("field_or_swizzle")
    indices = tuple(alphabet.index(c) for c in field)
    if any(i >= size for i in indices) or write and len(set(indices)) != len(indices):
        raise UnsupportedPositionError("swizzle_extent")
    return indices


class _Evaluation:
    def __init__(self, program: PositionProgram, inputs: dict[str, Any]) -> None:
        self.structures = dict(program.structures)
        self.inputs = inputs
        self.cache: dict[int, Any] = {}
        self.work = 0

    def coerce(self, kind: ShaderType, value: Any) -> Any:
        if kind.count or kind.name in self.structures:
            if isinstance(value, _External) and value.kind == kind:
                return value
            if not kind.count and isinstance(value, _Structure) and value.kind == kind.name:
                return value
            raise UnsupportedPositionError("aggregate_type")
        base, shape = _shape(kind.name)
        if not isinstance(value, _Value) or value.base != base or value.data.shape != shape:
            raise UnsupportedPositionError("assignment_type")
        return value

    def external(self, kind: ShaderType, value: Any) -> Any:
        if kind.count or kind.name in self.structures:
            return _External(kind, value)
        base, shape = _shape(kind.name)
        result = _value(base, value)
        if result.data.shape != shape:
            raise UnsupportedPositionError("input_shape")
        return result

    def numeric(self, value: Any) -> _Value:
        if not isinstance(value, _Value):
            raise UnsupportedPositionError("numeric_dependency")
        return value

    def field(self, value: Any, field: str, depth: int) -> Any:
        if isinstance(value, _Structure):
            fields = self.structures[value.kind]
            names = [name for name, _ in fields]
            if field not in names or len(fields) != len(value.arguments):
                raise UnsupportedPositionError("structure_field")
            index = names.index(field)
            return self.coerce(fields[index][1], self.get(value.arguments[index], depth + 1))
        if isinstance(value, _External):
            if value.kind.count or value.kind.name not in self.structures or not isinstance(value.data, dict):
                raise UnsupportedPositionError("structure_input")
            fields = dict(self.structures[value.kind.name])
            if field not in fields or field not in value.data:
                raise UnsupportedPositionError("missing_field")
            return self.external(fields[field], value.data[field])
        value = self.numeric(value)
        if value.data.ndim != 1:
            raise UnsupportedPositionError("swizzle_type")
        indices = _swizzle(field, len(value.data))
        data = value.data[list(indices)] if len(indices) > 1 else value.data[indices[0]]
        return _value(value.base, data)

    def indexed(self, value: Any, index: Any) -> Any:
        index = self.numeric(index)
        if index.base not in {"int", "uint"} or index.data.shape != ():
            raise UnsupportedPositionError("index_type")
        i = int(index.data)
        if isinstance(value, _External) and value.kind.count:
            if (not isinstance(value.data, (tuple, list)) or len(value.data) != value.kind.count
                    or not 0 <= i < value.kind.count):
                raise UnsupportedPositionError("array_index_or_extent")
            return self.external(ShaderType(value.kind.name), value.data[i])
        value = self.numeric(value)
        if value.data.ndim not in (1, 2):
            raise UnsupportedPositionError("index_type")
        size = value.data.shape[-1]
        if not 0 <= i < size:
            raise UnsupportedPositionError("index_extent")
        return _value(value.base, value.data[:, i] if value.data.ndim == 2 else value.data[i])

    def constructor(self, name: str, args: tuple, depth: int) -> Any:
        if name in self.structures:
            if len(args) != len(self.structures[name]):
                raise UnsupportedPositionError("structure_arguments")
            return _Structure(name, args)
        values = [self.numeric(self.get(arg, depth + 1)) for arg in args]
        if name in {"clamp", "min", "max", "abs", "sin", "cos", "sqrt"}:
            return self.function(name, values)
        base, shape = _shape(name)
        if not values:
            raise UnsupportedPositionError("constructor_arguments")
        if not shape:
            if len(values) != 1 or values[0].data.shape != ():
                raise UnsupportedPositionError("scalar_constructor")
            return _value(base, values[0].data, cast=True)
        if len(shape) == 2 and len(values) == 1:
            value = values[0]
            if value.data.shape == ():
                return _value(base, np.eye(*shape, dtype=np.float32) * value.data, cast=True)
            if value.data.ndim == 2 and value.data.shape == shape:
                return _value(base, value.data, cast=True)
            raise UnsupportedPositionError("matrix_conversion")
        if any(v.data.ndim > 1 for v in values):
            raise UnsupportedPositionError("constructor_shape")
        flat = np.concatenate([np.atleast_1d(v.data) for v in values])
        count = math.prod(shape)
        if len(values) == 1 and values[0].data.shape == () and len(shape) == 1:
            flat = np.full(count, values[0].data)
        if len(flat) != count:
            raise UnsupportedPositionError("constructor_extent")
        data = flat.reshape(shape[::-1]).T if len(shape) == 2 else flat
        return _value(base, data, cast=True)

    def function(self, name: str, values: list[_Value]) -> _Value:
        counts = {"clamp": 3, "min": 2, "max": 2, "abs": 1, "sin": 1, "cos": 1, "sqrt": 1}
        if len(values) != counts[name] or any(v.base != "float" or v.data.ndim > 1 for v in values):
            raise UnsupportedPositionError("function_arguments")
        shapes = {v.data.shape for v in values if v.data.shape}
        if len(shapes) > 1:
            raise UnsupportedPositionError("function_shape")
        arrays = [v.data for v in values]
        if name == "clamp":
            if np.any(arrays[1] > arrays[2]):
                raise UnsupportedPositionError("clamp_bounds")
            result = np.minimum(np.maximum(arrays[0], arrays[1]), arrays[2])
        elif name == "min":
            result = np.minimum(*arrays)
        elif name == "max":
            result = np.maximum(*arrays)
        else:
            with np.errstate(all="ignore"):
                result = {"abs": np.abs, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt}[name](arrays[0])
        return _value("float", result)

    def binary(self, op: str, left: Any, right: Any) -> _Value:
        a, b = self.numeric(left), self.numeric(right)
        if a.base != b.base or a.base == "bool":
            raise UnsupportedPositionError("binary_type")
        x, y = a.data, b.data
        if a.base in {"int", "uint"}:
            if x.ndim > 1 or y.ndim > 1 or x.shape and y.shape and x.shape != y.shape:
                raise UnsupportedPositionError("integer_shape")
            if op not in {"+", "-", "*"}:
                raise UnsupportedPositionError("integer_operation")
            wide = {"+": np.add, "-": np.subtract, "*": np.multiply}[op](x.astype(object), y.astype(object))
            lower, upper = (0, 2**32 - 1) if a.base == "uint" else (-2**31, 2**31 - 1)
            if np.any(wide < lower) or np.any(wide > upper):
                raise UnsupportedPositionError("integer_range")
            return _value(a.base, np.asarray(wide, dtype=np.int64))
        if op == "*" and x.ndim and y.ndim and (x.ndim == 2 or y.ndim == 2):
            if x.shape[-1] != y.shape[0]:
                raise UnsupportedPositionError("matrix_product_shape")
            result = x @ y
        else:
            if x.shape and y.shape and x.shape != y.shape:
                raise UnsupportedPositionError("component_shape")
            if op == "%" or op == "/" and np.any(y == 0):
                raise UnsupportedPositionError("unsupported_or_undefined_arithmetic")
            with np.errstate(all="ignore"):
                result = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}[op](x, y)
        return _value("float", result)

    def get(self, node: Expression, depth: int = 0) -> Any:
        if depth > 64:
            raise UnsupportedPositionError("dependency_depth")
        cached = self.cache.get(id(node))
        if cached is not None:
            return cached
        self.work += 1
        if self.work > 4096:
            raise UnsupportedPositionError("evaluation_budget")
        op, parts = node.operation, node.parts
        if op == "unknown":
            raise UnsupportedPositionError(parts[0])
        if op == "input":
            name, kind = parts
            if name not in self.inputs:
                raise UnsupportedPositionError("missing_input")
            value = self.external(kind, self.inputs[name])
        elif op == "typed":
            value = self.coerce(parts[0], self.get(parts[1], depth + 1))
        elif op == "literal":
            token = parts[0]
            if token in {"true", "false"}:
                value = _value("bool", token == "true")
            elif token[-1:] in {"u", "U"}:
                if not token[:-1].isdecimal():
                    raise UnsupportedPositionError("integer_literal")
                value = _value("uint", int(token[:-1]))
            elif any(c in token for c in ".eEfF"):
                value = _value("float", float(token.rstrip("fF")))
            else:
                value = _value("int", int(token))
        elif op == "call":
            value = self.constructor(parts[0], parts[1], depth)
        elif op == "field":
            value = self.field(self.get(parts[0], depth + 1), parts[1], depth)
        elif op == "index":
            value = self.indexed(self.get(parts[0], depth + 1), self.get(parts[1], depth + 1))
        elif op == "binary":
            value = self.binary(parts[0], self.get(parts[1], depth + 1), self.get(parts[2], depth + 1))
        elif op == "unary":
            operand = self.numeric(self.get(parts[1], depth + 1))
            if operand.base == "bool" or operand.base == "uint" and parts[0] == "-":
                raise UnsupportedPositionError("unary_type")
            value = _value(operand.base, operand.data if parts[0] == "+" else -operand.data)
        elif op == "write_field":
            original = self.numeric(self.get(parts[0], depth + 1))
            replacement = self.numeric(self.get(parts[2], depth + 1))
            if original.data.ndim != 1 or replacement.base != original.base:
                raise UnsupportedPositionError("write_type")
            indices = _swizzle(parts[1], len(original.data), write=True)
            expected = () if len(indices) == 1 else (len(indices),)
            if replacement.data.shape != expected:
                raise UnsupportedPositionError("write_shape")
            data = original.data.copy()
            data[list(indices)] = replacement.data
            value = _value(original.base, data)
        else:
            raise UnsupportedPositionError("unknown_ir")
        self.cache[id(node)] = value
        return value


def position_from_program(program: PositionProgram, inputs: dict[str, Any], *,
                          vertex: int = 0, instance: int = 0) -> PositionResult:
    """Missing, invalid or over-budget dependencies return an explicit unknown."""
    if (not isinstance(program, PositionProgram) or not isinstance(inputs, dict)
            or len(inputs) > 512 or len(program.structures) > 64):
        return PositionResult(None, "program_or_inputs")
    if any(type(v) is not int or not 0 <= v <= 2**31 - 1 for v in (vertex, instance)):
        return PositionResult(None, "draw_index")
    try:
        values = {**inputs, "gl_VertexID": vertex, "gl_InstanceID": instance}
        result = _Evaluation(program, values).get(program.position)
        if not isinstance(result, _Value) or result.base != "float" or result.data.shape != (4,):
            raise UnsupportedPositionError("position_type")
        return PositionResult(tuple(float(v) for v in result.data))
    except (UnsupportedPositionError, ValueError, TypeError, OverflowError, KeyError) as why:
        return PositionResult(None, str(why))


def position_from_source(source: str, inputs: dict[str, Any], **indices) -> PositionResult:
    try:
        program = parse_positions(source)
    except UnsupportedPositionError as why:
        return PositionResult(None, str(why))
    return position_from_program(program, inputs, **indices)


@invariant("perception.shader_positions_use_declared_operations", scope="perception",
           owner="core/perception/shader_position_math.py", observational=False)
def _shader_position_invariant() -> tuple:
    source = """#version 300 es
    in vec2 point; uniform mat4 transform;
    void main(){ gl_Position=transform*vec4(point,0.0,1.0);
                 gl_Position.xy=gl_Position.yx; }"""
    inputs = {"point": (2., 3.), "transform": np.eye(4, dtype=np.float32)}
    result = position_from_source(source, inputs)
    assert result.position == (3., 2., 0., 1.)
    assert position_from_source(source, {"point": (2., 3.)}).position is None
    assert position_from_source(source.replace("transform*", "secret*"), inputs).position is None
    return ()
