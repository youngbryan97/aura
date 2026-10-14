"""Parse a bounded straight-line GLSL ES position subset into immutable data.

No shader text executes on the host. References bind to the value at each
statement; a later swizzle write cannot change an earlier expression. Unsupported
control flow, preprocessing and calls decline the source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class UnsupportedPositionError(ValueError):
    """A source or dependency has semantics outside the admitted subset."""


@dataclass(frozen=True)
class ShaderType:
    name: str
    count: int = 0


@dataclass(frozen=True)
class Expression:
    operation: str
    parts: tuple


@dataclass(frozen=True)
class PositionProgram:
    position: Expression
    structures: tuple[tuple[str, tuple[tuple[str, ShaderType], ...]], ...]
    inputs: tuple[tuple[str, ShaderType], ...]
    blocks: tuple[tuple[str, str | None], ...] = ()
    outputs: tuple[tuple[str, Expression], ...] = ()
    input_qualifiers: tuple[tuple[str, tuple[str, ...]], ...] = ()


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[uUfF]?|[{}()\[\].,;=+*/%-]")
_QUALIFIERS = frozenset({"in", "out", "uniform", "const", "smooth", "flat", "centroid", "invariant", "highp"})
_NUMERIC = frozenset({"float", "int", "uint", "bool"} | {prefix + str(i) for prefix in ("vec", "ivec", "uvec", "bvec")
                      for i in (2, 3, 4)} | {"mat" + str(i) for i in (2, 3, 4)}
                     | {f"mat{c}x{r}" for c in (2, 3, 4) for r in (2, 3, 4)})
_OPAQUE = frozenset({"sampler2D"})
_PURE_CALLS = frozenset({"clamp", "min", "max", "abs", "sin", "cos", "sqrt", "texture"})


def _tokens(source: str) -> tuple[str, ...]:
    if not isinstance(source, str) or len(source.encode("utf-8")) > 32768:
        raise UnsupportedPositionError("source_budget")
    source = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    lines = source.splitlines()
    directives = [line.strip() for line in lines if line.lstrip().startswith("#")]
    if directives != ["#version 300 es"]:
        raise UnsupportedPositionError("preprocessor_or_version")
    source = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    tokens, previous = [], 0
    for match in _TOKEN.finditer(source):
        if source[previous:match.start()].strip():
            raise UnsupportedPositionError("unsupported_token")
        tokens.append(match.group())
        if len(tokens) > 8192:
            raise UnsupportedPositionError("token_budget")
        previous = match.end()
    if source[previous:].strip():
        raise UnsupportedPositionError("unsupported_token")
    return tuple(tokens)


class _Parser:
    def __init__(self, source: str) -> None:
        self.tokens, self.cursor = _tokens(source), 0
        self.structures: dict[str, tuple[tuple[str, ShaderType], ...]] = {}
        self.blocks: dict[str, str | None] = {}
        self.inputs = {"gl_InstanceID": ShaderType("int"), "gl_VertexID": ShaderType("int"), "gl_FragCoord": ShaderType("vec4")}
        self.output_names: list[str] = []
        self.input_qualifiers: dict[str, tuple[str, ...]] = {}
        self.variables = {name: Expression("input", (name, kind)) for name, kind in self.inputs.items()}
        self.variables["gl_Position"] = Expression("unknown", ("unassigned_position",))
        self.variables["gl_PointSize"] = Expression("unknown", ("unassigned_point_size",))
        self.types = {**self.inputs, "gl_Position": ShaderType("vec4"), "gl_PointSize": ShaderType("float")}
        self.main_seen = False

    def peek(self) -> str:
        return self.tokens[self.cursor] if self.cursor < len(self.tokens) else ""

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if not token or expected is not None and token != expected:
            raise UnsupportedPositionError("syntax")
        self.cursor += 1
        return token

    def identifier(self) -> str:
        value = self.take()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
            raise UnsupportedPositionError("identifier")
        return value

    def array_type(self, name: str) -> ShaderType:
        count = 0
        if self.peek() == "[":
            self.take("[")
            value = self.take()
            if not value.isdecimal() or not 1 <= int(value) <= 256:
                raise UnsupportedPositionError("array_extent")
            count = int(value)
            self.take("]")
        return ShaderType(name, count)

    def qualifiers(self) -> tuple[str, ...]:
        found = []
        if self.peek() == "layout":
            self.take("layout")
            self.take("(")
            while self.peek() != ")":
                # Storage layout belongs to reflected bytes, never guessed offsets.
                if self.take() not in {"std140", "column_major", "row_major", "location", "=", ","} and not self.tokens[self.cursor - 1].isdecimal():
                    raise UnsupportedPositionError("layout_qualifier")
            self.take(")")
        while self.peek() in _QUALIFIERS:
            found.append(self.take())
        if self.peek() in ("lowp", "mediump"):
            raise UnsupportedPositionError("unsupported_precision")
        return tuple(found)

    def fields(self) -> tuple[tuple[str, ShaderType], ...]:
        self.take("{")
        fields = []
        while self.peek() != "}":
            self.qualifiers()
            kind, name = self.identifier(), self.identifier()
            if kind not in _NUMERIC and kind not in self.structures:
                raise UnsupportedPositionError("field_type")
            fields.append((name, self.array_type(kind)))
            self.take(";")
            if len(fields) > 32 or len({n for n, _ in fields}) != len(fields):
                raise UnsupportedPositionError("field_budget_or_shadowing")
        self.take("}")
        return tuple(fields)

    def declare(self, name: str, kind: ShaderType, value: Expression | None, external: bool) -> None:
        if name in self.variables or len(self.variables) >= 512:
            raise UnsupportedPositionError("shadowing_or_variable_budget")
        self.types[name] = kind
        if external:
            if value is not None:
                raise UnsupportedPositionError("initialized_external")
            self.inputs[name] = kind
            self.variables[name] = Expression("input", (name, kind))
        else:
            self.variables[name] = Expression("typed", (kind, value or Expression("unknown", ("unassigned_variable",))))

    def parse(self) -> PositionProgram:
        while self.peek():
            if self.peek() == "precision":
                self.take("precision")
                self.take("highp")
                if self.take() not in {"float", "int"}:
                    raise UnsupportedPositionError("precision_type")
                self.take(";")
                continue
            if self.peek() == "struct":
                self.take("struct")
                name = self.identifier()
                if name in self.structures or name in _NUMERIC or len(self.structures) >= 32:
                    raise UnsupportedPositionError("structure_budget_or_shadowing")
                self.structures[name] = self.fields()
                self.take(";")
                continue
            qualifiers = self.qualifiers()
            kind = self.identifier()
            if kind == "void":
                if qualifiers or self.main_seen:
                    raise UnsupportedPositionError("function")
                self.take("main")
                self.take("(")
                self.take(")")
                self.main_seen = True
                self.body()
                continue
            if self.peek() == "{":
                if "uniform" not in qualifiers:
                    raise UnsupportedPositionError("interface_block")
                if kind in self.blocks or len(self.blocks) >= 32:
                    raise UnsupportedPositionError("block_budget_or_shadowing")
                fields = self.fields()
                if self.peek() == ";":
                    self.blocks[kind] = None
                    for name, field_type in fields:
                        self.declare(name, field_type, None, True)
                else:
                    instance = self.identifier()
                    self.blocks[kind] = instance
                    if kind in self.structures:
                        raise UnsupportedPositionError("block_shadowing")
                    self.structures[kind] = fields
                    self.declare(instance, self.array_type(kind), None, True)
                self.take(";")
                continue
            if kind not in _NUMERIC and kind not in self.structures and kind not in _OPAQUE:
                raise UnsupportedPositionError("global_type")
            name = self.identifier()
            declared = self.array_type(kind)
            expression = None
            if self.peek() == "=":
                self.take("=")
                expression = self.expression()
            self.take(";")
            external = "in" in qualifiers or "uniform" in qualifiers
            self.declare(name, declared, expression, external)
            if external:
                self.input_qualifiers[name] = qualifiers
            if "out" in qualifiers:
                self.output_names.append(name)
        if not self.main_seen:
            raise UnsupportedPositionError("missing_main")
        return PositionProgram(self.variables["gl_Position"], tuple(self.structures.items()), tuple(self.inputs.items()),
                               tuple(self.blocks.items()), tuple((n, self.variables[n]) for n in self.output_names),
                               tuple(self.input_qualifiers.items()))

    def body(self) -> None:
        self.take("{")
        returned = False
        while self.peek() != "}":
            if returned:
                raise UnsupportedPositionError("statements_after_return")
            if self.peek() == "return":
                self.take("return")
                self.take(";")
                returned = True
                continue
            self.qualifiers()
            name = self.identifier()
            if name in _NUMERIC or name in self.structures:
                variable = self.identifier()
                kind = self.array_type(name)
                value = None
                if self.peek() == "=":
                    self.take("=")
                    value = self.expression()
                self.take(";")
                self.declare(variable, kind, value, False)
                continue
            if name not in self.variables or name in self.inputs:
                raise UnsupportedPositionError("assignment_target")
            field = None
            if self.peek() == ".":
                self.take(".")
                field = self.identifier()
                if not re.fullmatch(r"[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4}", field) or len(set(field)) != len(field):
                    raise UnsupportedPositionError("write_swizzle")
            self.take("=")
            value = self.expression()
            self.take(";")
            if field is not None:
                value = Expression("write_field", (self.variables[name], field, value))
            self.variables[name] = Expression("typed", (self.types[name], value))
        self.take("}")

    def expression(self, minimum: int = 0, depth: int = 0) -> Expression:
        if depth > 48:
            raise UnsupportedPositionError("expression_depth")
        token = self.take()
        if token in ("+", "-"):
            left = Expression("unary", (token, self.expression(30, depth + 1)))
        elif token == "(":
            left = self.expression(0, depth + 1)
            self.take(")")
        elif re.fullmatch(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[uUfF]?", token):
            left = Expression("literal", (token,))
        elif token in ("true", "false"):
            left = Expression("literal", (token,))
        elif self.peek() == "(":
            if token not in _NUMERIC and token not in self.structures and token not in _PURE_CALLS:
                raise UnsupportedPositionError("unsupported_call")
            self.take("(")
            args = []
            if self.peek() != ")":
                while True:
                    args.append(self.expression(0, depth + 1))
                    if len(args) > 32:
                        raise UnsupportedPositionError("argument_budget")
                    if self.peek() != ",":
                        break
                    self.take(",")
            self.take(")")
            left = Expression("call", (token, tuple(args)))
        elif token in self.variables:
            left = self.variables[token]
        else:
            raise UnsupportedPositionError("unbound_symbol")
        precedence = {"+": 10, "-": 10, "*": 20, "/": 20, "%": 20}
        while self.peek():
            op = self.peek()
            if op == ".":
                self.take(".")
                left = Expression("field", (left, self.identifier()))
            elif op == "[":
                self.take("[")
                index = self.expression(0, depth + 1)
                self.take("]")
                left = Expression("index", (left, index))
            elif op in precedence and precedence[op] >= minimum:
                self.take()
                right = self.expression(precedence[op] + 1, depth + 1)
                left = Expression("binary", (op, left, right))
            else:
                break
        return left


def parse_positions(source: str) -> PositionProgram:
    """Unsupported source raises a reason before any host arithmetic occurs."""
    return _Parser(source).parse()
