"""Code that looks wrong: general bug patterns found in a program's syntax tree, each with the edits that would mend it.

A person debugging someone else's program does not read every line with equal
suspicion. Some shapes are almost never what was meant: a quantity multiplied
by a literal zero, two branches of one test that do exactly the same thing, a
wall that turns the ball back on one side and not the other, a width measured
along the vertical while a height sits unused beside it, the key called "up"
that moves things down. Each is a pattern in the tree, not a fact about any
particular program, and each suggests a small set of edits.

This finds them with tree-sitter, so it reads JavaScript (on its own or inside
a page's script tags) and Python the same way. A suspicion is only that: what
the program does, measured by running it
(core/self_modification/watching_a_program_run.py), decides which edits are
kept.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Edit", "Suspicion", "applied", "scripts_in", "what_looks_wrong"]


@dataclass(frozen=True)
class Edit:
    """Replace the bytes from ``start`` to ``end`` of the source with ``text``."""

    start: int
    end: int
    text: str

    def says(self, source: str) -> str:
        """What the change does, as it follows "I": each side set as code, so a ``*`` in it is not read as emphasis."""
        was = source[self.start:self.end]
        if not was:
            return f"added {_as_code(self.text)}"
        if not self.text:
            return f"removed {_as_code(was)}"
        return f"changed {_as_code(was)} to {_as_code(self.text)}"


def _as_code(text: str) -> str:
    shown = repr(text)[1:-1]
    return f"`` {shown} ``" if "`" in shown else f"`{shown}`"


@dataclass
class Suspicion:
    """One place that looks wrong, why, and the edits that would mend it."""

    pattern: str
    line: int
    why: str
    edits: list[Edit] = field(default_factory=list)
    function: str = ""


def applied(source: str, edits: list[Edit]) -> str:
    """The source with every edit made, latest first so the offsets hold."""
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        source = source[: edit.start] + edit.text + source[edit.end :]
    return source


# -- reading the program --------------------------------------------------


def scripts_in(source: str, suffix: str) -> list[tuple[int, str]]:
    """The code in a file, as (offset, code) pairs: the whole of a script, or each script tag of a page."""
    if suffix in (".js", ".mjs", ".py"):
        return [(0, source)]
    found = []
    for match in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", source, re.DOTALL | re.IGNORECASE):
        found.append((match.start(1), match.group(1)))
    return found


def _parser(language: str) -> Any:
    from tree_sitter import Language, Parser

    if language == "python":
        import tree_sitter_python as grammar
    else:
        import tree_sitter_javascript as grammar
    return Parser(Language(grammar.language()))


def _walk(node: Any) -> Iterator[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _text(node: Any, code: bytes) -> str:
    return code[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _enclosing_function(node: Any, code: bytes) -> str:
    while node is not None:
        if node.type in ("function_declaration", "method_definition", "function_definition"):
            name = node.child_by_field_name("name")
            return _text(name, code) if name is not None else ""
        node = node.parent
    return ""


@dataclass
class _Program:
    code: bytes
    root: Any
    offset: int
    text: str

    def at(self, node: Any) -> tuple[int, int]:
        """Character offsets of a node in the whole file."""
        before = self.code[: node.start_byte].decode("utf-8", "replace")
        inside = _text(node, self.code)
        start = self.offset + len(before)
        return start, start + len(inside)

    def line(self, node: Any) -> int:
        return self.text[: self.at(node)[0]].count("\n") + 1


# -- the patterns -----------------------------------------------------------


def _always_zero(program: _Program) -> Iterator[Suspicion]:
    """A product with a literal zero in it is a constant, whatever its other side says."""
    for node in _walk(program.root):
        if node.type not in ("binary_expression", "binary_operator"):
            continue
        operator = node.child_by_field_name("operator")
        left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
        if operator is None or _text(operator, program.code) != "*" or left is None or right is None:
            continue
        for zero, other in ((right, left), (left, right)):
            if _text(zero, program.code).strip() in ("0", "0.0"):
                start, end = program.at(node)
                kept = _text(other, program.code)
                if kept.startswith("(") and kept.endswith(")"):
                    kept = kept[1:-1]
                yield Suspicion(
                    "always zero", program.line(node),
                    f"`{_text(node, program.code)[:60]}` is always zero",
                    [Edit(start, end, kept)], _enclosing_function(node, program.code),
                )


def _siblings(names: set[str], name: str) -> list[str]:
    """Other names in the program that differ from this one only in their first word: playerScore, computerScore."""
    parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", name)
    if len(parts) < 2:
        return []
    rest = "".join(parts[1:])
    return sorted(
        other for other in names
        if other != name and other.endswith(rest) and len(other) > len(rest)
        and other[: -len(rest)].isidentifier() and other[: -len(rest)].lower() != parts[0].lower()
    )


def _branches(node: Any) -> list[Any]:
    """The bodies of an if / else-if chain, in order."""
    bodies = []
    while node is not None and node.type == "if_statement":
        consequence = node.child_by_field_name("consequence")
        if consequence is not None:
            bodies.append(consequence)
        alternative = node.child_by_field_name("alternative")
        node = None
        if alternative is not None:
            inner = [c for c in alternative.children if c.type in ("if_statement", "statement_block")]
            node = inner[0] if inner and inner[0].type == "if_statement" else None
            if inner and inner[0].type == "statement_block":
                bodies.append(inner[0])
    return bodies


def _statements(body: Any, code: bytes) -> list[str]:
    return [" ".join(_text(child, code).split()) for child in body.named_children]


def _mirrors(a: str, b: str) -> bool:
    """Two statements the same but for the signs of their numbers: serve(1) and serve(-1)."""
    plain = lambda text: re.sub(r"-\s*(?=\d)", "", text)  # noqa: E731
    return a != b and plain(a) == plain(b)


def _same_in_both_branches(program: _Program) -> Iterator[Suspicion]:
    """Branches of one test that should mirror each other, with a statement that does not.

    Two branches doing exactly the same thing, or doing mirrored things (one
    serves to the left, the other to the right) except for one statement left
    identical, which names something that has a counterpart (playerScore,
    computerScore). That statement was meant to be the counterpart's in one
    of them.
    """
    names = {_text(n, program.code) for n in _walk(program.root) if n.type == "identifier"}
    for node in _walk(program.root):
        if node.type != "if_statement" or (node.parent is not None and node.parent.type == "else_clause"):
            continue
        bodies = _branches(node)
        for index, first in enumerate(bodies):
            for second in bodies[index + 1 :]:
                one, other = _statements(first, program.code), _statements(second, program.code)
                if len(one) != len(other):
                    continue
                pairs = list(zip(one, other, strict=True))
                same = [k for k, (a, b) in enumerate(pairs) if a == b]
                mirrored = all(a == b or _mirrors(a, b) for a, b in pairs)
                if not same or not mirrored:
                    continue
                edits = []
                for body in (first, second):
                    for k in same:
                        for name_node in _walk(body.named_children[k]):
                            if name_node.type != "identifier":
                                continue
                            start, end = program.at(name_node)
                            edits += [Edit(start, end, sibling) for sibling in _siblings(names, _text(name_node, program.code))]
                if edits:
                    yield Suspicion(
                        "same in both branches", program.line(first.named_children[same[0]]),
                        "two branches that mirror each other do the same thing here: "
                        f"`{one[same[0]][:50]}`",
                        edits, _enclosing_function(node, program.code),
                    )


_UPPER = re.compile(r"^\s*(?P<thing>[\w.]+)\s*\+\s*[\w.]+\s*>\s*(?P<bound>[\w.]+)\s*$")
_LOWER = re.compile(r"^\s*(?P<thing>[\w.]+)\s*<\s*0\s*$")


def _one_sided_boundary(program: _Program) -> Iterator[Suspicion]:
    """A thing turned back at the far wall of an axis and not the near one, or the other way round."""
    for node in _walk(program.root):
        if node.type != "if_statement":
            continue
        condition = node.child_by_field_name("condition")
        body = node.child_by_field_name("consequence")
        if condition is None or body is None:
            continue
        test = _text(condition, program.code).strip("() ")
        upper = _UPPER.match(test)
        if upper is None:
            continue
        thing = upper.group("thing")
        function = node.parent
        while function is not None and function.type not in ("function_declaration", "statement_block"):
            function = function.parent
        scope = _text(function, program.code) if function is not None else program.text
        if re.search(rf"{re.escape(thing)}\s*<\s*0\b|{re.escape(thing)}\s*<=\s*0\b", scope):
            continue
        mirrored = _mirror(_text(body, program.code), thing)
        if mirrored is None:
            continue
        start, _end = program.at(node)
        indent = program.text[program.text.rfind("\n", 0, start) + 1 : start]
        yield Suspicion(
            "one-sided boundary", program.line(node),
            f"`{thing}` is turned back at the far edge and nothing turns it back at the near one",
            [Edit(start, start, f"if ({thing} < 0) {mirrored}\n{indent}")],
            _enclosing_function(node, program.code),
        )


def _mirror(body: str, thing: str) -> str | None:
    """The body of a far-edge check, rewritten for the near edge: placed at zero, sent the other way."""
    mirrored = re.sub(rf"({re.escape(thing)}\s*=\s*)[^;]+;", r"\g<1>0;", body)
    flipped = re.sub(r"=\s*-\s*Math\.abs\(", "= Math.abs(", mirrored)
    if flipped == mirrored:
        flipped = re.sub(r"=\s*Math\.abs\(", "= -Math.abs(", mirrored)
    return flipped if flipped != body else None


def _crossed_dimension(program: _Program) -> Iterator[Suspicion]:
    """A width beside a vertical coordinate, where the program has the height it pairs with elsewhere."""
    names = {_text(n, program.code) for n in _walk(program.root) if n.type == "identifier"}
    for node in _walk(program.root):
        if node.type != "binary_expression":
            continue
        text = _text(node, program.code)
        if len(text) > 80 or not re.search(r"\.y\b", text):
            continue
        for name_node in _walk(node):
            name = _text(name_node, program.code) if name_node.type == "identifier" else ""
            for wide, tall in (("_W", "_H"), ("WIDTH", "HEIGHT"), ("Width", "Height")):
                if name.endswith(wide) and name[: -len(wide)] + tall in names:
                    start, end = program.at(name_node)
                    yield Suspicion(
                        "crossed dimension", program.line(name_node),
                        f"`{name}` is a width, used here along y beside `{text[:50]}`",
                        [Edit(start, end, name[: -len(wide)] + tall)],
                        _enclosing_function(node, program.code),
                    )


_DIRECTIONS = {"up": ("y", -1), "down": ("y", 1), "left": ("x", -1), "right": ("x", 1)}


def _direction_against_its_name(program: _Program) -> Iterator[Suspicion]:
    """A branch on a key named for a direction that moves things the other way on a screen.

    Screens count down the page: up is less y, left is less x. A key named
    ArrowUp whose branch adds to a y is the key moving things down.
    """
    for node in _walk(program.root):
        if node.type != "if_statement":
            continue
        condition = node.child_by_field_name("condition")
        body = node.child_by_field_name("consequence")
        if condition is None or body is None:
            continue
        test = _text(condition, program.code).lower()
        named = [word for word in _DIRECTIONS if re.search(rf"\barrow{word}\b|['\"]{word}['\"]", test)]
        if len(named) != 1:
            continue
        axis, sign = _DIRECTIONS[named[0]]
        for statement in _walk(body):
            if statement.type != "augmented_assignment_expression":
                continue
            target = statement.child_by_field_name("left")
            operator = statement.child_by_field_name("operator")
            if target is None or operator is None or not _text(target, program.code).endswith(f".{axis}"):
                continue
            does = 1 if _text(operator, program.code) == "+=" else -1 if _text(operator, program.code) == "-=" else 0
            if does and does != sign:
                start, end = program.at(operator)
                yield Suspicion(
                    "direction against its name", program.line(statement),
                    f"the {named[0]} key moves `{_text(target, program.code)}` the other way",
                    [Edit(start, end, "-=" if does > 0 else "+=")],
                    _enclosing_function(node, program.code),
                )


_PATTERNS = (_always_zero, _same_in_both_branches, _one_sided_boundary, _crossed_dimension, _direction_against_its_name)


def what_looks_wrong(source: str, suffix: str) -> list[Suspicion]:
    """Every suspicion in a file, in the order they appear in it."""
    language = "python" if suffix == ".py" else "javascript"
    parser = _parser(language)
    found: list[Suspicion] = []
    for offset, code in scripts_in(source, suffix):
        encoded = code.encode("utf-8")
        program = _Program(encoded, parser.parse(encoded).root_node, offset, source)
        for pattern in _PATTERNS:
            found.extend(pattern(program))
    unique: dict[tuple[str, int, tuple], Suspicion] = {}
    for suspicion in found:
        unique.setdefault((suspicion.pattern, suspicion.line, tuple(suspicion.edits)), suspicion)
    return sorted(unique.values(), key=lambda s: s.line)
