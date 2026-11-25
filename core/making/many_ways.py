"""The ways a program in JavaScript could be put right, thousands of them, each a small change written by her own Python.

A change is made at a place in the program's syntax tree: a comparison turned, a bound moved, one name for another that is
in reach, a number for one that the rule itself gives, an expression the program writes elsewhere, a guard against what a
failure says was not there, a count that nothing adds to (the templates of PAR and TBar, Kim and others 2013, Liu and
others 2019; the ingredients of ssFix and CapGen; her own core/self_modification/fix_templates.py and ingredient_repair.py,
which do this for Python).

What lets thousands be tried is that the program is not loaded again for each. All of them are written into one program,
each behind its own number, and the page that holds it runs the tests with one of them at a time let through (mutant
schemata, Untch, Offutt and Harrold 1993). A way is tried in milliseconds where her model writes one in minutes. Where to
change it comes from what the failing tests run and the passing ones do not (Ochiai, as her Python repairs are localised).

Nothing here is taken for a mend. What is left after the tests is tried whole, and kept only where less fails
(core/making/maker.py).
"""
from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Making.ManyWays")

__all__ = ["FORCED", "Evidence", "Read", "Way", "conditions", "evidence_of", "forced", "likeliest", "made", "with_all", "ways"]

#: The most ways at one place, in all, and of those the most that are written into one program.
AT_A_PLACE, IN_ALL, WRITTEN_IN = 48, 60000, 7000

#: The ways of each sort, by how often that sort is what was wrong, as the studies of real mends have it; what she has
#: found of her own (core/making/what_has_worked.py) is counted over this.
SORTS = {
    "guard": 1.0, "name_for_name": 0.8, "field_for_field": 0.85, "comparison": 0.9, "number_from_the_rule": 0.95, "shift_by_one": 0.75,
    "boolean": 0.6, "arithmetic": 0.55, "negate": 0.6, "literal": 0.6, "words_for_words": 0.8, "time": 0.7, "kept_within": 0.55,
    "ingredient": 0.5, "swap_arguments": 0.5, "whole_number": 0.45, "same_kind": 0.5, "returned": 0.8, "list": 0.45, "counted": 0.5,
    "dropped": 0.25, "missing_import": 1.5, "template": 0.7, "condition": 0.65, "joined": 0.6, "unit": 0.4, "copied": 0.35, "row": 1.4,
}

_COMPARE = {"<": ("<=", ">", ">="), "<=": ("<", ">=", ">"), ">": (">=", "<", "<="), ">=": (">", "<=", "<"),
            "===": ("!==",), "!==": ("===",), "==": ("!=",), "!=": ("==",)}
_LOGIC = {"&&": ("||",), "||": ("&&",), "??": ("||",)}
_SUMS = {"+": ("-",), "-": ("+",), "*": ("/",), "/": ("*",), "%": ("/",)}
_AUGMENTED = {"+=": ("-=", "="), "-=": ("+=", "="), "*=": ("/=",), "/=": ("*=",), "=": ("+=", "-=")}
_ROUND = {"floor": ("round", "ceil"), "ceil": ("round", "floor"), "round": ("floor", "ceil"), "min": ("max",), "max": ("min",)}
_LISTS = {"push": ("unshift",), "unshift": ("push",), "some": ("every",), "every": ("some",), "find": ("filter",), "findIndex": ("indexOf",)}
#: What a place must be, for an expression to be put in it.
_VALUES = frozenset({"identifier", "member_expression", "subscript_expression", "call_expression", "binary_expression", "number", "ternary_expression",
                     "parenthesized_expression", "string", "unary_expression", "true", "false", "template_string", "array", "null"})


@dataclass(frozen=True)
class Way:
    """One change: where, what is put there, what sort it is, and the block of the program it is in."""

    id: int
    place: int
    sort: str
    start: int
    end: int
    text: str
    block: int
    says: str = ""


@dataclass(frozen=True)
class Evidence:
    """What the failing tests say: the numbers their rules give, the words they use it by, the names they read, and how it broke."""

    numbers: tuple[str, ...] = ()
    words: tuple[str, ...] = ()
    names: frozenset[str] = frozenset()
    broke: tuple[str, ...] = ()


_SAID_NUMBERS = {w: str(n) for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def evidence_of(failing: Sequence[tuple[str, str, str]]) -> Evidence:
    """The evidence in some failing tests: (its name, which is its rule; its text; why it failed)."""
    numbers: dict[str, None] = {}
    words: dict[str, None] = {}
    names: set[str] = set()
    broke: list[str] = []
    for name, text, why in failing:
        said = [*re.findall(r"(?<![\w.])\d+(?:\.\d+)?", name), *(_SAID_NUMBERS[w] for w in re.findall(r"[a-z]+", name.lower()) if w in _SAID_NUMBERS)]
        for number in said:
            numbers.setdefault(number)
        # What is done in a test is what the program has to answer to. What is expected of it is not for the program to say
        # back: LIVE 24 November 2026, "0.7" was expected to be seen, and "Score" was changed to "0.7".
        for _what, given in re.findall(r"\.(click|type|choose|enter|press|hold|send)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text):
            for _quote, word in re.findall(r"([\"'`])((?:\\.|(?!\1).){1,40})\1", given):
                words.setdefault(word)
        names.update(re.findall(r"\.([A-Za-z_$][\w$]*)", text))
        if why and re.search(r"is not (?:defined|a function|iterable)|of (?:undefined|null)|not found", why):
            broke.append(why)
    return Evidence(tuple(numbers)[:30], tuple(words)[:40], frozenset(names), tuple(broke))


def _parser() -> Any:
    import tree_sitter_javascript as grammar
    from tree_sitter import Language, Parser

    return Parser(Language(grammar.language()))


@dataclass
class Read:
    """A program, read: its tree, its functions, its blocks, and what each block is in."""

    code: bytes
    root: Any
    #: The byte at which each block begins, and the block's number.
    blocks: dict[tuple[int, int], int] = field(default_factory=dict)
    #: The state's fields, by the kind of thing each holds.
    fields: Mapping[str, str] = field(default_factory=dict)
    #: The program's own functions, and the names the state has of itself.
    functions: tuple[str, ...] = ()
    top: frozenset[str] = frozenset()
    #: For each name something in the state has, the others beside it that hold the same kind of thing.
    beside: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, code: str, state: Mapping[str, Any] | None = None, functions: Iterable[str] = ()) -> Read | None:
        try:
            tree = _parser().parse(code.encode())
        except ImportError as exc:
            logger.info("The program is not read for the ways it could be changed: %s", exc)
            return None
        if tree.root_node.has_error:
            return None
        read = cls(code.encode(), tree.root_node, fields=_fields(state or {}), functions=tuple(functions),
                   top=frozenset(map(str, state or {})), beside=_beside(state or {}))
        for node in _walk(tree.root_node):
            if node.type == "statement_block":
                read.blocks[(node.start_byte, node.end_byte)] = len(read.blocks)
            elif node.type == "arrow_function" and (body := node.child_by_field_name("body")) is not None and body.type != "statement_block":
                read.blocks[(body.start_byte, body.end_byte)] = len(read.blocks)
            elif (node.type in {"if_statement", "for_statement", "for_in_statement", "while_statement", "else_clause"}
                  and (alone := _alone(node)) is not None):
                read.blocks[(alone.start_byte, alone.end_byte)] = len(read.blocks)
        return read

    def text(self, node: Any) -> str:
        return self.code[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def block_of(self, node: Any) -> int:
        while node is not None:
            found = self.blocks.get((node.start_byte, node.end_byte))
            if found is not None:
                return found
            node = node.parent
        return -1


def _walk(node: Any) -> Iterator[Any]:
    stack = [node]
    while stack:
        at = stack.pop()
        yield at
        stack.extend(reversed(at.children))


def _alone(node: Any) -> Any:
    """The one statement a branch or a loop is of, where it has no braces."""
    name = "consequence" if node.type == "if_statement" else "body"
    body = node.child_by_field_name(name)
    if node.type == "else_clause":
        body = next((c for c in node.named_children if c.type != "comment"), None)
    if body is None or body.type in {"statement_block", "if_statement", "lexical_declaration", "variable_declaration", "function_declaration", "empty_statement"}:
        return None
    return body


def _fields(state: Mapping[str, Any]) -> dict[str, str]:
    """Every name the state has, at any depth, by the kind of thing it holds."""
    out: dict[str, str] = {}

    def of(value: Any) -> None:
        if isinstance(value, Mapping):
            for name, held in value.items():
                out.setdefault(str(name), _kind(held))
                of(held)
        elif isinstance(value, list):
            for item in value[:4]:
                of(item)

    of(state)
    return out


def _beside(state: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    found: dict[str, dict[str, None]] = {}

    def of(value: Any) -> None:
        if isinstance(value, Mapping):
            for name, held in value.items():
                found.setdefault(str(name), {}).update(dict.fromkeys(str(o) for o, other in value.items() if o != name and _kind(other) == _kind(held)))
                of(held)
        elif isinstance(value, list):
            for item in value[:4]:
                of(item)

    of(state)
    return {name: tuple(others) for name, others in found.items()}


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "truth"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "words"
    return "list" if isinstance(value, list) else "thing" if isinstance(value, Mapping) else "nothing"


def _can_be_put_in(node: Any) -> bool:
    """Whether an expression can be put where this one is: not where it is given to, called, declared, or is a name of something."""
    parent = node.parent
    if parent is None or node.type not in _VALUES:
        return False
    kind = parent.type
    if kind == "catch_clause" or (node.type == "parenthesized_expression" and kind in {"if_statement", "while_statement", "do_statement", "switch_statement"}):
        return False  # its brackets are the statement's own; what is in them is what can be put another in place of
    if kind in {"formal_parameters", "object_pattern", "array_pattern", "rest_pattern", "import_statement", "class_body", "for_in_statement", "update_expression",
                "labeled_statement", "pair_pattern", "assignment_pattern"}:
        return kind == "for_in_statement" and parent.child_by_field_name("right") == node or kind == "assignment_pattern" and parent.child_by_field_name("right") == node
    if kind in {"assignment_expression", "augmented_assignment_expression"}:
        return parent.child_by_field_name("right") == node
    if kind == "variable_declarator":
        return parent.child_by_field_name("value") == node
    if kind in {"call_expression", "new_expression"}:
        return False  # what is called; its arguments are in a node of their own
    if kind == "member_expression":
        return parent.child_by_field_name("object") == node and node.type not in {"string", "number"} and not _is_given_to(parent)
    if kind == "subscript_expression":
        return not _is_given_to(parent) or parent.child_by_field_name("index") == node
    if kind == "pair":
        return parent.child_by_field_name("value") == node
    if kind in {"method_definition", "field_definition", "arrow_function"}:
        return kind == "arrow_function" and parent.child_by_field_name("body") == node
    if kind == "unary_expression":
        return parent.children[0].type not in {"delete", "typeof"}
    if kind in {"expression_statement", "program", "computed_property_name", "template_literal"}:
        return False
    return True


def _is_given_to(node: Any) -> bool:
    """Whether a member is what something is given to, at the end of the chain it is in."""
    while node.parent is not None and node.parent.type in {"member_expression", "subscript_expression"} and node.parent.child_by_field_name("object") == node:
        node = node.parent
    parent = node.parent
    return parent is not None and (parent.type in {"assignment_expression", "augmented_assignment_expression"} and parent.child_by_field_name("left") == node
                                   or parent.type == "update_expression" or parent.type == "unary_expression" and parent.children[0].type == "delete")


def _is_truth(node: Any) -> bool:
    """Whether what is here is used as a truth."""
    if node.type == "binary_expression" and node.child_by_field_name("operator").type in {*_COMPARE, "in", "instanceof"}:
        return True
    parent = node.parent
    while parent is not None and parent.type == "parenthesized_expression":
        node, parent = parent, parent.parent
    if parent is None:
        return False
    if parent.type in {"if_statement", "while_statement", "ternary_expression", "do_statement", "for_statement"}:
        return parent.child_by_field_name("condition") == node
    if parent.type == "unary_expression":
        return parent.children[0].type == "!"
    return parent.type == "binary_expression" and parent.child_by_field_name("operator").type in {"&&", "||"}


def _function_of(node: Any) -> Any:
    """The function of the program's own it is in, the outermost."""
    found = None
    while node is not None:
        if node.type in {"method_definition", "function_declaration", "function_expression", "arrow_function"}:
            found = node
        node = node.parent
    return found


def _in_reach(read: Read, node: Any) -> list[str]:
    """The names that can be used here: what the functions it is in are given, and what was declared before it in a block it is in."""
    names: dict[str, None] = {}
    at = node
    while at is not None:
        parent = at.parent
        if at.type in {"method_definition", "function_declaration", "function_expression", "arrow_function"}:
            given = at.child_by_field_name("parameters") or at.child_by_field_name("parameter")
            for n in _walk(given) if given is not None else ():
                if n.type in {"identifier", "shorthand_property_identifier_pattern"}:
                    names.setdefault(read.text(n))
        if parent is not None and parent.type in {"for_in_statement", "for_statement"}:
            left = parent.child_by_field_name("left") or parent.child_by_field_name("initializer")
            for n in _walk(left) if left is not None and left.end_byte <= node.start_byte else ():
                if n.type in {"identifier", "shorthand_property_identifier_pattern"}:
                    names.setdefault(read.text(n))
        for sibling in (parent.named_children if parent is not None else ()):
            if sibling.start_byte >= at.start_byte:
                break
            if sibling.type in {"lexical_declaration", "variable_declaration"}:
                for declared in sibling.named_children:
                    name = declared.child_by_field_name("name")
                    for n in _walk(name) if name is not None else ():
                        if n.type in {"identifier", "shorthand_property_identifier_pattern"}:
                            names.setdefault(read.text(n))
        at = parent
    return list(names)


def _state_name(read: Read, node: Any) -> str:
    """What the state is called here: what the function it is in is first given, where it is asked for what the state has."""
    function = _function_of(node)
    given = function.child_by_field_name("parameters") if function is not None else None
    first = next((n for n in _walk(given) if n.type == "identifier"), None) if given is not None else None
    if first is None:
        return ""
    name = read.text(first)
    asked = set(re.findall(rf"(?<![\w$.]){re.escape(name)}\.([A-Za-z_$][\w$]*)", read.text(function)))
    return name if asked & set(read.fields) else ""


def _short(text: str, most: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= most else text[:most] + "…"


class _Collected:
    """The ways found so far: no two the same at a place, and none that leaves it as it was."""

    def __init__(self, read: Read, at_a_place: int = AT_A_PLACE) -> None:
        self.read, self.out, self.most = read, [], at_a_place
        self.places: dict[tuple[int, int], int] = {}
        self._seen: set[tuple[int, int, str]] = set()
        self._at: dict[tuple[int, int], int] = {}

    def add(self, sort: str, node: Any, text: str, says: str = "") -> None:
        self.span(sort, node.start_byte, node.end_byte, text, self.read.block_of(node), says or f"changed `{_short(self.read.text(node))}` to `{_short(text)}`")

    def span(self, sort: str, start: int, end: int, text: str, block: int, says: str) -> None:
        was = self.read.code[start:end].decode("utf-8", "replace")
        key = (start, end, " ".join(text.split()))
        spot = (start, end)
        if key in self._seen or " ".join(text.split()) == " ".join(was.split()) or len(self.out) >= IN_ALL or self._at.get(spot, 0) >= self.most:
            return
        self._seen.add(key)
        self._at[spot] = self._at.get(spot, 0) + 1
        place = self.places.setdefault(spot, len(self.places) + 1)
        self.out.append(Way(len(self.out) + 1, place, sort, start, end, text, block, says))


def ways(read: Read, evidence: Evidence, blocks: Mapping[int, float] | None = None) -> list[Way]:
    """Every way to change it, in the blocks that are suspected (all of them, where none is said to be)."""
    got = _Collected(read)
    nodes = [n for n in _walk(read.root) if blocks is None or read.block_of(n) in blocks]
    props: dict[str, dict[str, None]] = {}
    strings: dict[str, None] = {}
    truths: dict[int, list[str]] = {}
    values: dict[int, list[str]] = {}
    for n in _walk(read.root):
        if n.type == "member_expression":
            of, prop = n.child_by_field_name("object"), n.child_by_field_name("property")
            if of is not None and prop is not None and n.parent is not None and not (n.parent.type == "call_expression" and n.parent.child_by_field_name("function") == n):
                props.setdefault(read.text(of), {}).setdefault(read.text(prop))
        elif n.type == "string":
            strings.setdefault(read.text(n))
        if n.type in _VALUES and (function := _function_of(n)) is not None and n.end_byte - n.start_byte < 120 and _can_be_put_in(n):
            (truths if _is_truth(n) else values).setdefault(function.start_byte, []).append(read.text(n))
    for node in nodes:
        _at(read, got, node, evidence, props, list(strings), truths, values)
    _statements(read, got, nodes, evidence)
    _rows(read, got)
    return _that_can_be_read(got.out)


def _that_can_be_read(found: Sequence[Way]) -> list[Way]:
    """Those whose text is whole: an expression where one is put in place of another, statements where they are added."""
    parser, known, out = _parser(), {}, []
    for way in found:
        text = way.text if way.start == way.end else f"({way.text});"
        if text not in known:
            known[text] = not parser.parse(f"(() => {{ {text} }})".encode()).root_node.has_error
        if known[text]:
            out.append(way)
    return out


def likeliest(read: Read, found: Sequence[Way], blocks: Mapping[int, float], pays: Mapping[str, float] | None = None, most: int = WRITTEN_IN) -> list[Way]:
    """The likeliest of them, by how suspect the block each is in is and how often a mend is of its sort: as many as can be
    written into one program."""
    pays = pays or {}

    def worth(way: Way) -> float:
        return blocks.get(way.block, 0.0) * (SORTS.get(way.sort, 0.5) + pays.get(way.sort, 0.0))

    return sorted(found, key=worth, reverse=True)[:most]


def _at(read: Read, got: _Collected, node: Any, evidence: Evidence, props: Mapping[str, Mapping[str, None]], strings: Sequence[str],
        truths: Mapping[int, list[str]], values: Mapping[int, list[str]]) -> None:
    kind, text = node.type, read.text(node)
    if kind == "call_expression":
        _calls(read, got, node, text, evidence)
    if kind in {"assignment_expression", "augmented_assignment_expression"}:
        _given(read, got, node, text)
    if not _can_be_put_in(node):
        return
    if kind == "binary_expression":
        left, op, right = node.child_by_field_name("left"), node.child_by_field_name("operator"), node.child_by_field_name("right")
        a, o, b = read.text(left), op.type, read.text(right)
        for table, sort in ((_COMPARE, "comparison"), (_LOGIC, "boolean"), (_SUMS, "arithmetic")):
            for other in table.get(o, ()):
                got.add(sort, node, f"{a} {other} {b}")
        if o in {"===", "!==", "==", "!="}:
            got.add("same_kind", node, f"String({a}) {o} String({b})")
            got.add("same_kind", node, f"Number({a}) {o} Number({b})")
        if o in {"-", "+", "/", "%"}:
            got.add("swap_arguments", node, f"{b} {o} {a}")
        if o == "/":
            got.add("guard", node, f"({b} ? {a} / {b} : 0)")
        if o == "*" and re.fullmatch(r"\w+\.dt", b):
            got.add("time", node, a)
        elif o == "*" and re.fullmatch(r"\w+\.dt", a):
            got.add("time", node, b)
    if _is_truth(node):
        got.add("negate", node, f"!({text})")
        function = _function_of(node)
        for other in dict.fromkeys(_nearest(text, truths.get(function.start_byte if function is not None else -1, ()), 8)):
            if _names_are_in_reach(read, node, other):
                got.add("ingredient", node, other)
                got.add("joined", node, f"({text}) && ({other})")
                got.add("joined", node, f"({text}) || ({other})")
        if kind == "binary_expression" and node.child_by_field_name("operator").type in {"&&", "||"}:
            got.add("dropped", node, read.text(node.child_by_field_name("left")))
            got.add("dropped", node, read.text(node.child_by_field_name("right")))
        for name in _truths_of_the_state(read, node, evidence):
            got.add("joined", node, f"({text}) && {name}")
            got.add("joined", node, f"({text}) && !{name}")
            got.add("joined", node, f"({text}) || {name}")
    elif kind in {"binary_expression", "call_expression", "member_expression", "identifier", "ternary_expression"}:
        function = _function_of(node)
        for other in dict.fromkeys(_nearest(text, values.get(function.start_byte if function is not None else -1, ()), 6)):
            if _names_are_in_reach(read, node, other):
                got.add("ingredient", node, other)
    if kind == "number":
        _numbers(got, node, text, evidence)
    elif kind in {"true", "false"}:
        got.add("literal", node, "false" if kind == "true" else "true")
    elif kind == "string":
        bare = text[1:-1]
        near = [s for s in strings if s[1:-1] != bare]
        said = [f'"{w}"' for w in evidence.words if w != bare and '"' not in w]
        for other in dict.fromkeys([*_nearest(text, near, 5), *_nearest(text, said, 4)]):
            got.add("words_for_words", node, other)
    elif kind == "identifier":
        reach = [n for n in _in_reach(read, node) if n != text]
        if text in _in_reach(read, node):
            for other in reach[:14]:
                got.add("name_for_name", node, other)
        elif any(re.search(rf"(?<![\w$]){re.escape(text)} is not defined", why) for why in evidence.broke):
            for other in _nearest(text, [*reach, *(f"this.{f}" for f in read.functions), *_paths(read, node)], 8):
                got.add("guard", node, other)
            got.add("guard", node, "0")
    elif kind == "member_expression":
        _members(read, got, node, text, props, evidence)
    elif kind == "ternary_expression":
        test, yes, no = (read.text(node.child_by_field_name(f)) for f in ("condition", "consequence", "alternative"))
        got.add("negate", node, f"{test} ? {no} : {yes}")
    if kind in {"binary_expression", "call_expression"} and _is_a_sum(read, node) and not _is_truth(node):
        got.add("kept_within", node, f"Math.max(0, {text})")
        got.add("whole_number", node, f"Math.round({text})")
        got.add("whole_number", node, f"Math.floor({text})")
        got.add("arithmetic", node, f"-({text})")
        got.add("kept_within", node, f"Math.abs({text})")


def _is_a_sum(read: Read, node: Any) -> bool:
    if node.type == "binary_expression":
        return node.child_by_field_name("operator").type in {"+", "-", "*", "/", "%"} and node.parent.type != "binary_expression"
    return read.text(node).startswith("Math.")


def _nearest(text: str, others: Iterable[str], most: int) -> list[str]:
    """The likest of some others, by what is written."""
    scored = [(difflib.SequenceMatcher(None, text, other, autojunk=False).quick_ratio(), other) for other in dict.fromkeys(others) if other != text]
    scored.sort(key=lambda one: -one[0])
    return [other for score, other in scored[:most] if score > 0.3]


def _names_are_in_reach(read: Read, node: Any, text: str) -> bool:
    free = set(re.findall(r"(?<![\w$.'\"])([A-Za-z_$][\w$]*)(?!\s*:)", re.sub(r"([\"'`])(?:\\.|(?!\1).)*\1", "", text)))
    known = {*_in_reach(read, node), "this", "Math", "Number", "String", "Array", "Object", "JSON", "true", "false", "null", "undefined", "W", "H", "new", "typeof",
             "Infinity", "NaN", "Date", "Boolean", "parseInt", "parseFloat", "isNaN"}
    return free <= known


def _paths(read: Read, node: Any) -> list[str]:
    state = _state_name(read, node)
    return [f"{state}.{name}" for name in read.fields] if state else []


def _truths_of_the_state(read: Read, node: Any, evidence: Evidence) -> list[str]:
    state = _state_name(read, node)
    return [f"{state}.{name}" for name, kind in read.fields.items() if kind == "truth" and name in evidence.names][:4] if state else []


def _numbers(got: _Collected, node: Any, text: str, evidence: Evidence) -> None:
    try:
        n = float(text)
    except ValueError:
        return

    def as_written(value: float) -> str:
        return str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(round(value, 6))

    for other in evidence.numbers:
        got.add("number_from_the_rule", node, other)
    for other in (n + 1, n - 1):
        got.add("shift_by_one", node, as_written(other))
    for other in (n * 1000, n / 1000, n * 60, n / 60, n * 100, n / 100):
        got.add("unit", node, as_written(other))
    for other in (n * 2, n / 2, -n, 0.0, 1.0):
        got.add("literal", node, as_written(other))


def _members(read: Read, got: _Collected, node: Any, text: str, props: Mapping[str, Mapping[str, None]], evidence: Evidence) -> None:
    of, prop = node.child_by_field_name("object"), node.child_by_field_name("property")
    if of is None or prop is None:
        return
    base, name = read.text(of), read.text(prop)
    others: dict[str, None] = dict(props.get(base, {}))
    if base == _state_name(read, node):
        others.update(dict.fromkeys(read.fields))
    others.update(dict.fromkeys(read.beside.get(name, ())))
    mine = read.fields.get(name)
    if mine == "truth":
        got.add("negate", node, f"!{text}")
    for other in others:
        if other != name and (mine is None or read.fields.get(other) in {None, mine}):
            got.add("field_for_field", node, f"{base}.{other}")
    _guards(got, node, text, base, name, evidence)


def _guards(got: _Collected, node: Any, text: str, base: str, name: str, evidence: Evidence) -> None:
    if not any(f"reading '{name}'" in why or "is not iterable" in why for why in evidence.broke):
        return
    got.add("guard", node, f"({base} || {{}}).{name}")
    got.add("guard", node, f"({base} ? {base}.{name} : 0)")
    got.add("guard", node, f"(({base} || {{}}).{name} || [])")
    got.add("guard", node, f"(({base} || {{}}).{name} || 0)")


def _calls(read: Read, got: _Collected, node: Any, text: str, evidence: Evidence) -> None:
    called, given = node.child_by_field_name("function"), node.child_by_field_name("arguments")
    if called is None or given is None or not _a_call_can_be_put_in(node):
        return
    args = [read.text(a) for a in given.named_children if a.type != "comment"]
    inside = read.text(given)
    if called.type == "member_expression":
        of, prop = called.child_by_field_name("object"), called.child_by_field_name("property")
        base, name = read.text(of), read.text(prop)
        if base == "Math":
            for other in _ROUND.get(name, ()):
                got.add("whole_number", node, f"Math.{other}{inside}")
        for other in _LISTS.get(name, ()):
            got.add("list", node, f"{base}.{other}{inside}")
        if name == "sort":
            if args and (m := re.fullmatch(r"\(?\s*(\w+)\s*,\s*(\w+)\s*\)?\s*=>\s*([\s\S]+)", args[0])):
                got.add("list", node, f"{base}.sort(({m.group(2)}, {m.group(1)}) => {m.group(3)})")
            elif not args:
                got.add("list", node, f"{base}.sort((a, b) => a - b)")
            got.add("list", node, f"{base}.slice().sort{inside}")
        if name == "filter" and args and (m := re.fullmatch(r"(\(?[\w\s,{}]*?\)?)\s*=>\s*(?![{\s])([\s\S]*)", args[0])):
            got.add("negate", node, f"{base}.filter({m.group(1)} => !({m.group(2)}))")
        if name in {"find", "filter", "map", "some", "reduce", "forEach", "slice", "indexOf", "includes", "split", "trim", "toLowerCase", "push", "toFixed"} \
                and any(f"reading '{name}'" in why for why in evidence.broke):
            for other in (f"({base} || []).{name}{inside}", f"String({base} ?? '').{name}{inside}", f"({base} || 0).{name}{inside}", f"({base} ? {text} : null)"):
                got.add("guard", node, other)
        if any(re.search(rf"{re.escape(name)} is not a function", why) for why in evidence.broke):
            for other in _nearest(name, read.functions, 3):
                got.add("guard", node, f"this.{other}{inside}")
        if base == "this":
            for other in _nearest(name, [f for f in read.functions if f != name], 2):
                got.add("name_for_name", node, f"this.{other}{inside}")
    elif called.type == "identifier":
        name = read.text(called)
        if name in read.functions and name not in _in_reach(read, node):
            got.add("guard", node, f"this.{name}{inside}")
        elif any(f"{name} is not defined" in why for why in evidence.broke):
            for other in _nearest(name, read.functions, 3):
                got.add("guard", node, f"this.{other}{inside}")
    if len(args) >= 2 and not any(a.startswith("...") for a in args):
        for i in range(len(args) - 1):
            turned = [*args[:i], args[i + 1], args[i], *args[i + 2:]]
            got.add("swap_arguments", node, f"{read.text(called)}({', '.join(turned)})")


def _a_call_can_be_put_in(node: Any) -> bool:
    return _can_be_put_in(node) or (node.parent is not None and node.parent.type in {"expression_statement", "call_expression", "member_expression"}
                                    and not (node.parent.type == "member_expression" and _is_given_to(node.parent)) and node.parent.type != "call_expression")


def _given(read: Read, got: _Collected, node: Any, text: str) -> None:
    """What is given to something: another way of giving it, or kept within what it can be."""
    if node.parent is None or node.parent.type not in {"expression_statement", "sequence_expression", "parenthesized_expression", "for_statement", "arrow_function"}:
        return
    left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
    op = "=" if node.type == "assignment_expression" else read.text(node.child_by_field_name("operator"))
    a, b = read.text(left), read.text(right)
    if left.type not in {"member_expression", "subscript_expression", "identifier"}:
        return
    for other in _AUGMENTED.get(op, ()):
        got.add("arithmetic", node, f"{a} {other} {b}")
    reach = _in_reach(read, node)
    for event in ("e", "ev", "event"):
        if event in reach and op in {"+=", "-="} and ".dt" not in b:
            got.add("time", node, f"{a} {op} ({b}) * {event}.dt")
    if op in {"+=", "-="}:
        sign = op[0]
        got.add("kept_within", node, f"{a} = Math.max(0, {a} {sign} ({b}))")
        got.add("kept_within", node, f"{a} = Math.max(0, Math.min(W, {a} {sign} ({b})))")
        got.add("kept_within", node, f"{a} = Math.max(0, Math.min(H, {a} {sign} ({b})))")
    got.add("dropped", node, "0", f"left out `{_short(text)}`")


def _rows(read: Read, got: _Collected) -> None:
    """Controls for each thing in a list, named by its place in it, whose events put what is given nowhere the list is: each
    event put in the thing it is of. What a control is of is read where it is shown: the list it is made for each of, and what
    of it the control shows."""
    update = next((n for n in _walk(read.root) if n.type == "method_definition" and read.text(n.child_by_field_name("name")) == "update"), None)
    body = update.child_by_field_name("body") if update is not None else None
    given = [read.text(c) for c in update.child_by_field_name("parameters").named_children] if update is not None else []
    if body is None or len(given) < 2:
        return
    state, event = given[0], given[1]
    made: list[str] = []
    for call in _walk(read.root):
        found = _row_control(read, call) if call.type == "call_expression" else None
        if found is None:
            continue
        before, after, path, of_it = found
        named = f"/^{re.escape(before)}(\\d+){re.escape(after)}$/"
        put = f"{{ ...x, {of_it}: {event}.value }}" if of_it else f"{event}.value"
        new = f"{state}.{'.'.join(path)}.map((x, i) => (i === Number(at[1]) ? {put} : x))"
        for depth in range(len(path) - 1, -1, -1):
            new = f"{{ ...{'.'.join([state, *path[:depth]])}, {path[depth]}: {new} }}"
        made.append(f"if ({event}.type === \"input\") {{ const at = {named}.exec({event}.name); if (at) return {new}; }}")
    made = list(dict.fromkeys(made))
    block = read.blocks.get((body.start_byte, body.end_byte), -1)
    spot = body.start_byte + 1
    for text in ([" ".join(made)] if len(made) > 1 else []) + made:
        got.span("row", spot, spot, f" {text} ", block, "put what is given to each row in the row it is of")


def _row_control(read: Read, call: Any) -> tuple[str, str, list[str], str] | None:
    """For a control shown for each thing in a list: its name before and after the place, where in the state the list is, and
    what of the thing it shows ("" for the thing itself). None for any other call."""
    function, args = call.child_by_field_name("function"), call.child_by_field_name("arguments")
    if function is None or args is None or function.type != "member_expression" or read.text(function.child_by_field_name("property")) not in {"input", "number", "date", "area", "check", "select", "slider"}:
        return None
    parts = [c for c in args.named_children if c.type != "comment"]
    if len(parts) < 3 or parts[1].type != "template_string":
        return None
    holes = [c for c in parts[1].named_children if c.type == "template_substitution"]
    if len(holes) != 1 or holes[0].named_child_count != 1 or holes[0].named_children[0].type != "identifier":
        return None
    index = read.text(holes[0].named_children[0])
    whole, start = read.text(parts[1]), parts[1].start_byte
    at, end = holes[0].start_byte - start, holes[0].end_byte - start
    before, after = whole.encode()[1:at].decode(), whole.encode()[end:-1].decode()
    each = call.parent
    while each is not None:
        if each.type == "arrow_function" and (over := each.child_by_field_name("parameters")) is not None:
            names = [read.text(c) for c in over.named_children]
            if len(names) >= 2 and names[1] == index:
                break
        each = each.parent
    over = each.parent.parent if each is not None and each.parent is not None else None
    if over is None or over.type != "call_expression" or read.text(over.child_by_field_name("function").child_by_field_name("property") or over) not in {"map", "flatMap"}:
        return None
    thing = read.text(each.child_by_field_name("parameters").named_children[0])
    shows = re.findall(rf"(?<![\w$.]){re.escape(thing)}((?:\.[A-Za-z_$][\w$]*)?)(?![\w$(])", read.text(parts[2]))
    path = _in_the_state(read, over.child_by_field_name("function").child_by_field_name("object"))
    return (before, after, path, shows[0].lstrip(".")) if shows and path else None


def _in_the_state(read: Read, node: Any) -> list[str]:
    """Where in the state an expression is, by what its names were given; [] where it is nowhere plain."""
    function = _function_of(node)
    state = _state_name(read, node)
    text = read.text(node)
    for _ in range(4):
        if not re.fullmatch(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", text):
            return []
        head, *rest = text.split(".")
        if head == state:
            return rest
        given = next((read.text(n.child_by_field_name("value")) for n in _walk(function) if n.type == "variable_declarator" and n.child_by_field_name("value") is not None
                      and read.text(n.child_by_field_name("name")) == head), "") if function is not None else ""
        if not given:
            return []
        text = ".".join([given, *rest])
    return []


def _statements(read: Read, got: _Collected, nodes: Sequence[Any], evidence: Evidence) -> None:
    """What could be added to a block: a count that nothing adds to, a function's answer that is not given back, a statement
    the program writes in another."""
    for node in nodes:
        if node.type == "statement_block" and node.parent is not None and node.parent.type == "arrow_function":
            held = [c for c in node.named_children if c.type != "comment"]
            if held and held[-1].type == "expression_statement":
                inner = read.text(held[-1]).rstrip(";")
                got.span("returned", held[-1].start_byte, held[-1].start_byte, f"return {inner}; ", read.block_of(node), f"gave back `{_short(inner)}`")
        if node.type != "statement_block" or node.parent is None or node.parent.type == "method_definition" and not read.fields:
            continue
        state = _state_name(read, node)
        if not state:
            continue
        block = read.blocks[(node.start_byte, node.end_byte)]
        first, last = node.start_byte + 1, node.end_byte - 1
        numbers = list(evidence.numbers[:4]) or ["1"]
        for name in [n for n in read.fields if n in evidence.names][:8]:
            path, holds = f"{state}.{name}", read.fields[name]
            adds: list[str] = []
            if holds == "number":
                adds = [f"{path} += {n};" for n in dict.fromkeys([*numbers, "1"])] + [f"{path} -= {n};" for n in dict.fromkeys([*numbers, "1"])] + [f"{path} = 0;"]
            elif holds == "truth":
                adds = [f"{path} = true;", f"{path} = false;", f"{path} = !{path};"]
            elif holds == "words":
                adds = [f"{path} = '';"]
            elif holds == "list":
                adds = [f"{path} = [];"]
            for text in adds:
                for at in (first, last):
                    got.span("counted", at, at, f" {text} ", block, f"added `{text}`" + (" first" if at == first else " last"))


# ── what a place would have to be ─────────────────────────────────────
#
# A branch that mends a failing test when it is made to go the other way is where the program decides wrongly (angelic
# values, Nopol, Xuan and others 2017). What it should decide by is then worked out from what is in reach there: every
# test of one thing, or of one thing against another, by itself and joined to what was there, run like any other way. Her
# own core/self_modification/condition_synthesis.py does this for Python from values it recorded; here running is quick
# enough that they are simply run.

#: The sort of a way that only asks what a branch decided, and is no change to be made.
FORCED = "forced"


def forced(read: Read, blocks: Mapping[int, float], first: int) -> list[Way]:
    """For each branch in the suspected blocks, it made to go one way and the other."""
    got = _Collected(read)
    for node in _walk(read.root):
        if node.type in _VALUES and read.block_of(node) in blocks and _can_be_put_in(node) and _is_truth(node):
            got.add(FORCED, node, "true", "it is so")
            got.add(FORCED, node, "false", "it is not so")
    return [Way(first + way.id, first + way.place, way.sort, way.start, way.end, way.text, way.block, way.says) for way in got.out]


def conditions(read: Read, spans: Iterable[tuple[int, int]], evidence: Evidence, *, most: int = 2400) -> list[Way]:
    """What the branches at these places could be decided by: each test of what is in reach there, alone and with what was there."""
    got = _Collected(read, at_a_place=most)
    wanted = set(spans)
    for node in _walk(read.root):
        if (node.start_byte, node.end_byte) not in wanted or node.type not in _VALUES:
            continue
        wanted.discard((node.start_byte, node.end_byte))
        was = read.text(node)
        for test in _tests_in_reach(read, node, evidence):
            got.add("joined", node, f"({was}) && {test}")
            got.add("joined", node, f"({was}) || {test}")
            got.add("condition", node, test)
    return _that_can_be_read(got.out)


def _tests_in_reach(read: Read, node: Any, evidence: Evidence) -> list[str]:
    paths = paths_in_reach(read, node)
    numbers = [p for p, kind in paths.items() if kind in {"number", ""}]
    tests = [f"{p} > 0" for p in numbers] + [f"{p} <= 0" for p in numbers]
    tests += [p for p, kind in paths.items() if kind == "truth"] + [f"!{p}" for p, kind in paths.items() if kind == "truth"]
    tests += [t for p, kind in paths.items() if kind == "list" for t in (f"{p}.length > 0", f"{p}.length === 0")]
    tests += [t for p, kind in paths.items() if kind in {"words", "thing"} for t in (f"!!{p}", f"!{p}")]
    for number in evidence.numbers[:5]:
        tests += [f"{p} {op} {number}" for p in numbers for op in (">=", "<", ">", "<=")]
    tests += [f"{p} {op} {q}" for p in numbers for q in numbers if p != q and paths[p] == paths[q] for op in ("<", ">=")]
    for word in evidence.words[:6]:
        if '"' not in word:
            tests += [f'{p} === "{word}"' for p, kind in paths.items() if kind == "words"]
    return list(dict.fromkeys(tests))


def paths_in_reach(read: Read, node: Any, most: int = 26) -> dict[str, str]:
    """What can be asked for here, by what kind of thing it is where that is known: the state's fields that this function asks
    for, what its other names are asked for, and those names themselves. What is written nearest comes first."""
    reach = _in_reach(read, node)
    function = _function_of(node)
    text = read.text(function) if function is not None else ""
    state = _state_name(read, node)
    at = node.start_byte - (function.start_byte if function is not None else 0)
    found: dict[str, tuple[int, str]] = {}
    for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)((?:\.[A-Za-z_$][\w$]*)*)", text):
        base, rest = m.group(1), m.group(2)
        if base not in reach or text[m.end():m.end() + 1] == "(":
            rest = rest.rpartition(".")[0] if base in reach else rest
            if base not in reach:
                continue
        path = base + rest
        last = path.rsplit(".", 1)[-1]
        if rest.endswith(".length") or path == state or not rest and path in {"e", "ev", "event", "g", "ui"}:
            continue
        kind = read.fields.get(last, "") if rest else ""
        near = abs(m.start() - at)
        if path not in found or near < found[path][0]:
            found[path] = (near, kind)
    for name, kind in read.fields.items():
        if state and kind != "nothing":
            found.setdefault(f"{state}.{name}", (10**6, kind)) if any(name == top for top in read.top) else None
    ordered = sorted(found.items(), key=lambda one: one[1][0])[:most]
    return {path: kind for path, (_near, kind) in ordered}


# ── all of them in one program ────────────────────────────────────────


#: The sorts of way that may put another kind of thing where one is.
_OF_ANY_KIND = frozenset({"guard", "forced", "dropped", "returned", "time", "arithmetic", "kept_within"})


def with_all(read: Read, chosen: Sequence[Way]) -> str:
    """The program, as an expression that is the object of its functions, with every one of these written into it: each let
    through by its number, its blocks marked as they are run, and its loops counted."""
    at_place: dict[tuple[int, int], list[Way]] = {}
    for way in chosen:
        at_place.setdefault((way.start, way.end), []).append(way)
    inserted: dict[int, list[Way]] = {}
    for (start, end), held in list(at_place.items()):
        if start == end:
            inserted.setdefault(start, []).extend(held)
            del at_place[(start, end)]
    code = read.code

    def added(at: int) -> bytes:
        return "".join(f" if (__W === {way.id}) {{ {way.text} }}" for way in inserted.pop(at, ())).encode()

    def raw(start: int, end: int) -> bytes:
        """Text that stands between the parts of a node, with whatever is to be added in it."""
        out, at = b"", start
        for spot in sorted(s for s in inserted if start <= s < end or (s == end and s == start)):
            out += code[at:spot] + added(spot)
            at = spot
        return out + code[at:end]

    def written(node: Any) -> bytes:
        pieces, at = [], node.start_byte
        for child in node.children:
            pieces += [raw(at, child.start_byte), raw(child.start_byte, child.start_byte), written(child)]
            at = child.end_byte
        pieces.append(raw(at, node.end_byte) if node.children else code[node.start_byte:node.end_byte])
        body = b"".join(pieces)
        span = (node.start_byte, node.end_byte)
        block = read.blocks.get(span)
        if node.type == "statement_block" and block is not None:
            counts = " __N();" if node.parent is not None and node.parent.type in {"for_statement", "for_in_statement", "while_statement", "do_statement"} else ""
            opened, closed = body.index(b"{") + 1, body.rindex(b"}")
            return body[:opened] + f" __H[{block}] = 1;{counts}".encode() + body[opened:closed] + added(node.end_byte - 1) + body[closed:]
        held = at_place.get(span, ())
        if held and node.type != "statement_block":
            # What is put in place of something is the kind of thing that was seen there; a truth may be told by anything,
            # and a guard is for where what was there was not.
            place, plain = held[0].place, b"(" + code[node.start_byte:node.end_byte] + b")"
            free = [way for way in held if way.sort in _OF_ANY_KIND] if not _is_truth(node) else list(held)
            kept = [way for way in held if way not in free]
            others = lambda some: "".join(f"__W === {way.id} ? ({way.text}) : " for way in some).encode()  # noqa: E731
            chosen = others(free) + (f"__K({place}, ".encode() + others(kept) + plain + b")" if kept else plain)
            body = f"(__S === {place} ? (".encode() + chosen + f") : __R({place}, ".encode() + body + b"))"
        if block is not None and _is_alone(node):
            counts = " __N();" if node.parent is not None and node.parent.type in {"for_statement", "for_in_statement", "while_statement"} else ""
            return f"{{ __H[{block}] = 1;{counts} ".encode() + body + b" }"
        if block is not None:
            return f"(__H[{block}] = 1, ".encode() + body + b")"
        return body

    def _is_alone(node: Any) -> bool:
        return node.type.endswith("_statement") or node.type in {"throw_statement", "return_statement"}

    held = [child for child in read.root.children if child.type != "comment"]
    if not held or held[-1].type != "expression_statement":
        return ""
    before = b"\n".join(written(child) for child in held[:-1])
    return f"(() => {{ {before.decode('utf-8', 'replace')}\nreturn {written(held[-1]).decode('utf-8', 'replace').strip().rstrip(';')}; }})()"


def made(code: str, way: Way) -> str:
    """The program with one change made."""
    whole = code.encode()
    return (whole[:way.start] + way.text.encode() + whole[way.end:]).decode("utf-8", "replace")
