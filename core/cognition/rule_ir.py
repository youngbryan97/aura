"""One shape for every learned rule: ``Node(kind, parameters, [Node...])``.

Three things were learnable here and each was written down differently. A
positional program is ``kind``/``args``/``parts`` and interprets itself. An
ordering is a flat record of which level each cell belongs to. An ordering
composed with a move is a third shape with its own field names and its own
reader. Every consumer that wanted "the rule this turn learned" had to know
which of the three it was holding, and anything wanting to compose one with
another had to know both.

That is a representational limit rather than a missing feature. Composition
across the two axes was possible only where somebody had written a type for
that pair — ``Composed`` exists precisely because ordering-then-move needed
one — and a third axis would need three more.

So there is one node here. It applies to a state and returns a state, which
is the one thing all three genuinely have in common; a positional rule, whose
own interpreter answers per position, becomes the special case where the
answer is read for every position in turn. Composition stops being a type and
becomes an ordinary node with children, which is what makes an unplanned pair
expressible without anybody planning it.

Nothing is reimplemented. A node carries the JSON of whatever solved it and
hands the work back to that, so the solvers stay the authority on their own
semantics and this stays a spine rather than a second opinion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Node", "as_node"]

#: A rule that reads only positions. Its parameters are the positional
#: program's own JSON.
POSITIONAL = "positional"

#: A rule that reads the cells. Its parameters are the ordering's own JSON.
BY_KEY = "by_key"

#: Children applied in order, the output of each being the input of the next.
THEN = "then"


@dataclass(frozen=True)
class Node:
    """One learned rule, whatever kind of rule it turned out to be."""

    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)
    parts: tuple["Node", ...] = ()

    # ---------------------------------------------------------------- writing

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": dict(self.parameters),
            "parts": [part.to_json() for part in self.parts],
        }

    @classmethod
    def from_json(cls, raw: Any) -> "Node | None":
        if not isinstance(raw, dict):
            return None
        kind = str(raw.get("kind") or "").strip()
        if not kind:
            return None
        parameters = raw.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        parts: list[Node] = []
        for item in raw.get("parts") or ():
            child = cls.from_json(item)
            if child is None:
                return None
            parts.append(child)
        return cls(kind=kind, parameters=dict(parameters), parts=tuple(parts))

    # ---------------------------------------------------------------- reading

    def apply(self, state: Sequence[Any]) -> tuple[Any, ...] | None:
        """The state this rule turns ``state`` into, or None where it cannot.

        None rather than a guess. An ordering meets cells it never saw, a
        positional rule meets a length it was not written for, and a composed
        rule meets either — and in each case the honest answer is that this
        rule does not say.
        """

        cells = tuple(state)
        if self.kind == THEN:
            for part in self.parts:
                nxt = part.apply(cells)
                if nxt is None:
                    return None
                cells = tuple(nxt)
            return cells
        if self.kind == POSITIONAL:
            return self._positional(cells)
        if self.kind == BY_KEY:
            return self._by_key(cells)
        return None

    def _positional(self, cells: tuple[Any, ...]) -> tuple[Any, ...] | None:
        from core.cognition.primitive_invention import IndexProgram

        program = IndexProgram.from_json(self.parameters.get("program"))
        if program is None:
            return None
        size = len(cells)
        try:
            return tuple(cells[program(index, size)] for index in range(size))
        except (IndexError, TypeError, ValueError):
            return None

    def _by_key(self, cells: tuple[Any, ...]) -> tuple[Any, ...] | None:
        from core.cognition.value_order import Ordering

        ordering = Ordering.from_json(self.parameters.get("ordering"))
        if ordering is None:
            return None
        return ordering.apply(cells)

    def describe(self) -> str:
        """The rule in words, borrowed from whatever solved it."""

        if self.kind == THEN:
            return ", then ".join(part.describe() for part in self.parts)
        if self.kind == POSITIONAL:
            return _words_for(self.parameters.get("program"))
        if self.kind == BY_KEY:
            from core.cognition.value_order import Ordering

            ordering = Ordering.from_json(self.parameters.get("ordering"))
            return ordering.describe() if ordering is not None else "an unreadable order"
        return "an unreadable rule"


def as_node(rule: Any) -> Node | None:
    """Whatever this turn learned, as a node.

    Accepts a positional program, an ordering, an ordering composed with a
    move, or a node — so a caller can hold "the rule" without holding which
    kind of rule it is. Returns None for anything else, because inventing a
    node for an unknown object would put a rule in the library that nothing
    can interpret.
    """

    if rule is None:
        return None
    if isinstance(rule, Node):
        return rule

    from core.cognition.primitive_invention import IndexProgram
    from core.cognition.value_order import Composed, Ordering

    if isinstance(rule, IndexProgram):
        return Node(kind=POSITIONAL, parameters={"program": rule.to_json()})
    if isinstance(rule, Ordering):
        return Node(kind=BY_KEY, parameters={"ordering": rule.to_json()})
    if isinstance(rule, Composed):
        ordering = as_node(rule.ordering)
        move = as_node(rule.move)
        if ordering is None or move is None:
            return None
        # The pair that needed its own type is a node with two children now,
        # which is the whole point: the next pair will not need a third type.
        return Node(kind=THEN, parts=(ordering, move))
    # A solved relation wraps the program it found.
    inner = getattr(rule, "program", None) or getattr(rule, "index_program", None)
    if inner is not None and inner is not rule:
        return as_node(inner)
    return None


def _words_for(stored: Any) -> str:
    """A positional program in words.

    The words live beside the programs, in the table of forms, rather than on
    the program itself — so a program that has been saved and loaded has lost
    them. They are found again by value: a program compares equal to the one
    in the table it came from, whatever route it took to get here, and the
    table is the one place they are written.

    Sizes are tried in turn because a form is generated per length and most
    are the same value at any of them. A program nothing matches is described
    as itself rather than guessed at.
    """

    from core.cognition.primitive_invention import IndexProgram, _index_forms

    program = IndexProgram.from_json(stored)
    if program is None:
        return "an unreadable rule"
    for size in (4, 6, 5, 8, 3):
        for _name, described, candidate in _index_forms(size):
            if candidate == program:
                return str(described)
    return f"{program.kind} {', '.join(str(arg) for arg in program.args)}".strip()
