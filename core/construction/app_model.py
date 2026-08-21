"""What an app is, as data the runtime can reason about.

A request to build something is not a request for prose. The runtime has a
Python interpreter, a file system and a browser; asking a language model to
author the code puts the whole job on the one part of the system that cannot
check its own work. The model plans. The system builds.

So an app is a typed value here: named state, operations over that state, the
controls that trigger them and the views that show them. The operations have
one meaning, given once, in :func:`apply`. The same operations compile to
JavaScript for the browser, which is why a build can be tested before anyone
opens it — the semantics under test and the semantics that ship are the same
list of operations, read twice.

Nothing in this module knows what a timer or a shopping list is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Field",
    "Op",
    "Action",
    "Control",
    "View",
    "AppSpec",
    "initial_state",
    "apply",
    "UnknownOperation",
]

#: What a field can hold. Anything outside this cannot be compiled, which is
#: the point: an app model that cannot be compiled is rejected at the spec.
FieldKind = Literal["number", "text", "boolean", "list"]

#: The operations. Each one is defined once, in `apply`, and emitted once, by
#: the compiler. Adding one means adding both, and the equivalence test fails
#: until they agree.
OpName = Literal[
    "set",       # target <- value, or target <- the named input
    "add",       # target <- target + value (numbers)
    "toggle",    # target <- not target (booleans)
    "append",    # target <- target + [value or input]
    "remove",    # target <- target without the item at the given index
    "clear",     # target <- empty for its kind
    "count",     # target <- length of the source list
    "sum",       # target <- total of the numbers in the source list
]


class UnknownOperation(ValueError):
    """Raised when a spec names an operation the compiler cannot emit."""


@dataclass(frozen=True, slots=True)
class Field:
    """One named piece of state."""

    name: str
    kind: FieldKind = "number"
    initial: Any = None
    label: str = ""

    def start(self) -> Any:
        if self.initial is not None:
            return list(self.initial) if self.kind == "list" else self.initial
        return {"number": 0, "text": "", "boolean": False, "list": []}[self.kind]


@dataclass(frozen=True, slots=True)
class Op:
    """One state change. `value` is a literal; `source` names a field or input."""

    op: OpName
    target: str
    value: Any = None
    source: str = ""


@dataclass(frozen=True, slots=True)
class Action:
    """A named sequence of operations, run in order."""

    name: str
    ops: tuple[Op, ...] = ()
    label: str = ""


@dataclass(frozen=True, slots=True)
class Control:
    """Something the person operates, bound to an action."""

    kind: Literal["button", "text_input", "number_input", "checkbox"]
    action: str = ""
    label: str = ""
    #: For inputs: the name an action reads with `source`.
    input_name: str = ""


@dataclass(frozen=True, slots=True)
class View:
    """Something the person reads."""

    kind: Literal["value", "list", "boolean"]
    field: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class AppSpec:
    """A complete, compilable application."""

    title: str
    fields: tuple[Field, ...] = ()
    actions: tuple[Action, ...] = ()
    controls: tuple[Control, ...] = ()
    views: tuple[View, ...] = ()
    #: Runs `tick` every second when an action of that name exists.
    ticking: bool = False
    #: Keeps state across reloads in localStorage.
    persist: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def field_named(self, name: str) -> Field | None:
        return next((item for item in self.fields if item.name == name), None)

    def action_named(self, name: str) -> Action | None:
        return next((item for item in self.actions if item.name == name), None)

    def problems(self) -> tuple[str, ...]:
        """Everything about this spec that would not compile."""
        found: list[str] = []
        names = {item.name for item in self.fields}
        if not self.title.strip():
            found.append("the app has no title")
        if not self.fields:
            found.append("the app has no state")
        for action in self.actions:
            for op in action.ops:
                if op.target not in names:
                    found.append(f"action {action.name} writes unknown field {op.target}")
                if op.source and op.source not in names and not _is_input(self, op.source):
                    found.append(f"action {action.name} reads unknown source {op.source}")
        for control in self.controls:
            if control.kind == "button" and not self.action_named(control.action):
                found.append(f"button {control.label!r} triggers unknown action {control.action}")
        for view in self.views:
            if view.field not in names:
                found.append(f"view shows unknown field {view.field}")
        return tuple(dict.fromkeys(found))


#: Inputs the compiler supplies without anyone declaring them. A list view
#: renders a control per row and passes that row's position as `index`, so an
#: action that removes a row reads a source no control could have declared.
AMBIENT_INPUTS = ("index",)


def _is_input(spec: AppSpec, name: str) -> bool:
    if name in AMBIENT_INPUTS:
        return True
    return any(control.input_name == name for control in spec.controls)


def initial_state(spec: AppSpec) -> dict[str, Any]:
    return {item.name: item.start() for item in spec.fields}


def apply(
    spec: AppSpec, state: dict[str, Any], action_name: str, inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one action. The single definition of what these operations mean."""
    action = spec.action_named(action_name)
    if action is None:
        raise UnknownOperation(f"no action named {action_name}")
    given = dict(inputs or {})
    result = dict(state)
    for op in action.ops:
        result[op.target] = _apply_one(op, result, given)
    return result


def _read(op: Op, state: dict[str, Any], inputs: dict[str, Any]) -> Any:
    if op.source:
        return inputs[op.source] if op.source in inputs else state.get(op.source)
    return op.value


def _apply_one(op: Op, state: dict[str, Any], inputs: dict[str, Any]) -> Any:
    current = state.get(op.target)
    incoming = _read(op, state, inputs)
    if op.op == "set":
        return incoming
    if op.op == "add":
        return _number(current) + _number(incoming)
    if op.op == "toggle":
        return not bool(current)
    if op.op == "append":
        return list(current or []) + [incoming]
    if op.op == "remove":
        items = list(current or [])
        index = int(_number(incoming))
        if 0 <= index < len(items):
            items.pop(index)
        return items
    if op.op == "clear":
        return [] if isinstance(current, list) else (0 if isinstance(current, (int, float)) else "")
    if op.op == "count":
        return len(list(_read(op, state, inputs) or []))
    if op.op == "sum":
        return sum(_number(item) for item in list(_read(op, state, inputs) or []))
    raise UnknownOperation(f"unknown operation {op.op}")


def _number(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        return int(text) if text.lstrip("-").isdigit() else float(text)
    except (TypeError, ValueError):
        return 0
