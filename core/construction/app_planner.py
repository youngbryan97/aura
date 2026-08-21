"""Turn a request for an app into a spec the compiler can build.

The language model's part of this is planning: naming the state an app keeps,
the things a person can do to it, and what they see. That is language work.
It does not write the app, and nothing it returns is executed — the plan is
data, checked against a schema, repaired where it is thin, and compiled by
:mod:`core.construction.app_compiler`.

Repair is the reason a weak plan still produces a working app. A plan that
names state and no way to see it gets views; one that names an action and no
button gets a button; one that appends to a list nobody declared gets the
list. Each repair is recorded, so a build can say what it decided for itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from core.construction.app_model import (
    AMBIENT_INPUTS,
    Action,
    AppSpec,
    Control,
    Field,
    Op,
    View,
)

__all__ = ["PlannedApp", "spec_from_plan", "plan_schema", "repair_notes"]

_FIELD_KINDS = {"number", "text", "boolean", "list"}
_OPS = {"set", "add", "toggle", "append", "remove", "clear", "count", "sum"}
_CONTROLS = {"button", "text_input", "number_input", "checkbox"}


@dataclass(frozen=True, slots=True)
class PlannedApp:
    spec: AppSpec
    repairs: tuple[str, ...] = ()


def plan_schema() -> dict[str, Any]:
    """The shape of a plan, as a JSON schema for a typed tool call."""
    return {
        "type": "object",
        "required": ["title", "fields", "actions"],
        "properties": {
            "title": {"type": "string", "description": "What the app is called."},
            "fields": {
                "type": "array",
                "description": "The state the app keeps.",
                "items": {
                    "type": "object",
                    "required": ["name", "kind"],
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": sorted(_FIELD_KINDS)},
                        "label": {"type": "string"},
                        "initial": {},
                    },
                },
            },
            "actions": {
                "type": "array",
                "description": "What a person can do, as operations over the state.",
                "items": {
                    "type": "object",
                    "required": ["name", "ops"],
                    "properties": {
                        "name": {"type": "string"},
                        "label": {"type": "string"},
                        "ops": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["op", "target"],
                                "properties": {
                                    "op": {"type": "string", "enum": sorted(_OPS)},
                                    "target": {"type": "string"},
                                    "value": {},
                                    "source": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "inputs": {
                "type": "array",
                "description": "Text or number boxes an action reads by name.",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": ["text", "number"]},
                        "label": {"type": "string"},
                    },
                },
            },
        },
    }


def _identifier(value: object, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return text or fallback


def _title(value: object, request: str) -> str:
    text = str(value or "").strip()
    if text:
        return text[:60]
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]*", request) if len(word) > 2]
    return " ".join(words[:5]).capitalize() or "App"


def _kind_of(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in _FIELD_KINDS else "number"


def spec_from_plan(plan: dict[str, Any], request: str = "") -> PlannedApp:
    """Build a compilable spec from whatever the plan actually contains."""
    repairs: list[str] = []
    raw = dict(plan or {})

    fields: list[Field] = []
    seen: set[str] = set()
    for index, item in enumerate(list(raw.get("fields") or [])):
        item = dict(item or {})
        name = _identifier(item.get("name"), f"value_{index + 1}")
        if name in seen:
            continue
        seen.add(name)
        fields.append(
            Field(
                name=name,
                kind=_kind_of(item.get("kind")),
                initial=item.get("initial"),
                label=str(item.get("label") or "").strip(),
            )
        )

    inputs: list[Control] = []
    declared_inputs: set[str] = set()
    for index, item in enumerate(list(raw.get("inputs") or [])):
        item = dict(item or {})
        name = _identifier(item.get("name"), f"input_{index + 1}")
        if name in declared_inputs:
            continue
        declared_inputs.add(name)
        kind = "number_input" if str(item.get("kind") or "") == "number" else "text_input"
        inputs.append(
            Control(kind=kind, input_name=name, label=str(item.get("label") or name.replace("_", " ")))
        )

    actions: list[Action] = []
    for index, item in enumerate(list(raw.get("actions") or [])):
        item = dict(item or {})
        name = _identifier(item.get("name"), f"action_{index + 1}")
        ops: list[Op] = []
        for entry in list(item.get("ops") or []):
            entry = dict(entry or {})
            op = str(entry.get("op") or "").strip().lower()
            if op not in _OPS:
                repairs.append(f"dropped an operation named {op or 'nothing'}, which does not exist")
                continue
            target = _identifier(entry.get("target"), "")
            if not target:
                repairs.append(f"dropped an operation in {name} that named no field")
                continue
            ops.append(
                Op(
                    op=op,  # type: ignore[arg-type]
                    target=target,
                    value=entry.get("value"),
                    source=_identifier(entry.get("source"), "") if entry.get("source") else "",
                )
            )
        if not ops:
            repairs.append(f"dropped action {name}, which did nothing")
            continue
        actions.append(
            Action(name=name, ops=tuple(ops), label=str(item.get("label") or name.replace("_", " ")))
        )

    # Repair, in the order a missing piece blocks the next one.
    known = {item.name for item in fields}
    for action in actions:
        for op in action.ops:
            if op.target not in known:
                kind = "list" if op.op in {"append", "remove", "clear"} else "number"
                fields.append(Field(name=op.target, kind=kind))
                known.add(op.target)
                repairs.append(f"added the {kind} {op.target}, which {action.name} writes")
            if op.source and op.source not in known and op.source not in declared_inputs:
                if op.source in AMBIENT_INPUTS:
                    continue
                kind = "number_input" if op.op in {"add", "remove"} else "text_input"
                inputs.append(
                    Control(kind=kind, input_name=op.source, label=op.source.replace("_", " "))
                )
                declared_inputs.add(op.source)
                repairs.append(f"added a box for {op.source}, which {action.name} reads")

    buttons = [
        Control(kind="button", action=action.name, label=action.label or action.name.replace("_", " "))
        for action in actions
        if action.name != "tick"
    ]
    if actions and not buttons:
        repairs.append("the app had nothing to press")

    views: list[View] = []
    for item in fields:
        views.append(
            View(
                kind="list" if item.kind == "list" else "value",
                field=item.name,
                label=item.label or item.name.replace("_", " "),
            )
        )
    if not raw.get("fields"):
        repairs.append("the plan named no state, so it was read from the operations")

    spec = AppSpec(
        title=_title(raw.get("title"), request),
        fields=tuple(fields),
        actions=tuple(actions),
        controls=tuple(inputs + buttons),
        views=tuple(views),
        ticking=any(action.name == "tick" for action in actions),
        persist=True,
        notes=tuple(repairs),
    )
    return PlannedApp(spec=spec, repairs=tuple(dict.fromkeys(repairs)))


def repair_notes(planned: PlannedApp) -> str:
    """What the builder decided for itself, as a sentence."""
    if not planned.repairs:
        return ""
    return "Filled in: " + "; ".join(planned.repairs[:6]) + "."


def plan_from_json(text: str, request: str = "") -> PlannedApp | None:
    """Read a plan out of whatever the model returned, or None."""
    raw = str(text or "").strip()
    if not raw:
        return None
    for candidate in _json_objects(raw):
        try:
            loaded = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(loaded, dict) and (loaded.get("fields") or loaded.get("actions")):
            return spec_from_plan(loaded, request)
    return None


def _json_objects(text: str) -> list[str]:
    """Every balanced object in the text, longest first."""
    found: list[str] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start : end + 1])
                    break
    return sorted(found, key=len, reverse=True)
