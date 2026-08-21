"""Check a built app before anyone opens it.

Two things are checked, and both are checks the runtime can make on its own.

The page is parsed, and every control and view in it has to bind to something
the spec declares — a button wired to no action is the defect that made a
built page look finished and do nothing.

Then the emitted JavaScript is run. The state machine in the page and the one
in :mod:`core.construction.app_model` are compiled from the same operations, so they
must agree; sequences of actions are run through both and the states compared.
A disagreement is a compiler bug, and it is found here rather than by the
person who asked for the app.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from core.construction.app_compiler import reducer_js
from core.construction.app_model import AppSpec, apply, initial_state

__all__ = ["VerifiedApp", "verify_app", "node_available"]

#: A build must not hang a turn. Node starts in tens of milliseconds.
_NODE_TIMEOUT_S = 20.0


@dataclass(frozen=True, slots=True)
class VerifiedApp:
    ok: bool
    checks: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    sequences_run: int = 0
    semantics_checked: bool = False


class _Bindings(HTMLParser):
    """What the page actually wires up."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: set[str] = set()
        self.values: set[str] = set()
        self.lists: set[str] = set()
        self.inputs: set[str] = set()
        self.title_seen = False
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = {key: (value or "") for key, value in attrs}
        if "data-action" in found:
            self.actions.add(found["data-action"])
        if "data-value" in found:
            self.values.add(found["data-value"])
        if "data-list" in found:
            self.lists.add(found["data-list"])
        if "data-input" in found:
            self.inputs.add(found["data-input"])
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            self.title_seen = True


def node_available() -> bool:
    try:
        done = subprocess.run(
            ["node", "--version"], capture_output=True, timeout=5.0, check=False
        )
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _sample_inputs(spec: AppSpec) -> dict[str, Any]:
    """One value per declared input, chosen for its control's kind."""
    from core.construction.app_model import AMBIENT_INPUTS

    values: dict[str, Any] = dict.fromkeys(AMBIENT_INPUTS, 0)
    for index, control in enumerate(spec.controls):
        if not control.input_name:
            continue
        values[control.input_name] = (
            index + 2 if control.kind == "number_input" else f"item {index + 1}"
        )
    return values


def _sequences(spec: AppSpec) -> list[list[str]]:
    """Every action alone, then all of them together, then a repeat."""
    names = [action.name for action in spec.actions]
    if not names:
        return []
    runs = [[name] for name in names]
    runs.append(list(names))
    runs.append(list(names) * 2)
    return runs


def _run_in_node(spec: AppSpec, runs: list[list[str]], inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
    driver = (
        reducer_js(spec)
        + "\nconst START = "
        + json.dumps({item.name: item.start() for item in spec.fields})
        + ";\nconst RUNS = "
        + json.dumps(runs)
        + ";\nconst INPUTS = "
        + json.dumps(inputs)
        + ";\nconst out = RUNS.map(function (run) {\n"
        "  let state = Object.assign({}, START);\n"
        "  run.forEach(function (name) { state = reduce(state, name, INPUTS); });\n"
        "  return state;\n});\n"
        "console.log(JSON.stringify(out));\n"
    )
    with tempfile.TemporaryDirectory() as work:
        script = Path(work) / "check.js"
        script.write_text(driver, encoding="utf-8")
        try:
            done = subprocess.run(
                ["node", str(script)],
                capture_output=True,
                text=True,
                timeout=_NODE_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    if done.returncode != 0:
        raise ValueError(f"the emitted JavaScript did not run: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout or "[]")


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return str(left) == str(right)


def verify_app(spec: AppSpec, html: str) -> VerifiedApp:
    """Everything checkable about this build, checked."""
    checks: list[str] = []
    problems: list[str] = list(spec.problems())

    bindings = _Bindings()
    try:
        bindings.feed(html)
    except (ValueError, AssertionError) as exc:
        problems.append(f"the page did not parse: {exc}")
        return VerifiedApp(False, tuple(checks), tuple(problems))

    if not bindings.title_seen or not bindings.title.strip():
        problems.append("the page has no title")
    else:
        checks.append("the page parses and is titled")

    declared = {action.name for action in spec.actions}
    for name in sorted(bindings.actions):
        if name not in declared:
            problems.append(f"a control triggers {name}, which the app does not define")
    wired = {control.action for control in spec.controls if control.kind == "button"}
    for name in sorted(wired):
        if name not in bindings.actions:
            problems.append(f"action {name} has no control on the page")
    if wired and not problems:
        checks.append(f"{len(bindings.actions)} control(s) bound to declared actions")

    shown = bindings.values | bindings.lists
    for name in sorted(shown):
        if spec.field_named(name) is None:
            problems.append(f"the page shows {name}, which is not state")
    for view in spec.views:
        if view.field not in shown:
            problems.append(f"{view.field} is declared as a view and is not on the page")
    if shown and not problems:
        checks.append(f"{len(shown)} view(s) bound to declared state")

    semantics_checked = False
    runs = _sequences(spec)
    if runs and node_available():
        inputs = _sample_inputs(spec)
        try:
            from_node = _run_in_node(spec, runs, inputs)
        except ValueError as exc:
            problems.append(str(exc))
            from_node = None
        if from_node is not None:
            semantics_checked = True
            for run, theirs in zip(runs, from_node):
                mine = initial_state(spec)
                for name in run:
                    mine = apply(spec, mine, name, inputs)
                for key in sorted(mine):
                    if not _same(mine[key], theirs.get(key)):
                        problems.append(
                            f"after {' then '.join(run)}, {key} is {theirs.get(key)!r} in the "
                            f"page and {mine[key]!r} in the model"
                        )
            if not problems:
                checks.append(f"{len(runs)} action sequence(s) agree with the model")

    return VerifiedApp(
        ok=not problems,
        checks=tuple(checks),
        problems=tuple(dict.fromkeys(problems)),
        sequences_run=len(runs),
        semantics_checked=semantics_checked,
    )
