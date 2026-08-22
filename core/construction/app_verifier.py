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

import asyncio
import json
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from core.construction.app_compiler import reducer_js
from core.construction.app_model import AppSpec, apply, initial_state

__all__ = ["VerifiedApp", "verify_app", "node_available", "dom_driver_available"]

#: Loads a built page in a real DOM and clicks its own controls.
_DRIVER = Path(__file__).resolve().parents[2] / "tools" / "appcheck" / "drive_app.js"

#: A build must not hang a turn. Node starts in tens of milliseconds.
_NODE_TIMEOUT_S = 20.0


@dataclass(frozen=True, slots=True)
class VerifiedApp:
    ok: bool
    checks: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    sequences_run: int = 0
    semantics_checked: bool = False
    driven_in_dom: bool = False


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
        if found.get("data-row-action"):
            self.actions.add(found["data-row-action"])
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


def _run_node(
    argv: list[str],
    *,
    cwd: str | None = None,
    timeout: float = _NODE_TIMEOUT_S,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run node through the runtime's own gateway.

    The gateway is where a subprocess is recorded, bounded, reaped and told it
    needs no accelerator. Calling subprocess directly from here would put a
    build outside all of that, which is the mistake this module exists to
    stop making elsewhere.

    Outside a running runtime — a test, a tool — there is no gateway to use
    and the call is made directly.
    """
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    gateway = get_subprocess_gateway()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return gateway.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            read_only=True,
            input=stdin,
            check=False,
            source="construction.verify_app",
            accelerator_capability="none",
        )
    raise RuntimeError("verify_app runs off the loop; call it with asyncio.to_thread")


def node_available() -> bool:
    try:
        done = _run_node(["node", "--version"], timeout=5.0)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError, RuntimeError, ImportError):
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
    """Every action alone, then all together, then a repeat.

    An action on a row of a list has nothing to press until the list has a
    row, so it is preceded by whatever fills that list. Without that, the
    check reported a missing control for a page that was correct.
    """
    row_actions = {view.row_action: view.field for view in spec.views if view.row_action}
    plain = [action.name for action in spec.actions if action.name not in row_actions]
    if not plain and not row_actions:
        return []

    def fills(field: str) -> str | None:
        for action in spec.actions:
            if action.name in row_actions:
                continue
            if any(op.op == "append" and op.target == field for op in action.ops):
                return action.name
        return None

    runs = [[name] for name in plain]
    for name, field in row_actions.items():
        first = fills(field)
        runs.append([first, name] if first else [name])
    ordered = plain + [
        step for name in row_actions for step in ([fills(row_actions[name])] if fills(row_actions[name]) else []) + [name]
    ]
    runs.append(ordered)
    runs.append(ordered * 2)
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
    try:
        done = _run_node(["node", "--input-type=commonjs", "-"], stdin=driver)
    except (OSError, subprocess.SubprocessError, RuntimeError):
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
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return str(left) == str(right)


def dom_driver_available() -> bool:
    """Whether the page can be opened and clicked, rather than only reasoned about."""
    if not _DRIVER.is_file() or not node_available():
        return False
    try:
        done = _run_node(
            ["node", "-e", "require('jsdom')"], cwd=str(_DRIVER.parent), timeout=10.0
        )
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return False


def _as_rendered(spec: AppSpec, state: dict[str, Any]) -> dict[str, Any]:
    """What the page shows for this state, in the page's own formatting."""
    shown: dict[str, Any] = {}
    for view in spec.views:
        value = state.get(view.field)
        if isinstance(value, list):
            shown[view.field] = [
                json.dumps(item) if isinstance(item, (dict, list)) else str(item)
                for item in value
            ]
        elif isinstance(value, bool):
            shown[view.field] = "yes" if value else "no"
        elif isinstance(value, (int, float)):
            rounded = round(float(value) * 100) / 100
            shown[view.field] = str(int(rounded) if rounded == int(rounded) else rounded)
        else:
            shown[view.field] = str(value if value is not None else "")
    return shown


def _drive_in_dom(html: str, runs: list[list[str]], inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Open the page, click through each run, and report what it displays."""
    payload = json.dumps({"runs": runs, "inputs": inputs})
    try:
        done = _run_node(
            ["node", str(_DRIVER), "-", payload],
            cwd=str(_DRIVER.parent),
            timeout=_NODE_TIMEOUT_S * 3,
            stdin=html,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None
    if done.returncode != 0:
        raise ValueError(f"the page could not be opened: {done.stderr.strip()[:300]}")
    try:
        return json.loads(done.stdout or "{}")
    except (TypeError, ValueError):
        return None


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
    wired |= {view.row_action for view in spec.views if view.row_action}
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
            if len(from_node) != len(runs):
                problems.append(
                    "the emitted reducer returned "
                    f"{len(from_node)} result(s) for {len(runs)} action sequence(s)"
                )
            else:
                for run, theirs in zip(runs, from_node, strict=True):
                    mine = initial_state(spec)
                    for name in run:
                        mine = apply(spec, mine, name, inputs)
                    for key in sorted(mine):
                        if not _same(mine[key], theirs.get(key)):
                            problems.append(
                                f"after {' then '.join(run)}, {key} is {theirs.get(key)!r} "
                                f"in the page and {mine[key]!r} in the model"
                            )
            if not problems:
                checks.append(f"{len(runs)} action sequence(s) agree with the model")

    # The strongest check available: open the page and press its own buttons.
    driven = False
    if runs and not problems and dom_driver_available():
        inputs = _sample_inputs(spec)
        try:
            observed = _drive_in_dom(html, runs, inputs)
        except ValueError as exc:
            problems.append(str(exc))
            observed = None
        if observed is not None:
            driven = True
            for message in list(observed.get("errors") or [])[:4]:
                problems.append(f"the page reported: {message}")
            rendered = list(observed.get("rendered") or [])
            if len(rendered) != len(runs):
                problems.append(
                    "the DOM driver returned "
                    f"{len(rendered)} result(s) for {len(runs)} action sequence(s)"
                )
            else:
                for run, shown in zip(runs, rendered, strict=True):
                    state = initial_state(spec)
                    for name in run:
                        state = apply(spec, state, name, inputs)
                    expected = _as_rendered(spec, state)
                    for key in sorted(expected):
                        if key not in shown:
                            problems.append(
                                f"after {' then '.join(run)}, the page never showed {key}"
                            )
                        elif not _same_rendering(expected[key], shown[key]):
                            problems.append(
                                f"after {' then '.join(run)}, the page shows {key} as "
                                f"{shown[key]!r} and it should be {expected[key]!r}"
                            )
            if not problems:
                checks.append(f"{len(runs)} sequence(s) clicked through in a real DOM")

    return VerifiedApp(
        ok=not problems,
        checks=tuple(checks),
        problems=tuple(dict.fromkeys(problems)),
        sequences_run=len(runs),
        semantics_checked=semantics_checked,
        driven_in_dom=driven,
    )


def _same_rendering(expected: Any, shown: Any) -> bool:
    if isinstance(expected, list) and isinstance(shown, list):
        return len(expected) == len(shown) and all(
            str(a) == str(b) for a, b in zip(expected, shown, strict=True)
        )
    return str(expected) == str(shown)
