"""Every route a turn of the driver takes into the organism has a counterpart or says it is the instrument.

The battery passes only if the architecture has the property, not if the test
became another organ. The driver does reach into the organism outside the
phases: it beats the heart, steps the substrate and the layers on its own
count, trains the world model, retrieves, acts. Each of those is listed with
what the running organism does in its place, and this fails when a turn reaches
something that is not listed or a listed counterpart is not there.
"""

from __future__ import annotations

import ast
import pathlib
import re

from core.subject.driver import HARNESS_ROUTES

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER = ROOT / "core" / "subject" / "driver.py"
FREE_ROUTES = {"step_once", "perturb", "sustain", "on_frame", "note_effort", "capture", "_condition_index"}


def _turn_routes() -> set[str]:
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SubjectRuntime")
    methods = {node.name for node in runtime.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    turn = next(node for node in runtime.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "turn_once")
    routes: set[str] = set()
    for node in ast.walk(turn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in methods and node.attr.startswith("_"):
                routes.add(node.attr)
            if node.attr in {"after_phase", "read"}:
                routes.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FREE_ROUTES:
            routes.add(node.func.id)
    return routes


def test_every_route_a_turn_takes_is_listed() -> None:
    missing = sorted(_turn_routes() - set(HARNESS_ROUTES))
    assert not missing, f"routes into the organism with no stated counterpart: {missing}"


def test_the_inventory_finds_the_routes_it_is_about() -> None:
    routes = _turn_routes()
    assert {"_consciousness_tick", "_act", "_retrieve", "step_once"} <= routes, routes


def test_every_named_counterpart_exists() -> None:
    broken = []
    for route, counterpart in HARNESS_ROUTES.items():
        if counterpart.startswith(("instrument:", "experimental control")):
            continue
        match = re.match(r"([\w/]+\.py)(?::(\w+))?", counterpart)
        assert match, f"{route}: {counterpart!r} names no file"
        path = ROOT / match.group(1)
        if not path.exists():
            broken.append(f"{route}: {path} does not exist")
            continue
        name = match.group(2)
        if name and not re.search(rf"^\s*(async\s+def|def|class)\s+{name}\b", path.read_text(encoding="utf-8"), re.M):
            broken.append(f"{route}: {name} is not defined in {match.group(1)}")
    assert not broken, broken
