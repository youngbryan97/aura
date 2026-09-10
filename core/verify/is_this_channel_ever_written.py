"""A telemetry channel whose only writer nobody calls can never carry a reading.

``can_this_ever_fire`` asks whether a caller has switched a mechanism off by
always handing it nothing. This asks the question one step earlier, for the one
kind of code where a missing caller is invisible from the inside: a publisher.

A publisher is complete, correct, tested, and silent. It writes the channels of
its subsystem, it returns what it wrote, and its unit tests call it directly and
pass. Nothing else calls it. Every channel it owns then reads "declared, never
written" — which on a display is indistinguishable from an organ that is dead,
and which decays every claim bound to those channels.

Two were found by hand on 2026-09-10, both in the live desktop after hours of
uptime:

* ``core.phenomena_wiring.sample`` holds the only ``write`` for the
  twenty-three disposition channels. Its module had two callers, for ``boot``
  and ``snapshot``. Nothing named ``sample``.
* ``core.conation.wiring.tick`` publishes eight conative channels and delivers
  arousal to the soma. Its module had two callers, for ``boot`` and
  ``snapshot``. Nothing named ``tick``.

The shape is worth naming because the test suite cannot see it: calling the
publisher yourself is exactly what a unit test does, so the green test is
evidence about the publisher and no evidence at all about the runtime.

What counts as a caller here is deliberately generous — any mention of the
name in production code outside the function's own body, including passing it
as a value, since registering a callback is how most of these are wired. A
generous rule reports few things and each one is real.
"""

from __future__ import annotations

import ast
import json
import pathlib
from dataclasses import dataclass

#: Where production code lives. A test, a tool or a script calling the
#: publisher is precisely the evidence this module refuses to accept.
_PRODUCTION_PACKAGES = ("core", "interface", "skills", "security", "llm", "executors")

#: The functions that put a value on a channel. ``write`` is the module-level
#: one; the dictionary method carries the same name.
_WRITERS = ("write",)

#: How a channel gets declared.
_DECLARERS = ("channel", "declare_channel")


@dataclass(frozen=True)
class AChannelNobodyWrites:
    """One channel whose writers are all unreachable from production code."""

    channel: str
    writers: tuple[str, ...]
    declared_in: str

    def key(self) -> str:
        return self.channel

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "writers": list(self.writers),
            "declared_in": self.declared_in,
        }


def _production_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for package in _PRODUCTION_PACKAGES:
        base = root / package
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            parts = set(path.relative_to(root).parts)
            if parts & {"__pycache__", "tests", "test"}:
                continue
            files.append(path)
    return files


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("/", ".")


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "a.channel"`` bindings, which is how ids are held."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _resolve(node: ast.AST, local: dict[str, str], everywhere: dict[str, str]) -> str:
    """The channel name an argument stands for, where that is decidable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return local.get(node.id) or everywhere.get(node.id, "")
    if isinstance(node, ast.Attribute):
        # ``ch.CHANNEL_CARE_GINI`` — the constant lives in the imported module,
        # and a channel id is unique across the tree by contract, so the name
        # alone resolves it.
        return everywhere.get(node.attr, "")
    return ""


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """Line number -> the name of the innermost function that line is inside.

    Widest span first so a nested definition overwrites its parent's claim on
    the lines it owns. A ``write`` inside a closure belongs to the closure.
    """
    spans: list[tuple[int, int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        spans.append((end - node.lineno, node.lineno, end, node.name))
    owner: dict[int, str] = {}
    for _width, start, end, name in sorted(spans, reverse=True):
        for line in range(start, end + 1):
            owner[line] = name
    return owner


def _names_mentioned_outside(
    trees: dict[str, ast.Module], home: str, function: str
) -> bool:
    """Whether anything outside this function's own body mentions its name."""
    for module, tree in trees.items():
        for node in ast.walk(tree):
            mentioned = (
                isinstance(node, ast.Name) and node.id == function
            ) or (isinstance(node, ast.Attribute) and node.attr == function)
            if not mentioned:
                continue
            if module != home:
                return True
            if not _within(tree, function, node.lineno):
                return True
    return False


def _within(tree: ast.Module, function: str, line: int) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function:
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.lineno <= line <= end:
            return True
    return False


def channels_nobody_writes(root: str = "") -> tuple[AChannelNobodyWrites, ...]:
    """Every declared channel whose writers are all unreachable."""
    base = pathlib.Path(root or ".").resolve()
    files = _production_files(base)
    trees: dict[str, ast.Module] = {}
    constants: dict[str, str] = {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        module = _module_name(path, base)
        trees[module] = tree
        constants.update(_string_constants(tree))

    declared: dict[str, str] = {}
    writers: dict[str, set[tuple[str, str]]] = {}
    for module, tree in trees.items():
        local = _string_constants(tree)
        owners = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = ""
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called in _DECLARERS:
                for keyword in node.keywords:
                    if keyword.arg == "name":
                        name = _resolve(keyword.value, local, constants)
                        if name:
                            declared.setdefault(name, module)
                continue
            if called not in _WRITERS or not node.args:
                continue
            name = _resolve(node.args[0], local, constants)
            if not name:
                continue
            enclosing = owners.get(node.lineno, "")
            if enclosing:
                writers.setdefault(name, set()).add((module, enclosing))

    unreached: dict[tuple[str, str], bool] = {}
    findings: list[AChannelNobodyWrites] = []
    for name, where in declared.items():
        sites = writers.get(name) or set()
        if not sites:
            # No writer at all is a different defect and a louder one; it is
            # not what this module claims to find, so it is left alone.
            continue
        dead: list[str] = []
        for module, function in sorted(sites):
            key = (module, function)
            if key not in unreached:
                unreached[key] = not _names_mentioned_outside(trees, module, function)
            if unreached[key]:
                dead.append(f"{module}.{function}")
        if len(dead) == len(sites):
            findings.append(
                AChannelNobodyWrites(
                    channel=name, writers=tuple(dead), declared_in=where
                )
            )
    return tuple(sorted(findings, key=lambda f: f.channel))


def load_baseline(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reviewed = data.get("reviewed") or {}
    return {str(k): str(v) for k, v in reviewed.items()}


def report(root: str = "", baseline: pathlib.Path | None = None) -> dict[str, object]:
    findings = channels_nobody_writes(root)
    reviewed = load_baseline(baseline) if baseline else {}
    new = [f for f in findings if f.key() not in reviewed]
    return {
        "schema": "aura.verify.channel_writers.v1",
        "declared_with_unreachable_writers": len(findings),
        "reviewed": len(findings) - len(new),
        "new": [f.to_dict() for f in new],
        "all": [f.to_dict() for f in findings],
    }


__all__ = [
    "AChannelNobodyWrites",
    "channels_nobody_writes",
    "load_baseline",
    "report",
]
