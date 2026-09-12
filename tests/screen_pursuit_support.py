"""Reach a name across every module the pursuit is now built from.

`core/skills/screen_pursuit.py` was one 6,240-line module, so a test could
replace anything a pursuit touched with a single `setattr` on it, and could
assert a call site existed by reading that one file. It is now eight modules:
the screen primitives, the bearings, the looking, the decision, the acting, the
observing and the blockers each have their own.

Both habits break quietly against a split. A `setattr` on one module leaves the
others running the real thing, so the test passes for the wrong reason or fails
for a reason that is not the code's fault. A source read of one file finds
nothing and reports a missing call site, which is indistinguishable from the
call site having been deleted.

So a patch names the symbol and this sets it everywhere it lives, and the
source is asked for by what it is. Same shape as `chat_lane_support`, for the
same reason.
"""
from __future__ import annotations

import pathlib
from types import ModuleType
from typing import Any


def pursuit_modules() -> tuple[pathlib.Path, ...]:
    """Every module the pursuit is built from, screen_pursuit.py first."""
    here = pathlib.Path(__file__).resolve().parent.parent / "core" / "skills"
    main = here / "screen_pursuit.py"
    return (main, *sorted(here.glob("screen_pursuit_*.py")))


def pursuit_source() -> str:
    """Their text, concatenated. The order is stable so slices stay stable."""
    return "\n".join(path.read_text(encoding="utf-8") for path in pursuit_modules())


def _loaded() -> list[ModuleType]:
    import importlib

    return [
        importlib.import_module(f"core.skills.{path.stem}") for path in pursuit_modules()
    ]


def patch_pursuit(monkeypatch: Any, name: str, value: Any, *, raising: bool = True) -> int:
    """Set ``name`` to ``value`` in every pursuit module that binds it.

    Returns how many modules were patched, so a test can assert the name was
    found at all rather than silently patching nothing.
    """
    patched = 0
    for module in _loaded():
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    if patched == 0 and raising:
        raise AttributeError(
            f"no pursuit module binds {name!r}; a patch that lands nowhere is a "
            "test asserting against the real implementation"
        )
    return patched


def pursuit_loop_source() -> str:
    """`pursue_on_screen` and the four functions lifted out of it.

    A test that reads `inspect.getsource(pursue_on_screen)` is asking about the
    loop, not about one file. The decision, the acting, the observing and the
    blocker clearing were closures inside it until the module went back under
    the size gate's ceiling, and reading only what is left reports their call
    sites as missing.
    """
    import importlib
    import inspect

    parts = []
    for module_name, function_name in (
        ("core.skills.screen_pursuit", "pursue_on_screen"),
        ("core.skills.screen_pursuit_observing", "observe_the_screen"),
        ("core.skills.screen_pursuit_blockers", "clear_what_blocks_the_run"),
        ("core.skills.screen_pursuit_decision", "decide_the_next_move"),
        ("core.skills.screen_pursuit_acting", "carry_out_the_move"),
    ):
        module = importlib.import_module(module_name)
        parts.append(inspect.getsource(getattr(module, function_name)))
    return "\n".join(parts)


def pursuit_function_source(name: str) -> str:
    """The source of one function of the pursuit, wherever it lives.

    These assertions used to slice the one module from one `async def` to the
    next. That works while two functions sit next to each other in one file and
    stops the moment either moves — the next `def` is then in a different
    module and the slice silently covers something else.
    """
    import ast

    for path in pursuit_modules():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                lines = text.splitlines(keepends=True)
                return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise AttributeError(f"no pursuit module defines {name!r}")
