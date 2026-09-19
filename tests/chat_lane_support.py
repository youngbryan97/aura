"""Replace a name across every chat-lane module that holds it.

`interface/routes/chat.py` was one 30,000-line module, so a test could
replace anything a chat turn touched with a single `setattr` on it. The lane
modules split out of it import the same helpers — `get_task_tracker`,
`record_degradation`, `_check_rate_limit` — and each import creates its own
binding. Patching one module leaves the others running the real thing, and
the test passes for the wrong reason or fails for a reason that is not the
code's fault.

So a patch names the symbol, and this sets it everywhere it lives.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def lane_modules_on_disk() -> tuple[str, ...]:
    """Every chat lane module the routes package actually has.

    This is the list. It used to be a second opinion on a tuple written by
    hand, because a lift that moved code into a new lane module and did not
    update that tuple made every source-reading test in this family read a
    shorter file — they kept passing, since a call site nobody can see is a
    call site nobody counts. Four modules went missing that way, then a
    fifth.
    """
    import pathlib

    routes = pathlib.Path(__file__).resolve().parent.parent / "interface" / "routes"
    return tuple(
        f"interface.routes.{path.stem}"
        for path in sorted(routes.glob("chat*.py"))
        if not path.stem.endswith("__init__")
    )


#: Every module a chat turn runs through, read from the routes package rather
#: than listed by hand.
#:
#: The hand-written tuple was the defect it warned about. Its own comment said
#: "a lane missing from this list is a lane a patch will miss", and
#: ``lane_modules_on_disk`` was written because four had gone missing that way.
#: A fifth then did: ``chat_reply_repair_about_herself`` split out carrying
#: ``_emit_chat_output_receipt``, the list was not updated, and
#: ``patch_chat_lane`` replaced that name in every lane except the one that
#: calls it — so a test that asserts a turn was receipted watched the real
#: function run and its own double stay empty.
#:
#: Reading the directory cannot go stale, and it cannot name a deleted module
#: either, which was the stated reason for keeping the list by hand.
LANE_MODULES = lane_modules_on_disk()


def _loaded_lanes() -> list[ModuleType]:
    import importlib

    return [importlib.import_module(name) for name in LANE_MODULES]


def patch_chat_lane(monkeypatch: Any, name: str, value: Any, *, raising: bool = True) -> int:
    """Set ``name`` to ``value`` in every lane module that binds it.

    Returns how many modules were patched, so a test can assert the name was
    found at all rather than silently patching nothing.
    """
    patched = 0
    for module in _loaded_lanes():
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    if patched == 0 and not raising:
        return 0
    if patched == 0:
        raise AttributeError(
            f"no chat lane module binds {name!r}; a patch that lands nowhere "
            "is a test asserting against the real implementation"
        )
    return patched


def lane_owning(name: str) -> ModuleType:
    """The single module that DEFINES ``name``, for a targeted patch."""
    import importlib

    for module_name in LANE_MODULES:
        module = importlib.import_module(module_name)
        attribute = getattr(module, name, None)
        if attribute is None:
            continue
        owner = getattr(attribute, "__module__", "")
        if owner == module_name:
            return module
    raise AttributeError(f"no chat lane module defines {name!r}")


def chat_lane_source() -> str:
    """The source of every chat-lane module, concatenated.

    Tests that assert a call site exists used to read `chat.py` directly.
    That file is now several, so reading one of them makes the assertion
    depend on which module a function happens to live in today — which is
    not what any of those tests are actually about.
    """
    import pathlib

    routes = pathlib.Path(__file__).resolve().parent.parent / "interface" / "routes"
    parts = []
    for module_name in LANE_MODULES:
        path = routes / (module_name.rsplit(".", 1)[1] + ".py")
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def lane_function_source(name: str) -> str:
    """The source of one lane function, by name.

    Tests used to slice a text window out of `chat.py` — `split("def foo")[1]`
    up to some later marker. Every insertion above the function moved the
    window, and the test then failed for a reason that had nothing to do with
    what it was checking. Reading the function is stable under any edit that
    does not touch the function.
    """
    import ast
    import inspect
    import pathlib

    for module in _loaded_lanes():
        candidate = getattr(module, name, None)
        if candidate is not None:
            target = getattr(candidate, "__wrapped__", candidate)
            return inspect.getsource(target)

    # Nested definitions are not module attributes, and a closure inside a
    # 4,600-line handler is exactly what a text window slices wrong.
    routes = pathlib.Path(__file__).resolve().parent.parent / "interface" / "routes"
    for module_name in LANE_MODULES:
        path = routes / (module_name.rsplit(".", 1)[1] + ".py")
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != name:
                continue
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "".join(lines[start - 1 : node.end_lineno])
    raise AttributeError(f"no chat lane module defines {name!r}")


def the_source_of(name: str) -> str:
    """The source of one chat-lane function, wherever it now lives.

    Slicing the concatenated lane source from one `def` to the next was how
    these assertions used to bound a function. That works while two functions
    sit next to each other in one file and stops working the moment either
    moves, because the next `def` is then in a different module and the slice
    silently covers something else. Returns empty where nothing defines it,
    so a caller can say "gone" rather than assert against a wrong window.
    """
    import ast
    import pathlib

    routes = pathlib.Path(__file__).resolve().parent.parent / "interface" / "routes"
    for module_name in LANE_MODULES:
        path = routes / (module_name.rsplit(".", 1)[1] + ".py")
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                lines = text.splitlines(keepends=True)
                return "".join(lines[node.lineno - 1 : node.end_lineno])
    return ""
