"""A mend of hers loaded into the process she is running, without a restart (Master Todo F15).

Her own yes took a mend into her source, and the process she was running went on with the old code until she
restarted. A mend her loop makes changes the body of a function. Here the running function keeps its identity and
takes the new body: its ``__code__`` is replaced with the one compiled from the mended source. Every reference to it
(an import elsewhere, a bound method, a registered callback, an instance made before) runs the mended code from its
next call. Live-coding tools reload a function in place the same way (jurigged, Breuleux, 2021). Reloading the module
instead would leave each of those references on the old objects.

Only function bodies are loaded. When anything else in the file changed (a signature, a default, a decorator, a
statement at module or class level, a function added or removed), nothing is loaded and the change waits for her
next start, because a body that calls a helper the process never defined would fail when it ran. A function that
needs a different closure, or is a generator where it was not, is refused the same way. A load keeps the code it
replaced, so it can be undone.
"""
from __future__ import annotations

import ast
import copy
import inspect
import logging
import sys
from dataclasses import dataclass, field
from types import CodeType, FunctionType
from typing import Any

from core.runtime.effect_boundary import effect_sink
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.SelfModification.Loading")

#: The flags that say what kind of function a code object makes; a body swapped in must make the same kind.
_KIND = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_ITERABLE_COROUTINE


@dataclass(frozen=True)
class Swap:
    """One running function and the code it had before and has now."""

    module: str
    qualname: str
    function: FunctionType
    before: CodeType
    after: CodeType
    #: The exceptions the mended body raises by name, on purpose: one of these from it is the function answering.
    raises: frozenset[str] = frozenset()


@dataclass
class Loaded:
    """What one mend loaded into this process, and what of it waits for her next start."""

    module: str
    swaps: list[Swap] = field(default_factory=list)
    waits: list[str] = field(default_factory=list)
    undone: bool = False
    #: The goal of hers the mend came from, and the call it was mended for (core/runtime/a_call_that_failed.py).
    goal: str = ""
    mended_call: Any = None

    def undo(self) -> list[str]:
        """Put back the code each swap replaced; the functions put back."""
        if self.undone:
            return []
        for swap in self.swaps:
            swap.function.__code__ = swap.before
        self.undone = True
        with _LOCK:
            if self in _LOADS:
                _LOADS.remove(self)
        return [swap.qualname for swap in self.swaps]


_LOCK = checked_lock("into_the_running_process.loads")
_LOADS: list[Loaded] = []


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function in a module by its qualified name, as Python names it."""
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def walk(nodes: list[ast.AST], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                found[name] = node
                walk(node.body, f"{name}.<locals>.")
            elif isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.")
            else:
                # An if, a try, a with or a match at this level: what it holds keeps this level's names.
                walk([child for child in ast.iter_child_nodes(node)
                      if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case))], prefix)

    walk(list(getattr(tree, "body", [])), "")
    return found


def _skeleton(tree: ast.Module) -> str:
    """The module with every function body emptied: what must not change for a body to be loaded."""
    emptied = copy.deepcopy(tree)
    for node in ast.walk(emptied):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Pass()]
    return ast.dump(emptied)


def what_changed(before: str, after: str) -> tuple[list[str], str]:
    """The functions to load (qualified names), or why nothing can be: ``([...], "")`` or ``([], why)``.

    A function is loaded when its body changed or it moved in the file. A moved function runs as before, and its
    new line numbers keep her next fault's traceback true to the file.
    """
    try:
        old_tree, new_tree = ast.parse(before), ast.parse(after)
    except SyntaxError as exc:
        return [], f"the mended source does not parse: {exc.msg}"
    if _skeleton(old_tree) != _skeleton(new_tree):
        return [], ("code outside function bodies changed (a signature, a default, a decorator, a statement, or a "
                    "function added or removed)")
    old, new = _functions(old_tree), _functions(new_tree)
    changed = [name for name, node in new.items() if "<locals>" not in name and (
        ast.dump(ast.Module(body=node.body, type_ignores=[])) != ast.dump(ast.Module(body=old[name].body,
                                                                                    type_ignores=[]))
        or node.lineno != old[name].lineno)]
    return changed, ""


def _code_named(code: CodeType, qualname: str) -> CodeType | None:
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            if constant.co_qualname == qualname:
                return constant
            inner = _code_named(constant, qualname)
            if inner is not None:
                return inner
    return None


def _same_code(running: CodeType, made_from: CodeType) -> bool:
    """Whether the running code is what ``made_from`` compiles to, line numbers aside."""
    if running.co_code != made_from.co_code or running.co_names != made_from.co_names:
        return False
    if len(running.co_consts) != len(made_from.co_consts):
        return False
    for ours, theirs in zip(running.co_consts, made_from.co_consts, strict=True):
        if isinstance(ours, CodeType) and isinstance(theirs, CodeType):
            if not _same_code(ours, theirs):
                return False
        elif type(ours) is not type(theirs) or ours != theirs:
            return False
    return True


def _running_function(module: Any, qualname: str) -> FunctionType | None:
    """The function object this process runs for ``qualname``, under any staticmethod, classmethod, property or
    functools.wraps around it."""
    holder: Any = module
    for part in qualname.split("."):
        space = getattr(holder, "__dict__", None)
        holder = space.get(part) if isinstance(space, dict) and part in space else getattr(holder, part, None)
        if holder is None:
            return None
    if isinstance(holder, (staticmethod, classmethod)):
        holder = holder.__func__
    elif isinstance(holder, property):
        holder = holder.fget
    seen = 0
    while holder is not None and seen < 32:
        if isinstance(holder, FunctionType) and holder.__code__.co_qualname == qualname:
            return holder
        holder, seen = getattr(holder, "__wrapped__", None), seen + 1
    return None


def _raised_by_name(node: ast.AST) -> frozenset[str]:
    """The exception names a function's own ``raise`` statements name."""
    named: set[str] = set()
    for raised in (n for n in ast.walk(node) if isinstance(n, ast.Raise) and n.exc is not None):
        target = raised.exc.func if isinstance(raised.exc, ast.Call) else raised.exc
        if isinstance(target, ast.Name):
            named.add(target.id)
        elif isinstance(target, ast.Attribute):
            named.add(target.attr)
    return frozenset(named)


@effect_sink("self_modification.load_into_the_running_process", allowed_domains=("state_mutation",))
def load_into_this_process(module_name: str, before: str, after: str, *, goal: str = "",
                           mended_call: Any = None) -> Loaded:
    """Load a mend of ``module_name`` (its source ``before`` and ``after``) into the functions this process runs.

    All or nothing: either every changed body is swapped in or none is, so the process never runs half a mend. It
    changes the code she is running, so it runs only in a governed scope, as a module reload does
    (core/runtime/consequential_primitives.guarded_hot_reload). Undoing a load needs none: it only puts back what a
    governed load replaced.
    """
    loaded = Loaded(module=module_name, goal=goal, mended_call=mended_call)
    module = sys.modules.get(module_name)
    if module is None:
        loaded.waits.append(f"{module_name} is not loaded in this process; it is read from the mended file when "
                            "it is first imported")
        return loaded
    names, why = what_changed(before, after)
    if why:
        loaded.waits.append(f"{module_name}: {why}; it waits for her next start")
        return loaded
    if not names:
        loaded.waits.append(f"{module_name}: no function body changed, so there is nothing to load")
        return loaded
    filename = str(getattr(module, "__file__", "") or module_name)
    try:
        compiled = compile(after, filename, "exec", dont_inherit=True)
        made_from = compile(before, filename, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        loaded.waits.append(f"{module_name} does not compile: {exc}")
        return loaded
    planned: list[Swap] = []
    nodes = _functions(ast.parse(after))
    for name in names:
        new_code = _code_named(compiled, name)
        function = _running_function(module, name)
        if new_code is None or function is None:
            loaded.waits.append(f"{module_name}.{name} is not a function this process runs; nothing is loaded")
            return loaded
        old_code = function.__code__
        expected = _code_named(made_from, name)
        if expected is None or not _same_code(old_code, expected):
            # The process runs other code than the mend was made on (her source moved since she started), so the
            # mended body would bring that difference with it.
            loaded.waits.append(f"{module_name}.{name} is not running the code the mend was made on; nothing is "
                                "loaded, and it waits for her next start")
            return loaded
        if old_code.co_freevars != new_code.co_freevars or (old_code.co_flags & _KIND) != (new_code.co_flags & _KIND):
            loaded.waits.append(f"{module_name}.{name} needs a different closure or kind of function; nothing is "
                                "loaded, and it waits for her next start")
            return loaded
        planned.append(Swap(module_name, name, function, old_code, new_code, _raised_by_name(nodes[name])))
    for swap in planned:
        swap.function.__code__ = swap.after
    loaded.swaps = planned
    if planned:
        with _LOCK:
            _LOADS.append(loaded)
        logger.info("🩹 Loaded into her running process: %s", ", ".join(f"{module_name}.{s.qualname}" for s in planned))
    return loaded


def loads() -> list[Loaded]:
    """What has been loaded into this process and not undone, oldest first."""
    with _LOCK:
        return list(_LOADS)


def the_load_holding(module_name: str, qualname: str) -> Loaded | None:
    """The load that swapped this function, if one did."""
    with _LOCK:
        for loaded in reversed(_LOADS):
            if any(s.module == module_name and s.qualname == qualname for s in loaded.swaps):
                return loaded
    return None


def _arguments(call: Any) -> tuple[Any, ...] | None:
    if call is None or isinstance(call, str):
        return None
    return (getattr(call, "positional", None), getattr(call, "keywords", None), getattr(call, "star_args", None),
            getattr(call, "star_kwargs", None))


def undo_if_it_faulted(raised_in: tuple[str, str] | None, error_type: str, call: Any = None) -> Loaded | None:
    """Undo the load holding the function a fault was raised in, when the fault says the mend did not hold.

    It did not hold when the call it was mended for fails again with the same arguments, or when the function raises
    an exception its mended body does not raise by name, which the mend may have caused. An exception the body
    raises by name, for other arguments, is the function answering as it was written to.
    """
    if not raised_in:
        return None
    module_name, qualname = raised_in
    loaded = the_load_holding(module_name, qualname)
    if loaded is None:
        return None
    name = str(error_type or "").rsplit(".", 1)[-1]
    swap = next(s for s in loaded.swaps if s.module == module_name and s.qualname == qualname)
    again = _arguments(call) is not None and _arguments(call) == _arguments(loaded.mended_call)
    if name in swap.raises and not again:
        return None
    loaded.undo()
    logger.warning("🩹 Undone in her running process: %s.%s raised %s after its mend was loaded%s.", module_name,
                   qualname, name, f" ({loaded.goal})" if loaded.goal else "")
    return loaded


def undo_every_load_for_test() -> None:
    for loaded in loads():
        loaded.undo()


__all__ = ["Loaded", "Swap", "load_into_this_process", "loads", "the_load_holding", "undo_if_it_faulted",
           "what_changed"]
