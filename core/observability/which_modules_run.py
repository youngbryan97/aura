"""Which of her modules ran in this process, and when each function first did: F12's first measure.

Master Todo F12 asks of every consciousness module whether it drives what she does or says, measured by lesion. An
audit from outside asks the same of an architecture's named modules ("Real Computation or Merely Naming?",
arXiv:2610.41604). Before a lesion means anything the module has to run, and reading imports cannot say whether it
does: the package registers its own services, so a module nothing outside imports may run on every tick, and one
imported everywhere may never be called.

This watches with ``sys.monitoring`` (PEP 669). A start event is set only on the code objects of the watched
modules, and the first call of each function is recorded with its time and then disabled for that function, so once
a function has run it costs nothing. Modules imported after ``start`` are taken up every ``uptake_s`` seconds and
at each ``report``.
"""
from __future__ import annotations

import sys
import threading
import time
from types import CodeType, FunctionType, ModuleType
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["ModuleActivity", "get_module_activity", "start_if_asked"]

_NAME = "aura.which_modules_run"


def _code_objects(module: ModuleType) -> list[CodeType]:
    """Every function's code in a module: its functions, its classes' methods and properties, and what they nest."""
    found: list[CodeType] = []
    seen: set[int] = set()

    def add(code: CodeType) -> None:
        if id(code) in seen:
            return
        seen.add(id(code))
        found.append(code)
        for const in code.co_consts:
            if isinstance(const, CodeType):
                add(const)

    def from_value(value: Any) -> None:
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            add(value.__code__)
        elif isinstance(value, (staticmethod, classmethod)):
            from_value(value.__func__)
        elif isinstance(value, property):
            for part in (value.fget, value.fset, value.fdel):
                if part is not None:
                    from_value(part)

    for value in list(vars(module).values()):
        if isinstance(value, type) and value.__module__ == module.__name__:
            for member in list(vars(value).values()):
                from_value(member)
        else:
            from_value(value)
    return found


class ModuleActivity:
    """First calls of the functions in the modules whose names start with ``prefix``."""

    def __init__(self, prefix: str = "core.consciousness.", *, uptake_s: float = 30.0) -> None:
        self.prefix = prefix
        self.uptake_s = uptake_s
        self._stopping = threading.Event()
        self._lock = checked_lock("which_modules_run")
        self._tool: int | None = None
        self._watched: dict[str, list[CodeType]] = {}
        self._owner: dict[int, tuple[str, str]] = {}
        self._first: dict[tuple[str, str], float] = {}
        self.started_at = 0.0

    def start(self) -> bool:
        """Begin watching; False when every monitoring tool slot is taken."""
        monitoring = sys.monitoring
        for tool in (monitoring.COVERAGE_ID, 3, 4):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, _NAME)
                self._tool = tool
                break
        else:
            return False
        monitoring.register_callback(self._tool, monitoring.events.PY_START, self._started)
        self.started_at = time.time()
        self._take_up_modules()
        if self.uptake_s > 0:
            threading.Thread(target=self._keep_taking_up, name="which_modules_run", daemon=True).start()
        return True

    def _keep_taking_up(self) -> None:
        while not self._stopping.wait(self.uptake_s):
            if self._tool is None:
                return
            self._take_up_modules()

    def stop(self) -> None:
        self._stopping.set()
        if self._tool is None:
            return
        monitoring = sys.monitoring
        for codes in self._watched.values():
            for code in codes:
                monitoring.set_local_events(self._tool, code, 0)
        monitoring.register_callback(self._tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(self._tool)
        self._tool = None

    def _take_up_modules(self) -> None:
        for name, module in list(sys.modules.items()):
            if not name.startswith(self.prefix) or name in self._watched or not isinstance(module, ModuleType):
                continue
            codes = _code_objects(module)
            with self._lock:
                self._watched[name] = codes
                for code in codes:
                    self._owner[id(code)] = (name, code.co_qualname)
            tool = self._tool
            if tool is None:
                return
            for code in codes:
                sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)

    def _started(self, code: CodeType, _offset: int) -> Any:
        owner = self._owner.get(id(code))
        if owner is not None:
            with self._lock:
                self._first.setdefault(owner, time.time())
        return sys.monitoring.DISABLE

    def report(self) -> dict[str, Any]:
        """Per module: how many of its functions exist, which ran, and when each first did."""
        if self._tool is not None:
            self._take_up_modules()
        with self._lock:
            modules = {}
            for name, codes in sorted(self._watched.items()):
                ran = sorted((qual, at - self.started_at) for (mod, qual), at in self._first.items() if mod == name)
                modules[name] = {"functions": len(codes), "ran": len(ran), "first_calls": ran}
        never = [name for name, row in modules.items() if row["functions"] and not row["ran"]]
        return {"prefix": self.prefix, "watched_for_s": time.time() - self.started_at, "modules": modules,
                "modules_that_never_ran": never}


_ACTIVITY: ModuleActivity | None = None


def start_if_asked() -> ModuleActivity | None:
    """Watch her consciousness modules from launch when AURA_MODULE_ACTIVITY is set; the watch, or None."""
    global _ACTIVITY
    if _ACTIVITY is not None:
        return _ACTIVITY
    from core.runtime.flags import env_bool

    if not env_bool(
        "AURA_MODULE_ACTIVITY",
        default=False,
        description="Record which core.consciousness functions run, for Master Todo F12 (sys.monitoring).",
        owner="core/observability/which_modules_run.py",
    ):
        return None
    activity = ModuleActivity()
    if activity.start():
        _ACTIVITY = activity
    return _ACTIVITY


def get_module_activity() -> ModuleActivity | None:
    """The watch started at launch, if there is one."""
    return _ACTIVITY
