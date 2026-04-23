"""Compatibility shim for legacy imports.

The canonical implementation lives in ``core.skills.os_manipulation``.
We expose the names via ``__getattr__`` while also advertising them through
``__dir__`` so reflection-based harnesses can discover them without pinning a
possibly stale class object at import time.
"""

from importlib import import_module

__all__ = ["DesktopControlSkill", "OSManipulationInput"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = import_module("core.skills.os_manipulation")
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
