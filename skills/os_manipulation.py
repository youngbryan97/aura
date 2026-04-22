"""Compatibility shim for legacy imports.

The canonical implementation lives in ``core.skills.os_manipulation``.
"""

from importlib import import_module

__all__ = ["DesktopControlSkill", "OSManipulationInput"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = import_module("core.skills.os_manipulation")
    return getattr(module, name)
