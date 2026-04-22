"""Compatibility shim for legacy imports.

The canonical implementation lives in ``core.skills.computer_use``.
"""

from importlib import import_module

__all__ = ["ComputerUseParams", "ComputerUseSkill"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = import_module("core.skills.computer_use")
    return getattr(module, name)
