"""Compatibility shim for legacy imports.

The canonical implementation lives in ``core.skills.computer_use``.
We expose the names via ``__getattr__`` while also advertising them through
``__dir__`` so reflection-based harnesses can discover them without pinning a
possibly stale class object at import time.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.skills.computer_use import ComputerUseParams, ComputerUseSkill

__all__ = ["ComputerUseParams", "ComputerUseSkill"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = import_module("core.skills.computer_use")
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
