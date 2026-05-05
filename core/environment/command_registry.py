"""Environment command compiler registry."""
from __future__ import annotations

from .command import CommandCompiler


class CommandRegistry:
    def __init__(self) -> None:
        self._compilers: dict[str, CommandCompiler] = {}

    def register(self, compiler: CommandCompiler) -> None:
        self._compilers[compiler.environment_id] = compiler

    def get(self, environment_id: str) -> CommandCompiler:
        return self._compilers[environment_id]


_COMMAND_REGISTRY = CommandRegistry()


def get_command_registry() -> CommandRegistry:
    return _COMMAND_REGISTRY


__all__ = ["CommandRegistry", "get_command_registry"]
