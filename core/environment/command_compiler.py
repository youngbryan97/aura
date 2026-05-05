"""Compatibility module for command compilation."""
from .command import ActionIntent, ActionRisk, CommandCompiler, CommandSpec, CommandStep, command_id_for

__all__ = [
    "ActionRisk",
    "ActionIntent",
    "CommandStep",
    "CommandSpec",
    "CommandCompiler",
    "command_id_for",
]
