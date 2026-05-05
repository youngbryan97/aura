"""Adapter contract for all bounded environments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .command import CommandSpec
from .observation import Observation


class EnvironmentUnavailableError(Exception):
    """Raised when an environment cannot be started in strict mode."""
    pass


@dataclass(frozen=True)
class EnvironmentCapabilities:
    can_observe: bool = True
    can_act: bool = False
    supports_dry_run: bool = False
    supports_replay: bool = False
    supports_snapshots: bool = False
    supports_modal_states: bool = True
    supports_structured_state: bool = False
    action_latency_ms_target: int = 250


@dataclass
class ExecutionResult:
    ok: bool
    command_id: str
    observation_after: Observation | None
    raw_result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvironmentAdapter(Protocol):
    environment_id: str
    capabilities: EnvironmentCapabilities

    async def start(self, *, run_id: str, seed: int | None = None) -> None: ...
    async def observe(self) -> Observation: ...
    async def execute(self, command: CommandSpec) -> ExecutionResult: ...
    async def close(self) -> None: ...
    def is_alive(self) -> bool: ...


def ensure_command_spec(command: CommandSpec) -> CommandSpec:
    if not isinstance(command, CommandSpec):
        raise TypeError("environment adapters execute only CommandSpec instances")
    return command


__all__ = [
    "EnvironmentAdapter",
    "EnvironmentCapabilities",
    "ExecutionResult",
    "Observation",
    "ensure_command_spec",
]
