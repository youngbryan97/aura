"""Terminal-grid environment base adapter."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from core.environment.adapter import EnvironmentCapabilities, ExecutionResult, ensure_command_spec
from core.environment.command import CommandSpec
from core.environment.observation import Observation


@dataclass
class TerminalGridScreen:
    width: int = 80
    height: int = 24
    text: str = ""


class TerminalGridAdapter:
    environment_id = "terminal_grid:generic"
    capabilities = EnvironmentCapabilities(
        can_observe=True,
        can_act=True,
        supports_dry_run=True,
        supports_replay=True,
        supports_modal_states=True,
        supports_structured_state=True,
    )

    def __init__(self, *, screen_text: str | None = None) -> None:
        self.run_id = ""
        self.sequence_id = 0
        self.screen = TerminalGridScreen(text=screen_text or self._default_screen())
        self.commands: list[str] = []
        self._alive = False

    async def start(self, *, run_id: str, seed: int | None = None) -> None:
        self.run_id = run_id
        self.sequence_id = 0
        self._alive = True

    async def observe(self) -> Observation:
        self.sequence_id += 1
        return Observation(
            environment_id=self.environment_id,
            run_id=self.run_id,
            sequence_id=self.sequence_id,
            timestamp=time.time(),
            context_id="terminal",
            raw=self.screen.text,
            text=self.screen.text,
            structured={"width": self.screen.width, "height": self.screen.height},
            metadata={"adapter": self.__class__.__name__},
        )

    async def execute(self, command: CommandSpec) -> ExecutionResult:
        ensure_command_spec(command)
        if not self._alive:
            return ExecutionResult(False, command.command_id, None, error="adapter_not_alive")
        for step in command.steps:
            self.commands.append(f"{step.kind}:{step.value}")
            if step.kind == "wait":
                await asyncio.sleep(min(0.01, step.timeout_s))
        observation = await self.observe()
        return ExecutionResult(True, command.command_id, observation, raw_result={"commands": list(self.commands[-5:])})

    async def close(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    @staticmethod
    def _default_screen() -> str:
        lines = [""] * 24
        lines[1] = "   --------"
        lines[2] = "   |......|"
        lines[3] = "   |..@...|"
        lines[4] = "   |......|"
        lines[5] = "   --------"
        lines[22] = "[Aura the Agent] St:10 Dx:10 Co:10 In:10 Wi:10 Ch:10 Neutral"
        lines[23] = "Dlvl:1 $:0 HP:12(12) Pw:7(7) AC:10 Xp:1/0 T:1"
        return "\n".join(line[:80].ljust(80) for line in lines)


__all__ = ["TerminalGridAdapter", "TerminalGridScreen"]
