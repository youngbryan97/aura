import pytest
from core.environment.adapter import EnvironmentAdapter, ExecutionResult
from core.environment.observation import Observation

class ScriptedTerminalAdapter(EnvironmentAdapter):
    environment_id = "terminal_grid:test"

    def __init__(self, screens):
        self.screens = list(screens)
        self.index = 0
        self.executed = []
        self.mode = "fixture_replay"
        self.run_id = None
        self.alive = False

    async def start(self, *, run_id: str, seed: int | None = None) -> None:
        self.run_id = run_id
        self.alive = True

    async def observe(self) -> Observation:
        text = self.screens[min(self.index, len(self.screens) - 1)]
        return Observation(
            environment_id=self.environment_id,
            run_id=self.run_id or "test_run",
            sequence_id=self.index,
            raw=text,
            text=text,
            metadata={"mode": self.mode, "simulated": False},
        )

    async def execute(self, command) -> ExecutionResult:
        if not hasattr(command, 'command_id'):
            raise ValueError("Expected CommandSpec")
        self.executed.append(command)
        self.index += 1
        obs = await self.observe()
        return ExecutionResult(ok=True, command_id=command.command_id, observation_after=obs)

    async def close(self) -> None:
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive

@pytest.fixture
def fake_adapter():
    return ScriptedTerminalAdapter(["screen1", "screen2", "screen3"])
