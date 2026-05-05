import pytest

from core.environment.adapter import EnvironmentCapabilities, ensure_command_spec
from core.environment.command import ActionIntent, CommandSpec, CommandStep
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackTerminalGridAdapter


def test_adapter_declares_environment_id_and_capabilities():
    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    assert adapter.environment_id == "terminal_grid:nethack"
    assert isinstance(adapter.capabilities, EnvironmentCapabilities)
    assert adapter.capabilities.can_observe


def test_adapter_execute_accepts_only_command_spec():
    with pytest.raises(TypeError):
        ensure_command_spec("h")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_adapter_start_observe_execute_close_contract():
    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    compiler = NetHackCommandCompiler()
    await adapter.start(run_id="contract", seed=1)
    assert adapter.is_alive()
    obs = await adapter.observe()
    assert obs.environment_id == "terminal_grid:nethack"
    command = compiler.compile(ActionIntent(name="move", parameters={"direction": "west"}))
    result = await adapter.execute(command)
    assert result.ok
    assert result.observation_after is not None
    await adapter.close()
    await adapter.close()
    assert not adapter.is_alive()


def test_command_spec_validation_blocks_missing_fields():
    spec = CommandSpec(
        command_id="",
        environment_id="terminal_grid:nethack",
        intent=ActionIntent(name="wait"),
        preconditions=[],
        steps=[CommandStep("key", ".")],
        expected_effects=["turn_passed"],
    )
    with pytest.raises(ValueError):
        spec.validate()
