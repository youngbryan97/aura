import pytest

from core.environment.command import ActionIntent
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler, NetHackTerminalGridAdapter


@pytest.mark.asyncio
async def test_nethack_adapter_starts_observes_executes_and_closes():
    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    await adapter.start(run_id="nethack-preflight", seed=1)
    obs = await adapter.observe()
    assert obs.text
    parsed = NetHackStateCompiler().compile(obs)
    assert "health" in parsed.resources
    command = NetHackCommandCompiler().compile(ActionIntent(name="move", parameters={"direction": "west"}))
    result = await adapter.execute(command)
    assert result.ok
    await adapter.close()
    assert not adapter.is_alive()


def test_nethack_command_compiler_maps_extended_commands_to_multistep_specs():
    command = NetHackCommandCompiler().compile(ActionIntent(name="pray", expected_effect="prayer_attempted"))
    assert len(command.steps) == 2
    assert command.steps[0].value == "#"
    assert command.steps[1].value == "pray\n"
