import pytest

from core.environment import ActionIntent, EnvironmentKernel
from core.environment.replay import EnvironmentTraceReplay
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler, NetHackTerminalGridAdapter


@pytest.mark.asyncio
async def test_environment_kernel_observe_gate_act_trace_replay(tmp_path):
    trace_path = tmp_path / "kernel.jsonl"
    kernel = EnvironmentKernel(
        adapter=NetHackTerminalGridAdapter(force_simulated=True),
        state_compiler=NetHackStateCompiler(),
        command_compiler=NetHackCommandCompiler(),
        trace_path=trace_path,
    )
    await kernel.start(run_id="kernel", seed=1)
    try:
        frame = await kernel.step(ActionIntent(name="observe", expected_effect="information_gain"))
    finally:
        await kernel.close()
    assert frame.receipt is not None
    assert frame.gateway_decision is not None
    assert frame.gateway_decision.approved
    assert trace_path.exists()
    assert EnvironmentTraceReplay().load(trace_path).ok
