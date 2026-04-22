import logging
from unittest.mock import AsyncMock

import pytest

from core.orchestrator import RobustOrchestrator


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IntegrationTest")


@pytest.mark.asyncio
async def test_full_system_loop(monkeypatch):
    """User input should flow through the current InferenceGate entry point."""
    logger.info("🚀 Starting Full Mind/Body/Language Integration Test...")

    orchestrator = RobustOrchestrator()
    orchestrator._inference_gate = AsyncMock()
    orchestrator._inference_gate.generate = AsyncMock(
        return_value="I am fully operational."
    )

    class _Kernel:
        def is_ready(self) -> bool:
            return False

    monkeypatch.setattr(
        "core.kernel.kernel_interface.KernelInterface.get_instance",
        staticmethod(lambda: _Kernel()),
    )

    user_input = (
        "Please run a thorough analysis of the system health and provide a detailed "
        "report on memory usage and skill registration status. I need a deep dive "
        "into the logs and an assessment of potential performance bottlenecks in "
        "the cognitive cycle."
    )
    logger.info("🗣️  Input: '%s'", user_input)

    response = await orchestrator.process_user_input(user_input)

    assert response == "I am fully operational."
    orchestrator._inference_gate.generate.assert_awaited_once()
    gate_call = orchestrator._inference_gate.generate.await_args
    assert gate_call.args[0] == user_input
    context = gate_call.kwargs["context"]
    assert context["origin"] == "user"
    assert context["is_background"] is False
    assert isinstance(context["history"], list)

    assert orchestrator.conversation_history[-2]["role"] == "user"
    assert orchestrator.conversation_history[-2]["content"] == user_input
    assert orchestrator.conversation_history[-1]["role"] == "assistant"
    assert orchestrator.conversation_history[-1]["content"] == response
