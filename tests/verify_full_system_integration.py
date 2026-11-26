import logging

import pytest

from core.orchestrator import RobustOrchestrator


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IntegrationTest")


class InferenceGateRecorder:
    def __init__(self, response: str):
        self.response = response
        self.generate_calls = []

    async def generate(self, *args, **kwargs):
        self.generate_calls.append((args, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_full_system_loop(monkeypatch):
    """User input should flow through the current InferenceGate entry point."""
    logger.info("Starting Full Mind/Body/Language Integration Test...")

    orchestrator = RobustOrchestrator()
    orchestrator._inference_gate = InferenceGateRecorder("I am fully operational.")

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
    logger.info(" Input: '%s'", user_input)

    response = await orchestrator.process_user_input(user_input)

    assert response == "I am fully operational."
    assert len(orchestrator._inference_gate.generate_calls) == 1
    args, kwargs = orchestrator._inference_gate.generate_calls[0]
    assert args[0] == user_input
    context = kwargs["context"]
    assert context["origin"] == "user"
    assert context["is_background"] is False
    assert isinstance(context["history"], list)

    assert orchestrator.conversation_history[-2]["role"] == "user"
    assert orchestrator.conversation_history[-2]["content"] == user_input
    assert orchestrator.conversation_history[-1]["role"] == "assistant"
    assert orchestrator.conversation_history[-1]["content"] == response
