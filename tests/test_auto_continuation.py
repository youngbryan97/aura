import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.orchestrator.mixins.message_handling import MessageHandlingMixin

class DummyStatus:
    is_processing = False

class DummyOrchestrator(MessageHandlingMixin):
    def __init__(self):
        self.status = DummyStatus()
        self._inference_gate = MagicMock()
        self._inference_gate.generate = AsyncMock()
        self._inference_gate.SILENCE_SENTINEL = "<|SILENCE|>"
        self.conversation_history = []
        
        # Async Lock mock
        self._lock = asyncio.Lock()
        self._last_emitted_fingerprint = ""
        
    def _is_user_facing_origin(self, origin):
        return origin == "user"
        
    async def _ensure_inference_gate_ready(self, context=None):
        pass
        
    def _get_fingerprint(self, text):
        return text
        
    def _extend_foreground_quiet_window(self, amt):
        pass
        
    def _publish_telemetry(self, data):
        pass
        
    def _record_message_in_history(self, message, role):
        self.conversation_history.append({"role": role, "content": message})

@pytest.mark.asyncio
async def test_auto_continuation_triggers():
    orchestrator = DummyOrchestrator()
    
    # We will simulate a truncation (ends with a letter) followed by a completion (ends with a period)
    # The _inference_gate.generate will be called twice.
    orchestrator._inference_gate.generate.side_effect = [
        "This is the first part of a very long sentence that just cuts off", # ends with 'f' (no punctuation)
        " right here. And this is the end." # ends with '.' (punctuation)
    ]
    
    # We also have to mock ServiceContainer and KernelInterface so they don't intercept
    import core.kernel.kernel_interface as ki
    import core.container as container
    
    class MockKernel:
        def is_ready(self): return False
        
    ki.KernelInterface.get_instance = MagicMock(return_value=MockKernel())
    container.ServiceContainer.get = MagicMock(return_value=None)
    
    # Mock will
    import core.will as will
    class MockWillDecision:
        def is_approved(self): return True
        outcome = MagicMock(value="approved")
        reason = ""
        constraints = None
        
    class MockWill:
        _started = True
        def decide(self, *args, **kwargs): return MockWillDecision()
        
    will.get_will = MagicMock(return_value=MockWill())
    
    # Pad string to > 200 chars to trigger continuation
    long_first_part = "This is the first part of a very long sentence that just cuts off. " * 5 + "and then it cuts off"
    long_second_part = " right here. And this is the end."
    
    orchestrator._inference_gate.generate.side_effect = [
        long_first_part,
        long_second_part
    ]
    
    response = await orchestrator._process_user_input_core("Tell me a story.", origin="user")
    
    assert orchestrator._inference_gate.generate.call_count == 2
    assert response == long_first_part + long_second_part
    print("\nAuto-continuation successfully concatenated the response!")
