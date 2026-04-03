import asyncio
import time
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier

async def test_neural_reboot():
    print("Testing Neural Reboot in MLXLocalClient...")
    client = MLXLocalClient(model_path="test-model")
    
    # Mock the process
    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = True
    client._process = mock_proc
    client._init_done = True
    
    await client.reboot_worker()
    
    assert client._process is None
    assert client._init_done is False
    mock_proc.kill.assert_called_once()
    print("✅ Neural Reboot successfully purges worker.")

async def test_router_metal_recovery():
    print("Testing Router Metal Recovery trigger...")
    router = IntelligentLLMRouter()
    
    # Register a mock endpoint with a reboot_worker method
    mock_adapter = AsyncMock()
    mock_adapter.reboot_worker = AsyncMock()
    # Simulate a Metal failure on the first call
    mock_adapter.call.side_effect = [
        (False, "", {"error": "RESOURCE_EXHAUSTED: [metal::Device] error 3 - No such process"})
    ]
    
    endpoint = LLMEndpoint(name="MLX-Test", tier=LLMTier.PRIMARY)
    router.register_endpoint(endpoint)
    router.adapters["MLX-Test"] = mock_adapter
    
    # We need to mock ServiceContainer to avoid purge errors
    with patch("core.container.ServiceContainer.get", return_value=None):
        try:
            await router.think("test prompt", prefer_tier=LLMTier.PRIMARY)
        except Exception:
            pass # We expect it to fail after exhausts all endpoints
            
    # Check if proactive recovery was triggered
    mock_adapter.reboot_worker.assert_called_once()
    assert router._recovery_states["MLX-Test"] < time.time() + 20 # Should be ~15s
    print("✅ Router correctly triggers proactive reboot on Metal failure.")

if __name__ == "__main__":
    asyncio.run(test_neural_reboot())
    asyncio.run(test_router_metal_recovery())

