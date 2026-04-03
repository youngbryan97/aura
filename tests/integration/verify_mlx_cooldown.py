import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.brain.llm.mlx_client import MLXLocalClient, get_mlx_client

class TestMLXCooldown(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reset globals in mlx_client
        import core.brain.llm.mlx_client as mlx_module
        mlx_module._GLOBAL_LAST_SWAP_TIME = 0.0
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = ""
        mlx_module._client_instances = {}

    @patch("core.brain.llm.mlx_client.os.path.realpath")
    @patch("core.brain.llm.mlx_client.time.time")
    @patch("core.brain.llm.mlx_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_swap_cooldown_enforced(self, mock_sleep, mock_time, mock_realpath):
        # Setup paths
        primary_path = "/models/32B"
        deep_path = "/models/72B"
        
        # Mock realpath to return our test paths, handling *args/**kwargs for pathlib compatibility
        path_map = {
            "primary": primary_path,
            "deep": deep_path,
            primary_path: primary_path,
            deep_path: deep_path
        }
        mock_realpath.side_effect = lambda x, *args, **kwargs: path_map.get(x, x)
        
        # Patch model_registry functions that are imported locally in mlx_client
        with patch("core.brain.llm.model_registry.get_model_path") as mock_get_path, \
             patch("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
             patch("core.brain.llm.model_registry.DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit"):
            mock_get_path.side_effect = lambda m=None: "primary" if "32B" in str(m) or m is None else "deep"
            
            # 1. First load (72B)
            client_72b = MLXLocalClient(model_path=deep_path)
            
            # Mock _process to avoid actual spawning
            client_72b._process = MagicMock()
            client_72b._process.is_alive.return_value = True
            client_72b._init_done = True
            
            # Manually set the "last heavy model" and "swap time" as if it just loaded
            import core.brain.llm.mlx_client as mlx_module
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = deep_path
            mlx_module._GLOBAL_LAST_SWAP_TIME = 1000.0
            
            # 2. Immediate load of 32B (should trigger cooldown)
            client_32b = MLXLocalClient(model_path=primary_path)
            
            # Mock time to be just 5 seconds after 72B load
            mock_time.return_value = 1005.0
            
            # We need to mock _spawn_worker to avoid actual process creation
            with patch.object(MLXLocalClient, "_spawn_worker", return_value=MagicMock()):
                # Call _ensure_worker_alive (which is what generate/think calls)
                # We mock it to think the process is NOT alive so it tries to spawn
                client_32b._process = None 
                
                # Mock the queue wait to succeed immediately with a real dictionary
                with patch("core.brain.llm.mlx_client.run_io_bound", new_callable=AsyncMock) as mock_run_io:
                    mock_run_io.return_value = {"status": "ok"}
                    await client_32b._ensure_worker_alive()
            
            # Verify sleep was called for the remaining 25 seconds
            mock_sleep.assert_awaited_with(25.0)
            print("✅ Cooldown of 25s was correctly enforced.")

if __name__ == "__main__":
    unittest.main()
