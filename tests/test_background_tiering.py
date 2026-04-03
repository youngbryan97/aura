import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Aura.Test")

class TestBackgroundTiering(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from core.brain.llm_health_router import HealthAwareLLMRouter
        self.router = HealthAwareLLMRouter()
        self.router.logger = logger
        
        # Manually inject endpoints to avoid real registration overhead/side effects
        self.router.endpoints = {
            "local": MagicMock(name="local", model_name="32B", is_healthy=True, tier="local", is_local=True),
            "local_deep": MagicMock(name="local_deep", model_name="72B", is_healthy=True, tier="local_deep", is_local=True),
            "api_fast": MagicMock(name="api_fast", model_name="7B-Cloud", is_healthy=True, tier="api_fast", is_local=False),
            "api_deep": MagicMock(name="api_deep", model_name="GPT-4", is_healthy=True, tier="api_deep", is_local=False),
            "local_fast": MagicMock(name="local_fast", model_name="7B-Local", is_healthy=True, tier="local_fast", is_local=True)
        }
        
        for name, ep in self.router.endpoints.items():
            ep.name = name # Ensure name is set correctly
            ep.is_available.return_value = True
            
        self.router._call_endpoint = AsyncMock(return_value={"ok": True, "text": "Mocked Response"})

    async def test_automatic_background_tiering_by_flag(self):
        """Verify that is_background=True forces the tertiary tier."""
        with patch.object(self.router, '_call_endpoint', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"ok": True, "text": "Mocked Response"}
            
            await self.router.think("Hello", is_background=True)
            
            # Background routing should use the local 7B brainstem first.
            called_ep = mock_call.call_args[0][0]
            self.assertEqual(called_ep.name, "local_fast")
            self.assertEqual(called_ep.tier, "local_fast")

    async def test_automatic_background_tiering_by_origin(self):
        """Verify that origin='metabolic' forces the tertiary tier."""
        with patch.object(self.router, '_call_endpoint', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"ok": True, "text": "Mocked Response"}
            
            await self.router.think("Hello", origin="metabolic_cycle")
            
            called_ep = mock_call.call_args[0][0]
            self.assertEqual(called_ep.name, "local_fast")

    async def test_background_override_is_demoted(self):
        """Background tasks must stay on the 7B path even if they request primary."""
        with patch.object(self.router, '_call_endpoint', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"ok": True, "text": "Mocked Response"}
            
            await self.router.think("Hello", prefer_tier="primary", is_background=True)
            
            called_ep = mock_call.call_args[0][0]
            self.assertEqual(called_ep.name, "local_fast")

    async def test_background_inference_is_suppressed_while_foreground_user_turn_is_active(self):
        """Background jobs should back off instead of contending with an active user reply."""
        mock_orch = MagicMock()
        mock_orch.status.is_processing = True
        mock_orch._current_origin = "api"
        mock_orch._current_task_is_autonomous = False
        mock_orch._foreground_user_quiet_until = 0.0

        with patch("core.container.ServiceContainer.get", return_value=mock_orch):
            result = await self.router.think("Hello", origin="sovereign_pruner", is_background=True)

        self.assertIsNone(result)
        self.router._call_endpoint.assert_not_called()

    async def test_background_inference_is_suppressed_during_quiet_window(self):
        """Background jobs should also back off immediately after a user-facing turn completes."""
        mock_orch = MagicMock()
        mock_orch.status.is_processing = False
        mock_orch._current_origin = ""
        mock_orch._current_task_is_autonomous = False
        mock_orch._foreground_user_quiet_until = 9999999999.0

        with patch("core.container.ServiceContainer.get", return_value=mock_orch):
            result = await self.router.think("Hello", origin="sovereign_pruner", is_background=True)

        self.assertIsNone(result)
        self.router._call_endpoint.assert_not_called()

if __name__ == "__main__":
    unittest.main()
