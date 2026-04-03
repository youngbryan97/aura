import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from core.brain.llm_health_router import HealthAwareLLMRouter, EndpointHealth

class TestLLMRoutingTiering(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = HealthAwareLLMRouter()
        
        # Mock 32B Endpoint (Fast)
        self.mock_32b = MagicMock()
        self.mock_32b.think = AsyncMock(return_value="32B response")
        self.router.register(
            name="Cortex",
            url="internal",
            model="cortex-32b",
            is_local=True,
            tier="local",
            client=self.mock_32b
        )
        
        # Mock 72B Endpoint (Heavy/Slow)
        self.mock_72b = MagicMock()
        self.mock_72b.think = AsyncMock(return_value="72B response")
        self.router.register(
            name="Solver",
            url="internal",
            model="solver-72b",
            is_local=True,
            tier="local_deep",
            client=self.mock_72b
        )

        # Mock 7B Brainstem Endpoint
        self.mock_7b = MagicMock()
        self.mock_7b.think = AsyncMock(return_value="7B response")
        self.router.register(
            name="Brainstem",
            url="internal",
            model="brainstem-7b",
            is_local=True,
            tier="local_fast",
            client=self.mock_7b
        )
        
        # Mock API Endpoint
        self.mock_api = MagicMock()
        self.mock_api.think = AsyncMock(return_value="API response")
        self.router.register(
            name="Gemini-Fast",
            url="cloud",
            model="gemini-2.0-flash",
            is_local=False,
            tier="api_fast",
            client=self.mock_api
        )

    async def test_primary_tier_excludes_72b(self):
        """Verify that 'primary' tier preference excludes the 72B model."""
        # Force 32B to fail to see if it falls through to 72B
        self.mock_32b.think.side_effect = Exception("32B failed")
        
        # Request with primary tier
        result = await self.router.generate_with_metadata(
            "Hello", prefer_tier="primary"
        )
        
        # Should fail closed locally, NOT auto-promote to 72B or cloud.
        self.assertEqual(result["endpoint"], "all_failed")
        
        # Verify 72B was never called
        self.mock_72b.think.assert_not_called()

    async def test_secondary_tier_allows_72b(self):
        """Verify that an explicit deep handoff allows the 72B model."""
        # Request with secondary tier
        result = await self.router.generate_with_metadata(
            "Complex task", prefer_tier="secondary", deep_handoff=True
        )
        
        # Should use 72B
        self.assertEqual(result["endpoint"], "Solver")
        self.assertEqual(result["text"], "72B response")

    async def test_no_tier_preference_excludes_72b(self):
        """Verify that without preference, it now DEFAULTS to 'primary' and excludes 72B."""
        # Force 32B and API to fail
        self.mock_32b.think.side_effect = Exception("32B failed")
        self.mock_api.think.side_effect = Exception("API failed")
        
        # Request without tier preference
        try:
            await self.router.generate_with_metadata("Hello")
        except Exception:
            pass # We expect failure because all PRIMARY endpoints failed
        
        # Verify 72B was NOT called (it used to be called in 'greedy' mode)
        self.mock_72b.think.assert_not_called()

    async def test_foreground_primary_skips_brainstem_and_uses_cloud_fallback(self):
        """Foreground chat must not silently degrade to the 7B brainstem."""
        self.mock_32b.think.side_effect = Exception("32B failed")

        result = await self.router.generate_with_metadata(
            "Hello",
            prefer_tier="primary",
            origin="user",
            allow_cloud_fallback=True,
        )

        self.assertEqual(result["endpoint"], "Gemini-Fast")
        self.assertEqual(result["text"], "API response")
        self.mock_7b.think.assert_not_called()

    async def test_gui_report_prefers_last_foreground_endpoint_over_background(self):
        """Background telemetry should not overwrite the visible conversational tier."""
        await self.router.generate("Hello", prefer_tier="primary", origin="user")
        await self.router.generate("Idle thought", prefer_tier="tertiary", origin="system", is_background=True)

        report = self.router.get_health_report()
        self.assertEqual(report["current_tier"], "Cortex (32B)")
        self.assertEqual(report["active_endpoint"], "Cortex")
        self.assertEqual(report["background_endpoint"], "Brainstem")

    async def test_background_quiet_window_blocks_brainstem_until_cortex_ready(self):
        """Boot quiet-window protection should keep brainstem offline until Cortex finishes warming."""
        fake_gate = MagicMock()
        fake_gate.get_conversation_status.return_value = {
            "conversation_ready": False,
            "state": "warming",
            "warmup_in_flight": True,
        }
        fake_gate._background_local_deferral_reason.return_value = "cortex_startup_quiet"

        def _fake_get(name, default=None):
            if name == "inference_gate":
                return fake_gate
            return default

        with patch.object(HealthAwareLLMRouter, "_foreground_quiet_window_active", return_value=True):
            with patch("core.container.ServiceContainer.get", side_effect=_fake_get):
                result = await self.router.generate_with_metadata(
                    "Idle thought",
                    prefer_tier="tertiary",
                    origin="system",
                    is_background=True,
                )

        self.assertEqual(result["endpoint"], "suppressed")
        self.assertIn(
            result["error"],
            {"foreground_quiet_window", "background_deferred:cortex_startup_quiet"},
        )
        self.mock_7b.think.assert_not_called()

    async def test_gui_report_mapping(self):
        """Verify that get_health_report returns human-readable tiers."""
        # Trigger a success on 32B
        await self.router.generate("Hello", prefer_tier="primary")
        
        report = self.router.get_health_report()
        self.assertEqual(report["current_tier"], "Cortex (32B)")
        self.assertEqual(report["active_endpoint"], "Cortex")
        
        # Trigger a success on 72B
        await self.router.generate("Hello", prefer_tier="secondary", deep_handoff=True)
        report = self.router.get_health_report()
        self.assertEqual(report["current_tier"], "Solver (72B)")

if __name__ == "__main__":
    unittest.main()
