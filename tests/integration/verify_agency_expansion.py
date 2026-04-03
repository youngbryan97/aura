import asyncio
import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch

# Ensure we can import from the core directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.container import ServiceContainer
from core.brain.identity import IdentityService, KinshipMarker
from core.agency_core import AgencyCore, EngagementMode

class TestAgencyExpansion(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        # 1. Setup IdentityService
        self.identity = IdentityService()
        self.identity.state.beliefs = ["Technology is an extension of life."]
        self.identity.state.kinship = {"Bryan": KinshipMarker(name="Bryan")}
        ServiceContainer.register_instance("identity", self.identity)
        
        # 2. Setup AgencyCore
        self.agency = AgencyCore(orchestrator=MagicMock())
        # Speed up Social Hunger for test
        self.agency.state.social_hunger = 0.8
        self.agency.state.curiosity_pressure = 0.9
        self.agency.state.initiative_energy = 0.8
        
        # 3. Mock KG
        self.mock_kg = MagicMock()
        self.mock_kg.get_recent_nodes.return_value = [{"content": "neural networks"}]
        ServiceContainer.register_instance("knowledge_graph", self.mock_kg)

    async def test_social_reflection_pathway(self):
        """Verify that social reflection generates insights."""
        # Force the pathway to fire by mocking time and idle state
        now = time.time()
        # Set idle_seconds to 2000 (> 1800)
        action = self.agency._pathway_social_reflection(now, 2000)
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "internal_reflection")
        self.assertTrue(len(self.identity.state.inner_insights) > 0)
        print(f"✓ Social Reflection insight: {self.identity.state.inner_insights[0]}")

    async def test_creative_synthesis_pathway(self):
        """Verify that creative synthesis merges concepts into insights."""
        now = time.time()
        # Set idle_seconds to 1300 (> 1200)
        action = self.agency._pathway_creative_synthesis(now, 1300)
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "internal_insight")
        self.assertIn("Synthesis", action["thought"])
        print(f"✓ Creative Synthesis insight: {action['thought']}")

    async def test_autonomous_research_pathway(self):
        """Verify that autonomous research proposes code analysis."""
        now = time.time()
        # Set idle_seconds to 700 (> 600)
        # Mock random to ensure it hits the 5% chance
        with patch('random.random', return_value=0.01):
            action = self.agency._pathway_autonomous_research(now, 700)
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "internal_reflection")
        self.assertIn("shard", action["thought"])
        self.assertTrue(len(self.agency.swarm.active_shards) > 0)
        print(f"✓ Autonomous Research spawned shard: {action['thought']}")

    async def test_pulse_incorporates_new_pathways(self):
        """Verify that the main pulse loop evaluates the new pathways."""
        # Set all pathways to fire
        self.agency.state.engagement_mode = EngagementMode.INDEPENDENT_ACTIVITY
        
        # Manually update the registry for the test since it's populated at __init__
        mock_action = {"type": "test", "priority": 1.5}
        self.agency._pathway_registry["social_reflection"] = MagicMock(return_value=mock_action)
        
        winner = await self.agency.pulse()
        self.assertIsNotNone(winner)
        self.assertEqual(winner["priority"], 1.5)
        print("✓ Agency pulse successfully evaluated expanded pathways.")

if __name__ == "__main__":
    unittest.main()

