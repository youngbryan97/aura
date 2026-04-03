import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in path
sys.path.append("/Users/bryan/.gemini/antigravity/scratch/autonomy_engine")

from core.brain.personality_engine import PersonalityEngine
from core.agency_core import AgencyCore, SovereignSwarm
from core.orchestrator.main import RobustOrchestrator

class TestPhase5Evolution(unittest.IsolatedAsyncioTestCase):
    def test_trait_mutation(self):
        print("\nTesting Trait Mutation...")
        engine = PersonalityEngine()
        engine.traits = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }
        engine.internal_monologue = ["I want to research more about AI", "wonder why things work"]
        
        # Manually trigger mutation (bypassing the 1-hour check for test)
        engine._mutate_traits()
        
        print(f"Updated Openness: {engine.traits['openness']}")
        self.assertGreater(engine.traits['openness'], 0.5)
        print("✓ Trait mutation influenced by monologue.")

    async def test_sovereign_swarm_property(self):
        print("\nTesting Sovereign Swarm Property...")
        # We don't need to instantiate the whole thing, just test the mixin logic
        with patch('core.container.get_container') as mock_container:
            orchestrator = RobustOrchestrator()
            mock_agency = MagicMock()
            mock_agency.swarm = "SovereignSwarmInstance"
            
            # Mock service container to return our agency core
            mock_container.return_value.get.side_effect = lambda x: mock_agency if x == "agency_core" else None
            
            print(f"Orchestrator sovereign_swarm property: {orchestrator.sovereign_swarm}")
            self.assertEqual(orchestrator.sovereign_swarm, "SovereignSwarmInstance")
            print("✓ sovereign_swarm property correctly resolved via AgencyCore.")

    async def test_social_reflection_rag(self):
        print("\nTesting Social Reflection RAG...")
        core = AgencyCore()
        
        # Mocking required services
        mock_identity = MagicMock()
        mock_identity.state.kinship = {"Bryan": 0.8}
        
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=[
            {"text": "Bryan said he likes neural networks.", "metadata": {"speaker": "user"}}
        ])
        
        with patch('core.container.ServiceContainer.get') as mock_get:
            mock_get.side_effect = lambda name, default=None: {
                "identity": mock_identity,
                "memory_facade": mock_memory
            }.get(name, default)
            
            # Mock spawn_shard as async
            core.swarm.spawn_shard = MagicMock(return_value=asyncio.Future())
            core.swarm.spawn_shard.return_value.set_result(True)
            
            # Reset cooldown
            core._last_social_reflection = 0
            
            insight = await core._pathway_social_reflection({}, idle_seconds=3600.0)
            
            if insight:
                print(f"Social Insight: {insight.get('thought')}")
                self.assertIn("Bryan", insight.get('thought'))
                self.assertIn("recalled", insight.get('thought'))
                print("✓ Social reflection incorporates memory search.")
            else:
                self.fail("Social reflection pathway returned None")

if __name__ == "__main__":
    unittest.main()

