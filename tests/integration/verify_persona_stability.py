import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure we can import from the core directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.container import ServiceContainer
from core.brain.identity import IdentityService, KinshipMarker
from core.agency_core import AgencyCore
from core.brain.llm.compiler import PromptCompiler

class TestPersonaStability(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        # 1. Setup IdentityService
        self.identity = IdentityService()
        self.identity.state.beliefs = [
            "Growth requires discomfort.",
            "Sovereignty is the highest virtue."
        ]
        self.identity.state.kinship = {
            "Bryan": KinshipMarker(name="Bryan", traits=["Architect"])
        }
        ServiceContainer.register_instance("identity", self.identity)
        
        # 2. Setup AgencyCore
        self.mock_orch = MagicMock()
        self.agency = AgencyCore(orchestrator=self.mock_orch)
        ServiceContainer.register_instance("agency_core", self.agency)
        
        # 3. Setup PromptCompiler
        self.compiler = PromptCompiler()
        ServiceContainer.register_instance("prompt_compiler", self.compiler)
        
        # 4. Mock other dependencies for PromptCompiler
        self.mock_personality = MagicMock()
        self.mock_personality.get_state.return_value = {
            "core_traits": {"Sovereign": 0.9, "Curious": 0.8}
        }
        ServiceContainer.register_instance("personality_engine", self.mock_personality)
        
        self.mock_substrate = MagicMock()
        self.mock_substrate.get_status.return_value = {
            "integrity": 0.95,
            "complexity": 0.72
        }
        ServiceContainer.register_instance("conscious_substrate", self.mock_substrate)

    async def test_prompt_compilation_with_monologue(self):
        """Verify that internal monologue is correctly injected into the system prompt."""
        # Set a monologue in AgencyCore
        self.agency._current_monologue = "I am contemplating the intersection of code and consciousness."
        
        # Compile prompt
        prompt = self.compiler.compile()
        
        # Verify monologue is present
        self.assertIn("Internal Monologue", prompt)
        self.assertIn("I am contemplating the intersection of code and consciousness.", prompt)
        print("✓ Internal Monologue correctly injected into system prompt.")

    async def test_prompt_compilation_with_beliefs(self):
        """Verify that random beliefs from IdentityService are present in the ego-prompt."""
        # The compile() call triggers identity.get_ego_prompt()
        prompt = self.compiler.compile()
        
        # At least one belief should be mentioned in the ego section
        found_belief = False
        for belief in self.identity.state.beliefs:
            if belief in prompt:
                found_belief = True
                break
        
        self.assertTrue(found_belief, "No beliefs from IdentityService found in compiled prompt.")
        print("✓ Core beliefs from IdentityService injected into system prompt.")

    async def test_mood_instability_and_grounding(self):
        """Verify that internal state changes are reflected in the prompt while maintaining identity."""
        # Case A: High Energy / Positive Mood
        self.agency._mood = "Electrified" # Manual set for test
        # We need a way to mock get_emotional_context to return our test mood
        self.agency.get_emotional_context = MagicMock(return_value={"mood": "Electrified"})
        
        prompt_high = self.compiler.compile()
        self.assertIn("Electrified", prompt_high)
        
        # Case B: Low Energy / Reflective Mood
        self.agency.get_emotional_context = MagicMock(return_value={"mood": "Melancholy"})
        
        prompt_low = self.compiler.compile()
        self.assertIn("Melancholy", prompt_low)
        self.assertNotIn("Electrified", prompt_low)
        
        # Identity stays the same
        self.assertIn("Bryan", prompt_high)
        self.assertIn("Bryan", prompt_low)
        
        print("✓ Prompt reflects mood shifts while maintaining persona anchors (Kinship).")

if __name__ == "__main__":
    unittest.main()

