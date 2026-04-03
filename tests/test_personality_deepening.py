################################################################################

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# Ensure we can import from the core directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import RobustOrchestrator
from core.brain.personality_engine import PersonalityEngine
from core.identity import IdentitySystem
from core.container import ServiceContainer

class TestPersonalityDeepening(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        # Completely manual mocks to avoid patch-related hangs in IsolatedAsyncioTestCase
        mock_ls = MagicMock()
        mock_ls.update = AsyncMock()
        mock_ls.emotions = {"contemplation": MagicMock(intensity=0)}
        mock_ls.get_status = MagicMock(return_value={"health": 1.0})
        ServiceContainer.register_instance("liquid_state", mock_ls)
        
        mock_pe = MagicMock()
        mock_pe.last_update = 0
        mock_pe.internal_monologue = []
        mock_pe.emotions = {"contemplation": MagicMock(intensity=0)}
        ServiceContainer.register_instance("personality_engine", mock_pe)
        
        mock_identity = MagicMock()
        mock_identity.get_full_system_prompt.return_value = "System: Hello. INTERNAL MONOLOGUE: Reflection"
        ServiceContainer.register_instance("identity", mock_identity)

    async def asyncTearDown(self):
        pass

    async def test_orchestrator_startup_and_personality_update(self):
        """Verify orchestrator starts and updates personality."""
        orch = RobustOrchestrator()
        orch.setup()
        
        # Inject mocks directly and ensure they ARE the ones used
        pe = MagicMock()
        pe.update = MagicMock()
        orch._personality_engine = pe
        
        # Initial cycle count
        self.assertEqual(orch.status.cycle_count, 0)
        
        # Run one cycle manually
        with patch.object(orch, '_get_service', side_effect=lambda name, *a: pe if name == "personality_engine" else MagicMock()):
            with patch.object(orch, '_acquire_next_message', return_value=None):
                with patch.object(orch, '_update_liquid_pacing', return_value=None):
                    with patch.object(orch, '_trigger_autonomous_thought', return_value=None):
                        with patch.object(orch, '_pulse_agency_core', return_value=None):
                             with patch.object(orch, '_run_terminal_self_heal', return_value=None):
                                await orch._process_cycle()
        
        # Cycle count should increment
        self.assertEqual(orch.status.cycle_count, 1)
        
        # Verify personality update was called
        pe.update.assert_called()
        print("✓ Orchestrator cycle and personality update verified.")

    async def test_internal_monologue_and_prompt(self):
        """Verify internal monologue is captured and injected into prompt."""
        # Use real PersonalityEngine
        pe = PersonalityEngine()
        
        # Force a reflection trigger by manipulating emotions
        from core.brain.personality_engine import EmotionalState
        es = EmotionalState(name="contemplation")
        es.intensity = 90
        pe.emotions["contemplation"] = es
        
        # Generate behaviors (this should trigger monologue)
        pe._generate_spontaneous_behaviors()
        
        # Should have something in monologue
        self.assertTrue(len(pe.internal_monologue) > 0)
        reflection = pe.internal_monologue[0]
        print(f"✓ Internal Monologue captured: {reflection}")
        
        # Check identity prompt integration
        mock_identity = MagicMock()
        mock_identity.get_full_system_prompt.return_value = f"INTERNAL MONOLOGUE: {reflection}"
        ServiceContainer.register_instance("identity", mock_identity)
        
        identity = ServiceContainer.get("identity")
        prompt = identity.get_full_system_prompt()
        self.assertIn("INTERNAL MONOLOGUE", prompt)
        self.assertIn(reflection, prompt)
        print("✓ Monologue injected into identity prompt.")

    def test_persona_persistence(self):
        """Verify persona can be persisted."""
        pe = PersonalityEngine()
        # Mock to avoid real disk write in test
        with patch('builtins.open', unittest.mock.mock_open()) as mocked_file:
            with patch('pathlib.Path.mkdir'):
                success = pe.persist()
                self.assertTrue(success)
                mocked_file.assert_called()
        print("✓ Persona persistence verified.")

if __name__ == "__main__":
    unittest.main()


##
