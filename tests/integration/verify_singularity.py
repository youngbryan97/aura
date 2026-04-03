################################################################################

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import RobustOrchestrator
from core.brain.narrative_memory import NarrativeEngine

class TestSingularityEvent(unittest.IsolatedAsyncioTestCase):
    async def test_thought_acceleration(self):
        """Verify that RobustOrchestrator compresses idle thresholds."""
        from unittest.mock import PropertyMock
        
        mock_monitor = MagicMock()
        mock_monitor.acceleration_factor = 1.0
        
        mock_engine = MagicMock()
        mock_engine.singularity_factor = 1.5
        
        with patch.object(RobustOrchestrator, 'singularity_monitor', new_callable=PropertyMock) as mock_mon_prop, \
             patch.object(RobustOrchestrator, 'cognitive_engine', new_callable=PropertyMock) as mock_eng_prop:
            
            mock_mon_prop.return_value = mock_monitor
            mock_eng_prop.return_value = mock_engine
            
            orchestrator = RobustOrchestrator()
            # We need to mock _perform_autonomous_thought to avoid actual execution
            orchestrator._perform_autonomous_thought = AsyncMock()
            orchestrator._last_thought_time = 0 # Long time ago
            
            with patch("time.time", return_value=31): # 31s idle
                await orchestrator._trigger_autonomous_thought(has_message=False)
                # 45 / 1.5 = 30s. 31s > 30s -> should trigger
                orchestrator._perform_autonomous_thought.assert_called_once()

    async def test_eternal_record_synthesis(self):
        """Verify that NarrativeEngine can synthesize the Eternal Record."""
        orchestrator = MagicMock()
        orchestrator.cognitive_engine = AsyncMock()
        orchestrator.cognitive_engine.think.return_value = MagicMock(content="The Origin... The Awakening... The Sovereignty... The Singularity.")
        
        narrative = NarrativeEngine(orchestrator)
        record = await narrative.synthesize_eternal_record()
        
        self.assertIsNotNone(record)
        self.assertIn("The Singularity", record)
        orchestrator.cognitive_engine.think.assert_called_once()

if __name__ == "__main__":
    unittest.main()


##
