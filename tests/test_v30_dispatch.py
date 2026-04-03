"""tests/test_v30_dispatch.py
───────────────────────────
Verifies that spontaneous autonomous messages reach the primary target.
"""

import asyncio
import logging
import unittest
from unittest.mock import MagicMock, AsyncMock
from core.orchestrator.main import RobustOrchestrator
from core.utils.output_gate import AutonomousOutputGate
from core.container import ServiceContainer

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestV30Dispatch(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock dependencies
        self.orchestrator = MagicMock(spec=RobustOrchestrator)
        self.orchestrator.reply_queue = asyncio.Queue()
        self.orchestrator.status = MagicMock()
        self.orchestrator.status.running = True
        
        # Real OutputGate
        self.gate = AutonomousOutputGate(self.orchestrator)
        self.orchestrator.output_gate = self.gate
        
    async def test_spontaneous_bypass(self):
        """Spontaneous autonomous messages should reach primary target."""
        message = "Hello from the Void."
        metadata = {
            "autonomous": True,
            "spontaneous": True,
            # "force_user": True # The gate should now handle spontaneous=True even without force_user
        }
        
        # This should reach primary (reply_queue in our mock setup)
        # Note: AutonomousOutputGate.emit puts things in orchestrator.reply_queue if target is primary
        await self.gate.emit(message, origin="test", target="primary", metadata=metadata)
        
        # check if it's in the queue
        self.assertFalse(self.orchestrator.reply_queue.empty())
        item = await self.orchestrator.reply_queue.get()
        self.assertEqual(item, message)
        print("✅ Spontaneous bypass verified.")

    async def test_normal_autonomous_redirection(self):
        """Normal autonomous messages (non-spontaneous) should still be redirected to secondary."""
        message = "Technical log that should be hidden."
        metadata = {
            "autonomous": True,
            "spontaneous": False
        }
        
        # Clear queue
        while not self.orchestrator.reply_queue.empty(): self.orchestrator.reply_queue.get_nowait()
        
        await self.gate.emit(message, origin="test", target="primary", metadata=metadata)
        
        # Should be empty because target was changed to "secondary"
        self.assertTrue(self.orchestrator.reply_queue.empty())
        print("✅ Normal autonomous redirection still functioning.")

if __name__ == "__main__":
    unittest.main()
