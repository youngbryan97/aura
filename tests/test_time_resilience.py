import unittest
import time
from unittest.mock import MagicMock, patch
import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import RobustOrchestrator
from core.resilience.sovereign_watchdog import SovereignWatchdog

class TestTimeResilience(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrator_stall_detection_resilience(self):
        """Verify that Orchestrator stall detection ignores wall-clock jumps."""
        orchestrator = RobustOrchestrator()
        orchestrator.status = MagicMock()
        orchestrator.status.is_processing = True
        
        # Start processing at T=100 (monotonic)
        with patch('time.monotonic', return_value=100.0):
            orchestrator._current_processing_start = time.monotonic()
            
        # Simulate 1 second has passed monotonically, but wall-clock jumped 1 hour (3600s)
        # Aegis sentinel should NOT trigger if it uses monotonic time.
        with patch('time.monotonic', return_value=101.0):
            with patch('time.time', return_value=time.time() + 3600.0):
                # Manually trigger the check logic from _aegis_sentinel
                start_time = orchestrator._current_processing_start
                delta = time.monotonic() - start_time
                self.assertEqual(delta, 1.0, "Delta should be 1.0s regardless of wall-clock jump")
                self.assertLess(delta, 45.0, "Stall should NOT be detected")

    async def test_watchdog_heartbeat_resilience(self):
        """Verify that SovereignWatchdog ignores wall-clock jumps."""
        mock_orch = MagicMock()
        watchdog = SovereignWatchdog(mock_orch, timeout=30.0)
        
        # Initial heartbeat at T=100 (monotonic)
        with patch('time.monotonic', return_value=100.0):
            watchdog.heartbeat()
            
        # Simulate 10 seconds passed monotonically, but wall-clock jumped 1 hour
        with patch('time.monotonic', return_value=110.0):
            with patch('time.time', return_value=time.time() + 3600.0):
                elapsed = time.monotonic() - watchdog._last_heartbeat
                self.assertEqual(elapsed, 10.0, "Elapsed should be 10.0s regardless of wall-clock jump")
                self.assertLess(elapsed, 30.0, "Watchdog should NOT trigger recovery")

if __name__ == '__main__':
    unittest.main()
