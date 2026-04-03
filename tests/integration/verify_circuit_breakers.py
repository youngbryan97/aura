import asyncio
import unittest
from pathlib import Path
from core.resilience.immunity_hyphae import CircuitBreaker

class TestMetabolicCircuitBreakers(unittest.TestCase):
    def test_circuit_breaker_trip(self):
        cb = CircuitBreaker("test_component", threshold=3, window=60)
        
        # Initial state should be CLOSED
        self.assertEqual(cb.state, "CLOSED")
        
        # Report 2 failures
        cb.report_failure()
        cb.report_failure()
        self.assertEqual(cb.state, "CLOSED")
        
        # 3rd failure should trip it
        cb.report_failure()
        self.assertEqual(cb.state, "OPEN")
        self.assertTrue(cb.is_quarantined())

    def test_circuit_breaker_recovery(self):
        # Very short recovery time for testing
        cb = CircuitBreaker("test_component", threshold=1, recovery_time=0.1)
        
        cb.report_failure()
        self.assertEqual(cb.state, "OPEN")
        
        # Wait for recovery
        import time
        time.sleep(0.2)
        
        # State should be HALF-OPEN
        self.assertEqual(cb.state, "HALF-OPEN")
        self.assertFalse(cb.is_quarantined())
        
        # Success should close it
        cb.report_success()
        self.assertEqual(cb.state, "CLOSED")

if __name__ == "__main__":
    unittest.main()
