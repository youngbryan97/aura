import asyncio
import time
import unittest
from core.soma.resilience_engine import ResilienceEngine, ResilienceState

class TestResilienceEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = ResilienceEngine()

    async def test_failure_accumulation(self):
        """Test that repeated failures lead to DEPLETION."""
        # Initial state should be RESTED
        self.assertEqual(self.engine.profile.state, ResilienceState.RESTED)
        
        # Record a high-stakes failure
        self.engine.record_failure("planning", severity=0.8, stakes=1.0)
        # Should now be in FRICTION or STRAIN
        self.assertIn(self.engine.profile.state, [ResilienceState.FRICTION, ResilienceState.STRAIN])
        
        # Multiple failures
        for _ in range(5):
            self.engine.record_failure("tool_execution", severity=0.9, stakes=1.0)
            
        # Should definitely be in DEPLETION now
        self.assertEqual(self.engine.profile.state, ResilienceState.DEPLETION)
        self.assertGreater(self.engine.profile.depletion, 0.7)
        
        # Effort modifier should be 0.0 in DEPLETION
        self.assertEqual(self.engine.get_effort_modifier(), 0.0)

    async def test_success_recovery(self):
        """Test that success reduces frustration."""
        self.engine.record_failure("planning", severity=0.8, stakes=1.0)
        initial_frustration = self.engine.profile.frustration
        
        self.engine.record_success("planning")
        self.assertLess(self.engine.profile.frustration, initial_frustration)

    async def test_decay(self):
        """Test natural decay of frustration."""
        self.engine.record_failure("social", severity=0.5, stakes=0.5)
        f1 = self.engine.profile.frustration
        
        # Manually trigger decay logic
        self.engine.profile.last_update -= 1800 # simulate 30 minutes passed
        self.engine._apply_decay()
        self.assertLess(self.engine.profile.frustration, f1)
        self.assertAlmostEqual(self.engine.profile.frustration, f1 * 0.5, places=2)
        # Wait, _decay_loop is a 'while True'. I should test the math directly or extract the internal update.
        
    async def test_effort_modulation(self):
        """Test that effort modifier scales correctly."""
        # RESTED = 1.0
        self.assertEqual(self.engine.get_effort_modifier(), 1.0)
        
        # Force FRICTION
        self.engine.profile.frustration = 0.3
        self.engine._update_state()
        self.assertEqual(self.engine.profile.state, ResilienceState.FRICTION)
        self.assertEqual(self.engine.get_effort_modifier(), 0.85)
        
        # Force DEPLETION
        self.engine.profile.depletion = 0.8
        self.engine._update_state()
        self.assertEqual(self.engine.profile.state, ResilienceState.DEPLETION)
        self.assertEqual(self.engine.get_effort_modifier(), 0.0)

if __name__ == "__main__":
    unittest.main()

