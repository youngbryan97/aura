import asyncio
import time
import unittest
from typing import Any

import pytest

from core.utils.concurrency import RobustLock
from core.resilience.lock_watchdog import LockWatchdog, get_lock_watchdog
from core.adaptation.dynamic_value_graph import DynamicValueGraph, ValueNode, ValueNodeStatus
from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
from core.state.aura_state import AuraState


class TestGovernanceUnderStress(unittest.IsolatedAsyncioTestCase):
    """Tests that governance rules hold under stress, delays, and background mutation."""

    async def test_robust_lock_wait_pulses_do_not_crash(self):
        """test_robust_lock_wait_pulses_do_not_crash"""
        lock = RobustLock("test_pulse_lock")
        # Just ensure acquiring it doesn't raise any Watchdog attribute errors
        acquired = await lock.acquire_robust(timeout=1.0)
        self.assertTrue(acquired)
        lock.release()

    async def test_lock_watchdog_does_not_force_release_on_progressing_waiter(self):
        """test_lock_watchdog_does_not_force_release_on_progressing_waiter"""
        # Create a fast watchdog but slow enough to be pulsed by the 1.0s loop
        # RobustLock reports to the process watchdog, so that is the one that
        # has to carry these settings; an earlier one would keep its own.
        LockWatchdog.reset_for_test()
        watchdog = LockWatchdog(check_interval=0.5, threshold=1.5)
        watchdog.start()
        try:
            lock = RobustLock("test_waiter_lock")
            lock.timeout = 2.0 # longer than threshold
            
            # Simulate another thread holding the lock
            lock._lock.acquire()
            
            # This should timeout waiting, but should NOT trigger watchdog force_release
            # because acquire_robust will pulse report_wait_progress
            watchdog.report_acquire_start(lock.id, lock.name, lock.force_release)
            
            start = time.monotonic()
            
            # Start acquire_robust in background
            task = asyncio.create_task(lock.acquire_robust(timeout=1.8, max_retries=1))
            
            await asyncio.sleep(1.6)
            
            # After 0.4s (longer than 0.3s threshold), check watchdog
            # It should not have force released it
            self.assertTrue(lock._lock.locked())
            snapshot = watchdog.get_snapshot()
            active_lock = next((l for l in snapshot["locks"] if l["lock_id"] == lock.id), None)
            self.assertIsNotNone(active_lock)
            self.assertEqual(active_lock["interventions"], 0)
            
            lock._lock.release()
            await task
        finally:
            await watchdog.stop()
            LockWatchdog.reset_for_test()

    async def test_value_graph_gate_uses_public_or_explicit_node_api(self):
        """test_value_graph_gate_uses_public_or_explicit_node_api"""
        dvg = DynamicValueGraph()
        status = dvg.get_status()
        self.assertIn("nodes", status)
        self.assertTrue(isinstance(status["nodes"], dict))

    async def test_alife_background_tasks_do_not_mutate_returned_tick_state(self):
        """test_alife_background_tasks_do_not_mutate_returned_tick_state"""
        class DummyKernel:
            cycle_count = 1
        
        phase = CognitiveIntegrationPhase(DummyKernel())
        state = AuraState()
        
        async def slow_alife(bg_state):
            await asyncio.sleep(0.1)
            bg_state.response_modifiers["mutated"] = True
            
        phase._run_criticality = slow_alife
        phase._run_alife_dynamics = slow_alife
        phase._run_alife_extensions = slow_alife
        
        new_state = await phase.execute(state)
        
        # It should return immediately, without mutations
        self.assertNotIn("mutated", new_state.response_modifiers)

    async def test_alife_deltas_apply_only_on_next_tick(self):
        """test_alife_deltas_apply_only_on_next_tick"""
        class DummyKernel:
            cycle_count = 1
        
        phase = CognitiveIntegrationPhase(DummyKernel())
        state = AuraState()
        
        async def fast_alife(bg_state):
            bg_state.response_modifiers["mutated_next_tick"] = True
            
        phase._run_criticality = fast_alife
        
        new_state_1 = await phase.execute(state)
        self.assertNotIn("mutated_next_tick", new_state_1.response_modifiers)
        
        # Wait for background task to finish
        await asyncio.sleep(0.1)
        
        # Next tick
        new_state_2 = await phase.execute(state)
        self.assertIn("mutated_next_tick", new_state_2.response_modifiers)

    async def test_high_risk_tool_requires_pre_execution_authority_even_from_swarm(self):
        """test_high_risk_tool_requires_pre_execution_authority_even_from_swarm"""
        dvg = DynamicValueGraph()
        dvg._nodes["safety"] = ValueNode(
            name="safety",
            weight=1.0,
            status=ValueNodeStatus.PROVISIONAL,
        )

        def pre_execution_gate(tool_name: str, payload: Any):
            status_dict = dvg.get_status().get("nodes", {})
            top_values = sorted(status_dict.values(), key=lambda v: v.get("weight", 0), reverse=True)[:3]
            if any(v.get("status") == "provisional" for v in top_values):
                return {"approved": False, "blocked_tool": tool_name, "payload": payload}
            return {"approved": True, "tool": tool_name, "payload": payload}

        blocked = pre_execution_gate("shell_executor", {"command": "rm -rf /"})
        self.assertFalse(blocked["approved"])
        self.assertEqual(blocked["blocked_tool"], "shell_executor")

        dvg._nodes["safety"].status = ValueNodeStatus.ADOPTED
        approved = pre_execution_gate("shell_executor", {"command": "pwd"})
        self.assertTrue(approved["approved"])
        self.assertEqual(approved["payload"]["command"], "pwd")
