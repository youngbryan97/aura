################################################################################

"""tests/test_ghost_probe.py
Unit test for Ghost Probe deployment logic.
"""
import asyncio
import os
import unittest
from unittest.mock import MagicMock
from core.collective.probe_manager import ProbeManager
from core.skills.ghost_probe import GhostProbeSkill
from core.container import ServiceContainer

class TestGhostProbe(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        self.orchestrator = MagicMock()
        # Mock loop for run_in_executor
        self.orchestrator.loop = asyncio.get_running_loop()
        self.manager = ProbeManager(self.orchestrator)
        ServiceContainer.register_instance("probe_manager", self.manager)
        self.skill = GhostProbeSkill(self.orchestrator)

    async def asyncTearDown(self):
        """Ensure all probes are killed."""
        probe_ids = list(self.manager.probes.keys())
        for pid in probe_ids:
            try:
                await self.manager.cleanup_probe(pid)
            except Exception:
                pass

    
