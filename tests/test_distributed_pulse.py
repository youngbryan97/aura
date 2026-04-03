################################################################################

"""tests/test_distributed_pulse.py
Unit test for PulseManager distributed discovery logic.
"""
import asyncio
import json
import socket
import unittest
from unittest.mock import MagicMock, patch
import time

from core.senses.pulse_manager import PulseManager

class TestPulseDiscovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = MagicMock()
        self.orchestrator.loop = None  # Will be set in async context
        self.orchestrator.peers = {}
        self.pulse_manager = PulseManager(self.orchestrator)
        self.pulse_manager.running = True

