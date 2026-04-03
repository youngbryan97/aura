import asyncio
import logging
import time
import sys
import pytest
import os

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.orchestrator.main import RobustOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.SmokeTest")

