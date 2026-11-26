################################################################################


import asyncio
import logging
import sys
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pytest

from core.container import ServiceContainer

# Setup path to import server.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Lightweight local dependencies
class MockOrchestrator:
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.reply_queue = asyncio.Queue()
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

class MockAgentThread:
    def __init__(self):
        self.orchestrator = MockOrchestrator()
        self.started = False
        self.joined_timeout = None

    def start(self):
        self.started = True
        self.orchestrator.start()

    def join(self, timeout=None):
        self.joined_timeout = timeout

# Import server module (without running it)
import interface.server as server

# Inject mocks
server.AgentThread = MockAgentThread

@pytest.mark.asyncio
async def test_server_globals(monkeypatch):
    """Verify server lifespan uses the current no-double-boot runtime contract."""
    print("Testing Server Lifespan Runtime Contract...")

    monkeypatch.setenv("AURA_GUI_PROXY", "1")

    async with server.lifespan(server.app):
        assert not hasattr(server, "aura_agent")
        assert server.main_loop is asyncio.get_running_loop()
        assert ServiceContainer.get("mycelial_network", default=None) is not None
        assert server._event_bridge_task is not None
        assert server._event_bridge_task.done() is False

    assert server.main_loop is None
    assert server._event_bridge_task is None

if __name__ == "__main__":
    asyncio.run(test_server_globals())


##
