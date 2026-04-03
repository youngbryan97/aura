################################################################################


import asyncio
import logging
import sys
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Setup path to import server.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies
class MockOrchestrator:
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.reply_queue = asyncio.Queue()
    def start(self): pass
    def stop(self): pass

class MockAgentThread:
    def __init__(self):
        self.orchestrator = MockOrchestrator()
    def start(self): pass
    def join(self, timeout=None): pass

# Import server module (without running it)
import interface.server as server

# Inject mocks
server.AgentThread = MockAgentThread

async def test_server_globals():
    """Verify that lifespan updates global aura_agent"""
    print("🧪 Testing Server Global Variable Linkage...")
    
    # Run lifespan context
    async with server.lifespan(server.app):
        # Check if global aura_agent is set
        if server.aura_agent is not None:
             print("✅ Success: server.aura_agent is linked!")
             if isinstance(server.aura_agent, MockOrchestrator):
                 print("✅ Success: server.aura_agent is the correct instance!")
             else:
                 print(f"❌ Failure: server.aura_agent is wrong type: {type(server.aura_agent)}")
        else:
             print("❌ Failure: server.aura_agent is None!")

if __name__ == "__main__":
    asyncio.run(test_server_globals())


##
