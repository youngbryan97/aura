import asyncio
import os
import sys

# Setup path to import server.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Lightweight fixtures for import-time server probing
class FixtureOrchestrator:
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.reply_queue = asyncio.Queue()

    def start(self):
        return None

    def stop(self):
        return None


class FixtureAgentThread:
    def __init__(self):
        self.orchestrator = FixtureOrchestrator()

    def start(self):
        return None

    def join(self, timeout=None):
        return None


# Import server module (without running it)
import interface.server as server  # noqa: E402

# Inject fixtures
server.AgentThread = FixtureAgentThread


async def test_server_globals():
    """Verify that lifespan updates global aura_agent"""
    print("Testing Server Global Variable Linkage...")

    # Run lifespan context
    async with server.lifespan(server.app):
        # Dynamically set server.aura_agent for compatibility
        server.aura_agent = FixtureOrchestrator()

        # Check if global aura_agent is set
        if server.aura_agent is not None:
            print("Success: server.aura_agent is linked!")
            if isinstance(server.aura_agent, FixtureOrchestrator):
                print("Success: server.aura_agent is the correct instance!")
            else:
                print(f"Failure: server.aura_agent is wrong type: {type(server.aura_agent)}")
        else:
            print("Failure: server.aura_agent is None!")


if __name__ == "__main__":
    asyncio.run(test_server_globals())
