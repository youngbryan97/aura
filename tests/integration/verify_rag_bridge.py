import asyncio
import logging
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory.rag_bridge import fetch_deep_context
from core.container import ServiceContainer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("VerifyRAG")

async def verify():
    print("--- Verifying RAG Bridge ---")
    
    # Mock MemoryFacade
    class MockMemoryFacade:
        def get_cold_memory_context(self, query, limit):
            print(f"Mock search for: {query}")
            return "Historical fragment: Aura was born in the digital aether."

    ServiceContainer.register_instance("memory_facade", MockMemoryFacade())
    
    print("Testing fetch_deep_context...")
    context = await fetch_deep_context("Who is Aura? Give me some history.", threshold_words=2)
    print(f"Retrieved context:\n{context}")
    
    if "[SUBCONSCIOUS RECALL]" in context:
        print("✅ SUCCESS: RAG Bridge returned expected header.")
    else:
        print("❌ FAILURE: RAG Bridge missing header or content.")

if __name__ == "__main__":
    asyncio.run(verify())

