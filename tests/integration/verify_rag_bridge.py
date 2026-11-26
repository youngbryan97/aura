import asyncio
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.memory.rag_bridge import fetch_deep_context
from core.container import ServiceContainer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("VerifyRAG")

async def verify():
    print("--- Verifying RAG Bridge ---")
    
    class MemoryFacadeFixture:
        def search(self, query, limit):
            print(f"Fixture search for: {query}")
            return [
                {
                    "text": "Historical fragment: Aura was born in the digital aether.",
                    "timestamp": time.time(),
                    "similarity_score": 0.91,
                    "reinforcement_count": 2,
                }
            ][:limit]

    ServiceContainer.register_instance("memory_facade", MemoryFacadeFixture())
    
    print("Testing fetch_deep_context...")
    context = await fetch_deep_context("Who is Aura? Give me some history.", threshold_words=2)
    print(f"Retrieved context:\n{context}")
    
    if "[SUBCONSCIOUS TEMPORAL RECALL]" in context:
        print("SUCCESS: RAG Bridge returned expected header.")
    else:
        print("FAILURE: RAG Bridge missing header or content.")
        raise AssertionError("RAG Bridge missing temporal recall header")

if __name__ == "__main__":
    asyncio.run(verify())
