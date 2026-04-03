################################################################################

import asyncio
from typing import Any
import logging
from core.container import ServiceContainer
from core.service_registration import register_all_services

class MockOrchestrator:
    def __init__(self):
        self.enqueued = []
    
    def enqueue_from_thread(self, message: Any, origin: str = "user"):
        if isinstance(message, dict) and "origin" not in message:
            message["origin"] = origin
        self.enqueued.append(message)

async def main():
    orc = MockOrchestrator()
    register_all_services()
    
    print("\n--- Test 1: Dict Queue Injection ---")
    orc.enqueue_from_thread({
        "content": "A sudden rush of insight regarding quantum computing!",
        "context": {"urgency": "HIGH", "emotion": "EXCITED"}
    }, origin="impulse")
    print("Enqueued:", orc.enqueued)
    assert orc.enqueued[0]["origin"] == "impulse"
    
    print("\n--- Test 2: AffectEngine Context Extraction ---")
    affect = ServiceContainer.get("affect_engine")
    if affect:
        print(f"Current Affect State: {affect.state.dominant_emotion}")
        await affect.modify(0.8, 0.8, 0.8, source="integration_test")
        print(f"New Affect State after trigger: {affect.state.dominant_emotion}")
    
    print("\n--- Test 3: Archiver ---")
    archiver = ServiceContainer.get("archive_engine")
    if archiver:
        res = await archiver.archive_vital_logs()
        print(f"Archive Result: {res}")

if __name__ == "__main__":
    asyncio.run(main())


##
