import asyncio
import threading
import time
import unittest
from typing import List
from core.state_registry import get_registry, UnifiedState

class TestStateRegistryConcurrency(unittest.TestCase):
    def test_hybrid_concurrency(self):
        registry = get_registry()
        received_versions: List[int] = []
        lock = threading.Lock()

        # 1. Add Async Listener
        async def async_listener(state: UnifiedState):
            with lock:
                received_versions.append(state.version)
        
        registry.subscribe(async_listener)

        # 2. Add Sync Listener
        def sync_listener(state: UnifiedState):
            # This will be run in a thread by the dispatcher
            pass
        registry.subscribe(sync_listener)

        async def run_test():
            # Start dispatcher
            registry.ensure_dispatcher()
            
            # Update from main loop
            await registry.update(phi=0.1)
            
            # Update from multiple threads
            def thread_task(val):
                registry.sync_update(phi=val)
            
            threads = []
            for i in range(10):
                t = threading.Thread(target=thread_task, args=(i/10.0,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
                
            # Allow some time for dispatcher to catch up
            await asyncio.sleep(0.5)
            
            return received_versions

        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        versions = loop.run_until_complete(run_test())
        loop.close()

        print(f"Received versions: {versions}")
        
        # Verify monotonicity and reception
        self.assertGreater(len(versions), 0, "No versions received")
        # Since we use a queue, they MUST be in order of submission (roughly)
        # but version numbers are incremented under lock, so they must be unique and increasing.
        for i in range(len(versions) - 1):
            self.assertLess(versions[i], versions[i+1], f"Non-monotonic versions: {versions}")
            
        # Check instrumentation
        self.assertGreaterEqual(registry.update_count, 11)
        self.assertEqual(registry.failed_notifications, 0)

if __name__ == "__main__":
    unittest.main()
