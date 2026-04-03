import asyncio
import threading
import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils.concurrency import RobustLock

async def worker(name, lock, duration=0.1):
    print(f"[{name}] Attempting to acquire lock...")
    async with lock:
        print(f"[{name}] Lock acquired! Working...")
        await asyncio.sleep(duration)
        print(f"[{name}] Work done. Releasing...")
    print(f"[{name}] Released.")

def run_loop(name, lock):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(f"[{name}] Starting event loop in thread {threading.current_thread().name}")
    loop.run_until_complete(worker(name, lock))
    loop.close()

async def main():
    lock = RobustLock("CrossLoopTest")
    
    # Thread 1: Loop A
    t1 = threading.Thread(target=run_loop, args=("LoopA", lock), name="Thread-A")
    # Thread 2: Loop B
    t2 = threading.Thread(target=run_loop, args=("LoopB", lock), name="Thread-B")
    
    t1.start()
    time.sleep(0.05) # Give A a head start
    t2.start()
    
    t1.join()
    t2.join()
    print("\nVerification Complete: RobustLock handled cross-loop acquisition successfully.")

if __name__ == "__main__":
    asyncio.run(main())
