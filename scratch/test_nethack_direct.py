import asyncio
from core.container import ServiceContainer
from core.adapters.nethack_adapter import NetHackAdapter

async def test_nethack():
    adapter = NetHackAdapter()
    print("Starting adapter...")
    adapter.start()
    await asyncio.sleep(2)
    print("Getting screen...")
    screen = adapter.get_screen_text()
    print("Screen Content:")
    print("-" * 20)
    print(screen)
    print("-" * 20)
    print("Stopping adapter...")
    adapter.stop()

if __name__ == "__main__":
    asyncio.run(test_nethack())
