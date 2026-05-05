
import sys
import os
# Add current dir to path
sys.path.append("/Users/bryan/.aura/live-source")

from core.adapters.nethack_adapter import NetHackAdapter
import time

adapter = NetHackAdapter()
print("Starting adapter...")
adapter.start()
print("Adapter started.")
time.sleep(2)
screen = adapter.get_screen_text()
print("Screen Captured:")
print(screen)
print(f"Is alive: {adapter.is_alive()}")
adapter.close()
