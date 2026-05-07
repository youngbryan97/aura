"""
Vision Service (The Eyes)
Runs in the Background (Sandbox).
Captures screen content and updates 'sensory_memory.json'.
"""
import time
import json
import base64
import os
import sys
import asyncio
from datetime import datetime

# Try to import mss for screenshots
try:
    import mss
    mss_available = True
except ImportError:
    mss_available = False

async def run_vision_loop():
    print("Vision Service Starting (Async)...")
    
    if not mss_available:
        print("Error: 'mss' not installed. Please run 'install_package mss'.")
        return

    sys.stdout.flush()
    
    # mss.mss() is a context manager, we wrap the whole loop to reuse the connection
    def _capture_and_save(sct, monitor):
        # 1. Capture Screen
        screenshot = sct.grab(monitor)
        # 2. Convert to PNG bytes
        png = mss.tools.to_png(screenshot.rgb, screenshot.size)
        # 3. Base64 Encode
        b64_data = base64.b64encode(png).decode('utf-8')
        # 4. Write to Shared Memory
        memory = {
            "timestamp": datetime.now().isoformat(),
            "type": "visual",
            "status": "active",
            "image_data": b64_data,
            "description": "Screen capture active. Vision analysis (LLaVA/Ollama) is initialized and waiting for integration."
        }
        with open("vision_memory.tmp", "w") as f:
            json.dump(memory, f)
        os.replace("vision_memory.tmp", "sensory_vision.json")
        return True

    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1] # Primary monitor
            while True:
                try:
                    await asyncio.to_thread(_capture_and_save, sct, monitor)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Frame captured.")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Vision Error: {e}")
                    sys.stdout.flush()
                
                await asyncio.sleep(5.0) # Non-blocking sleep
    except Exception as e:
        print(f"Fatal Vision Error: {e}")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(run_vision_loop())
    except KeyboardInterrupt:
        print("Vision Service Stopping.")
