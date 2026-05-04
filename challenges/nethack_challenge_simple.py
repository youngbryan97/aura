print("SHELL LEVEL")
import os
import sys
print(f"ALIVE (PID {os.getpid()})")
sys.stdout.flush()
import asyncio
print(f"DEBUG: asyncio imported (PID {os.getpid()})")
sys.stdout.flush()
import logging
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

print(f"DEBUG: Importing NetHackAdapter... (PID {os.getpid()})")
sys.stdout.flush()
from core.adapters.nethack_adapter import NetHackAdapter
print(f"DEBUG: Importing create_orchestrator... (PID {os.getpid()})")
sys.stdout.flush()
from core.orchestrator.main import create_orchestrator
print(f"DEBUG: Importing ServiceContainer... (PID {os.getpid()})")
sys.stdout.flush()
from core.container import ServiceContainer

async def run():
    print(f"DEBUG: Entering run() coroutine (PID {os.getpid()})")
    sys.stdout.flush()
    print(">>> SIMPLE CHALLENGE STARTING <<<")
    print(f"DEBUG: Calling create_orchestrator... (PID {os.getpid()})")
    sys.stdout.flush()
    orchestrator = create_orchestrator()
    print(f"DEBUG: create_orchestrator returned. (PID {os.getpid()})")
    sys.stdout.flush()
    await orchestrator.start()
    
    adapter = NetHackAdapter()
    adapter.start(name="AuraSimple")
    ServiceContainer.register_instance("nethack_adapter", adapter)
    
    print(">>> LOOP STARTING <<<")
    while True:
        obs = adapter.get_observation()
        obs_text = obs.get("text", "")
        print(f"DEBUG: Prompt: {repr(obs_text[:50])}")
        
        full_prompt = f"{obs_text}\n\n[EMBODIED CONTROL CONTRACT] Somatic reflex matcher v3 ACTIVE."
        
        response = await orchestrator.process_user_input_priority(
            full_prompt, origin="embodied_motor_reflex"
        )
        
        if response:
            print(f"DEBUG: Response: {response}")
        
        await asyncio.sleep(2)

print(f"DEBUG: Function definitions complete. (PID {os.getpid()})")
sys.stdout.flush()
print(f"DEBUG: REACHED LINE 54 (PID {os.getpid()})")
sys.stdout.flush()
print(f"DEBUG: __name__ = {__name__} (PID {os.getpid()})")
sys.stdout.flush()

if __name__ == "__main__":
    print(f"DEBUG: Starting asyncio.run(run())... (PID {os.getpid()})")
    sys.stdout.flush()
    asyncio.run(run())
