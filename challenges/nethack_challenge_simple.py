print("SHELL LEVEL")
import os
import sys
print(f"ALIVE (PID {os.getpid()})")
sys.stdout.flush()
import asyncio
import logging
from pathlib import Path
import traceback
import re

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
        try:
            obs = adapter.get_observation()
            obs_text = obs.get("text", "")
            print(f"DEBUG: Prompt: {repr(obs_text[:100])}...") # Print more of the screen
            
            full_prompt = f"{obs_text}\n\n[EMBODIED CONTROL CONTRACT] Somatic reflex matcher v3 ACTIVE."
            
            response = await orchestrator.process_user_input_priority(
                full_prompt, origin="embodied_motor_reflex"
            )
            
            if response:
                print(f"DEBUG: Response: {response}")
                # Simple parser to extract the action (usually a single character)
                # Aura usually responds with the key she wants to press.
                # If she says "I should move north", we might need a regex.
                # But for now, let's assume she returns the character.
                if "[SOMATIC:key=" in response:
                    action_match = re.search(r"key=['\"](.*?)['\"]", response)
                    action = action_match.group(1) if action_match else None
                else:
                    # If no explicit token, ONLY accept single characters from the movement set.
                    # NEVER parse the first letter of a sentence.
                    clean_resp = response.strip()
                    valid_moves = "hjklubny><." # NetHack standard moves
                    if len(clean_resp) == 1 and clean_resp.lower() in valid_moves:
                        action = clean_resp.lower()
                    else:
                        print(f"DEBUG: Ignoring non-action chatter: {repr(response)}")
                        action = None
                
                if action:
                    print(f"DEBUG: Executing action: {repr(action)}")
                    adapter.send_action(action)
                else:
                    # If she's just talking, we might want to send a 'wait' (.) or just skip
                    pass
            
            await asyncio.sleep(1) # Faster loop for reflexes
        except Exception as e:
            print(f"ERROR in loop: {e}")
            print("TRACEBACK:")
            print(traceback.format_exc())
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
